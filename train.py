import argparse
import os
import time
import socket
import logging
from datetime import datetime
from functools import partial
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models
from models.losses import CrossEntropyLossSoft
from datasets.data import get_dataset, get_transform
from optimizer import get_optimizer_config, get_lr_scheduler
from utils import setup_logging, setup_gpus, save_checkpoint
from utils import AverageMeter, accuracy

import wandb

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    if args.wandb_log:
        wandb.init(project=args.project, entity="alelab", name=args.results_dir.split('/')[-1])
        wandb.config.update(args)
    hostname = socket.gethostname()
    setup_logging(os.path.join(args.results_dir, 'log_{}.txt'.format(hostname)))
    logging.info("running arguments: %s", args)

    best_gpu = setup_gpus()
    torch.cuda.set_device(best_gpu)
    torch.backends.cudnn.benchmark = True

    train_transform = get_transform(args.dataset, 'train')
    train_data = get_dataset(args.dataset, args.train_split, train_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    model = models.__dict__[args.model]([1,2,4,8,32], train_data.num_classes).cuda()
    model.bn_to_cuda()

    lr_decay = list(map(int, args.lr_decay.split(',')))
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = None
    best_prec1 = None
    if args.resume and args.resume != 'None':
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(best_gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay, checkpoint['epoch'])
            logging.info("loaded resume checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    elif args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(best_gpu))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_soft = CrossEntropyLossSoft().cuda()
    sum_writer = SummaryWriter(args.results_dir + '/summary')

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_loss, train_prec1, train_prec5 = forward(train_loader, model, criterion, criterion_soft, epoch, args, True,
                                                       optimizer, sum_writer)
        model.eval()
        train_loss, train_prec1, train_prec5, train_pearson, train_pearson_hl, train_l2, train_l2_hl, train_ce = forward(train_loader, model, criterion, criterion_soft, epoch, args, False)
        val_loss, val_prec1, val_prec5, val_pearson, val_pearson_hl, val_l2, val_l2_hl, val_ce = forward(val_loader, model, criterion, criterion_soft, epoch, args, False)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

        if best_prec1 is None:
            is_best = True
            best_prec1 = val_prec1[-1]
        else:
            is_best = val_prec1[-1] > best_prec1
            best_prec1 = max(val_prec1[-1], best_prec1)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            },
            is_best,
            path=args.results_dir + '/ckpt')

        if sum_writer is not None:
            sum_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
            for bw, tl, tp1, tp5, vl, vp1, vp5 in zip(bit_width_list, train_loss, train_prec1, train_prec5, val_loss,
                                                      val_prec1, val_prec5):
                sum_writer.add_scalar('train_loss_{}'.format(bw), tl, global_step=epoch)
                sum_writer.add_scalar('train_prec_1_{}'.format(bw), tp1, global_step=epoch)
                sum_writer.add_scalar('train_prec_5_{}'.format(bw), tp5, global_step=epoch)
                sum_writer.add_scalar('val_loss_{}'.format(bw), vl, global_step=epoch)
                sum_writer.add_scalar('val_prec_1_{}'.format(bw), vp1, global_step=epoch)
                sum_writer.add_scalar('val_prec_5_{}'.format(bw), vp5, global_step=epoch)
        if args.wandb_log:
            for bw, tl, tp1, tp5, tce, vl, vp1, vp5, vce in zip([1,2,4,8,32,], train_loss, train_prec1, train_prec5, train_ce, val_loss,
                                                      val_prec1, val_prec5, val_ce):
                wandb.log({f'train_loss_{bw}':tl, "epoch":epoch})
                wandb.log({f'train_acc_{bw}':tp1, "epoch":epoch})
                wandb.log({f'test_loss_{bw}':vl, "epoch":epoch})
                wandb.log({f'test_acc_{bw}':vp1, "epoch":epoch})
                wandb.log({f'train_CE_{bw}':tce, "epoch":epoch})
                wandb.log({f'test_CE_{bw}':vce, "epoch":epoch})
            if args.eval_constraint:
                for bw, tr_p, tr_p_hl, tr_l2, tr_l2_hl,te_p, te_p_hl, te_l2, te_l2_hl \
                in zip([2,4,8, 32], train_pearson, train_pearson_hl, train_l2, train_l2_hl, val_pearson, val_pearson_hl, val_l2, val_l2_hl):
                    for l in range(model.get_num_layers()):
                        wandb.log({f'train_pearson_layer_{l}_bw_{bw}':tr_p[l], "epoch":epoch})
                        wandb.log({f'train_pearson_layer_{l}_bw_{bw}_hl':tr_p_hl[l], "epoch":epoch})
                        wandb.log({f'test_pearson_layer_{l}_bw_{bw}':te_p[l], "epoch":epoch})
                        wandb.log({f'test_pearson_layer_{l}_bw_{bw}_hl':te_p_hl[l], "epoch":epoch})
                        wandb.log({f'train_l2_layer_{l}_bw_{bw}':tr_l2[l], "epoch":epoch})
                        wandb.log({f'train_l2_layer_{l}_bw_{bw}_hl':tr_l2_hl[l], "epoch":epoch})
                        wandb.log({f'test_l2_layer_{l}_bw_{bw}':te_l2[l], "epoch":epoch})
                        wandb.log({f'test_l2_layer_{l}_bw_{bw}_hl':te_l2_hl[l], "epoch":epoch})

        logging.info('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                     '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(
                         epoch, train_loss[-1], train_prec1[-1], train_prec5[-1], val_loss[-1], val_prec1[-1],
                         val_prec5[-1]))


def forward(data_loader, model, criterion, criterion_soft, epoch, args, training=True, optimizer=None, sum_writer=None):
    if training:
        bit_width_list = list(map(int, args.bit_width_list.split(',')))
        bit_width_list.sort()
    else:
        bit_width_list = [1, 2, 4, 8, 32]
    losses = [AverageMeter() for _ in bit_width_list]
    top1 = [AverageMeter() for _ in bit_width_list]
    top5 = [AverageMeter() for _ in bit_width_list]
    l2_meter = [[AverageMeter() for _ in range(model.get_num_layers())] for b in bit_width_list]
    l2_hl_meter = [[AverageMeter() for _ in range(model.get_num_layers())] for b in bit_width_list]
    pearson_meter = [[AverageMeter() for _ in range(model.get_num_layers())] for b in bit_width_list]
    pearson_hl_meter = [[AverageMeter() for _ in range(model.get_num_layers())] for b in bit_width_list]
    CE_meter = [AverageMeter() for b in bit_width_list]
    for i, (input, target) in enumerate(data_loader):
        if not training:
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                if args.eval_constraint:
                    bw = bit_width_list[-1]
                    model.apply(lambda m: setattr(m, 'wbit', bw))
                    model.apply(lambda m: setattr(m, 'abit', bw))
                    output = model(input)
                    act_full = model.get_activations(input)
                    act_full_norm = model.norm_act(act_full)
                    target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
                    for bw, am_l, am_t1, am_t5, l2m, l2hlm, rm, rhlm, cem in zip(bit_width_list, losses, top1,
                                                                    top5, l2_meter,l2_hl_meter,pearson_meter, pearson_hl_meter, CE_meter):
                        model.apply(lambda m: setattr(m, 'wbit', bw))
                        model.apply(lambda m: setattr(m, 'abit', bw))
                        output = model(input)
                        loss = criterion(output, target)
                        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                        act_q = model.get_activations(input)
                        act_q_norm = model.eval_layers(input, act_full)
                        am_l.update(loss.item(), input.size(0))
                        am_t1.update(prec1.item(), input.size(0))
                        am_t5.update(prec5.item(), input.size(0))
                        act_q = model.get_activations(input)
                        act_q_norm = model.norm_act(act_q)
                        act_full_fromq = model.eval_layers(input, act_q)
                        act_full_fromq_norm = model.norm_act(act_full_fromq)
                        cem.update(criterion_soft(output, target_soft).item(), input.size(0))

                        for l in range(model.get_num_layers()):
                            l2m[l].update(torch.mean(torch.square(act_q[l]-act_full_fromq[l])).item(), input.size(0))
                            l2hlm[l].update(torch.mean(torch.square(act_q[l]-act_full[l])).item(), input.size(0))
                            rm[l].update(torch.mean((act_q_norm[l]*act_full_fromq_norm[l])).item(), input.size(0))
                            rhlm[l].update(torch.mean((act_q_norm[l]*act_full_norm[l])).item(), input.size(0))
                else:
                     with torch.no_grad():
                        model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
                        model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
                        for bw, am_l, am_t1, am_t5 in zip(bit_width_list, losses, top1, top5):
                            model.apply(lambda m: setattr(m, 'wbit', bw))
                            model.apply(lambda m: setattr(m, 'abit', bw))
                            output = model(input)
                            loss = criterion(output, target)
                            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                            am_l.update(loss.item(), input.size(0))
                            am_t1.update(prec1.item(), input.size(0))
                            am_t5.update(prec5.item(), input.size(0))

        else:
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            optimizer.zero_grad()

            # train full-precision supervisor
            model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
            model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses[-1].update(loss.item(), input.size(0))
            top1[-1].update(prec1.item(), input.size(0))
            top5[-1].update(prec5.item(), input.size(0))
            # Comp acts for slack
            with torch.no_grad():
                model.eval()
                act_full = model.get_activations(input)
                model.train()
            # train less-bit-wdith models
            target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
            for bw, am_l, am_t1, am_t5 in zip(bit_width_list[:-1][::-1], losses[:-1][::-1],\
                                                     top1[:-1][::-1], top5[:-1][::-1]):
                model.apply(lambda m: setattr(m, 'wbit', bw))
                model.apply(lambda m: setattr(m, 'abit', bw))
                output = model(input)
                loss = criterion_soft(output, target_soft)
                loss.backward()
                # recursive supervision
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)

                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                am_l.update(loss.item(), input.size(0))
                am_t1.update(prec1.item(), input.size(0))
                am_t5.update(prec5.item(), input.size(0))
            optimizer.step()

            if i % args.print_freq == 0:
                logging.info('epoch {0}, iter {1}/{2}, bit_width_max loss {3:.2f}, prec1 {4:.2f}, prec5 {5:.2f}'.format(
                    epoch, i, len(data_loader), losses[-1].val, top1[-1].val, top5[-1].val))
    if training:
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5]
    else:
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], \
             [[l.avg for l in _] for _ in pearson_meter], [[l.avg for l in _] for _ in pearson_hl_meter], \
             [[l.avg for l in _] for _ in l2_meter], [[l.avg for l in _] for _ in l2_hl_meter], [_.avg for _ in CE_meter]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--results-dir', default='./results', help='results dir')   
    parser.add_argument('--dataset', default='imagenet', help='dataset name or folder')
    parser.add_argument('--train_split', default='train', help='train split name')
    parser.add_argument('--model', default='resnet18', help='model architecture')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--optimizer', default='sgd', help='optimizer function used')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay', default='100,150,180', help='lr decay steps')
    parser.add_argument('--weight-decay', default=3e-4, type=float, help='weight decay')
    parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
    parser.add_argument('--pretrain', default=None, help='path to pretrained full-precision checkpoint')
    parser.add_argument('--resume', default=None, help='path to latest checkpoint')
    parser.add_argument('--bit_width_list', default='4', help='bit width list')
    parser.add_argument('--wandb_log',  action='store_true')
    parser.add_argument('--eval_constraint',  action='store_true')
    parser.add_argument('--project',  default='Baselines', type=str, help='wandb Project name')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)