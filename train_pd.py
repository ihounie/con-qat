import argparse
import os
import time
import socket
import logging
from datetime import datetime
from functools import partial
from functools import reduce
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
import copy

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
parser.add_argument('--lr_dual', default=0.1, type=float, help='dual learning rate')
parser.add_argument('--lr_decay', default='100,150,180', help='lr decay steps')
parser.add_argument('--epsilon_out', default=0.1, type=float, help='output crossentropy constraint level')
parser.add_argument('--weight-decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--pretrain', default=None, help='path to pretrained full-precision checkpoint')
parser.add_argument('--resume', default=None, help='path to latest checkpoint')
parser.add_argument('--bit_width_list', default='4', help='bit width list')
parser.add_argument('--layerwise_constraint',  action='store_true')
parser.add_argument('--wandb_log',  action='store_true')
args = parser.parse_args()

def get_param_by_name(module,access_string):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def get_bit_width_list(args):
    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    # Add Full precision if not Passed
    if 32 not in bit_width_list:
        bit_width_list += [32]
    return bit_width_list

def main():
    #####################
    #   LOGGING
    #####################
    if args.wandb_log:
        wandb.init(project="con-qat", name=args.results_dir.split('/')[-1])
        wandb.config.update(args)
    hostname = socket.gethostname()
    setup_logging(os.path.join(args.results_dir, 'log_{}.txt'.format(hostname)))
    logging.info("running arguments: %s", args)
    ########################
    # choose and config GPU
    ########################
    best_gpu = setup_gpus()
    torch.cuda.set_device(best_gpu)
    torch.backends.cudnn.benchmark = True
    ########################
    #   DATALOADERS
    #######################
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
    ######################
    #   BIT WIDTHs
    ######################
    bit_width_list = get_bit_width_list(args)
    #####################
    # MODEL and OPT
    #####################
    model = models.__dict__[args.model](bit_width_list, train_data.num_classes).cuda()
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

    criterion = nn.CrossEntropyLoss().cuda()#Loss
    criterion_soft = CrossEntropyLossSoft().cuda()#Unused
    sum_writer = SummaryWriter(args.results_dir + '/summary')   
    num_layers = model.get_num_layers()
    #########################
    # DUAL INIT
    ########################
    # Tensor w/One dual variable per layer
    # For each low precision bitwidth
    if args.layerwise_constraint:
        lambdas = {bw:torch.ones(num_layers, requires_grad=False).cuda() for bw in bit_width_list[:-1]}
    else:
        lambdas = {bw:torch.ones(1, requires_grad=False).cuda() for bw in bit_width_list[:-1]}
    #########################
    # TRAIN LOOP
    ########################
    for epoch in range(args.start_epoch, args.epochs):
        ################
        #  Train Epoch
        ################
        model.train()
        train_loss, train_prec1, train_prec5, lambdas = forward(train_loader, model,lambdas, criterion, criterion_soft, epoch, True,
                                                       optimizer, sum_writer)
        #######################
        #   Test
        #######################
        print("Evaluating Model...")              
        model.eval()
        val_loss, val_prec1, val_prec5 = forward(val_loader, model, lambdas, criterion, criterion_soft, epoch, False)
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        #####################
        #   Model Saving
        #####################
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
        ###############################
        #   Local (Tensorboard) Logging
        ###############################
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
        ###############################
        #   W&B Logging
        ###############################
        if args.wandb_log:
            for bw, tl, tp1, vl, vp1 in zip(bit_width_list, train_loss, train_prec1, val_loss, val_prec1):
                wandb.log({f'train_loss_{bw}':tl, "epoch":epoch})
                wandb.log({f'train_acc_{bw}':tp1, "epoch":epoch})
                wandb.log({f'test_loss_{bw}':vl, "epoch":epoch})
                wandb.log({f'test_acc_{bw}':vp1, "epoch":epoch})
                # If low precision, log associated Dual Variables
                if bw != bit_width_list[-1]:
                    hist = wandb.Histogram(np_histogram=(lambdas[bw].cpu().numpy(), [float(l) for l in range(len(lambdas[bw])+1)]) )
                    wandb.log({"dual_vars": hist, "epoch":epoch })
                    for l in range(len(lambdas[bw])-1):
                        wandb.log({f"dual_layer_{l}_bw_{bw}": lambdas[bw][l].item(), "epoch":epoch })
                    wandb.log({f"dual_CE_bw_{bw}": lambdas[bw][-1].item(), "epoch":epoch })
                    print(f"Dual CE bw {bw}: {lambdas[bw][-1].item()}")
        ####################
        # STDOUT printing
        ####################
        logging.info('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                     '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(
                         epoch, train_loss[-1], train_prec1[-1], train_prec5[-1], val_loss[-1], val_prec1[-1],
                         val_prec5[-1]))

def forward(data_loader, model, lambdas, criterion,criterion_soft, epoch, training=True, optimizer=None, sum_writer=None, train_bn=True):
    # Save state to return model in its initial state
    initial_model_state = model.training
    bit_width_list = get_bit_width_list(args)
    if args.layerwise_constraint:
        epsilon = {b: [ 1/((2**b)-1) for _ in range(model.get_num_layers())]+[args.epsilon_out] for b in bit_width_list}
    else:
        epsilon = {b: [args.epsilon_out] for b in bit_width_list}
    losses = [AverageMeter() for _ in bit_width_list]
    top1 = [AverageMeter() for _ in bit_width_list]
    top5 = [AverageMeter() for _ in bit_width_list]
    slack_meter = [[AverageMeter() for _ in range(len(epsilon[b]))] for b in bit_width_list[:-1]]
    b_norm_layers = model.get_bn_layers()
    for i, (input, target) in enumerate(data_loader):
        if training:
            if not model.training:
                model.train()
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            # compute Full precision forward pass and loss
            # since bitwidth list is sorted in ascending order, the last elem is the highest prec
            model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
            model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
            output = model(input)
            loss = criterion(output, target)
            # Evaluate and log Acc
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses[-1].update(loss.item(), input.size(0))
            top1[-1].update(prec1.item(), input.size(0))
            top5[-1].update(prec5.item(), input.size(0))
            # Evaluate slack
            # We exlude the highest precision (bit_width_list[:-1])
            target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
            for j, bitwidth in enumerate(bit_width_list[:-1]):
                # Forward pass to update bn stats
                with torch.no_grad():
                    # Set model to Low Precision
                    model.apply(lambda m: setattr(m, 'wbit', bitwidth))
                    model.apply(lambda m: setattr(m, 'abit', bitwidth))
                    # Compute forward
                    out_q = model(input)
                    loss_q = criterion(out_q, target)
                    # Compute and log loss and acc
                    prec1_q, prec5_q = accuracy(out_q.data, target, topk=(1, 5)) 
                    losses[j].update(loss_q.item(), input.size(0))
                    top1[j].update(prec1_q.item(), input.size(0))
                    top5[j].update(prec5_q.item(), input.size(0))                
                # low precision clone to compute grads
                model_q = copy.deepcopy(model)
                # compute forward with Low precision model and
                # Grads enabled
                act_q = model_q.get_activations(input)
                # Set model to Full Precision
                model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
                model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
                # Freeze bn statistics to use quantized activations
                out_q = model_q(input)
                # Init Slacks
                slacks = torch.zeros_like(lambdas[bitwidth])
                # Output Constraint
                slacks[-1] = criterion_soft(out_q, target_soft) - epsilon[bitwidth][-1]
                slack_meter[j][-1].update(slacks[-1].item(), input.size(0))
                if args.layerwise_constraint: 
                    # Compute Full Prec layer outputs with Low Prec Activations as inputs
                    act_full = model.eval_layers(input, act_q)
                    # This will be vectorised
                    for l, (full, q) in enumerate(zip(act_full, act_q)):
                        if not l in b_norm_layers:
                            slacks[l] = torch.mean(torch.abs(full-q)) - epsilon[bitwidth][l]
                            slack_meter[j][l].update(slacks[l].item(), input.size(0))
                loss += torch.sum(lambdas[bitwidth] * slacks)
                target_soft = torch.nn.functional.softmax(out_q.detach(), dim=1)

            loss.backward()
            # Copy BN gradients of low precision copy
            # To Main model
            for name, param in model_q.named_parameters():
                if "bn" in name and str(bit_width_list[-1]) not in name:
                    get_param_by_name(model,name).grad = param.grad
            # GD Step
            optimizer.step()
            # Logging and printing
            if i % args.print_freq == 0:
                logging.info('epoch {0}, iter {1}/{2}, bit_width_max loss {3:.2f}, prec1 {4:.2f}, prec5 {5:.2f}, slacklCE{6:.2f}'.format(
                    epoch, i, len(data_loader), losses[-1].val, top1[-1].val, top5[-1].val, slack_meter[0][-1].val))
                for tacc, tl, bw in zip(top1, losses, bit_width_list):
                    wandb.log({f'trainloss_{bw}':tl.avg, "epoch":epoch})
                    wandb.log({f'trainacc_{bw}':tacc.avg, "epoch":epoch})
                for sl, bw in zip(slack_meter, bit_width_list[:-1]):
                    for l in range(len(lambdas[bw])-1):
                        wandb.log({f"slack_layer_{l}_bw_{bw}": sl[l].avg, "epoch":epoch })
                    wandb.log({f"slack_CE_bw_{bw}": sl[-1].avg, "epoch":epoch})     
                for bw in bit_width_list[:-1]:
                    wandb.log({'model_bn_mean': model.bn.bn_dict[str(bw)].running_mean, 'model_bn_var': model.bn.bn_dict[str(bw)].running_var})
                for name, param in model.named_parameters():
                    if "bn" in name:
                        wandb.log({name: param.data, f'{name}_grad': param.grad})

        else:
            # Just compute forward passes
            model.eval()
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                for bw, am_l, am_t1, am_t5 in zip(bit_width_list, losses, top1, top5):
                    model.apply(lambda m: setattr(m, 'wbit', bw))
                    model.apply(lambda m: setattr(m, 'abit', bw))
                    output = model(input)
                    loss = criterion(output, target)
                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    am_l.update(loss.item(), input.size(0))
                    am_t1.update(prec1.item(), input.size(0))
                    am_t5.update(prec5.item(), input.size(0))
    if training:
        # Dual Update
        print("Peforming Dual Update...")
        for bw_idx, bitwidth in enumerate(bit_width_list[:-1]):
            model.eval()
            with torch.no_grad():
                slacks = torch.zeros_like(lambdas[bitwidth])
                for i, (input, target) in enumerate(data_loader):
                    input = input.cuda()
                    # Pairwise CE Constraint between neighboring precision levels
                    # (If only one low precision level is used, its just high and Low)
                    # compute targets
                    model.apply(lambda m: setattr(m, 'wbit', bit_width_list[bw_idx-1]))
                    model.apply(lambda m: setattr(m, 'abit', bit_width_list[bw_idx-1]))
                    output = model(input)
                    target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
                    # Set model to Low Precision
                    model.apply(lambda m: setattr(m, 'wbit', bitwidth))
                    model.apply(lambda m: setattr(m, 'abit', bitwidth))
                    # Compute forward
                    out_q = model(input)
                    # Eval slack
                    slacks[-1] += criterion_soft(out_q, target_soft) - epsilon[bitwidth][-1]
                    if args.layerwise_constraint:
                        model.apply(lambda m: setattr(m, 'wbit', bitwidth))
                        model.apply(lambda m: setattr(m, 'abit', bitwidth))
                        act_q = model.get_activations(input)
                        model.apply(lambda m: setattr(m, 'wbit', 32))
                        model.apply(lambda m: setattr(m, 'abit', 32))
                        act_full = model.eval_layers(input, act_q)
                        # This will be vectorised
                        for l, (full, q) in enumerate(zip(act_full, act_q)):
                            if not l in b_norm_layers:
                                const_vec = torch.abs(full-q)
                                if const_vec.dim()>1:
                                    const = torch.mean(const_vec, axis=[l for l in range(1, const_vec.dim())])
                                else:
                                    const = const_vec
                                slacks[l] += torch.sum(const-epsilon[bitwidth][l])
                slacks = slacks/len(data_loader.dataset)
                lambdas[bitwidth] = torch.nn.functional.relu(lambdas[bitwidth] + args.lr_dual*slacks)
        if initial_model_state:
            model.train()
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], lambdas
    else:
        if initial_model_state:
            model.train()
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5]


if __name__ == '__main__':
    main()