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
from pathlib import Path
import random
import numpy as np

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
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr_dual', default=0.01, type=float, help='dual learning rate')
parser.add_argument('--lr_decay', default='100,150,180', help='lr decay steps')
parser.add_argument('--epsilon_out', default=0.1, type=float, help='output crossentropy constraint level')
parser.add_argument('--weight-decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--pretrain', default=None, help='path to pretrained full-precision checkpoint')
parser.add_argument('--resume', default=None, help='path to latest checkpoint')
parser.add_argument('--bit_width_list', default='4', help='bit width list')
parser.add_argument('--layerwise_constraint',  action='store_true')
parser.add_argument('--grads_wrt_high',  action='store_true')
parser.add_argument('--normalise_constraint',  action='store_true')
parser.add_argument('--constraint_norm', default='L2', help='L2, L1')
parser.add_argument('--pearson', action='store_true', help="use pearson instead of dif norm")
parser.add_argument('--wandb_log',  action='store_true')
parser.add_argument('--project',  default='ConQAT', type=str, help='wandb Project name')
parser.add_argument('--epsilonlw', default=1/(2**8-1), type = float, help='layer constraint tightness')
parser.add_argument('--seed', default=42, type=int)

args = parser.parse_args()

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    
    seed_everything(args.seed)
    #####################
    #   LOGGING
    #####################
    if args.wandb_log:
        wandb.init(project=args.project, entity="alelab", name=args.results_dir.split('/')[-1])
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

    criterion = nn.CrossEntropyLoss().cuda()#Loss
    criterion_soft = CrossEntropyLossSoft().cuda()#Unused
    sum_writer = SummaryWriter(args.results_dir + '/summary')   
    num_layers = model.get_num_layers()
    #########################
    # EPSILON and DUAL INIT
    ########################
    # Tensor w/One dual variable per layer
    # For each low precision bitwidth
    if args.layerwise_constraint:
        lambdas = {bw:torch.zeros(num_layers+1, requires_grad=False).cuda() for bw in bit_width_list[:-1]}
        for bw in bit_width_list[:-1]:
            lambdas[bw][-1] = 1
    else:
        lambdas = {bw:torch.ones(1, requires_grad=False).cuda() for bw in bit_width_list[:-1]}
    
    epsilon = {b: [ args.epsilonlw for _ in range(model.get_num_layers())]+[args.epsilon_out] for b in bit_width_list}

    if args.wandb_log:
        wandb.config.update({"epsilon":epsilon})

    #########################
    # Constraint norm
    ########################
    if args.constraint_norm=="L2":
        norm_func = torch.square
    elif args.constraint_norm=="L1":
        norm_func = torch.abs
    else:
        raise NotImplementedError  
    #########################
    # TRAIN LOOP
    ########################
    for epoch in range(args.start_epoch, args.epochs):
        ################
        #  Train Epoch
        ################
        model.train()
        train_loss, train_prec1, train_prec5, slack_train, lambdas = forward(train_loader, model,lambdas, criterion, criterion_soft, epoch, True,
                                                                optimizer, sum_writer,constraint_norm=norm_func, epsilon=epsilon,
                                                                bit_width_list=bit_width_list)
        #######################
        #   Test
        #######################
        print("Evaluating Model...")              
        model.eval()
        val_loss, val_prec1, val_prec5, slack_val = forward(val_loader, model, lambdas, criterion, criterion_soft, epoch, False, bit_width_list=bit_width_list,constraint_norm=norm_func,epsilon=epsilon)
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
            for bw, tl, tp1, tsl, vl, vp1, vsl in zip(bit_width_list, train_loss, train_prec1, slack_train, val_loss, val_prec1, slack_val):
                wandb.log({f'train_loss_{bw}':tl, "epoch":epoch})
                wandb.log({f'train_acc_{bw}':tp1, "epoch":epoch})
                wandb.log({f'test_loss_{bw}':vl, "epoch":epoch})
                wandb.log({f'test_acc_{bw}':vp1, "epoch":epoch})
                # If low precision, log associated Dual Variables
                if bw != bit_width_list[-1]:
                    hist = wandb.Histogram(np_histogram=(lambdas[bw].cpu().numpy(), [float(l) for l in range(len(lambdas[bw])+1)]) )
                    wandb.log({"dual_vars": hist, "epoch":epoch })
                    for l in range(len(lambdas[bw])):
                        wandb.log({f"dual_layer_{l}_bw_{bw}": lambdas[bw][l].item(), "epoch":epoch })
                    for l in range(len(epsilon[bw])-1):
                        wandb.log({f'slack_layer_{l}_bw_{bw}_train':tsl[l], "epoch":epoch})
                        wandb.log({f'slack_layer_{l}_bw_{bw}_test':vsl[l], "epoch":epoch})
                    wandb.log({f'slack_CE_bw_{bw}_train': tsl[-1], "epoch":epoch})
                    wandb.log({f'slack_CE_bw_{bw}_test': vsl[-1], "epoch":epoch})
                    wandb.log({f"dual_CE_bw_{bw}": lambdas[bw][-1].item(), "epoch":epoch })
                    print(f"Dual CE bw {bw}: {lambdas[bw][-1].item()}")

        ####################
        # STDOUT printing
        ####################
        logging.info('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                     '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(
                         epoch, train_loss[-1], train_prec1[-1], train_prec5[-1], val_loss[-1], val_prec1[-1],
                         val_prec5[-1]))
        
    weights_path = os.path.join(args.results_dir, 'trained_model')
    Path(weights_path).mkdir(parents=True, exist_ok=True)
    weights_path = os.path.join(weights_path, str(wandb.run.id)+'.pt')
    torch.save(model.state_dict(), weights_path)
    wandb.save(weights_path, policy = 'now')

def forward(data_loader, model, lambdas, criterion,criterion_soft, epoch, training=True, 
             optimizer=None, sum_writer=None, train_bn=True, epsilon=None, 
             bit_width_list = None, constraint_norm=torch.square):
    stats_keys = []
    for key in model.state_dict().keys():
        if "mean" in key or "var" in key:
            stats_keys.append(key)
    def copy_stats(source = None, target=None, keys=stats_keys):
        state_dict = source.state_dict()
        bn_stats = {key:state_dict[key] for key in keys}
        target.load_state_dict(bn_stats,strict=False)
    if bit_width_list is None:
        bit_width_list = get_bit_width_list(args)
    # Save state to return model in its initial state
    initial_model_state = model.training
    losses = [AverageMeter() for _ in bit_width_list]
    top1 = [AverageMeter() for _ in bit_width_list]
    top5 = [AverageMeter() for _ in bit_width_list]
    slack_meter = [[AverageMeter() for _ in range(model.get_num_layers()+1)] for b in bit_width_list]
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
            act_full = model.get_activations(input)
            if args.normalise_constraint or args.pearson:
                act_full = model.norm_act(act_full)
            output = act_full[-1]
            loss = criterion(output, target)
            # Evaluate slack
            # We exlude the highest precision (bit_width_list[:-1])
            if args.grads_wrt_high:
                target_soft = torch.nn.functional.softmax(output, dim=1)
            else:
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
            for j, bitwidth in enumerate(bit_width_list[:-1]):
                model.apply(lambda m: setattr(m, 'wbit', bitwidth))
                model.apply(lambda m: setattr(m, 'abit', bitwidth))
                # low precision clone to compute grads
                model_q = copy.deepcopy(model)
                # compute activations with Low precision model from high prec acts
                act_q = model_q.eval_layers(input, act_full)
                # Low prec predictions
                out_q = model_q(input)
                if args.normalise_constraint or args.pearson:
                    act_q = model.norm_act(act_q)
                # Update low precision bnorm stats in main model
                with torch.no_grad():
                    model(input) 
                # Init Slacks
                slacks = torch.zeros_like(lambdas[bitwidth])
                # Output Constraint
                slacks[-1] = criterion_soft(out_q, target_soft) - epsilon[bitwidth][-1]
                if args.layerwise_constraint:
                    # This will be vectorised
                    for l, (full, q) in enumerate(zip(act_full, act_q)):
                        if not l in b_norm_layers:
                            if args.pearson:
                                slacks[l] = torch.mean((1-full*q)) - epsilon[bitwidth][l]
                            else:
                                slacks[l] = torch.mean(constraint_norm(full-q)) - epsilon[bitwidth][l]
                            slack_meter[j][l].update(slacks[l].item(), input.size(0))
                loss += torch.sum(lambdas[bitwidth] * slacks)
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
                logging.info('epoch {0}, iter {1}/{2}'.format(epoch, i, len(data_loader)))
        else:
            # Just compute forward passes
            model.eval()
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
                model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
                act_full = model.get_activations(input)
                output = model(input)
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
                for bw, am_l, am_t1, am_t5, slm in zip(bit_width_list, losses, top1, top5, slack_meter):
                    model.apply(lambda m: setattr(m, 'wbit', bw))
                    model.apply(lambda m: setattr(m, 'abit', bw))
                    output = model(input)
                    loss = criterion(output, target)
                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    am_l.update(loss.item(), input.size(0))
                    am_t1.update(prec1.item(), input.size(0))
                    am_t5.update(prec5.item(), input.size(0))
                    # compute activations with Low precision model from high prec acts
                    act_q = model.eval_layers(input, act_full)
                    # Eval slack
                    slm[-1].update(criterion_soft(output, target_soft).item(), input.size(0))
                    for l in range(model.get_num_layers()):
                        slack =  torch.mean(torch.square(act_q[l]-act_full[l])) - epsilon[bw][l]
                        slm[l].update(slack.item(), input.size(0))
    if training:
        # Dual Update
        model.eval()
        print("Peforming Dual Update...")
        with torch.no_grad():
            slacks = {bitwidth: torch.zeros(len(epsilon[bitwidth])).cuda() for bitwidth in bit_width_list[:-1]}
            for i, (input, target) in enumerate(data_loader):
                input, target = input.cuda(), target.cuda(non_blocking=True)
                # High precision
                model.apply(lambda m: setattr(m, 'wbit',bit_width_list[-1] ))
                model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
                act_full = model.get_activations(input)
                if args.normalise_constraint or args.pearson:
                    model.train()
                    act_full = model.norm_act(act_full)
                    model.eval()
                output = act_full[-1]
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses[-1].update(loss.item(), input.size(0))
                top1[-1].update(prec1.item(), input.size(0))
                top5[-1].update(prec5.item(), input.size(0))
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
                # Low precision
                for bw_idx, bitwidth in enumerate(bit_width_list[:-1]):
                    model.apply(lambda m: setattr(m, 'wbit', bitwidth))
                    model.apply(lambda m: setattr(m, 'abit', bitwidth))
                    # compute activations with Low precision model from high prec acts
                    act_q = model_q.eval_layers(input, act_full)
                    # Low prec predictions
                    out_q = model_q(input)
                    # Eval slack
                    slacks[bitwidth][-1] += (criterion_soft(out_q, target_soft) - epsilon[bitwidth][-1])*input.size(0)
                    model.apply(lambda m: setattr(m, 'wbit', bitwidth))
                    model.apply(lambda m: setattr(m, 'abit', bitwidth))
                    if args.normalise_constraint or args.pearson:
                        model.train()
                        act_q= model.norm_act(act_q)
                        model.eval()
                    # This will be vectorised
                        for l, (full, q) in enumerate(zip(act_full, act_q)):
                            if not l in b_norm_layers:
                                if args.pearson:
                                    const_vec = (1-full*q)
                                else:
                                    const_vec = constraint_norm(full-q)
                                if const_vec.dim()>1:
                                    const = torch.mean(const_vec, axis=[l for l in range(1, const_vec.dim())])
                                else:
                                    const = const_vec
                                slacks[bitwidth][l] += torch.sum(const-epsilon[bitwidth][l])
            for bw_idx, bitwidth in enumerate(bit_width_list[:-1]):
                slacks[bitwidth] = slacks[bitwidth]/len(data_loader.dataset)
                if args.layerwise_constraint:
                    lambdas[bitwidth] = torch.nn.functional.relu(lambdas[bitwidth] + args.lr_dual*slacks[bitwidth])
                else:
                    lambdas[bitwidth] = torch.nn.functional.relu(lambdas[bitwidth] + args.lr_dual*slacks[bitwidth][-1])
                for l in range(len(slacks[bitwidth])):
                    slack_meter[bw_idx][l].update(slacks[bitwidth][l].item(), input.size(0))
        if initial_model_state:
            model.train()
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], [[l.avg for l in _] for _ in slack_meter], lambdas
    else:
        if initial_model_state:
            model.train()
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], [[l.avg for l in _] for _ in slack_meter]


if __name__ == '__main__':
    main()