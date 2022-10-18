import argparse
import copy
import os
from pathlib import Path
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

import models
from models.losses import CrossEntropyLossSoft
from datasets.data import get_dataset, get_transform
from optimizer import get_optimizer_config, get_lr_scheduler
from utils import setup_logging, setup_gpus, save_checkpoint
from utils import AverageMeter, accuracy, get_param_by_name, get_bit_width_list, log_epoch_end, seed_everything

import wandb


def main(args):
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
    # Train/Validation Data partitioning
    val_size = int(len(train_data)*args.val_frac)
    train_size = len(train_data) - val_size
    val_data, train_data = torch.utils.data.random_split(train_data, [val_size,train_size])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_transform = get_transform(args.dataset, 'val')
    
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    test_data = get_dataset(args.dataset, 'test', val_transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    ###################
    #   BITWIDTH
    ####################
    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    # Add 32 BN layers for evaluation only
    if 32 not in bit_width_list:
        bw_list = bit_width_list + [32]
    else:
        bw_list = bit_width_list
    #####################
    # MODEL and OPT
    #####################
    model = models.__dict__[args.model](bit_width_list, test_data.num_classes).cuda()
    num_layers = model.get_num_layers()
    layer_names = []
    block_l_names = ["conv0", "conv1", "shortcut"]
    for l in range(num_layers):
        layer_names.append(f"Block_{l//3}_{block_l_names[l%3]}")
    layer_names.append("CE")
    model = models.__dict__[args.model](bw_list, test_data.num_classes).cuda()
    model.bn_to_cuda()

    lr_decay = list(map(int, args.lr_decay.split(',')))
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = None
    best_prec1 = None
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)
    # LOSS
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_soft = CrossEntropyLossSoft().cuda()
    #########################
    # For Comparison Purposes ONLY
    ########################
    epsilon = {b: [ 0 for _ in range(model.get_num_layers()+1)] for b in bit_width_list}

    for epoch in range(args.start_epoch, args.epochs):
        #######################
        #   TRAINING
        #######################
        model.train()
        train_loss, train_prec1, train_prec5, slack_train = forward(train_loader, model, criterion, criterion_soft, epoch, args, True,
                                                       optimizer=optimizer, bit_width_list=bit_width_list)
        #######################
        #   VAL & TESTING
        #######################
        print("Evaluating Model...")              
        model.eval()
        val_loss, val_prec1, val_prec5, slack_val = forward(val_loader, model, criterion, criterion_soft, epoch, args, False, bit_width_list=bit_width_list,epsilon=epsilon)
        test_loss, test_prec1, test_prec5, _ = forward(test_loader, model, criterion, criterion_soft, epoch, args, False,  bit_width_list=bit_width_list,epsilon=epsilon, eval_slacks=False)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        ####################################
        #   Early Stopping & Model Saving
        ####################################
        if best_prec1 is None:
            is_best = True
            best_prec1 = val_prec1[-1]
            weights_folder = os.path.join(args.results_dir, 'trained_model')
            weights_path = os.path.join(weights_folder, str(wandb.run.id)+'.pt')
            Path(weights_folder).mkdir(parents=True, exist_ok=True)
        else:
            is_best = val_prec1[-1] > best_prec1
            best_prec1 = max(val_prec1[-1], best_prec1)
        if is_best:
            weights_folder = os.path.join(args.results_dir, 'trained_model')
            weights_path = os.path.join(weights_folder, str(wandb.run.id)+'.pt')
            torch.save(model.state_dict(), weights_path)

       ###############################
        #   LOGGING
        ###############################
        if args.wandb_log:
            log_epoch_end(bit_width_list, train_loss, train_prec1, slack_train, val_loss, val_prec1, slack_val,test_loss, test_prec1, epoch, None, epsilon, layer_names)
            if is_best:
                log_epoch_end(bit_width_list, train_loss, train_prec1, slack_train, val_loss, val_prec1, slack_val,test_loss, test_prec1, epoch, None, epsilon,layer_names, prefix='best')

        logging.info('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                     '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(
                         epoch, train_loss[-1], train_prec1[-1], train_prec5[-1], val_loss[-1], val_prec1[-1],
                         val_prec5[-1]))

def forward(data_loader, model, criterion, criterion_soft, epoch, args, training=True, optimizer=None, eval_slacks=True, bit_width_list=[8, 32], epsilon=None):
    # Save state to return model in its initial state
    initial_model_state = model.training
    losses = [AverageMeter() for _ in bit_width_list]
    top1 = [AverageMeter() for _ in bit_width_list]
    top5 = [AverageMeter() for _ in bit_width_list]
    slack_meter = [[AverageMeter() for _ in range(model.get_num_layers()+1)] for b in bit_width_list]
    for i, (input, target) in enumerate(data_loader):
        if not training:
            # Just compute forward passes
            model.eval()
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                if eval_slacks:
                    model.apply(lambda m: setattr(m, 'wbit', 32))
                model.apply(lambda m: setattr(m, 'abit', 32))
                model.eval()
                output = model(input)
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
                for bw, am_l, am_t1, am_t5, slm in zip(bit_width_list, losses, top1, top5, slack_meter):
                    model.apply(lambda m: setattr(m, 'wbit', bw))
                    model.apply(lambda m: setattr(m, 'abit', bw))
                    # compute activations with Low precision model
                    zq_for_hp, zq_for_const, output = model.get_activations(input)
                    # Log Loss and acc
                    loss = criterion(output, target)
                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    am_l.update(loss.item(), input.size(0))
                    am_t1.update(prec1.item(), input.size(0))
                    am_t5.update(prec5.item(), input.size(0))
                    # compute activations with High precision model
                    model.apply(lambda m: setattr(m, 'wbit', 32))
                    model.apply(lambda m: setattr(m, 'abit', 32))
                    model.eval()
                    z_full_for_const = model.eval_layers(input, zq_for_hp)
                    # Log slacks
                    slm[-1].update(criterion_soft(output, target_soft).item(), input.size(0))
                    for l in range(model.get_num_layers()):
                        slack =  torch.mean(torch.square(zq_for_const[l]-z_full_for_const[l])) - epsilon[bw][l]
                        slm[l].update(slack.item(), input.size(0))
                else:
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
            model.apply(lambda m: setattr(m, 'wbit', bit_width_list[0]))
            model.apply(lambda m: setattr(m, 'abit', bit_width_list[0]))
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            # high precision clone to update bn params
            model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
            model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
            model_high = copy.deepcopy(model)
            output_high = model(input)
            loss_high = criterion(output_high, target)
            loss_high.backward()
            # Copy BN gradients of low precision copy
            # To Main model
            for name, param in model_high.named_parameters():
                if "bn" in name and str(bit_width_list[-1]) not in name:
                    get_param_by_name(model,name).grad = param.grad
            # Update stats BNORM HP
            with torch.no_grad():
                model(input)

            optimizer.step()
            
            if i % args.print_freq == 0:
                logging.info('epoch {0}, iter {1}/{2}, bit_width_max loss {3:.2f}, prec1 {4:.2f}, prec5 {5:.2f}'.format(
                    epoch, i, len(data_loader), losses[-1].val, top1[-1].val, top5[-1].val))
    if initial_model_state:
            model.train()
    return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], [[l.avg for l in _] for _ in slack_meter]


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
    parser.add_argument('--constraint_norm', default='L2', help='L2, L1 for evaluation only')
    parser.add_argument('--wandb_log',  action='store_true')
    parser.add_argument('--val_frac', default=0.1, type=float, help='Validation Fraction')
    parser.add_argument('--project',  default='Baselines', type=str, help='wandb Project name')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)