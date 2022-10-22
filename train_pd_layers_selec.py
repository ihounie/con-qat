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
from torch.nn.functional import softmax
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from pathlib import Path
import random
import numpy as np

import models
from models.losses import CrossEntropyLossSoft
from datasets.data import get_dataset, get_transform
from optimizer import get_optimizer_config, get_lr_scheduler
from utils import setup_logging, setup_gpus, save_checkpoint
from utils import AverageMeter, accuracy, get_param_by_name, get_bit_width_list, log_epoch_end, seed_everything
import wandb
import copy

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--results-dir', default='./results', help='results dir')
parser.add_argument('--dataset', default='imagenet', help='dataset name or folder')
parser.add_argument('--train_split', default='train', help='train split name')
parser.add_argument('--model', default='resnet18', help='model architecture')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--optimizer', default='sgd', help='optimizer function used')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr_dual', default=0.01, type=float, help='dual learning rate')
parser.add_argument('--lr_decay', default='50, 75, 90', help='lr decay steps')
parser.add_argument('--val_frac', default=0.1, type=float, help='Validation Fraction')
parser.add_argument('--epsilon_out', default=0.1, type=float, help='output crossentropy constraint level')
parser.add_argument('--weight-decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--pretrain', default=None, help='path to pretrained full-precision checkpoint')
parser.add_argument('--resume', default=None, help='path to latest checkpoint')
parser.add_argument('--bit_width_list', default='4', help='bit width list')
parser.add_argument('--layerwise_constraint',  action='store_true')
parser.add_argument('--constraint_norm', default='L2', help='L2, L1')
parser.add_argument('--wandb_log',  action='store_true')
parser.add_argument('--project',  default='ConQAT', type=str, help='wandb Project name')
parser.add_argument('--epsilonlw', default=1/(2**8-1), type = float, help='layer constraint tightness')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--no_quant_layer', default='', help='comma separated list of layers')

args = parser.parse_args()

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
    ######################
    #   BIT WIDTHs
    ######################
    bit_width_list = get_bit_width_list(args)
    #####################
    # MODEL and OPT
    #####################
    if len(args.no_quant_layer):
        no_quant_layers = args.no_quant_layer.split(',')
    else:
        no_quant_layers = []
    model = models.__dict__[args.model](bit_width_list, test_data.num_classes, unquantized=no_quant_layers).cuda()
    num_layers = model.get_num_layers()
    layer_names = model.get_names()
    layer_names.append("CE")
    lr_decay = list(map(int, args.lr_decay.split(',')))
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = None
    best_lag = None
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)
    criterion = nn.CrossEntropyLoss().cuda()#Loss
    criterion_soft = CrossEntropyLossSoft().cuda()#Unused
    print("*"*20)
    print("Number of intermediate outputs Evaluated: ", num_layers)
    print("*"*20)
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
                                                                optimizer, constraint_norm=norm_func, epsilon=epsilon,
                                                                bit_width_list=bit_width_list)
        #######################
        #   Test
        #######################
        print("Evaluating Model...")              
        model.eval()
        val_loss, val_prec1, val_prec5, slack_val = forward(val_loader, model, lambdas, criterion, criterion_soft, epoch, False, bit_width_list=bit_width_list,constraint_norm=norm_func,epsilon=epsilon)
        test_loss, test_prec1, test_prec5, _ = forward(test_loader, model, lambdas, criterion, criterion_soft, epoch, False, bit_width_list=bit_width_list,constraint_norm=norm_func,epsilon=epsilon, eval_slacks=False)
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        ####################################
        #   Early Stopping and Model Saving
        ####################################
        val_lagrangian = val_loss[-1]
        if args.layerwise_constraint:
            for l in range(len(lambdas)):
                val_lagrangian += lambdas[bit_width_list[0]][l]*slack_val[0][l]
        else:
            val_lagrangian += lambdas[bit_width_list[0]][0]*slack_val[0][-1]
        if best_lag is None:
            is_best = True
            best_lag = val_lagrangian
            weights_folder = os.path.join(args.results_dir, 'trained_model')
            Path(weights_folder).mkdir(parents=True, exist_ok=True)
        else:
            is_best = val_lagrangian < best_lag
        if is_best:
            weights_folder = os.path.join(args.results_dir, 'trained_model')
            weights_path = os.path.join(weights_folder, str(wandb.run.id)+'.pt')
            torch.save(model.state_dict(), weights_path)
        ###############################
        #   W&B Logging
        ###############################
        if args.wandb_log:
            log_epoch_end(bit_width_list, train_loss, train_prec1, slack_train, val_loss, val_prec1, slack_val,test_loss, test_prec1, epoch, lambdas, epsilon, layer_names)
            if is_best:
                wandb.log({"best_val_lagrangian": val_lagrangian})
                log_epoch_end(bit_width_list, train_loss, train_prec1, slack_train, val_loss, val_prec1, slack_val,test_loss, test_prec1, epoch, lambdas, epsilon,layer_names, prefix='best')
            
        ####################
        # STDOUT printing
        ####################
        logging.info('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                     '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(
                         epoch, train_loss[-1], train_prec1[-1], train_prec5[-1], val_loss[-1], val_prec1[-1],
                         val_prec5[-1]))
    if args.wandb_log:
        weights_folder = os.path.join(args.results_dir, 'trained_model')
        weights_path = os.path.join(weights_folder, str(wandb.run.id)+'.pt')
        wandb.save(weights_path, policy = 'now')

def forward(data_loader, model, lambdas, criterion,criterion_soft, epoch, training=True, 
             optimizer=None, train_bn=True, epsilon=None, 
             bit_width_list = None, constraint_norm=torch.square, eval_slacks=True):
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
    for i, (input, target) in enumerate(data_loader):
        if training:
            if not model.training:
                model.train()
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            # compute Full precision forward pass and loss
            # since bitwidth list is sorted in ascending order, the last elem is the highest prec
            model.apply(lambda m: setattr(m, 'wbit', 32))
            model.apply(lambda m: setattr(m, 'abit', 32))
            out_full = model(input)
            loss = criterion(out_full, target)
            # Evaluate slack
            # We exlude the highest precision (bit_width_list[:-1]) 
            for j, bitwidth in enumerate(bit_width_list[:-1]):
                model.apply(lambda m: setattr(m, 'wbit', bitwidth))
                model.apply(lambda m: setattr(m, 'abit', bitwidth)) 
                # low precision clone to compute grads
                model_q = copy.deepcopy(model)
                # Update low precision bnorm stats in main model
                with torch.no_grad():
                    model(input)
                # compute activations with Low precision model
                zq_for_hp, zq_for_const, out_q = model_q.get_activations(input)
                # compute high precision activations with quantized inputs
                model.eval()
                model.apply(lambda m: setattr(m, 'wbit', 32))
                model.apply(lambda m: setattr(m, 'abit', 32))
                z_full_for_const = model.eval_layers(input, zq_for_hp)
                model.train()
                # Init Slacks
                slacks = torch.zeros_like(lambdas[bitwidth])
                # Output Constraint
                slacks[-1] = criterion_soft(out_q, softmax(out_full, dim=1)) - epsilon[bitwidth][-1]
                if args.layerwise_constraint:
                    # This will be vectorised
                    for l, (full, q) in enumerate(zip(z_full_for_const, zq_for_const)):
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
        elif eval_slacks:
            # Used for validation
            # Just compute forward passes
            # And eval slacks
            model.eval()
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
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
            #used for testing, no slacks
            model.eval()
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                for bw, am_l, am_t1, am_t5 in zip(bit_width_list, losses, top1, top5):
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
    if training:
        # Dual Update
        model.eval()
        print("Peforming Dual Update...")
        with torch.no_grad():
            slacks = {bitwidth: torch.zeros(len(epsilon[bitwidth])).cuda() for bitwidth in bit_width_list[:-1]}
            for i, (input, target) in enumerate(data_loader):
                input, target = input.cuda(), target.cuda(non_blocking=True)
                # High precision ACC and Loss (Only For logging purposes)
                model.apply(lambda m: setattr(m, 'wbit', 32 ))
                model.apply(lambda m: setattr(m, 'abit', 32))
                output = model(input)
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
                    # Compute forward
                    zq_for_hp, zq_for_const, outputs_q = model.get_activations(input)
                    # Compute and log loss and acc
                    loss = criterion(outputs_q, target)
                    prec1, prec5 = accuracy(outputs_q.data, target, topk=(1, 5))
                    losses[bw_idx].update(loss.item(), input.size(0))
                    top1[bw_idx].update(prec1.item(), input.size(0))
                    top5[bw_idx].update(prec5.item(), input.size(0))
                    # Eval slack
                    slacks[bitwidth][-1] += (criterion_soft(outputs_q, target_soft) - epsilon[bitwidth][-1])*input.size(0)
                    # compute activations with High precision model
                    model.apply(lambda m: setattr(m, 'wbit', 32))
                    model.apply(lambda m: setattr(m, 'abit', 32))
                    z_full_for_const = model.eval_layers(input, zq_for_hp)
                    # This will be vectorised
                    for l, (full, q) in enumerate(zip(z_full_for_const, zq_for_const)):
                        const_vec = constraint_norm(full-q)
                        const = torch.mean(const_vec, axis=[l for l in range(1, const_vec.dim())])
                        slacks[bitwidth][l] += torch.sum(const-epsilon[bitwidth][l])
            for bw_idx, bitwidth in enumerate(bit_width_list[:-1]):
                slacks[bitwidth] = slacks[bitwidth]/len(data_loader.dataset)
                if args.layerwise_constraint:
                    lambdas[bitwidth] = torch.nn.functional.relu(lambdas[bitwidth] + args.lr_dual*slacks[bitwidth])
                else:
                    lambdas[bitwidth] = torch.nn.functional.relu(lambdas[bitwidth] + args.lr_dual*slacks[bitwidth][-1])
                for l in range(len(slacks[bitwidth])):
                    slack_meter[bw_idx][l].update(slacks[bitwidth][l].item(), len(data_loader.dataset))
                
        if initial_model_state:
            model.train()
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], [[l.avg for l in _] for _ in slack_meter], lambdas
    else:
        if initial_model_state:
            model.train()
        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], [[l.avg for l in _] for _ in slack_meter]


if __name__ == '__main__':
    main()