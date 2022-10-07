import argparse
import os
import time
import socket
import logging
from datetime import datetime
from functools import partial

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
from train import forward

import wandb

parser = argparse.ArgumentParser(description='Evaluating Constraints for a Pretrained model')
parser.add_argument('--dataset', default='imagenet', help='dataset name or folder')
parser.add_argument('--train_split', default='train', help='train split name')
parser.add_argument('--model', default='resnet18', help='model architecture')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--model_path', default=None, help='path to latest checkpoint')
parser.add_argument('--bit_width_list', default='4', help='bit width list')
parser.add_argument('--wandb_log',  action='store_true')
parser.add_argument('--copy_bn',  action='store_true')
args = parser.parse_args()

def main():
    if args.wandb_log:
        wandb.init(project="con-qat", name="eval_"+args.model_path.split('/')[-1])
        wandb.config.update(args)
    hostname = socket.gethostname()
    setup_logging(os.path.join(args.model_path, 'log_{}.txt'.format(hostname)))
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
    model = models.__dict__[args.model](bit_width_list, train_data.num_classes).cuda()
    args.model_path = os.path.join(args.model_path,'ckpt', 'model_latest.pth.tar')
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cuda:{}'.format(best_gpu))
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        param_dict = checkpoint['state_dict']
        new_param_dict = {}
        if args.copy_bn:
            for k, v in param_dict.items():
                new_param_dict[k] = v
                for bw in bit_width_list:
                    if "bn" in  k:
                        new_param_dict[k.replace('32', str(bw))] = v       
        model.load_state_dict(new_param_dict)
        logging.info("loaded resume checkpoint '%s' (epoch %s)", args.model_path, checkpoint['epoch'])
    else:
        print(args.model_path)
        raise ValueError('Pretrained model path error!')
    # Please don't average over vector dimension
    criterion = torch.nn.MSELoss(reduction="none").cuda()
    sum_writer = None#SummaryWriter(args.results_dir + '/summary')
    model.eval()
    val_loss, val_prec1, val_prec5 = forward(val_loader, model, nn.CrossEntropyLoss().cuda(), None, 0, args, False)
    if args.wandb_log:
            for bw, vl, vp1, vp5 in zip(bit_width_list,val_loss,val_prec1, val_prec5):
                wandb.log({f'test_loss_{bw}':vl, "epoch":200})
                wandb.log({f'test_acc_{bw}':vp1, "epoch":200})
    if False:
        for bw in bit_width_list:
            constraint_train = eval_constraint(train_loader, model, criterion, bitwidth=bw).cpu().numpy()
            constraint_test = eval_constraint(val_loader, model, criterion, bitwidth=bw).cpu().numpy()
            for layer, (c_train, c_test) in enumerate(zip(constraint_train, constraint_test)):
                if args.wandb_log:
                    wandb.log({f"l1_train_layer_{layer}_bw_{bw}":c_train,f"l1_test_layer_{layer}_bw_{bw}":c_test})
                print(f"l1_train_layer_{layer}_bw_{bw}: {c_train}")
                print(f"l1_test_layer_{layer}_bw_{bw}: {c_test}")


def eval_constraint(data_loader, model, criterion, bitwidth=4):
    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    constraint = []
    model.eval()
    layers = model.get_num_layers()
    distance_all_layers = torch.zeros(layers).cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.cuda()
            model.apply(lambda m: setattr(m, 'wbit', bitwidth))
            model.apply(lambda m: setattr(m, 'abit', bitwidth))
            act_q = model.get_activations(input)
            model.apply(lambda m: setattr(m, 'wbit', 32))
            model.apply(lambda m: setattr(m, 'abit', 32))
            act_full = model.eval_layers(input, act_q)
            for l, (full, q) in enumerate(zip(act_full, act_q)):
                dist = torch.abs(full-q)
                # mean taken only accross features, not batch dim (first)
                if dist.dim()>1:
                    mean_dist = torch.mean(dist, axis=[l for l in range(1, dist.dim())])
                #Accumulate distances
                distance_all_layers[l] += torch.sum(mean_dist)
    # divide over number of samples to get mean
    return distance_all_layers/len(data_loader.dataset)


if __name__ == '__main__':
    main()