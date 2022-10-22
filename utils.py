from functools import reduce
import os
import torch
import numpy as np
import logging
import shutil
import gpustat
import random
import wandb


class AverageMeter:
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.float().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_logging(log_file):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def setup_gpus():
    """Adapted from https://github.com/bamos/setGPU/blob/master/setGPU.py
    """
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    best_gpu = min(pairs, key=lambda x: x[1])[0]
    return best_gpu


def save_checkpoint(state, is_best, path, name='model_latest.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = path + '/' + name
    torch.save(state, save_path)
    logging.info('checkpoint saved to {}'.format(save_path))
    if is_best:
        shutil.copyfile(save_path, path + '/model_best.pth.tar')

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

def log_epoch_end(bit_width_list, train_loss, train_prec1, slack_train,
                 val_loss, val_prec1, slack_val,test_loss, test_prec1, epoch, lambdas, epsilon, layer_names, prefix=''):
    for bw, tl, tp1, tsl, vl, vp1, vsl, tel, tep1  in zip(bit_width_list, train_loss, train_prec1, slack_train,
                                                                val_loss, val_prec1, slack_val,
                                                                test_loss, test_prec1):
        wandb.log({prefix+f'train_loss_{bw}':tl, "epoch":epoch})
        wandb.log({prefix+f'train_acc_{bw}':tp1, "epoch":epoch})
        wandb.log({prefix+f'val_loss_{bw}':vl, "epoch":epoch})
        wandb.log({prefix+f'val_acc_{bw}':vp1, "epoch":epoch})
        wandb.log({prefix+f'test_loss_{bw}':tel, "epoch":epoch})
        wandb.log({prefix+f'test_acc_{bw}':tep1, "epoch":epoch})
        # If low precision, log associated Dual Variables
        if bw != bit_width_list[-1]:
            if lambdas is not None:
                hist = wandb.Histogram(np_histogram=(lambdas[bw].cpu().numpy(), [float(l) for l in range(len(lambdas[bw])+1)]) )
                wandb.log({prefix+"dual_vars": hist, "epoch":epoch })
                if len(lambdas[bw])>1:
                    for l in range(len(lambdas[bw])):
                        wandb.log({prefix+f"dual_{layer_names[l]}_bw_{bw}": lambdas[bw][l].item(), "epoch":epoch })
                else:
                    wandb.log({prefix+f"dual_CE_bw_{bw}": lambdas[bw].item(), "epoch":epoch })
            for l in range(len(epsilon[bw])):
                wandb.log({prefix+f'slack_{layer_names[l]}_bw_{bw}_train':tsl[l], "epoch":epoch})
                wandb.log({prefix+f'slack_{layer_names[l]}_bw_{bw}_val':vsl[l], "epoch":epoch})
            if prefix=='' and lambdas is not None:
                print(prefix+f"Dual CE bw {bw}: {lambdas[bw][-1].item()}")