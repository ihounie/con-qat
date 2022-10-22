# Refer to https://arxiv.org/abs/1512.03385
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn, batchnorm1d_fn, Conv2d_FULL

__all__ = ['resnet20q']


class Activate(nn.Module):
    def __init__(self, bit_list, quantize=False):
        super(Activate, self).__init__()
        self.bit_list = bit_list
        self.abit = self.bit_list[-1]
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.bit_list)

    def forward(self, x):
        x = torch.clip(x, min=0.0, max=1.0)
        if self.quantize:
            x = self.quan(x)
        return x


class PreActBasicBlockQ(nn.Module):
    """Pre-activation version of the BasicBlock.
    """
    def __init__(self, bit_list, in_planes, out_planes, stride=1, block_num = 0, unquantized = []):
        super(PreActBasicBlockQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.unquantized = unquantized

        Conv2d = conv2d_quantize_fn(self.bit_list)
        NormLayer = batchnorm_fn(self.bit_list)

        self.bn0 = NormLayer(in_planes)
        self.act0 = Activate(self.bit_list)
        if "conv0" in unquantized:
            self.conv0 = Conv2d_FULL(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(out_planes)
        self.act1 = Activate(self.bit_list)
        if "conv1" in unquantized:
            self.conv1 = Conv2d_FULL(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip_conv = None
        if stride != 1:
            if "shortcut" in unquantized:
                self.skip_conv = Conv2d_FULL(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            else:
                self.skip_conv = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = NormLayer(out_planes)

    def forward(self, x):
        out = self.bn0(x)
        out = self.act0(out)

        if self.skip_conv is not None:
            shortcut = self.skip_conv(out)
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = x

        out = self.conv0(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv1(out)
        out += shortcut
        return out

    def get_activations(self, input):
        zq_for_hp = []# inputs for HP
        zq_for_const = []# For constraint Eval
        out = self.bn0(input)
        out = self.act0(out)
        if self.skip_conv is not None:
            a = self.skip_conv(out)
            shortcut = self.skip_bn(a)
        else:
            shortcut = input 
        pre, out = self.conv0(out, pre=True)
        zq_for_hp.append(pre)#quantized input of conv1
        out = self.bn1(out)
        out = self.act1(out)
        pre, out = self.conv1(out, pre=True)
        zq_for_const.append(pre)#quantized out of conv1
        zq_for_hp.append(pre)#quantized input of conv2
        a = self.conv1.quan_a(out)
        zq_for_const.append(a.detach())#quantized out of conv2
        if self.skip_conv is not None:
            a = self.skip_conv.quan_a(a)#quantized out of shortcut
            zq_for_const.append(a.detach())
        out += shortcut
        return zq_for_hp, zq_for_const, out

    def eval_layers(self, z):
        act = []
        if self.skip_conv is not None:
            a = self.skip_conv(z[0])
        out = self.conv0(z[0])
        out = torch.clip(out, 0.0, 1.0)
        act.append(out)
        out = self.conv1(z[1])
        out = torch.clip(out, 0.0, 1.0)
        act.append(out)
        if self.skip_conv is not None:
            a = torch.clip(a, 0.0, 1.0)
            act.append(a)
        return act

    def get_layer(self, l):
        if l=="conv0":
            return self.conv0
        elif l=="conv1":
            return self.conv1
        elif l=="shortcut":
            if self.skip_conv is not None:
                return self.skip_conv
            else:
                print('No Shortcut Layer on this block')
                raise
    def get_bn_layers(self):
        bn_layers = [self.bn0, self.bn1]
        if self.skip_conv is not None:
            bn_layers.append(self.skip_bn)
        return bn_layers



class PreActResNet(nn.Module):
    def __init__(self, block, num_units, bit_list, num_classes, expand=5, unquantized = []):
        super(PreActResNet, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.expand = expand
        self.bn_act_norm = []
        NormLayer = batchnorm_fn(self.bit_list)
        NormLayer1d = batchnorm1d_fn(self.bit_list)
        ep = self.expand
        self.conv0 = nn.Conv2d(3, 16 * ep, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_act_norm.append(NormLayer(16 * ep, affine=False))
        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16 * ep] * num_units[0] + [32 * ep] * num_units[1] + [64 * ep] * num_units[2]
        in_planes = 16 * ep
        self.max_layers = 3*len(channels)#If all had shortcut convs
        self.layers = nn.ModuleList()
        self.unquantized = self.process_unquantized_list(unquantized)
        for i, (stride, channel) in enumerate(zip(strides, channels)):
            self.layers.append(block(self.bit_list, in_planes, channel, stride, unquantized = self.unquantized[i]))
            in_planes = channel
            self.bn_act_norm.append(NormLayer(channel, affine=False))

        self.bn = NormLayer(64 * ep)
        self.bn_act_norm.append(NormLayer(64 * ep, affine=False))
        self.bn_act_norm.append(NormLayer1d(64 * ep, affine=False))
        self.fc = nn.Linear(64 * ep, num_classes)
        self.name_idx_dict = self.get_name_idx_dict()
        self.names = self.get_names()
        self.bn_layers = self.get_bn_layers()

    def forward(self, x):
        out = self.conv0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.fc(out)
        return out
    
    def get_activations(self, x):
        zq_for_hp, zq_for_const = [], []
        out = self.conv0(x)
        for layer in self.layers:
            hp, const, out = layer.get_activations(out)
            zq_for_hp += hp
            zq_for_const += const
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.fc(out)
        return zq_for_hp, zq_for_const, out
    
    def eval_layers(self,input, zq_for_hp):
        z = []
        idx=0
        for layer in self.layers:
            # Each block has 2 activation inputs (input to block and inpu to second conv)
            out = layer.eval_layers(zq_for_hp[idx:idx+2])
            idx += 2
            z += out
        return z
    
    def process_unquantized_list(self, uq_list):
        u_q = [[] for _ in range(self.max_layers)]
        for layer in uq_list:
            u_q[int(layer.split("_")[1])].append(layer.split("_")[2])
        return u_q
    
    def get_layer(self, l):
        name = self.names[l]
        num_block = int(name.split("_")[1])
        return self.layers[num_block].get_layer(name.split("_")[2])

    def get_names(self):
        layer_names = []
        block_l_names = ["conv0", "conv1", "shortcut"]
        l = 0
        for l in range(27):
            if l%3 ==2:
                if l//3 == 3 or l//3==6:
                    layer_names.append(f"Block_{l//3}_{block_l_names[l%3]}")
            else:
                layer_names.append(f"Block_{l//3}_{block_l_names[l%3]}")
        return layer_names
    
    def get_name_idx_dict(self):
        return {name:l for l, name in enumerate(self.get_names())}
    
    def get_idx_from_name(self, name):
        return self.name_idx_dict[name]

    def norm_act(self,activations):
        norm_act = []
        for bn, act in zip(self.bn_act_norm, activations):
            norm_act.append(bn(act))
        return norm_act

    def get_num_layers(self):
        num_layers = 0 # conv0 doesnt count
        for layer in self.layers:#conv layers
            # each Layer has two convs
            num_layers += 2
        # Two layers have a shortcut conv
        num_layers += 2
        # Output Not counted
        return num_layers
    
    def get_bn_layers(self):
        bn_layers = [self.bn]
        for layer in self.layers:
            bn_layers.append(layer.get_bn_layers())
        return bn_layers

# For CIFAR10
def resnet20q(bit_list, num_classes=10, unquantized =[]):
    return PreActResNet(PreActBasicBlockQ, [3, 3, 3], bit_list, num_classes=num_classes, unquantized=unquantized)

