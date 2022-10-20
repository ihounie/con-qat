# Refer to https://arxiv.org/abs/1512.03385
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn, batchnorm1d_fn

__all__ = ['resnet20q', 'resnet50q']


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
    def __init__(self, bit_list, in_planes, out_planes, stride=1):
        super(PreActBasicBlockQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.bit_list)
        NormLayer = batchnorm_fn(self.bit_list)

        self.bn0 = NormLayer(in_planes)
        self.act0 = Activate(self.bit_list)
        self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(out_planes)
        self.act1 = Activate(self.bit_list)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_conv = None
        if stride != 1:
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
            shortcut = self.skip_conv(out)
            shortcut = self.skip_bn(shortcut)
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
            a = self.skip_conv.quan_a(shortcut)
        else:
            a = torch.clip(shortcut, 0.0, 1.0)
        zq_for_const.append(a.detach())

        out += shortcut
        return zq_for_hp, zq_for_const, out

    def eval_layers(self, z):
        act = []
        if self.skip_conv is not None:
            shortcut = self.skip_conv(z[0])
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = z[0]
        out = self.conv0(z[0])
        out = torch.clip(out, 0.0, 1.0)
        act.append(out)
        out = self.conv1(z[1])
        out = torch.clip(out, 0.0, 1.0)
        act.append(out)
        shortcut = torch.clip(shortcut, 0.0, 1.0)
        act.append(shortcut)
        return act

class PreActResNet(nn.Module):
    def __init__(self, block, num_units, bit_list, num_classes, expand=5):
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
        self.layers = nn.ModuleList()
        for stride, channel in zip(strides, channels):
            self.layers.append(block(self.bit_list, in_planes, channel, stride))
            in_planes = channel
            self.bn_act_norm.append(NormLayer(channel, affine=False))

        self.bn = NormLayer(64 * ep)
        self.bn_act_norm.append(NormLayer(64 * ep, affine=False))
        self.bn_act_norm.append(NormLayer1d(64 * ep, affine=False))
        self.fc = nn.Linear(64 * ep, num_classes)

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
    
    def get_layer(self, l):
        if l==0:
            return self.conv0
        else: # This will throu an out of index if l= last two layers
            return self.layers[l-1]

    def norm_act(self,activations):
        norm_act = []
        for bn, act in zip(self.bn_act_norm, activations):
            norm_act.append(bn(act))
        return norm_act

    def bn_to_cuda(self):
        for bn in self.bn_act_norm:
            bn.cuda()

    def get_num_layers(self):
        num_layers = 0 # conv0 doesnt count
        for layer in self.layers:#conv layers
            # each Layer has three constrainable outputs
            num_layers +=3
        # Output Not counted
        return num_layers
        
class PreActBottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, bit_list, in_planes, out_planes, stride=1, downsample=None):
        super(PreActBottleneckQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.bit_list)
        norm_layer = batchnorm_fn(self.bit_list)

        self.bn0 = norm_layer(in_planes)
        self.act0 = Activate(self.bit_list)
        self.conv0 = Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_planes)
        self.act1 = Activate(self.bit_list)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.act2 = Activate(self.bit_list)
        self.conv2 = Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample

    def forward(self, x):        
        shortcut = self.downsample(x) if self.downsample is not None else x
        out = self.conv0(self.act0(self.bn0(x)))
        out = self.conv1(self.act1(self.bn1(out)))
        out = self.conv2(self.act2(self.bn2(out)))
        out += shortcut
        return out


class PreActResNetBottleneck(nn.Module):
    def __init__(self, block, layers, bit_list, num_classes):
        super(PreActResNetBottleneck, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        self.norm_layer = batchnorm_fn(self.bit_list)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn = self.norm_layer(512 * block.expansion)
        self.act = Activate(self.bit_list)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.bit_list, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.bit_list, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.act(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# For CIFAR10
def resnet20q(bit_list, num_classes=10):
    return PreActResNet(PreActBasicBlockQ, [3, 3, 3], bit_list, num_classes=num_classes)


# For ImageNet
def resnet50q(bit_list, num_classes=1000):
    return PreActResNetBottleneck(PreActBottleneckQ, [3, 4, 6, 3], bit_list, num_classes=num_classes)
