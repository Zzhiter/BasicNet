import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict


# Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 卷积块：conv + bn + Mish
class BasicConv(nn.Module):
    def __init__(self, in_channals, out_channals, \
                 kernel_size, stride = 1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channals, out_channals,\
                              kernel_size, stride, kernel_size // 2, bias = False)
        self.bn = nn.BatchNorm2d(out_channals)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# CSPdarknet种resblockbody的组成部分，内部的残差块
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channals = None,\
                 residual_activation = nn.Identity()):
        super(Resblock, self).__init__()

        if hidden_channals == None:
            hidden_channals = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channals, 1),
            BasicConv(hidden_channals, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

class Resblock_body(nn.Module):
    def __init__(self):

    def forward(self, x):
    def forward(self, x):


class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32

    def forward(self, x):

        return out1, out2, out3
