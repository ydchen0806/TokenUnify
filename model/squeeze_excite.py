import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from em_net.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm3d

# -- squeeze-and-excitation layer --
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                SynchronizedBatchNorm1d(channel // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel // reduction, channel),
                SynchronizedBatchNorm1d(channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class SELayerCS(nn.Module):
    # Squeeze-and-excitation layer (channel & spatial)
    def __init__(self, channel, reduction=4):
        super(SELayerCS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                SynchronizedBatchNorm1d(channel // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel // reduction, channel),
                SynchronizedBatchNorm1d(channel),
                nn.Sigmoid())

        self.sc = nn.Sequential(
                nn.Conv3d(channel, 1, kernel_size=(1, 1, 1)),
                SynchronizedBatchNorm3d(1),
                nn.ELU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 8, 8), stride=(1, 8, 8)),
                conv3d_bn_elu(1, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear', align_corners=False),
                nn.Conv3d(1, channel, kernel_size=(1, 1, 1)),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid())     

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        z = self.sc(x)
        return (x * y) + (x * z)    

