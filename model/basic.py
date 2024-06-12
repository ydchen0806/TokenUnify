import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# from em_net.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

# handy combination of existing modules

# -- conv layers for deployment: easy to use
def conv3d_pad(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1),
               bias=False):
    # padding with replication
    # the size of the padding should be a 6-tuple        
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return nn.Sequential(
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=0, dilation=dilation, bias=bias))

def conv3d_bn_non(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1),
                  bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes))              

def conv3d_bn_elu(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1),
                  bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes),
            nn.ELU(inplace=True))                                   

# -- conv layers for development: more flexible
# -- weight initialization--
def init_conv(m, init_mode):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
        if init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight)
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif init_mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight)
        elif init_mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def getConv2d(in_planes, out_planes, kernel_size, stride, padding, 
                  bias, pad_mode='zero', init_mode='', dilation_size=(1,1)):
    out = []
    if pad_mode == 'zero': # 0-padding
        out = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, \
                         dilation=dilation_size, padding=padding, stride=stride, bias=bias)]
    elif pad_mode == 'replicate': # replication-padding
        # need 6 values
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        out = [nn.ReplicationPad2d(padding),
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, dilation=dilation_size, bias=bias)]
    if len(out)==0: 
        raise ValueError('Unknown padding option {}'.format(mode))
    else:
        if init_mode!='': # do conv init
            init_conv(out[-1], init_mode)
        return out

def getConv3d(in_planes, out_planes, kernel_size, stride, padding, 
                  bias, pad_mode='zero', init_mode='', dilation_size=(1,1,1)):
    out = []
    if pad_mode == 'zero': # 0-padding
        out = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, \
                         dilation=dilation_size, padding=padding, stride=stride, bias=bias)]
    elif pad_mode == 'replicate': # replication-padding
        # need 6 values
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        out = [nn.ReplicationPad3d(padding),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, dilation=dilation_size, bias=bias)]
    if len(out)==0: 
        raise ValueError('Unknown padding option {}'.format(mode))
    else:
        if init_mode!='': # do conv init
            init_conv(out[-1], init_mode)
        return out


def getRelu(mode='relu'):
    if mode == 'relu':
        return nn.ReLU(inplace=True)
    elif mode == 'elu':
        return nn.ELU(inplace=True)
    elif mode[:5] == 'leaky':
        return nn.LeakyReLU(inplace=True, negative_slope=float(mode[5:]))
    raise ValueError('Unknown ReLU option {}'.format(mode))

def getBN(out_planes, dim=1, mode='sync', bn_momentum=0.1):
    if mode == 'async':
        if dim == 1:
            return nn.BatchNorm1d(out_planes, momentum=bn_momentum)
        elif dim == 2:
            return nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        elif dim == 3:
            return nn.BatchNorm3d(out_planes, momentum=bn_momentum)
    elif mode == 'sync':
        if dim == 1:
            return SynchronizedBatchNorm1d(out_planes, momentum=bn_momentum)
        elif dim == 2:
            return SynchronizedBatchNorm2d(out_planes, momentum=bn_momentum)
        elif dim == 3:
            return SynchronizedBatchNorm3d(out_planes, momentum=bn_momentum)
    raise ValueError('Unknown BatchNorm option: '+str(mode))

def conv3dBlock(in_planes, out_planes, kernel_size=[(3,3,3)], stride=[1], padding=[0], bias=[True], pad_mode=['zero'], bn_mode=[''], relu_mode=[''], init_mode='kaiming_normal', bn_momentum=0.1, dilation_size=None):
    # easy to make VGG style layers
    layers = []
    if dilation_size is None:
        dilation_size = [(1, 1, 1)]*len(in_planes)
    for i in range(len(in_planes)):
        if in_planes[i]>0:
            layers += getConv3d(in_planes[i], out_planes[i], kernel_size[i], stride[i], padding[i], bias[i], pad_mode[i], init_mode, dilation_size[i])
        if bn_mode[i] != '':
            layers.append(getBN(out_planes[i], 3, bn_mode[i], bn_momentum))
        if relu_mode[i] != '':
            layers.append(getRelu(relu_mode[i]))
    return nn.Sequential(*layers)

def conv2dBlock(in_planes, out_planes, kernel_size=[(3,3)], stride=[1], padding=[0], bias=[True], pad_mode=['zero'], bn_mode=[''], relu_mode=[''], init_mode='kaiming_normal', bn_momentum=0.1, dilation_size=None):
    # easy to make VGG style layers
    layers = []
    if dilation_size is None:
        dilation_size = [(1, 1)]*len(in_planes)
    for i in range(len(in_planes)):
        if in_planes[i]>0:
            layers += getConv2d(in_planes[i], out_planes[i], kernel_size[i], stride[i], padding[i], bias[i], pad_mode[i], init_mode, dilation_size[i])
        if bn_mode[i] != '':
            layers.append(getBN(out_planes[i], 2, bn_mode[i], bn_momentum))
        if relu_mode[i] != '':
            layers.append(getRelu(relu_mode[i]))
    return nn.Sequential(*layers)



def upsampleBlock(in_planes, out_planes, up=(1,2,2), mode='bilinear',
                 kernel_size = (1,1,1), stride = (1,1,1), padding = (0,0,0), bias=True, init_mode=''):
        # Upsampling
        out = None
        if mode == 'bilinear':
            out = [nn.Upsample(scale_factor=up, mode='trilinear', align_corners=True),
                nn.Conv3d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=bias)]
        elif mode == 'nearest':
            out = [nn.Upsample(scale_factor=up, mode='nearest'),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        elif mode == 'transpose': # dense version
            out = [nn.ConvTranspose3d(
                          in_planes, out_planes, kernel_size=kernel_size,
                          stride=up, bias=bias)]
        elif mode == 'transposeS': # sparse version
            out = [nn.ConvTranspose3d(
                              in_planes, in_planes, kernel_size=up,
                              stride=up, bias=bias, groups=in_planes),
                    nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, bias=bias)]
        if out is None:
            raise ValueError('Unknown upsampling mode {}'.format(mode))
        else:
            out = nn.Sequential(*out)
            for m in range(len(out._modules)):
                init_conv(out._modules[str(m)], init_mode)
            return out

def upsample2dBlock(in_planes, out_planes, up=(2,2), mode='bilinear',
                 kernel_size = (1,1), stride = (1,1), padding = (0,0), bias=True, init_mode=''):
        # Upsampling
        out = None
        if mode == 'bilinear':
            out = [nn.Upsample(scale_factor=up, mode='bilinear', align_corners=True),
                nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=bias)]
        elif mode == 'nearest':
            out = [nn.Upsample(scale_factor=up, mode='nearest'),
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        elif mode == 'transpose': # dense version
            out = [nn.ConvTranspose2d(
                          in_planes, out_planes, kernel_size=kernel_size,
                          stride=up, bias=bias)]
        elif mode == 'transposeS': # sparse version
            out = [nn.ConvTranspose2d(
                              in_planes, in_planes, kernel_size=up,
                              stride=up, bias=bias, groups=in_planes),
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=bias)]
        if out is None:
            raise ValueError('Unknown upsampling mode {}'.format(mode))
        else:
            out = nn.Sequential(*out)
            for m in range(len(out._modules)):
                init_conv(out._modules[str(m)], init_mode)
            return out



# -- merge layers--
def merge_crop(x1, x2):
    # x1 bigger
    offset = [(x1.size()[x]-x2.size()[x])//2 for x in range(2, x1.dim())]
    return torch.cat([x2, x1[:, :, offset[0]:offset[0]+x2.size(2),
                          offset[1]:offset[1]+x2.size(3), offset[2]:offset[2]+x2.size(4)]], 1)

def merge_add(x1, x2):
    # x1 bigger
    offset = [(x1.size()[x]-x2.size()[x])//2 for x in range(2, x1.dim())]
    return x2 + x1[:, :, offset[0]:offset[0]+x2.size(2), offset[1]:offset[1]+x2.size(3), offset[2]:offset[2]+x2.size(4)]
