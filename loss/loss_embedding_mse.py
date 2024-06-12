import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def embedding_loss_norm1(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])

    affs1 = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])

    affs2 = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = affs0.clone().detach()
    affs[:, 1:2, :, shift:, :] = affs1.clone().detach()
    affs[:, 2:3, :, :, shift:] = affs2.clone().detach()

    return loss, affs


def ema_embedding_loss_norm1(embedding, ema_embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    ema_embedding = F.normalize(ema_embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :]*ema_embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])

    affs1 = torch.sum(embedding[:, :, :, shift:, :]*ema_embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])

    affs2 = torch.sum(embedding[:, :, :, :, shift:]*ema_embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = affs0.clone().detach()
    affs[:, 1:2, :, shift:, :] = affs1.clone().detach()
    affs[:, 2:3, :, :, shift:] = affs2.clone().detach()

    return loss, affs


def inf_embedding_loss_norm1(embedding, shift=1):
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    affs1 = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    affs2 = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)

    affs = torch.zeros((B,3,D,H,W), dtype=embedding.dtype, device=embedding.device)
    affs[:, 0:1, shift:, :, :] = affs0.clone().detach()
    affs[:, 1:2, :, shift:, :] = affs1.clone().detach()
    affs[:, 2:3, :, :, shift:] = affs2.clone().detach()

    return affs


def embedding_loss_norm2(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    affs0 = (affs0 + 1) / 2
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])
    # loss0 = criterion(affs0, target[:, 0:1, shift:, :, :])

    affs1 = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    affs1 = (affs1 + 1) / 2
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])
    # loss1 = criterion(affs1, target[:, 1:2, :, shift:, :])

    affs2 = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    affs2 = (affs2 + 1) / 2
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])
    # loss2 = criterion(affs2, target[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = affs0.clone().detach()
    affs[:, 1:2, :, shift:, :] = affs1.clone().detach()
    affs[:, 2:3, :, :, shift:] = affs2.clone().detach()

    if fill:
        affs[:, 0, :shift, :, :] = affs0[:, 0, :shift, :, :]
        affs[:, 1, :, :shift, :] = affs1[:, 0, :, :shift, :]
        affs[:, 2, :, :, :shift] = affs2[:, 0, :, :, :shift]
    return loss, affs


def ema_embedding_loss_norm2(embedding, ema_embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    ema_embedding = F.normalize(ema_embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :]*ema_embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    affs0 = (affs0 + 1) / 2
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])
    # loss0 = criterion(affs0, target[:, 0:1, shift:, :, :])

    affs1 = torch.sum(embedding[:, :, :, shift:, :]*ema_embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    affs1 = (affs1 + 1) / 2
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])
    # loss1 = criterion(affs1, target[:, 1:2, :, shift:, :])

    affs2 = torch.sum(embedding[:, :, :, :, shift:]*ema_embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    affs2 = (affs2 + 1) / 2
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])
    # loss2 = criterion(affs2, target[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = affs0.clone().detach()
    affs[:, 1:2, :, shift:, :] = affs1.clone().detach()
    affs[:, 2:3, :, :, shift:] = affs2.clone().detach()

    if fill:
        affs[:, 0, :shift, :, :] = affs0[:, 0, :shift, :, :]
        affs[:, 1, :, :shift, :] = affs1[:, 0, :, :shift, :]
        affs[:, 2, :, :, :shift] = affs2[:, 0, :, :, :shift]
    return loss, affs


def embedding_single_offset_loss(embedding, order, shift, target, weightmap, criterion):
    B, C, D, H, W = embedding.shape
    order_shift = order % 3
    if order_shift == 0:
        affs_temp = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    elif order_shift == 1:
        affs_temp = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    elif order_shift == 2:
        affs_temp = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    else:
        raise NotImplementedError

    # affs_temp = (affs_temp + 1) / 2
    # affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    if order_shift == 0:
        loss_temp = criterion(affs_temp, target[:, order:order+1, shift:, :, :], weightmap[:, order:order+1, shift:, :, :])
    elif order_shift == 1:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, shift:, :], weightmap[:, order:order+1, :, shift:, :])
    elif order_shift == 2:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, :, shift:], weightmap[:, order:order+1, :, :, shift:])
    else:
        raise NotImplementedError

    return loss_temp, affs_temp

def embedding_loss_norm5(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    '''
    Based on embedding_loss_norm4, but compute each channel separately 
    '''
    embedding = F.normalize(embedding, p=2, dim=1)

    affs = torch.zeros_like(target)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    loss = 0
    for i, shift in enumerate(shifts):
        loss_temp, affs_temp = embedding_single_offset_loss(embedding, i, shift, target, weightmap, criterion)
        if i < 3:
            loss += loss_temp * affs0_weight
        else:
            loss += loss_temp
        if i % 3 == 0:
            affs[:, i:i+1, shift:, :, :] = affs_temp.clone().detach()
        elif i % 3 == 1:
            affs[:, i:i+1, :, shift:, :] = affs_temp.clone().detach()
        elif i % 3 == 2:
            affs[:, i:i+1, :, :, shift:] = affs_temp.clone().detach()
        else:
            raise NotImplementedError

    return loss, affs


def inf_embedding_single_offset_loss(embedding, order, shift):
    B, C, D, H, W = embedding.shape
    order_shift = order % 3
    if order_shift == 0:
        affs_temp = torch.sum(embedding[:, :, shift:, :, :]*embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    elif order_shift == 1:
        affs_temp = torch.sum(embedding[:, :, :, shift:, :]*embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    elif order_shift == 2:
        affs_temp = torch.sum(embedding[:, :, :, :, shift:]*embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    else:
        raise NotImplementedError

    return affs_temp


def inf_embedding_loss_norm5(embedding):
    '''
    Based on embedding_loss_norm4, but compute each channel separately 
    '''
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    # affs = torch.zeros_like(target)
    affs = torch.zeros((B,12,D,H,W), dtype=embedding.dtype, device=embedding.device)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    for i, shift in enumerate(shifts):
        affs_temp = inf_embedding_single_offset_loss(embedding, i, shift)
        if i % 3 == 0:
            affs[:, i:i+1, shift:, :, :] = affs_temp.clone().detach()
        elif i % 3 == 1:
            affs[:, i:i+1, :, shift:, :] = affs_temp.clone().detach()
        elif i % 3 == 2:
            affs[:, i:i+1, :, :, shift:] = affs_temp.clone().detach()
        else:
            raise NotImplementedError

    return affs


def ema_embedding_single_offset_loss(embedding, ema_embedding, order, shift, target, weightmap, criterion):
    B, C, D, H, W = embedding.shape
    order_shift = order % 3
    if order_shift == 0:
        affs_temp = torch.sum(embedding[:, :, shift:, :, :]*ema_embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    elif order_shift == 1:
        affs_temp = torch.sum(embedding[:, :, :, shift:, :]*ema_embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    elif order_shift == 2:
        affs_temp = torch.sum(embedding[:, :, :, :, shift:]*ema_embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
    else:
        raise NotImplementedError

    # affs_temp = (affs_temp + 1) / 2
    # affs_temp = torch.clamp(affs_temp, 0.0, 1.0)

    if order_shift == 0:
        loss_temp = criterion(affs_temp, target[:, order:order+1, shift:, :, :], weightmap[:, order:order+1, shift:, :, :])
    elif order_shift == 1:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, shift:, :], weightmap[:, order:order+1, :, shift:, :])
    elif order_shift == 2:
        loss_temp = criterion(affs_temp, target[:, order:order+1, :, :, shift:], weightmap[:, order:order+1, :, :, shift:])
    else:
        raise NotImplementedError

    return loss_temp, affs_temp

def ema_embedding_loss_norm5(embedding, ema_embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    '''
    Based on embedding_loss_norm4, but compute each channel separately 
    '''
    embedding = F.normalize(embedding, p=2, dim=1)
    ema_embedding = F.normalize(ema_embedding, p=2, dim=1)

    affs = torch.zeros_like(target)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    loss = 0
    for i, shift in enumerate(shifts):
        loss_temp, affs_temp = ema_embedding_single_offset_loss(embedding, ema_embedding, i, shift, target, weightmap, criterion)
        if i < 3:
            loss += loss_temp * affs0_weight
        else:
            loss += loss_temp
        if i % 3 == 0:
            affs[:, i:i+1, shift:, :, :] = affs_temp.clone().detach()
        elif i % 3 == 1:
            affs[:, i:i+1, :, shift:, :] = affs_temp.clone().detach()
        elif i % 3 == 2:
            affs[:, i:i+1, :, :, shift:] = affs_temp.clone().detach()
        else:
            raise NotImplementedError

    return loss, affs

def invert_offsets(offsets):
    return [[-off for off in offset] for offset in offsets]

def shift_tensor(tensor, offset):
    """ Shift a tensor by the given (spatial) offset.
    Arguments:
        tensor [torch.Tensor] - 4D (=2 spatial dims) or 5D (=3 spatial dims) tensor.
            Needs to be of float type.
        offset (tuple) - 2d or 3d spatial offset used for shifting the tensor
    """

    ndim = len(offset)
    assert ndim in (2, 3)
    diff = tensor.dim() - ndim

    # don't pad for the first dimensions
    # (usually batch and/or channel dimension)
    slice_ = diff * [slice(None)]

    # torch padding behaviour is a bit weird.
    # we use nn.ReplicationPadND
    # (torch.nn.functional.pad is even weirder and ReflectionPad is not supported in 3d)
    # still, padding needs to be given in the inverse spatial order

    # add padding in inverse spatial order
    padding = []
    for off in offset[::-1]:
        # if we have a negative offset, we need to shift "to the left",
        # which means padding at the right border
        # if we have a positive offset, we need to shift "to the right",
        # which means padding to the left border
        padding.extend([max(0, off), max(0, -off)])

    # add slicing in the normal spatial order
    for off in offset:
        if off == 0:
            slice_.append(slice(None))
        elif off > 0:
            slice_.append(slice(None, -off))
        else:
            slice_.append(slice(-off, None))

    # pad the spatial part of the tensor with replication padding
    slice_ = tuple(slice_)
    padding = tuple(padding)
    padder = nn.ReplicationPad2d if ndim == 2 else nn.ReplicationPad3d
    padder = padder(padding)
    shifted = padder(tensor)

    # slice the oadded tensor to get the spatially shifted tensor
    shifted = shifted[slice_]
    assert shifted.shape == tensor.shape

    return shifted

def embedding_loss_norm6(embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    offsets = shift
    offsets_ = invert_offsets(offsets)

    shifted = torch.cat([shift_tensor(embedding, off).unsqueeze(1) for off in offsets_], dim=1)
    affs = torch.sum(embedding.unsqueeze(1)*shifted, dim=2, keepdim=False)
    loss = criterion(affs, target, weightmap)
    return loss, affs

def ema_embedding_loss_norm6(embedding, ema_embedding, target, weightmap, criterion, affs0_weight=1, shift=1, fill=True):
    embedding = F.normalize(embedding, p=2, dim=1)
    ema_embedding = F.normalize(ema_embedding, p=2, dim=1)
    offsets = shift
    offsets_ = invert_offsets(offsets)

    shifted = torch.cat([shift_tensor(ema_embedding, off).unsqueeze(1) for off in offsets_], dim=1)
    affs = torch.sum(embedding.unsqueeze(1)*shifted, dim=2, keepdim=False)
    loss = criterion(affs, target, weightmap)
    return loss, affs
