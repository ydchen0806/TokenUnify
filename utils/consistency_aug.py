import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F

def simple_augment(data, rule):
    assert np.size(rule) == 4
    assert data.ndim == 3
    # z reflection
    if rule[0]:
        data = data[::-1, :, :]
    # x reflection
    if rule[1]:
        data = data[:, :, ::-1]
    # y reflection
    if rule[2]:
        data = data[:, ::-1, :]
    # transpose in xy
    if rule[3]:
        data = np.transpose(data, (0, 2, 1))
    return data

def simple_augment_torch(data, rule):
    assert np.size(rule) == 4
    assert len(data.shape) == 4
    # z reflection
    if rule[0]:
        data = torch.flip(data, [1])
    # x reflection
    if rule[1]:
        data = torch.flip(data, [3])
    # y reflection
    if rule[2]:
        data = torch.flip(data, [2])
    # transpose in xy
    if rule[3]:
        data = data.permute(0, 1, 3, 2)
    return data

def simple_augment_reverse(data, rule):
    assert np.size(rule) == 4
    assert len(data.shape) == 5
    # transpose in xy
    if rule[3]:
        # data = np.transpose(data, (0, 1, 2, 4, 3))
        data = data.permute(0, 1, 2, 4, 3)
    # y reflection
    if rule[2]:
        # data = data[:, :, :, ::-1, :]
        data = torch.flip(data, [3])
    # x reflection
    if rule[1]:
        # data = data[:, :, :, :, ::-1]
        data = torch.flip(data, [4])
    # z reflection
    if rule[0]:
        # data = data[:, :, ::-1, :, :]
        data = torch.flip(data, [2])
    return data

def order_aug(imgs, num_patch=4):
    assert imgs.shape[-1] % num_patch == 0
    patch_size = imgs.shape[-1] // num_patch
    new_imgs = np.zeros_like(imgs, dtype=np.float32)
    # ran_order = np.random.shuffle(np.arange(num_patch**2))
    ran_order = np.random.permutation(num_patch**2)
    for k in range(num_patch**2):
        xid_new = k // num_patch
        yid_new = k % num_patch
        order_id = ran_order[k]
        xid_old = order_id // num_patch
        yid_old = order_id % num_patch
        new_imgs[:, xid_new*patch_size:(xid_new+1)*patch_size, yid_new*patch_size:(yid_new+1)*patch_size] = \
            imgs[:, xid_old*patch_size:(xid_old+1)*patch_size, yid_old*patch_size:(yid_old+1)*patch_size]
    return new_imgs

def gen_mask(imgs, model_type='superhuman', min_mask_counts=40, max_mask_counts=60, min_mask_size=[3, 5, 5], max_mask_size=[7, 20, 20]):
    if model_type == 'mala':
        net_crop_size = [14, 106, 106]
    else:
        net_crop_size = [0, 0, 0]
    crop_size = list(imgs.shape)
    mask = np.ones_like(imgs, dtype=np.float32)
    mask_counts = random.randint(min_mask_counts, max_mask_counts)
    mask_size_z = random.randint(min_mask_size[0], max_mask_size[0])
    mask_size_xy = random.randint(min_mask_size[1], max_mask_size[1])
    for k in range(mask_counts):
        mz = random.randint(net_crop_size[0], crop_size[0]-mask_size_z-net_crop_size[0])
        my = random.randint(net_crop_size[1], crop_size[1]-mask_size_xy-net_crop_size[1])
        mx = random.randint(net_crop_size[2], crop_size[2]-mask_size_xy-net_crop_size[2])
        mask[mz:mz+mask_size_z, my:my+mask_size_xy, mx:mx+mask_size_xy] = 0
    return mask

def resize_3d(imgs, det_size, mode='linear'):
    new_imgs = []
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        if mode == 'linear':
            temp = cv2.resize(temp, (det_size, det_size), interpolation=cv2.INTER_LINEAR)
        elif mode == 'nearest':
            temp = cv2.resize(temp, (det_size, det_size), interpolation=cv2.INTER_NEAREST)
        else:
            raise AttributeError('No this interpolation mode!')
        new_imgs.append(temp)
    new_imgs = np.asarray(new_imgs)
    return new_imgs

def add_gauss_noise(imgs, min_std=0.1, max_std=0.1, norm_mode='norm'):
    if min_std == max_std:
        std = min_std
    else:
        std = random.uniform(min_std, max_std)
    gaussian = np.random.normal(0, std, (imgs.shape))
    imgs = imgs + gaussian
    if norm_mode == 'norm':
        imgs = (imgs-np.min(imgs)) / (np.max(imgs)-np.min(imgs))
    elif norm_mode == 'trunc':
        imgs[imgs<0] = 0
        imgs[imgs>1] = 1
    else:
        pass
    return imgs

def add_gauss_blur(imgs, kernel_size=5, sigma=0):
    outs = []
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        temp = cv2.GaussianBlur(temp, (kernel_size,kernel_size), sigma)
        outs.append(temp)
    outs = np.asarray(outs, dtype=np.float32)
    outs[outs < 0] = 0
    outs[outs > 1] = 1
    return outs

def add_intensity(imgs, contrast_factor=0.1, brightness_factor=0.1):
    # imgs *= 1 + (np.random.rand() - 0.5) * contrast_factor
    # imgs += (np.random.rand() - 0.5) * brightness_factor
    # imgs = np.clip(imgs, 0, 1)
    # imgs **= 2.0**(np.random.rand()*2 - 1)
    imgs *= 1 + contrast_factor
    imgs += brightness_factor
    imgs = np.clip(imgs, 0, 1)
    return imgs

def interp_5d(data, det_size, mode='bilinear'):
    assert len(data.shape) == 5, "the dimension of data must be 5!"
    out = []
    depth = data.shape[2]
    for k in range(depth):
        temp = data[:,:,k,:,:]
        if mode == 'bilinear':
            temp = F.interpolate(temp, size=(det_size, det_size), mode='bilinear', align_corners=True)
        elif mode == 'nearest':
            temp = F.interpolate(temp, size=(det_size, det_size), mode='nearest')
        out.append(temp)
    out = torch.stack(out, dim=2)
    return out

def convert_consistency_scale(gt, det_size):
    B, C, D, H, W = gt.shape
    gt = gt.detach().clone()
    out_gt = []
    masks = []
    for k in range(B):
        gt_temp = gt[k]
        det_size_temp = det_size[k]
        if det_size_temp[0] == gt_temp.shape[-1]:
            mask = torch.ones_like(gt_temp)
            out_gt.append(gt_temp)
            masks.append(mask)
        elif det_size_temp[0] > gt_temp.shape[-1]:
            shift = int((det_size_temp[0] - gt_temp.shape[-1]) // 2)
            gt_padding = torch.zeros((1, C, D, int(det_size_temp[0]), int(det_size_temp[0]))).float().cuda()
            mask = torch.zeros_like(gt_padding)
            gt_padding[0,:,:,shift:-shift,shift:-shift] = gt_temp
            mask[0,:,:,shift:-shift,shift:-shift] = 1
            # gt_padding = F.interpolate(gt_padding, size=(D, int(gt_temp.shape[-1]), int(gt_temp.shape[-1])), mode='trilinear', align_corners=True)
            gt_padding = interp_5d(gt_padding, int(gt_temp.shape[-1]), mode='bilinear')
            mask = F.interpolate(mask, size=(D, int(gt_temp.shape[-1]), int(gt_temp.shape[-1])), mode='nearest')
            gt_padding = torch.squeeze(gt_padding, dim=0)
            mask = torch.squeeze(mask, dim=0)
            out_gt.append(gt_padding)
            masks.append(mask)
        else:
            shift = int((gt_temp.shape[-1] - det_size_temp[0]) // 2)
            mask = torch.zeros_like(gt_temp)
            mask[:,:,shift:-shift,shift:-shift] = 1
            gt_padding = gt_temp[:,:,shift:-shift,shift:-shift]
            gt_padding = gt_padding[None, ...]
            # gt_padding = F.interpolate(gt_padding, size=(D, int(gt_temp.shape[-1]), int(gt_temp.shape[-1])), mode='trilinear', align_corners=True)
            gt_padding = interp_5d(gt_padding, int(gt_temp.shape[-1]), mode='bilinear')
            gt_padding = torch.squeeze(gt_padding, dim=0)
            out_gt.append(gt_padding)
            masks.append(mask)
    out_gt = torch.stack(out_gt, dim=0)
    masks = torch.stack(masks, dim=0)
    return out_gt, masks

def convert_consistency_flip(gt, rules):
    B, C, D, H, W = gt.shape
    gt = gt.detach().clone()
    rules = rules.data.cpu().numpy()
    out_gt = []
    for k in range(B):
        gt_temp = gt[k]
        rule = rules[k]
        gt_temp = simple_augment_torch(gt_temp, rule)
        out_gt.append(gt_temp)
    out_gt = torch.stack(out_gt, dim=0)
    return out_gt

if __name__ == "__main__":
    test = np.random.random((3,3,18,160,160)).astype(np.float32)
    det_size = np.asarray([[160],[320],[80]], dtype=np.float32)
    test = torch.tensor(test).to('cuda:0')
    det_size = torch.tensor(det_size).to('cuda:0')
    out_gt, masks = convert_consistency_scale(test, det_size)
    print(out_gt.shape)


