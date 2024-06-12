import os
import math
import numpy as np 
from PIL import Image

def show(img3d):
    # only used for image with shape [18, 160, 160, 3]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size*row, size*column, 3), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                if np.max(img3d[index]) > 1:
                    img = (img3d[index]).astype(np.uint8)
                else:
                    img = (img3d[index] * 255).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all

def show_one(img3d):
    # only used for image with shape [18, 160, 160]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size*row, size*column), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                img = (img3d[index] * 255).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all

def show_CE(img3d):
    # only used for image with shape [18, 160, 160]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size*row, size*column), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                img = (img3d[index]).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all

def show_twoImage(raw, affs):
    raw = (raw * 255).astype(np.uint8)
    affs = (affs * 255).astype(np.uint8)
    img1 = raw[0]
    img2 = raw[1]
    blank = np.zeros_like(img1, dtype=np.uint8)
    if affs.shape[0] == 1:
        cat1 = np.concatenate([img1, affs[0]], axis=1)
        cat2 = np.concatenate([img2, blank], axis=1)
    elif affs.shape[0] == 3:
        cat1 = np.concatenate([img1, affs[0], blank], axis=1)
        cat2 = np.concatenate([img2, affs[1], affs[2]], axis=1)
    elif affs.shape[0] == 5:
        cat1 = np.concatenate([img1, affs[0], affs[1], affs[4]], axis=1)
        cat2 = np.concatenate([img2, affs[2], affs[3], blank], axis=1)
    else:
        raise NotImplementedError
    im_cat = np.concatenate([cat1, cat2], axis=0)
    return im_cat

def training_show(iters, inputs, label, pred_bound, cache_path, if_skele=None, skele=None, pred_skele=None):
    img_input = np.repeat(inputs[0].data.cpu().numpy(), 3, 0)
    img_input = np.transpose(img_input, (1,2,3,0))
    img_input = show(img_input)
    input_placehplder = np.zeros_like(img_input, dtype=np.uint8)
    im_cat1 = np.concatenate([img_input, input_placehplder], axis=1)

    img_label = label[0][0:3].data.cpu().numpy()
    img_label = np.transpose(img_label, (1,2,3,0))
    img_label = show(img_label)

    img_pred_bound = pred_bound[0][0:3].data.cpu().numpy()
    img_pred_bound = np.transpose(img_pred_bound, (1,2,3,0))
    img_pred_bound = show(img_pred_bound)
    im_cat2 = np.concatenate([img_pred_bound, img_label], axis=1)

    if if_skele is not None:
        img_skele = np.repeat(skele[0, 0:1].data.cpu().numpy(), 3, 0)
        img_skele = np.transpose(img_skele, (1,2,3,0))
        img_skele = show(img_skele)

        img_pred_skele = np.repeat(pred_skele[0, 0:1].data.cpu().numpy(), 3, 0)
        img_pred_skele = np.transpose(img_pred_skele, (1,2,3,0))
        img_pred_skele = show(img_pred_skele)
        im_cat3 = np.concatenate([img_pred_skele, img_skele], axis=1)

        im_cat = np.concatenate([im_cat1, im_cat2, im_cat3], axis=0)
    else:
        im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def training_show_pretrain(iters, pred, label, cache_path, loss_mode='CrossEntropy'):
    img_input = pred[0].data.cpu().numpy()
    if loss_mode == 'CrossEntropy':
        img_input = show_CE(img_input)
    else:
        img_input[img_input < 0] = 0
        img_input[img_input > 1] = 1
        img_input = show_one(img_input)
    img_label = label[0].data.cpu().numpy()
    img_label = show_one(img_label)
    im_cat = np.concatenate([img_input, img_label], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))


def show_inpaining(iters, pred, label, mask, cache_path):
    pred = pred[0].data.cpu().numpy()
    label = label[0].data.cpu().numpy()
    mask = mask[0].data.cpu().numpy()
    inputs = label * mask
    inputs = np.squeeze(inputs)
    pred = np.squeeze(pred)
    inputs = inputs[14:-14, 106:-106, 106:-106]
    pred[pred < 0] = 0; pred[pred > 1] =1
    pred_img = show_one(pred)
    inputs_img = show_one(inputs)
    im_cat = np.concatenate([inputs_img, pred_img], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))


def show_affs(iters, inputs, pred, target, cache_path, model_type='mala'):
    pred = pred[0].data.cpu().numpy()
    inputs = inputs[0].data.cpu().numpy()
    target = target[0].data.cpu().numpy()
    inputs = np.squeeze(inputs)
    if model_type == 'mala':
        inputs = inputs[14:-14, 106:-106, 106:-106]
    inputs = inputs[:,:,:,np.newaxis]
    inputs = np.repeat(inputs, 3, 3)
    pred = np.transpose(pred, (1,2,3,0))
    target = np.transpose(target, (1,2,3,0))
    inputs[inputs<0]=0; inputs[inputs>1]=1
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    inputs_img = show(inputs)
    pred_img = show(pred)
    target_img = show(target)
    im_cat = np.concatenate([inputs_img, pred_img, target_img], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))


def show_affs2(iters, inputs, pred, target, cache_path):
    pred = pred[0].data.cpu().numpy()
    inputs = inputs[0].data.cpu().numpy()
    target = target[0].data.cpu().numpy()
    inputs = (inputs * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)
    img1 = inputs[0]
    img2 = inputs[1]
    if target.shape[0] == 5:
        cat1 = np.concatenate([img1, pred[0], pred[1], pred[4]], axis=1)
        cat2 = np.concatenate([img2, target[0], target[1], target[4]], axis=1)
    elif target.shape[0] == 3:
        cat1 = np.concatenate([img1, pred[0], pred[1], pred[2]], axis=1)
        cat2 = np.concatenate([img2, target[0], target[1], target[2]], axis=1)
    elif target.shape[0] == 1:
        cat1 = np.concatenate([img1, pred[0]], axis=1)
        cat2 = np.concatenate([img2, target[0]], axis=1)
    else:
        raise NotImplementedError
    im_cat = np.concatenate([cat1, cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))


def class_color(lb):
    d, h, w = lb.shape
    lb_color1 = np.zeros((d, h, w), dtype=np.uint8)
    lb_color2 = np.zeros((d, h, w), dtype=np.uint8)
    lb_color3 = np.zeros((d, h, w), dtype=np.uint8)
    ids_0 = lb == 0
    ids_1 = lb == 1
    lb_color1[ids_0] = 0; lb_color2[ids_0] = 0; lb_color3[ids_0] = 255
    lb_color1[ids_1] = 0; lb_color2[ids_1] = 255; lb_color3[ids_1] = 0
    lb_color = np.concatenate([lb_color1[:,:,:,np.newaxis], lb_color2[:,:,:,np.newaxis], lb_color3[:,:,:,np.newaxis]], axis=3)
    return lb_color


def show_affs_pseudo(iters, inputs, pred, target, mask, cache_path, model_type='mala'):
    pred = pred[0].data.cpu().numpy()
    inputs = inputs[0].data.cpu().numpy()
    target = target[0].data.cpu().numpy()
    mask = mask[0].data.cpu().numpy()
    inputs = np.squeeze(inputs)
    if model_type == 'mala':
        inputs = inputs[14:-14, 106:-106, 106:-106]
    inputs = inputs[:,:,:,np.newaxis]
    inputs = np.repeat(inputs, 3, 3)
    pred = np.transpose(pred, (1,2,3,0))
    target = np.transpose(target, (1,2,3,0))
    affs_z = class_color(target[:, :, :, 0]) * mask[0][:,:,:,np.newaxis]
    affs_y = class_color(target[:, :, :, 1]) * mask[1][:,:,:,np.newaxis]
    affs_x = class_color(target[:, :, :, 2]) * mask[2][:,:,:,np.newaxis]
    inputs_img = show(inputs)
    pred_img = show(pred)
    # target_img = show(target)
    mask = np.transpose(mask, (1,2,3,0))
    mask_img = show(mask)
    affs_z_img = show(affs_z)
    affs_y_img = show(affs_y)
    affs_x_img = show(affs_x)
    # im_cat = np.concatenate([inputs_img, pred_img, target_img], axis=1)
    im_cat1 = np.concatenate([inputs_img, pred_img], axis=1)
    im_cat2 = np.concatenate([mask_img, affs_z_img], axis=1)
    im_cat3 = np.concatenate([affs_y_img, affs_x_img], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2, im_cat3], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def show_affs_whole(iters, out_affs, gt_affs, cache_path):
    out_affs = out_affs[:, -1, ...]
    gt_affs = gt_affs[:, -1, ...]
    out_affs = (out_affs * 255).astype(np.uint8)
    out_affs = np.transpose(out_affs, (1,2,0))
    gt_affs = (gt_affs * 255).astype(np.uint8)
    gt_affs = np.transpose(gt_affs, (1,2,0))
    im_cat = np.concatenate([out_affs, gt_affs], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def show_affs_whole2(iters, out_affs, gt_affs, cache_path, mode):
    if mode == 'x-y-z-2':
        out_affs = out_affs[:, -1, ...]
        gt_affs = gt_affs[:, -1, ...]
        out_affs = (out_affs * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        cat1 = np.concatenate([out_affs[0], out_affs[1], out_affs[2]], axis=1)
        cat2 = np.concatenate([gt_affs[0], gt_affs[1], gt_affs[2]], axis=1)
        im_cat = np.concatenate([cat1, cat2], axis=0)
    else:
        if mode == 'x-y-z':
            out_affs = out_affs[-1, -1, ...]
            gt_affs = gt_affs[0, -1, ...]
        elif mode == 'x-y':
            out_affs = out_affs[-1, ...]
            gt_affs = gt_affs[2, -1, ...]
        elif mode == 'z':
            out_affs = out_affs[-1, ...]
            gt_affs = gt_affs[0, -1, ...]
        else:
            raise NotImplementedError
        
        out_affs = (out_affs * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        im_cat = np.concatenate([out_affs, gt_affs], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def show_affs_consistency(iters, inputs, pred, target, inputs_u, out_u1, out_u2, cache_path):
    pred = pred[0].data.cpu().numpy()
    inputs = inputs[0].data.cpu().numpy()
    target = target[0].data.cpu().numpy()
    inputs = np.squeeze(inputs)
    inputs = inputs[14:-14, 106:-106, 106:-106]
    inputs = inputs[:,:,:,np.newaxis]
    inputs = np.repeat(inputs, 3, 3)
    pred = np.transpose(pred, (1,2,3,0))
    target = np.transpose(target, (1,2,3,0))
    inputs_img = show(inputs)
    pred_img = show(pred)
    target_img = show(target)
    im_cat1 = np.concatenate([inputs_img, pred_img, target_img], axis=1)

    out_u1 = out_u1[0].data.cpu().numpy()
    inputs_u = inputs_u[0].data.cpu().numpy()
    out_u2 = out_u2[0].data.cpu().numpy()
    inputs_u = np.squeeze(inputs_u)
    inputs_u = inputs_u[14:-14, 106:-106, 106:-106]
    inputs_u = inputs_u[:,:,:,np.newaxis]
    inputs_u = np.repeat(inputs_u, 3, 3)
    out_u1 = np.transpose(out_u1, (1,2,3,0))
    out_u2 = np.transpose(out_u2, (1,2,3,0))
    inputs_u_img = show(inputs_u)
    out_u1_img = show(out_u1)
    out_u2_img = show(out_u2)
    im_cat2 = np.concatenate([inputs_u_img, out_u1_img, out_u2_img], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def draw_fragments_3d(pred):
    d,m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the neurons number of pred is %d" % size)
    color_pred = np.zeros([d, m, n, 3])
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,:,i] = color_val[idx]
    color_pred = color_pred
    return color_pred