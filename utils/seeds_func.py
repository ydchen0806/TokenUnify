import os
import numpy as np
import h5py
from scipy import ndimage
import cv2
import mahotas
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# generate affinity
def seg_to_affgraph(seg, nhood=np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                         seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                        * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                        * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )

    return aff



# generate seeds 
def gen_seeds(labels, affs_xy, min_size=10):
    # remove some neurons whose size is smaller than min_size
    ids, count = np.unique(labels, return_counts=True)
    for i, icount in enumerate(count):
        if icount < min_size: 
            labels[labels == ids[i]] = 0 
    
    boundary = np.ones_like(affs_xy)
    boundary[1:-1, 1:-1] = affs_xy[1:-1, 1:-1]
    boundary[boundary != 0] = 1 

    distance = mahotas.distance(boundary<0.5)
    seeds = np.zeros_like(labels)
    ite = 1
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = labels == label
        label_mask = label_mask.astype(np.int)
        temp_dis = np.multiply(distance, label_mask)
        max_where = np.where(temp_dis == np.max(temp_dis))
        seeds[max_where[0][0], max_where[1][0]] = ite
        ite += 1
    return seeds, boundary


def gen_seeds_2(labels, affs_xy, min_size=10):
    # remove some neurons whose size is smaller than min_size
    ids, count = np.unique(labels, return_counts=True)
    for i, icount in enumerate(count):
        if icount < min_size: 
            labels[labels == ids[i]] = 0 
    
    boundary = np.ones_like(affs_xy)
    boundary[1:-1, 1:-1] = affs_xy[1:-1, 1:-1]
    boundary[boundary != 0] = 1 

    distance = mahotas.distance(boundary<0.5)
    seeds = np.zeros_like(labels)
    # ite = 1
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = labels == label
        label_mask = label_mask.astype(np.int)
        temp_dis = np.multiply(distance, label_mask)
        max_where = np.where(temp_dis == np.max(temp_dis))
        seeds[max_where[0][0], max_where[1][0]] = label
        # ite += 1
    return seeds


# erosion labels
def erosion_labels(gt, steps=1):
    self_background = 0
    foreground = np.zeros(shape=gt.shape, dtype=np.bool)
    for label in np.unique(gt):
        if label == self_background:
            continue
        label_mask = gt==label
        # Assume that masked out values are the same as the label we are
        # eroding in this iteration. This ensures that at the boundary to
        # a masked region the value blob is not shrinking.
        eroded_label_mask = ndimage.binary_erosion(label_mask, iterations=steps, border_value=1)
        foreground = np.logical_or(eroded_label_mask, foreground)
    background = np.logical_not(foreground)
    gt[background] = self_background
    return gt 


# draw fragments
def draw_fragments(picture, raw=None, alpha=0.3):
    m,n = picture.shape
    ids = np.unique(picture)
    size = len(ids)
    print("The number of nuerons is %d" % size)
    color = np.zeros([m, n, 3])
    idx = np.searchsorted(ids, picture)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color[:,:,i] = color_val[idx]
    color = color / 255
    if raw is not None:
        plt.figure()
        plt.subplots(figsize=(10,10))
        plt.imshow(raw)
        plt.imshow(color, alpha=alpha)
        plt.axis('off')
        plt.show()
    else:
        plt.figure()
        plt.subplots(figsize=(10,10))
        plt.imshow(color)
        plt.axis('off')
        plt.show()


# thresdholding
def binary_thresholding(img, t=0.5):
    if np.max(img) > 1.0:
        img = img / 255.0
    img[img >= t] = 1
    img[img < t] = 0
    return img


# draw seeds
def draw_seeds(raw, seeds):
    plt.figure(figsize=(10,10))
    plt.imshow(raw, cmap='gray')
    seeds_listx, seeds_listy = np.where(seeds != 0)
    plt.scatter(seeds_listy, seeds_listx, c='r')
    plt.axis('off')
    plt.show()


def draw_seeds_v2(raw, seeds):
    plt.figure(figsize=(10,10))
    plt.imshow(raw, cmap='gray')
    seeds_listx = seeds[:, 0].astype(np.int)
    seeds_listy = seeds[:, 1].astype(np.int)
    plt.scatter(seeds_listy, seeds_listx, c='r')
    plt.axis('off')
    plt.show()


def draw_box(img, box):
    img_box = img.copy()
    if len(img_box.shape) == 2:
        img_box = img_box[:,:,np.newaxis]
        img_box = np.concatenate([img_box, img_box, img_box], axis=2)
    for i in range(1, box.shape[0]):
        position = box[i]
        x1 = position[0]
        y1 = position[1]
        x2 = x1 + position[2]
        y2 = y1 + position[3]
        img_box = cv2.rectangle(img_box, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    plt.figure(figsize=(10,10))
    plt.imshow(img_box)
    plt.axis('off')
    plt.show()


# draw affinity
def draw_general(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def make_summary_plot(it, raw, output, net_output, seeds, target):
    """
    This function create and save a summary figure
    """
    f, axarr = plt.subplots(2, 2, figsize=(8, 9.5))
    f.suptitle("RW summary, Iteration: " + repr(it))

    axarr[0, 0].set_title("Ground Truth Image")
    axarr[0, 0].imshow(raw[0].detach().numpy(), cmap="gray")
    axarr[0, 0].imshow(target[0, 0].detach().numpy(), alpha=0.6, vmin=-3, cmap="prism_r")
    seeds_listx, seeds_listy = np.where(seeds[0].data != 0)
    axarr[0, 0].scatter(seeds_listy,
                        seeds_listx, c="r")
    axarr[0, 0].axis("off")

    axarr[0, 1].set_title("LRW output (white seed)")
    axarr[0, 1].imshow(raw[0].detach().numpy(), cmap="gray")
    axarr[0, 1].imshow(np.argmax(output[0][0].detach().numpy(), 0), alpha=0.6, vmin=-3, cmap="prism_r")
    axarr[0, 1].axis("off")

    axarr[1, 0].set_title("Vertical Diffusivities")
    axarr[1, 0].imshow(net_output[0, 0].detach().numpy(), cmap="gray")
    axarr[1, 0].axis("off")

    axarr[1, 1].set_title("Horizontal Diffusivities")
    axarr[1, 1].imshow(net_output[0, 1].detach().numpy(), cmap="gray")
    axarr[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("./results/%04i.png"%it)
    plt.close()


def draw_fragments_seeds(out_path, k, pred, pred_seed, gt, gt_seed, f_txt, raw=None, alpha=0.8):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("k = %d, the neurons number of pred is %d" % (k, size))
    f_txt.write("k = %d, the neurons number of pred is %d" % (k, size))
    f_txt.write('\n')
    color_pred = np.zeros([m, n, 3])
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    color_pred = color_pred / 255
    if pred_seed is not None:
        pred_seeds_listx, pred_seeds_listy = np.where(pred_seed != 0)
    
    ids = np.unique(gt)
    size = len(ids)
    print("k = %d, the neurons number of gt is %d" % (k, size))
    f_txt.write("k = %d, the neurons number of gt is %d" % (k, size))
    f_txt.write('\n')
    color_gt= np.zeros([m, n, 3])
    idx = np.searchsorted(ids, gt)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_gt[:,:,i] = color_val[idx]
    color_gt = color_gt / 255
    if gt_seed is not None:
        gt_seeds_listx, gt_seeds_listy = np.where(gt_seed != 0)

    if raw is not None:
        plt.figure(figsize=(20,20),dpi=100)
        plt.subplot(121)
        plt.imshow(raw)
        plt.imshow(color_pred, alpha=alpha)
        if pred_seed is not None:
            plt.scatter(pred_seeds_listy, pred_seeds_listx, c='k', marker='.')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(raw)
        plt.imshow(color_gt, alpha=alpha)
        if gt_seed is not None:
            plt.scatter(gt_seeds_listy, gt_seeds_listx, c='k', marker='.')
        plt.axis('off')
        # plt.show()
        plt.savefig(os.path.join(out_path, str(k).zfill(4)+'.png'), bbox_inches = 'tight')
    else:
        plt.figure(figsize=(20,20),dpi=100)
        plt.subplot(121)
        plt.imshow(color_pred)
        if pred_seed is not None:
            plt.scatter(pred_seeds_listy, pred_seeds_listx, c='b')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(color_gt)
        if gt_seed is not None:
            plt.scatter(gt_seeds_listy, gt_seeds_listx, c='b')
        plt.axis('off')
        # plt.show()
        plt.savefig(os.path.join(out_path, str(k).zfill(4)+'.png'), bbox_inches = 'tight')
    plt.close('all')


def draw_fragments_noseeds(out_path, k, pred, gt=None, raw=None, alpha=0.8):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("k = %d, the neurons number of pred is %d" % (k, size))
    color_pred = np.zeros([m, n, 3])
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    color_pred = color_pred / 255
    
    if gt is not None:
        ids = np.unique(gt)
        size = len(ids)
        print("k = %d, the neurons number of gt is %d" % (k, size))
        color_gt= np.zeros([m, n, 3])
        idx = np.searchsorted(ids, gt)
        for i in range(3):
            color_val = np.random.randint(0, 255, ids.shape)
            if ids[0] == 0:
                color_val[0] = 0
            color_gt[:,:,i] = color_val[idx]
        color_gt = color_gt / 255

    if gt is not None:
        if raw is not None:
            plt.figure(figsize=(20,20),dpi=100)
            plt.subplot(121)
            plt.imshow(raw)
            plt.imshow(color_pred, alpha=alpha)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(raw)
            plt.imshow(color_gt, alpha=alpha)
            plt.axis('off')
        else:
            plt.figure(figsize=(20,20),dpi=100)
            plt.subplot(121)
            plt.imshow(color_pred)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(color_gt)
            plt.axis('off')
    else:
        if raw is not None:
            plt.figure(figsize=(20,20),dpi=100)
            plt.imshow(raw)
            plt.imshow(color_pred, alpha=alpha)
            plt.axis('off')
        else:
            plt.figure(figsize=(20,20),dpi=100)
            plt.imshow(color_pred)
            plt.axis('off')
    plt.savefig(os.path.join(out_path, str(k).zfill(4)+'.png'), bbox_inches = 'tight')
    plt.close('all')


def draw_fragments_3d(out_path, pred, gt=None, raw=None, alpha=0.8):
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
    color_pred = color_pred / 255
    
    if gt is not None:
        ids = np.unique(gt)
        size = len(ids)
        print("the neurons number of gt is %d" % size)
        color_gt= np.zeros([d, m, n, 3])
        idx = np.searchsorted(ids, gt)
        for i in range(3):
            color_val = np.random.randint(0, 255, ids.shape)
            if ids[0] == 0:
                color_val[0] = 0
            color_gt[:,:,:,i] = color_val[idx]
        color_gt = color_gt / 255

    if gt is not None:
        if raw is not None:
            for k in range(d):
                plt.figure(figsize=(20,20),dpi=100)
                plt.subplot(121)
                plt.imshow(raw[k])
                plt.imshow(color_pred[k], alpha=alpha)
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(raw[k])
                plt.imshow(color_gt[k], alpha=alpha)
                plt.axis('off')
                plt.savefig(os.path.join(out_path, str(k).zfill(4)+'.png'), bbox_inches = 'tight')
                plt.close('all')
        else:
            for k in range(d):
                plt.figure(figsize=(20,20),dpi=100)
                plt.subplot(121)
                plt.imshow(color_pred[k])
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(color_gt[k])
                plt.axis('off')
                plt.savefig(os.path.join(out_path, str(k).zfill(4)+'.png'), bbox_inches = 'tight')
                plt.close('all')
    else:
        if raw is not None:
            for k in range(d):
                plt.figure(figsize=(20,20),dpi=100)
                plt.imshow(raw[k])
                plt.imshow(color_pred[k], alpha=alpha)
                plt.axis('off')
                plt.savefig(os.path.join(out_path, str(k).zfill(4)+'.png'), bbox_inches = 'tight')
                plt.close('all')
        else:
            for k in range(d):
                plt.figure(figsize=(20,20),dpi=100)
                plt.imshow(color_pred[k])
                plt.axis('off')
                plt.savefig(os.path.join(out_path, str(k).zfill(4)+'.png'), bbox_inches = 'tight')
                plt.close('all')


if __name__ == "__main__":
    in_path1 = '../data/snemi3d/AC4_inputs.h5'
    in_path2 = '../data/snemi3d/AC4_labels.h5'
    f = h5py.File(in_path1, 'r')
    raw = f['main'][:]
    f.close()

    f = h5py.File(in_path2, 'r')
    labels = f['main'][:]
    f.close()

    out_path = '../data/snemi3d/AC4'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    draw_fragments_3d(out_path, labels, None, raw)