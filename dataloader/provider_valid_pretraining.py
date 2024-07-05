import os
import h5py
import random
import numpy as np
from torch.utils.data import Dataset


class Provider_valid(Dataset):
    def __init__(self, cfg, if_overlap=True):
        # basic settings
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        self.if_overlap = if_overlap

        # basic settings
        # the input size of network
        if cfg.MODEL.model_type == 'superhuman':
            self.crop_size = [18, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'mala':
            self.crop_size = [53, 268, 268]
            self.net_padding = [14, 106, 106]  # the edge size of patch reduced by network
        else:
            raise AttributeError('No this model type!')

        # the output size of network
        # for mala: [25, 56, 56]
        # for superhuman: [18, 160, 160]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # training dataset files (h5), may contain many datasets
        if cfg.DATA.dataset_name == 'cremi-A' or cfg.DATA.dataset_name == 'cremi':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs_interp.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-B':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs_interp.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-C':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs_interp.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
            self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'snemi3d-ac3' or cfg.DATA.dataset_name == 'snemi3d':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']
        elif cfg.DATA.dataset_name == 'snemi3d-ac4':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
        elif cfg.DATA.dataset_name == 'fib-25':
            self.sub_path = 'fib'
            self.train_datasets = ['fib_inputs.h5']
            self.train_labels = ['fib_labels.h5']
        else:
            raise AttributeError('No this dataset type!')

        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        # split validation data
        self.test_split = cfg.DATA.test_split

        self.if_order_aug = False
        self.if_mask_aug = cfg.DATA.if_mask_aug_unlabel
        if self.if_mask_aug:
            if 'snemi3d' in cfg.DATA.dataset_name and cfg.MODEL.model_type == 'superhuman':
                mask_path = os.path.join(cfg.DATA.data_folder, 'snemi3d', 'masks_snemi3d_suhu.h5')
            elif 'cremi' in cfg.DATA.dataset_name and cfg.MODEL.model_type == 'superhuman':
                mask_path = os.path.join(cfg.DATA.data_folder, 'cremi', 'masks_cremi_suhu.h5')
            elif 'fib' in cfg.DATA.dataset_name and cfg.MODEL.model_type == 'superhuman':
                mask_path = os.path.join(cfg.DATA.data_folder, 'fib', 'masks_fib_suhu.h5')
            else:
                raise AttributeError('Please prepare corresponding mask!')
            f = h5py.File(mask_path, 'r')
            self.masks = f['main'][:]
            f.close()

        # load dataset
        self.dataset = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()
            # The default is to use the top 25 sections of volumes as validation set to select models for simplicity
            data = data[:self.test_split]
            self.dataset.append(data)

        self.origin_data_shape = list(self.dataset[0].shape)

        # padding by 'reflect' mode for mala network
        if cfg.MODEL.model_type == 'mala':
            raise NotImplementedError
        else:
            if self.if_overlap:
                self.stride = [15, 80, 80]
                if 'cremi' in cfg.DATA.dataset_name:
                    self.valid_padding = [4, 48, 48]
                    self.num_zyx = [2, 13, 13]
                elif 'snemi3d' in cfg.DATA.dataset_name:
                    self.valid_padding = [4, 48, 48]
                    self.num_zyx = [2, 13, 13]
                elif 'fib' in cfg.DATA.dataset_name:
                    self.valid_padding = [4, 20, 20]
                    self.num_zyx = [2, 6, 6]
                else:
                    raise AttributeError('No this dataset type!')
            else:
                raise NotImplementedError

        for k in range(len(self.dataset)):
            self.dataset[k] = np.pad(self.dataset[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                    (self.valid_padding[1], self.valid_padding[1]), \
                                                    (self.valid_padding[2], self.valid_padding[2])), mode='reflect')

        # the training dataset size
        self.raw_data_shape = list(self.dataset[0].shape)

        self.reset_output()
        self.weight_vol = self.get_weight()
        if not self.if_overlap:
            self.weight_vol = np.ones_like(self.weight_vol, dtype=np.float32)

        # the number of inference times
        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset * len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        pos_data = index // self.num_per_dataset
        pre_data = index % self.num_per_dataset
        pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
        pos_x = pos_xy // self.num_zyx[2]
        pos_y = pos_xy % self.num_zyx[2]

        # find position
        fromz = pos_z * self.stride[0]
        endz = fromz + self.crop_size[0]
        if endz > self.raw_data_shape[0]:
            endz = self.raw_data_shape[0]
            fromz = endz - self.crop_size[0]
        fromy = pos_y * self.stride[1]
        endy = fromy + self.crop_size[1]
        if endy > self.raw_data_shape[1]:
            endy = self.raw_data_shape[1]
            fromy = endy - self.crop_size[1]
        fromx = pos_x * self.stride[2]
        endx = fromx + self.crop_size[2]
        if endx > self.raw_data_shape[2]:
            endx = self.raw_data_shape[2]
            fromx = endx - self.crop_size[2]
        self.pos = [fromz, fromy, fromx]

        imgs = self.dataset[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()
        imgs = imgs.astype(np.float32) / 255.0
        gt = imgs.copy()

        if self.if_order_aug:
            imgs = order_aug(imgs)
        if self.if_mask_aug:
            mask = self.masks[index]
            imgs = imgs * mask

        imgs = imgs[np.newaxis, ...]
        gt = gt[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        gt = np.ascontiguousarray(gt, dtype=np.float32)
        return imgs, gt

    def __len__(self):
        return self.iters_num

    def reset_output(self):
        if self.model_type == 'superhuman':
            self.out_affs = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)
            self.weight_map = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)
        else:
            self.out_affs = np.zeros(tuple([1]+self.origin_data_shape), dtype=np.float32)
            self.weight_map = np.zeros(tuple([1]+self.origin_data_shape), dtype=np.float32)

    def get_weight(self, sigma=0.2, mu=0.0):
        zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                 np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                 np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def add_vol(self, affs_vol):
        fromz, fromy, fromx = self.pos
        if self.model_type == 'superhuman':
            self.out_affs[:, fromz:fromz+self.out_size[0], \
                             fromx:fromx+self.out_size[1], \
                             fromy:fromy+self.out_size[2]] += affs_vol * self.weight_vol
            self.weight_map[:, fromz:fromz+self.out_size[0], \
                               fromx:fromx+self.out_size[1], \
                               fromy:fromy+self.out_size[2]] += self.weight_vol
        else:
            self.out_affs[:, fromz:fromz+self.out_size[0], \
                             fromx:fromx+self.out_size[1], \
                             fromy:fromy+self.out_size[2]] = affs_vol

    def get_results(self):
        if self.model_type == 'superhuman':
            self.out_affs = self.out_affs / self.weight_map
            self.out_affs = self.out_affs[:, self.valid_padding[0]:-self.valid_padding[0], \
                                             self.valid_padding[1]:-self.valid_padding[1], \
                                             self.valid_padding[2]:-self.valid_padding[2]]
        return self.out_affs

    def get_gt_affs(self):
        out_data = self.dataset[0].copy()
        out_data = out_data[np.newaxis, ...]
        out_data = out_data[:, self.valid_padding[0]:-self.valid_padding[0], \
                                self.valid_padding[1]:-self.valid_padding[1], \
                                self.valid_padding[2]:-self.valid_padding[2]]
        out_data = out_data.astype(np.float32) / 255.0
        return out_data

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

if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    import time
    import torch
    from PIL import Image

    seed = 555
    np.random.seed(seed)
    random.seed(seed)
    cfg_file = 'pretraining_snemi3d.yaml'
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict( yaml.load(f) )
    
    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    data = Provider_valid(cfg)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True)

    t = time.time()
    for k, batch in enumerate(dataloader, 0):
        inputs, target = batch
        inputs = inputs.data.numpy()
        target = target.data.numpy()
        data.add_vol(inputs[0])
    out_affs = data.get_results()
    for k in range(out_affs.shape[1]):
        affs_xy = out_affs[0, k]
        affs_xy = (affs_xy * 255).astype(np.uint8)
        Image.fromarray(affs_xy).save(os.path.join(out_path, str(k).zfill(4)+'.png'))
    print(time.time() - t)