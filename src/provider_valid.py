import os
import cv2
import h5py
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_affinity import seg_to_aff
# from utils.affinity_official import seg2affs
import psutil
from utils.affinity_ours import gen_affs_mutex_3d

class Provider_valid(Dataset):
    def __init__(self, cfg, valid_data=None, num_z=18, test=False, test_split=None):
        # basic settings
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        self.if_dilate = cfg.DATA.if_dilate
        self.shift_channels = cfg.DATA.shift_channels
        self.num_z = num_z
        self.test = test
        if valid_data is not None:
            valid_dataset_name = valid_data
        else:
            try:
                valid_dataset_name = cfg.DATA.valid_dataset
                print('valid on valid dataset!')
            except:
                valid_dataset_name = cfg.DATA.dataset_name
                print('valid on train dataset!')

        # basic settings
        # the input size of network
        if cfg.MODEL.model_type == 'superhuman':
            self.crop_size = [18, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'mala':
            self.crop_size = [53, 268, 268]
            self.net_padding = [14, 106, 106]  # the edge size of patch reduced by network
        elif cfg.MODEL.model_type == 'segmamba':
            # self.crop_size = [16,256,256]
            self.crop_size = [16,160,160]
            # self.crop_size = [32,320,320]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'unetr':
            self.crop_size = [32, 160, 160]
            self.net_padding = [0, 0, 0]
        else:
            self.crop_size = [16, 160, 160]
            self.net_padding = [0, 0, 0]

        # the output size of network
        # for mala: [25, 56, 56]
        # for superhuman: [18, 160, 160]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # training dataset files (h5), may contain many datasets
        if valid_dataset_name == 'cremiA':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif valid_dataset_name == 'cremiB':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif valid_dataset_name == 'cremiC':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif valid_dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs.h5']
            self.train_labels = ['cremiC_labels.h5']
            # self.sub_path = 'cremi'
            # self.train_datasets = ['cremiA_inputs.h5', 'cremiB_inputs.h5', 'cremiC_inputs.h5']
            # self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif valid_dataset_name == 'wafer4':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer4_raw.h5']
            self.train_labels = ['wafer4_seg.h5']

        elif valid_dataset_name == 'wafer26':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer26_raw.h5']
            self.train_labels = ['wafer26_seg.h5']
        elif valid_dataset_name == 'wafer25':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer25_raw.h5']
            self.train_labels = ['wafer25_seg.h5']

        elif valid_dataset_name == 'isbi':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_inputs.h5']
            self.train_labels = ['isbi_labels.h5']
        elif valid_dataset_name == 'isbi_test':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_test_inputs.h5']
            self.train_labels = ['isbi_test_labels.h5']
        elif valid_dataset_name == 'ac3':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']
        elif valid_dataset_name == 'ac4':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
        elif valid_dataset_name == 'fib':
            self.sub_path = 'fib'
            self.train_datasets = ['fib_inputs.h5']
            self.train_labels = ['fib_labels.h5']
        elif valid_dataset_name == 'wafer':
            self.sub_path = 'wafer'
            # self.train_datasets = ['wafer4_raw.h5', 'wafer25_raw.h5']
            # self.train_labels = ['wafer4_seg.h5', 'wafer25_seg.h5']
            self.train_datasets = ['wafer4_inputs.h5']
            self.train_labels = ['wafer4_labels.h5']
            # self.train_datasets = ['wafer25_inputs.h5', 'wafer26_inputs.h5', 'wafer26_2_inputs.h5', 'wafer36_inputs.h5']
            # self.train_labels = ['wafer25_labels.h5', 'wafer26_labels.h5', 'wafer26_2_labels.h5', 'wafer36_labels.h5']
        elif valid_dataset_name == 'wafer36_2':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer36_2_inputs.h5']
            self.train_labels = ['wafer36_2_labels.h5']
        else:
            raise AttributeError('No this dataset type!')

        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        # split validation data
        # if valid_dataset_name == 'snemi3d-ac4':
        #     self.test_split = 100
        # else:
        if test_split is None:
            self.test_split = cfg.DATA.test_split
        else:
            self.test_split = test_split
        if valid_dataset_name == 'isbi_test' or valid_dataset_name == 'ac3':
            self.test_split = 100
        print('the number of valid(test) = %d' % self.test_split)

        # load dataset
        self.dataset = []
        self.labels = []
        self.labels_origin = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()
            
            if valid_dataset_name == 'ac3':
                print('top data')
                data = data[:self.test_split]
            elif valid_dataset_name == 'wafer':  # 20240422
                print('whole wafer4 data')
                data = np.where(data > 255, 255, data.astype('uint8'))
            elif valid_dataset_name == 'wafer36_2':  # 20240511
                print('whole wafer36_2 data')
            else:
                data = data[-self.test_split:]
            self.dataset.append(data)

            # load labels
            f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
            label = f_label['main'][:]
            f_label.close()
            
            if valid_dataset_name == 'ac3':
                print('top label')
                label = label[:self.test_split]
            elif valid_dataset_name == 'wafer':  # 20240422
                print('whole wafer4 label')
            elif valid_dataset_name == 'wafer36_2':  # 20240511
                print('whole wafer36_2 label')
            else:
                label = label[-self.test_split:]           
            self.labels_origin.append(label.copy())
            if self.if_dilate:
                # temp = genSegMalis(temp, 1)
                label = seg_widen_border(label, tsz_h=1)
            self.labels.append(label)
        self.origin_data_shape = list(self.dataset[0].shape)

        # generate gt affinity
        self.gt_affs = []
        for k in range(len(self.labels)):
            temp = self.labels[k]
            # self.gt_affs.append(seg_to_affgraph(temp, mknhood3d(1), pad='replicate').astype(np.float32))
            self.gt_affs.append(seg_to_aff(temp).astype(np.float32))

        # padding by 'reflect' mode for inference
        if cfg.MODEL.model_type == 'mala':
            self.stride = self.out_size           # [25, 56, 56]
            self.valid_padding = self.net_padding # [14, 106, 106]
            assert self.dataset[0].shape[0] % 25 == 0, "the shape of test data must be 25*"
            padding_z = self.dataset[0].shape[0] // 25
            if 'fib' in valid_dataset_name:
                padding_xy = 10
            else:
                padding_xy = 23
            self.num_zyx = [padding_z, padding_xy, padding_xy]
        else:
            if 'fib' in valid_dataset_name:
                padding_xy = 20
                num_xy = 6
            elif 'cremiA' in valid_dataset_name or 'cremiB' in valid_dataset_name or 'cremiC' in valid_dataset_name or 'wafer4' in valid_dataset_name or 'wafer' in valid_dataset_name:
                padding_xy = 15
                num_xy = 15
            else:
                padding_xy = 48
                num_xy = 13
            if self.dataset[0].shape[0] == 125:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                # self.num_zyx = [15, num_xy, num_xy]
                self.num_zyx = [13, num_xy, num_xy]  # 20240422 num_x=13 num_xy=15
            elif self.dataset[0].shape[0] == 100:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [10, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 80:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [8, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 50:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [5, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 25:
                # for rapid inference
                self.stride = [15, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [2, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 20:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]  # 48
                self.num_zyx = [2, num_xy, num_xy]  # 13
            else:
                raise NotImplementedError

        # only for superhuman and the num-z = 10
        if self.num_z < 18:
            raise NotImplementedError

        for k in range(len(self.dataset)):
            self.dataset[k] = np.pad(self.dataset[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                    (self.valid_padding[1], self.valid_padding[1]), \
                                                    (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
            self.labels[k] = np.pad(self.labels[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                    (self.valid_padding[1], self.valid_padding[1]), \
                                                    (self.valid_padding[2], self.valid_padding[2])), mode='reflect')

        # the training dataset size
        self.raw_data_shape = list(self.dataset[0].shape)

        self.reset_output()
        self.weight_vol = self.get_weight()
        if self.num_z < 18:
            raise NotImplementedError

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
        lb = self.labels[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()

        if self.num_z < 18:
            raise NotImplementedError

        # convert label to affinity
        if self.model_type == 'mala':
            lb = lb[self.net_padding[0]:-self.net_padding[0], \
                    self.net_padding[1]:-self.net_padding[1], \
                    self.net_padding[2]:-self.net_padding[2]]
        # lb = genSegMalis(lb, 1)
        # lb_affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)
        if self.shift_channels is None:
            lb_affs = seg_to_aff(lb).astype(np.float32)
        else:
            lb_affs = gen_affs_mutex_3d(lb, shift=self.shift_channels,
                                        padding=True, background=True)

        # generate weights map for affinity
        # weight_factor = np.sum(lb_affs) / np.size(lb_affs)
        # weight_factor = np.clip(weight_factor, 1e-3, 1)
        # weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)
        weightmap = weight_binary_ratio(lb_affs)

        imgs = imgs.astype(np.float32) / 255.0
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        return imgs, lb_affs, weightmap

    def __len__(self):
        return self.iters_num

    def reset_output(self):
        if self.shift_channels is None:
            channel = 3
        else:
            channel = len(self.shift_channels)
        if not self.model_type == 'mala':
            self.out_affs = np.zeros(tuple([channel]+self.raw_data_shape), dtype=np.float32)
            self.weight_map = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)
        else:
            self.out_affs = np.zeros(tuple([channel]+self.origin_data_shape), dtype=np.float32)
            self.weight_map = np.zeros(tuple([1]+self.origin_data_shape), dtype=np.float32)


    def get_weight(self, sigma=0.2, mu=0.0):
        if self.num_z < 18:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, 18, dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        else:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def add_vol(self, affs_vol):
        fromz, fromy, fromx = self.pos
        if self.num_z < 16:
            raise NotImplementedError

        if not self.model_type == 'mala':
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
        if not self.model_type == 'mala':
            self.out_affs = self.out_affs / self.weight_map
            if self.valid_padding[0] == 0:
                self.out_affs = self.out_affs[:, :, \
                                                self.valid_padding[1]:-self.valid_padding[1], \
                                                self.valid_padding[2]:-self.valid_padding[2]]
            else:
                self.out_affs = self.out_affs[:, self.valid_padding[0]:-self.valid_padding[0], \
                                                self.valid_padding[1]:-self.valid_padding[1], \
                                                self.valid_padding[2]:-self.valid_padding[2]]
        return self.out_affs

    def get_gt_affs(self, num_data=0):
        return self.gt_affs[num_data].copy()

    def get_gt_lb(self, num_data=0):
        # lbs = self.labels_origin[num_data].copy()
        # lbs = lbs[self.valid_padding[0]:-self.valid_padding[0], \
        #           self.valid_padding[1]:-self.valid_padding[1], \
        #           self.valid_padding[2]:-self.valid_padding[2]]
        lbs = self.labels_origin[num_data]
        return lbs

    def get_raw_data(self, num_data=0):
        out = self.dataset[num_data].copy()
        return out[self.valid_padding[0]:-self.valid_padding[0], \
                    self.valid_padding[1]:-self.valid_padding[1], \
                    self.valid_padding[2]:-self.valid_padding[2]]


if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    import time
    import torch
    from utils.show import show_one
    from sklearn.metrics import f1_score

    seed = 555
    np.random.seed(seed)
    random.seed(seed)
    cfg_file = 'seg_onlylb_suhu_wbce_lr01_snemi3d_data25.yaml'
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict( yaml.load(f) )
    
    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    data = Provider_valid(cfg)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True)
    
    gt_affs = data.get_gt_affs()
    pred = np.random.random(tuple(gt_affs.shape)).astype(np.float32)
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    gt_affs = gt_affs.astype(np.uint8)
    pred = pred.astype(np.uint8)
    gt_affs = gt_affs.flatten()
    pred = pred.flatten()
    f1 = f1_score(1 - gt_affs, 1- pred)
    print(f1)

    # t = time.time()
    # for k, batch in enumerate(dataloader, 0):
    #     inputs, target, wrightmap = batch
    #     target = target.data.numpy()
    #     data.add_vol(target[0])
    # out_affs = data.get_results()
    # for k in range(out_affs.shape[1]):
    #     affs_xy = out_affs[2, k]
    #     affs_xy = (affs_xy * 255).astype(np.uint8)
    #     Image.fromarray(affs_xy).save(os.path.join(out_path, str(k).zfill(4)+'.png'))
    #     # print('single cost time: ', time.time()-t1)
    #     # tmp_data = np.squeeze(tmp_data)
    #     # if cfg.MODEL.model_type == 'mala':
    #     #     tmp_data = tmp_data[14:-14,106:-106,106:-106]
    #     # affs_xy = affs[2]
    #     # weightmap_xy = weightmap[2]

    #     # img_data = show_one(tmp_data)
    #     # img_affs = show_one(affs_xy)
    #     # img_weight = show_one(weightmap_xy)
    #     # im_cat = np.concatenate([img_data, img_affs, img_weight], axis=1)
    #     # Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
    # print(time.time() - t)