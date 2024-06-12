import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from glob import glob
import os
import h5py

class TotalDataProvider(data.Dataset):
    def __init__(self, data_path, partition_number=0, total_partitions=6):
        self.data_path = data_path
        self.partition_number = partition_number
        self.total_partitions = total_partitions
        self.data_list = sorted(glob(os.path.join(data_path, '*h5')))

        # Calculate partition size
        total_size = len(self.data_list)
        self.partition_size = total_size // self.total_partitions
        self.start_index = self.partition_size * partition_number
        self.end_index = (self.start_index + self.partition_size 
                          if partition_number < self.total_partitions - 1 
                          else total_size)

    def __len__(self):
        return self.end_index - self.start_index

    def scaler01(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def __getitem__(self, index):
        # Adjust index for the partition
        adjusted_index = self.start_index + index
        data_path = self.data_list[adjusted_index]
        with h5py.File(data_path, 'r') as f:
            data = f['main'][:]
            data = self.scaler01(data)
            data = torch.from_numpy(data).float()
            data = data.unsqueeze(0)
        return data, data_path

if __name__ == '__main__':
    import sys
    sys.path.append('/data/ydchen/VLP/wafer4')
    from model_superhuman2 import UNet_PNI
    from monai.inferers import sliding_window_inference
    from omegaconf import OmegaConf
    import random 
    from matplotlib import pyplot as plt
    from collections import OrderedDict

    cfg = OmegaConf.load('/data/ydchen/VLP/wafer4/config/seg_3d_ac4_data80.yaml')
    device = torch.device('cuda:0')
    model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                    out_planes=cfg.MODEL.output_nc,
                    filters=cfg.MODEL.filters,
                    upsample_mode=cfg.MODEL.upsample_mode,
                    decode_ratio=cfg.MODEL.decode_ratio,
                    merge_mode=cfg.MODEL.merge_mode,
                    pad_mode=cfg.MODEL.pad_mode,
                    bn_mode=cfg.MODEL.bn_mode,
                    relu_mode=cfg.MODEL.relu_mode,
                    init_mode=cfg.MODEL.init_mode)
    ckpt_path = os.path.join('/LSEM/wafer_seg/model_trained_superhuman/model/model-132000.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    data_path = '/LSEM/user/chenyinda/total_mec_seg_final/affinity'
    save_dir = '/data/ydchen/VLP/wafer4/data/temp'
    os.makedirs(save_dir, exist_ok=True)
    dataset = TotalDataProvider(data_path)
    provider = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(provider):
            data, path = batch
            data = data.to(device)
            data = data.squeeze()
            if not 'affinity' in data_path:
                output = sliding_window_inference(data, (18, 160, 160), 8, model, overlap=0.25)
            else:
                output = data
            print(output.shape)
            total_len = output.shape[1]
            for slice_num in range(0, total_len, 10):
                
                # slice_num = random.randint(0, total_len-1)
                plt.subplot(1, 2, 1)
                plt.imshow(data[:, slice_num, :, :].cpu().numpy().transpose(1,2,0))
                plt.subplot(1, 2, 2)
                plt.imshow(output[:, slice_num, :, :].cpu().numpy().transpose(1, 2, 0))
                plt.savefig(os.path.join(save_dir, f'{i}_{slice_num}.png'))


            break