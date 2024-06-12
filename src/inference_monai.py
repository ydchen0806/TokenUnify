
import argparse
import datetime
import json
import numpy as np
import os
import time
import timm.optim.optim_factory as optim_factory
from pathlib import Path
from loss.loss import WeightedMSE
import torch
import torch.backends.cudnn as cudnn
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util_mamba.lr_decay as lrd
import util_mamba.misc as misc
from util_mamba.datasets import build_dataset
from util_mamba.pos_embed import interpolate_pos_embed
from util_mamba.misc import NativeScalerWithGradNormCount as NativeScaler
from data_provider_labeled import Train as Trainset
import yaml
from attrdict import AttrDict
from utils.show import show_one
from utils.shift_channels import shift_func
from segmamba import SegMamba
from monai.inferers import sliding_window_inference
import waterz
import h5py
from utils.fragment import watershed, randomlabel, relabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import cv2
from model_superhuman2 import UNet_PNI


from model_superhuman_pea import UNet_PNI_embedding_deep as UNet_PNI_pea
from loss.loss_embedding_mse import embedding_loss_norm1, embedding_loss_norm5
from loss.loss import BCELoss, WeightedBCE, MSELoss, WeightedMSE
import torch.nn.functional as F
from collections import OrderedDict


from model_unetr import UNETR
from unet3d_mala import UNet3D_MALA

# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144

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

if __name__ == "__main__":
    
    out_path = os.path.join('/h3cstore_ns/hyshi/EM_mamba_new/inference')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(1)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)


    device = torch.device('cuda')

    # cfg_file = 'segmamba_3d_ac4_data80'
    cfg_file = 'seg_all_3d_ac4_data80'
    with open('/h3cstore_ns/hyshi/configs/' + cfg_file + '.yaml', 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    dataset = Trainset(cfg, [16,160,160])  # [16,256,256]
    valid_data, valid_label, valid_affs = dataset.valid_provide(valid_dataset=cfg.DATA.valid_dataset)
    valid_data = valid_data / 255

    if cfg.MODEL.model_type == 'superhuman':
        print("load superhuman model!")
        model = UNet_PNI(
            in_planes=cfg.MODEL.input_nc,
            out_planes=cfg.MODEL.output_nc,
            filters=cfg.MODEL.filters,
            upsample_mode=cfg.MODEL.upsample_mode,
            decode_ratio=cfg.MODEL.decode_ratio,
            pad_mode=cfg.MODEL.pad_mode,
            bn_mode=cfg.MODEL.bn_mode,
            relu_mode=cfg.MODEL.relu_mode,
            init_mode=cfg.MODEL.init_mode
            )
    elif cfg.MODEL.model_type == 'unetr':
        print("load UNETR model!")
        model = UNETR(
                in_channels=cfg.MODEL.input_nc,
                out_channels=cfg.MODEL.output_nc,
                img_size=cfg.MODEL.unetr_size,
                patch_size=cfg.MODEL.patch_size,
                feature_size=16,
                hidden_size=768,
                mlp_dim=2048,
                num_heads=8,
                pos_embed='perceptron',
                norm_name='instance',
                conv_block=True,
                res_block=True,
                kernel_size=cfg.MODEL.kernel_size,
                skip_connection=False,
                show_feature=False,
                dropout_rate=0.1)  #model_unetr.py的UNETR
    elif cfg.MODEL.model_type == 'segmamba':
        print("load segmamba model!")
        model = SegMamba(in_chans=1, out_chans=3)
    elif cfg.MODEL.model_type == 'mala':
        print("load mala model!")
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, 
                            if_sigmoid=cfg.MODEL.if_sigmoid,
                            init_mode=cfg.MODEL.init_mode_mala)

    # model = SegMamba(in_chans=1, out_chans=3)

    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_ac34_lr5_8gpu_b20_16_160_160_pre_ar_11_320_3090/checkpoint-270.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_ac34_lr5_8gpu_b20_16_160_160_pre_mae399/checkpoint-40.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_ac34_lr5_8gpu_b20_16_160_160_3090/checkpoint-330.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_ac3_lr5_b12_32_160_160_gaussian_8gpu/checkpoint-420.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_ac34_lr5_8gpu_b12_32_160_160_3090_MAE399_250_1/checkpoint-160.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac34_lr5_8gpu_b20_3090_MAE399/checkpoint-40.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_18_160_160_8gpu/checkpoint-352.pth')['model'])


    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_b20_16_160_160_pre_ar_11_370/checkpoint-1150.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_8gpu_b20_16_160_160_pre_1/checkpoint-1290.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_pre_monai_wafer_lr5_b12_32_160_160_8gpu0426/checkpoint-730.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_monai_wafer_lr5_b12_32_160_160_8gpu/checkpoint-460.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_8gpu_b10_16_160_160_xu799/checkpoint-540.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_monai_wafer_lr5_b20_18_160_160_gaussian_8gpu_0424/checkpoint-350.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_pre_wafer_lr5_8gpu_0426_3/checkpoint-700.pth')['model'])


    model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_b20_16_160_160_pre_ar_11_370/checkpoint-1160.pth')['model'])



    model = model.to(device)


    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmambaV3_ac3_lr5_b6_16_160_160_gaussian_8gpu/checkpoint-696.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_anisov1_ac3_lr5_b20_16_160_160_gaussian_8gpu/checkpoint-520.pth')['model'])
    # model = model.to(device)

    model.eval()
    with torch.no_grad():
        if not isinstance(valid_data, torch.Tensor):
            valid_data = valid_data.astype(np.float32)
            # valid_label = valid_label.astype(np.float32)
            valid_affs = valid_affs.astype(np.float32)
            valid_data = torch.tensor(valid_data, dtype = torch.float32)
            # valid_label = torch.tensor(valid_label, dtype = torch.float32)
            valid_affs = torch.tensor(valid_affs, dtype = torch.float32)
            valid_data = valid_data.unsqueeze(0).unsqueeze(0)
            valid_affs = valid_affs.unsqueeze(0)
            # valid_data = torch.squeeze(valid_data, 0)
        valid_data = valid_data.to(device, non_blocking=True)
        # valid_label = valid_label.to(device, non_blocking=True)
        valid_affs = valid_affs.to(device, non_blocking=True)
        pred_data = sliding_window_inference(valid_data, (16,160,160), 20, model, overlap=0.25, mode="gaussian") # parameters: data_raw_valid, model_input_size, valid batch size, model, 0.25
        
        assert pred_data.shape == valid_affs.shape, f"pred_data shape: {pred_data.shape}, valid_affs shape: {valid_affs.shape}"
        # pred_data = sliding_window_inference(valid_data, (16,256,256), 64, model)
        valid_mse_loss = torch.nn.functional.mse_loss(pred_data, valid_affs)
        pred_data = pred_data.squeeze(0)
        pred_data = pred_data.cpu().numpy()
        valid_affs = valid_affs.squeeze(0)
        valid_affs = valid_affs.cpu().numpy()
    valid_data = valid_data.squeeze(0).squeeze(0)
    valid_data = valid_data.cpu().numpy()
    # np.save('/h3cstore_ns/hyshi/valid_data_wafer4.npy', valid_data)
    # np.save('/h3cstore_ns/hyshi/valid_label_wafer4.npy', valid_label)


    # np.savez('/h3cstore_ns/hyshi/InferenceAC3_monai/mamba3_ar11_270.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3_monai/mamba3_MAE_40.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3_monai/mamba3_random_330.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3_monai/unetr_random_420.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3_monai/unetr_MAE_160.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3_monai/superhuman_MAE_40.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3_monai/superhuman_random_352_b2.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)

    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_monai/mamba3_ar11_1150.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_monai/mamba3_MAE_1290.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_monai/unetr_MAE_730.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_monai/unetr_random_460.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_monai/mamba3_random_540.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_monai/superhuman_random_350.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_monai/superhuman_MAE_700.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
 
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer36_2_monai/mamba3_ar11_1150.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer36_2_monai/mamba3_MAE_1290.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer36_2_monai/unetr_MAE_730.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer36_2_monai/unetr_random_460.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer36_2_monai/mamba3_random_540.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer36_2_monai/superhuman_random_350.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer36_2_monai/superhuman_MAE_700.npz', pred_affs=pred_data, gt_seg=valid_label, gt_affs=valid_affs)


    np.savez('/h3cstore_ns/hyshi/wafer4_errorbar/mamba_ar11/mamba_ar11_1160.npz', pred_affs=pred_data, gt_seg=valid_label , gt_affs=valid_affs)



    fragments = watershed(pred_data, 'maxima_distance')
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    seg_waterz = list(waterz.agglomerate(pred_data, [0.50],
                fragments=fragments,
                scoring_function=sf,
                discretize_queue=256))[0]
    arand_waterz = adapted_rand_ref(valid_label, seg_waterz, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(valid_label, seg_waterz, ignore_labels=(0))
    voi_sum_waterz = voi_split + voi_merge
    print(f"Validation MSE Loss: {valid_mse_loss}, ARAND: {arand_waterz}, VOI: {voi_sum_waterz}, VOI_MERGE: {voi_merge}, VOI_SPLIT: {voi_split}")