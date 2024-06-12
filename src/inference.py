# import os
# import torch
# from segmamba import SegMamba
# from attrdict import AttrDict
# import yaml
# import numpy as np
# import waterz
# import h5py
# from utils.fragment import watershed, randomlabel, relabel
# # import evaluate as ev
# from skimage.metrics import adapted_rand_error as adapted_rand_ref
# from skimage.metrics import variation_of_information as voi_ref
# import torch
# import cv2
# import os
# from provider_valid import Provider_valid
# import time
# from tqdm import tqdm
# from utils.lmc import mc_baseline


import argparse
import datetime
import json
import numpy as np
import os
import time
# import timm.optim.optim_factory as optim_factory
from pathlib import Path
from loss.loss import WeightedMSE
import torch
import torch.backends.cudnn as cudnn
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter

# import timm

# assert timm.__version__ == "0.3.2" # version check
# from timm.models.layers import trunc_normal_
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

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
from model_superhuman2 import UNet_PNI


from model_superhuman_pea import UNet_PNI_embedding_deep as UNet_PNI_pea
from loss.loss_embedding_mse import embedding_loss_norm1, embedding_loss_norm5
from loss.loss import BCELoss, WeightedBCE, MSELoss, WeightedMSE
import torch.nn.functional as F
from collections import OrderedDict


from model_unetr import UNETR
from unet3d_mala import UNet3D_MALA


from engine_finetune import train_one_epoch, evaluate


import waterz
import h5py
from utils.fragment import watershed, randomlabel, relabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import cv2
from provider_valid import Provider_valid
import time
from tqdm import tqdm

# from utils.lmc import mc_baseline


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


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pred_model_path', type=str, default='/h3cstore_ns/hyshi/EM_pretrain/mamba_seg_EM/EM_1/checkpoint-160.pth')
#     # parser.add_argument('--post_model_path', type=str, default='/h3cstore_ns/screen_out2/results_deep_refine_unet_1029_total_a40/checkpoint-799.pth')
#     # parser.add_argument('--input_dir', type=str, default='/h3cstore_ns/screen_generate/test/kodak')
#     parser.add_argument('--output_dir', type=str, default='/h3cstore_ns/hyshi/EM_mamba_new/result')
#     parser.add_argument('--batch_size', type=int, default=1)
#     # parser.add_argument('--num_workers', type=int, default=1)
#     # parser.add_argument('--patch_size', type=int, default=224)
#     # parser.add_argument('--stride', type=int, default=112)
#     # parser.add_argument('--hdf_path', default='/img_video/img/COCOunlabeled2017.hdf5', type=str,
#     #                 help='dataset path')
#     # parser.add_argument('--gpu', type=int, default=0)
#     parser.add_argument('--device', default='cuda')
#     # parser.add_argument('--log_dir', type=str, default='log')
#     # parser.add_argument('--save_dir', type=str, default='save')
#     # parser.add_argument('--norm_pix_loss', type=bool, default=False)
#     # parser.add_argument('--model', type=str, default='mae_vit_base_patch16_deeper')
#     # parser.add_argument('--input_size', type=int, default=224)
#     args = parser.parse_args()
#     return args

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
    # cfg_file = 'segmamba_3d_valid_ac3'
    cfg_file = 'seg_all_3d_ac4_data80'
    with open('/h3cstore_ns/hyshi/configs/' + cfg_file + '.yaml', 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    valid_provider = Provider_valid(cfg, test_split=cfg.DATA.test_split)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)


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
        # model = UNETR(
        #         in_channels=cfg.MODEL.input_nc,
        #         out_channels=cfg.MODEL.output_nc,
        #         img_size=cfg.MODEL.unetr_size,
        #         patch_size=cfg.MODEL.patch_size,
        #         feature_size=[16, 32, 64, 128],
        #         hidden_size=512,
        #         mlp_dim=2048,
        #         num_heads=8,
        #         pos_embed='perceptron',
        #         norm_name='instance',
        #         conv_block=True,
        #         res_block=True,
        #         kernel_size=cfg.MODEL.kernel_size,
        #         skip_connection=False,
        #         show_feature=False,
        #         dropout_rate=0.1)
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
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_monai_wafer_lr5_b20_18_160_160_gaussian_8gpu_0424/checkpoint-350.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_pre_wafer_lr5_8gpu_0426_3/checkpoint-700.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_monai_wafer_lr5_b12_32_160_160_8gpu/checkpoint-460.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_pre_monai_wafer_lr5_b12_32_160_160_8gpu0426/checkpoint-730.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_8gpu_b10_16_160_160_xu799/checkpoint-540.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_8gpu_b20_16_160_160_pre_1/checkpoint-1290.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_b20_16_160_160_pre_ar_11_370/checkpoint-1150.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_8gpu_b20_16_160_160_pre_ar_10_390_3090/checkpoint-1150.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_lr5_8gpu_b20_16_160_160_pre_ar_01_399_3090_xu360/checkpoint-600.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_16_160_160_pre_ar_01_370/checkpoint-940.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_wafer_0516/checkpoint-1420.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/mala_old_infer_wafer_lr5_8gpu_0426/checkpoint-220.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/mala_pre_wafer_lr5_8gpu_0504/checkpoint-750.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_18_160_160_8gpu/checkpoint-352.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac34_lr5_8gpu_b20_3090_MAE399/checkpoint-40.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_pre_monai_wafer_lr5_b12_32_160_160_8gpu0426/checkpoint-730.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/unetr_ac3_lr5_b12_32_160_160_gaussian_8gpu/checkpoint-420.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_ac34_lr5_8gpu_b20_16_160_160_3090/checkpoint-330.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_ac34_lr5_8gpu_b20_16_160_160_pre_mae399/checkpoint-40.pth')['model'])
    # model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/segmamba_155kernel_ac34_lr5_8gpu_b20_16_160_160_pre_ar_11_320_3090/checkpoint-270.pth')['model'])


    model.load_state_dict(torch.load('/h3cstore_ns/hyshi/EM_mamba_new/result/mala_pre_wafer_lr5_8gpu_0504/checkpoint-760.pth')['model'])



    model = model.to(device)


    # checkpoint = torch.load('/h3cstore_ns/hyshi/EM_seg/models/2024-05-13--19-42-04_seg_3d_ac4_data80/model-085000.ckpt')

    # new_state_dict = OrderedDict()
    # state_dict = checkpoint['model_weights']
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     # name = k
    #     new_state_dict[name] = v
    
    # model.load_state_dict(new_state_dict)
    # model = model.to(device)


    model.eval()
    loss_all = []
    f_txt = open('scores.txt', 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        # inputs = torch.from_numpy(inputs).cuda()
        # target = torch.from_numpy(target).cuda()
        # weightmap = torch.from_numpy(weightmap).cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        with torch.no_grad():
            pred = model(inputs)
        tmp_loss = torch.nn.functional.mse_loss(pred, target)
        losses_valid.append(tmp_loss.item())
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    epoch_loss = sum(losses_valid) / len(losses_valid)
    output_affs = valid_provider.get_results()
    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()
    valid_provider.reset_output()
    gt_seg = gt_seg.astype(np.uint32)

    # # save
    # if True:
    #     print('save affs...')
    #     f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
    #     f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
    #     f.close()

    # save for post-process
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/superhuman_random_350.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/superhuman_MAE_700.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/unetr_random_460.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/unetr_MAE_730.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/mamba3_random_540.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/mamba3_MAE_1290.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/mamba3_ar11_1150.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/mamba3_ar10_1150.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/mamba3_ar00_600.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/mamba3_ar01_940.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4/mamba3_random_1420.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_0518/mala_random_220.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceWafer4_0518/mala_MAE_750.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/superhuman_random_352.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/superhuman_MAE_40.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/unetr_MAE_730.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/unetr_random_420.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/mamba3_random_330.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/mamba3_MAE_40.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/mamba3_ar11_270.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/InferenceAC3/mala_MAE_85000.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)

    # np.savez('/h3cstore_ns/hyshi/OldInferenceWafer4/mamba3_ar11_1150.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/OldInferenceWafer36_2/mamba3_MAE_1290.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/OldInferenceWafer36_2/mamba3_ar11_1150.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/OldInferenceWafer36_2/mala_MAE_750.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/OldInferenceWafer36_2/mala_random_220.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)
    # np.savez('/h3cstore_ns/hyshi/OldInferenceWafer4/mala_random_220.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)



    np.savez('/h3cstore_ns/hyshi/wafer4_errorbar/mala_MAE/mala_MAE_760.npz', pred_affs=output_affs, gt_seg=gt_seg, gt_affs=gt_affs)



    # data_zip = np.load('/h3cstore_ns/hyshi/Best_Inference_result/superhuman_MAE_700.npz')
    # output_affs = data_zip['affs']
    # gt_seg = data_zip['seg']


    # segmentation
    print('Segmentation...')
    fragments = watershed(output_affs, 'maxima_distance')
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    # sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
    segmentation = list(waterz.agglomerate(output_affs, [0.50],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]
    segmentation = relabel(segmentation).astype(np.uint64)
    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('model-%d, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
        (1, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('model-%d, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
        (1, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')
    f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
    f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    f.close()

    # segmentation = mc_baseline(output_affs)
    # segmentation = relabel(segmentation).astype(np.uint64)
    # print('the max id = %d' % np.max(segmentation))
    # f = h5py.File(os.path.join(out_affs, 'seg_lmc.hdf'), 'w')
    # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    # f.close()

    # arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    # voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    # voi_sum = voi_split + voi_merge
    # print('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #     (voi_split, voi_merge, voi_sum, arand))
    # f_txt.write('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #     (voi_split, voi_merge, voi_sum, arand))
    # f_txt.write('\n')


    # output_affs_prop = output_affs.copy()

    # show
    # if True:
    #     print('show affs...')
    #     output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
    #     gt_affs = (gt_affs * 255).astype(np.uint8)
    #     for i in range(output_affs_prop.shape[1]):
    #         cat1 = np.concatenate([output_affs_prop[0,i], output_affs_prop[1,i], output_affs_prop[2,i]], axis=1)
    #         cat2 = np.concatenate([gt_affs[0,i], gt_affs[1,i], gt_affs[2,i]], axis=1)
    #         im_cat = np.concatenate([cat1, cat2], axis=0)
    #         cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), im_cat)
        
    #     print('show seg...')
    #     segmentation[gt_seg==0] = 0
    #     color_seg = draw_fragments_3d(segmentation)
    #     color_gt = draw_fragments_3d(gt_seg)
    #     for i in range(color_seg.shape[0]):
    #         im_cat = np.concatenate([color_seg[i], color_gt[i]], axis=1)
    #         cv2.imwrite(os.path.join(seg_img_path, str(i).zfill(4)+'.png'), im_cat)
    # print('Done')




    # dataset = Train(cfg, [16, 256, 256])
    # valid_data, valid_label, valid_affs = dataset.valid_provide()
    # valid_data = valid_data / 255


    # # ckpt_path = args.pred_model_path
    # # checkpoint = torch.load(ckpt_path)

    # # new_state_dict = OrderedDict()
    # # state_dict = checkpoint['model_weight']
    # # for k, v in state_dict.items():
    # #     name = k[7:] # remove module
    # #     # name = k
    # #     new_state_dict[name] = v

    # # model.load_state_dict(new_state_dict)


    # model.eval()

    # if not isinstance(valid_data, torch.Tensor):
    #     valid_data = valid_data.astype(np.float32)
    #     valid_affs = valid_affs.astype(np.float32)
    #     valid_affs = torch.tensor(valid_data, dtype=torch.float32)
    #     valid_data = torch.tensor(valid_affs, dtype=torch.float32)
    #     valid_data = valid_data.unsqueeze(0).unsqueeze(0)
    #     valid_affs = valid_affs.unsqueeze(0)
    # valid_data = valid_data.to(device, non_blocking=True)
    # valid_affs = valid_affs.to(device, non_blocking=True)
    # pred_data = sliding_window_inference(valid_data, (16,256,256), 1, model, overlap=0.25) # parameters: data_raw_valid, model_input_size, valid batch size, model, 0.25
    # assert pred_data.shape == valid_affs.shape, f"pred_data shape: {pred_data.shape}, valid_affs shape: {valid_affs.shape}"
    # valid_mse_loss = torch.nn.functional.mse_loss(pred_data, valid_affs)
    # pred_data = pred_data.squeeze(0)
    # pred_data = pred_data.cpu().numpy()
    # fragments = watershed(pred_data, 'maxima_distance')
    # # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    # # seg_waterz = list(waterz.agglomerate(pred_data, [0.50],
    # #             fragments=fragments,
    # #             scoring_function=sf,
    # #             discretize_queue=256))[0]
    # # arand_waterz = adapted_rand_ref(valid_label, seg_waterz, ignore_labels=(0))[0]
    # # voi_split, voi_merge = voi_ref(valid_label, seg_waterz, ignore_label=(0))
    # # voi_sum_waterz = voi_split + voi_merge
    # # # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
    # # # epoch_1000x = int(epoch * 1000)
    # # print(f"Validation MSE Loss: {valid_mse_loss}, ARAND: {arand_waterz}, VOI: {voi_sum_waterz}, VOI_MERGE: {voi_merge}, VOI_SPLIT: {voi_split}")



