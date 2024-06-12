from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_provider_labeled import Provider
from provider_valid import Provider_valid
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss
from utils.show import show_affs, show_affs_whole
from unet3d_mala import UNet3D_MALA
from model_superhuman2 import UNet_PNI, UNet_PNI_Noskip
from utils.utils import setup_seed, execute
from utils.shift_channels import shift_func
# from utils.lmc import mc_baseline
# from lmc import mc_baseline
import waterz

from utils.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
from segmamba import SegMamba

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    valid_provider = Provider_valid(cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, is_train=True):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if is_train:
        
        if cfg.MODEL.model_type == 'mala':
            print('load mala model!')
            model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid,
                                init_mode=cfg.MODEL.init_mode_mala).to(device)
        elif cfg.MODEL.model_type == 'segmamba':
            print('load segmamba model!')
            model = SegMamba(in_chans=cfg.MODEL.input_nc,
                            out_chans=cfg.MODEL.output_nc).to(device)
        else:
            print('load superhuman model!')
            model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                            out_planes=cfg.MODEL.output_nc,
                            filters=cfg.MODEL.filters,
                            upsample_mode=cfg.MODEL.upsample_mode,
                            decode_ratio=cfg.MODEL.decode_ratio,
                            merge_mode=cfg.MODEL.merge_mode,
                            pad_mode=cfg.MODEL.pad_mode,
                            bn_mode=cfg.MODEL.bn_mode,
                            relu_mode=cfg.MODEL.relu_mode,
                            init_mode=cfg.MODEL.init_mode).to(device)

    else:
        if cfg.MODEL.model_type == 'mala':
            raise AttributeError('have not implemented yet!')
            
            print('load mala as feature extractor!')
        elif cfg.MODEL.model_type == 'superhuman':
            model = UNet_PNI_Noskip(in_planes=cfg.MODEL.input_nc,
                            out_planes=1,
                            filters=cfg.MODEL.filters,
                            upsample_mode=cfg.MODEL.upsample_mode,
                            decode_ratio=cfg.MODEL.decode_ratio,
                            merge_mode=cfg.MODEL.merge_mode,
                            pad_mode=cfg.MODEL.pad_mode,
                            bn_mode=cfg.MODEL.bn_mode,
                            relu_mode=cfg.MODEL.relu_mode,
                            init_mode=cfg.MODEL.init_mode).to(device)

    if cfg.MODEL.train_model_path:
        print(f'Load pre-trained model from {cfg.MODEL.train_model_path} ...')
        # ckpt_path = os.path.join('../models', \
        #                          cfg.MODEL.trained_model_name, \
        #                          'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
        ckpt_path = cfg.MODEL.train_model_path
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['model_weights']

        pretrained_model_dict = OrderedDict()
        for name, weight in pretrained_dict.items():
            if 'module' in name:
                name = name[7:]
            pretrained_model_dict[name] = weight


        from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
        model_dict = model.state_dict()
        # if cfg.MODEL.load_encoder:
        if not is_train:
            encoder_dict = OrderedDict()
            if cfg.MODEL.if_skip == 'True':
                print('Load the parameters of encoder and decoder!')
                encoder_dict = {k: v for k, v in pretrained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
            else:
                print('Load the parameters of encoder!')
                encoder_dict = {k: v for k, v in pretrained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
        else:
            encoder_dict = pretrained_model_dict
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)

    print('Done (time: %.2fs)' % (time.time() - t1))
    return model


def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device = torch.device('cuda:0')

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")

    best_voi = 100000
    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1

        t1 = time.time()
        inputs, target, weightmap = train_provider.next()

        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()
        pred = model(inputs)

        ##############################
        # LOSS
        loss = criterion(pred, target, weightmap)
        # clip loss
        loss = torch.clamp(loss, min=0, max=5)

        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                             % (iters, sum_loss * 1, current_lr, sum_time,
                                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(
                                    np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                             % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(
                                    np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, inputs, pred[:, :3], target[:, :3], cfg.cache_path, model_type=cfg.MODEL.model_type)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                         shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                for k, batch in enumerate(dataloader, 0):
                    inputs, target, weightmap = batch
                    inputs = inputs.cuda()
                    target = target.cuda()
                    weightmap = weightmap.cuda()
                    with torch.no_grad():
                        pred = model(inputs)
                    tmp_loss = criterion(pred, target, weightmap)
                    losses_valid.append(tmp_loss.item())
                    valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = valid_provider.get_results()
                gt_affs = valid_provider.get_gt_affs().copy()
                gt_seg = valid_provider.get_gt_lb()
                valid_provider.reset_output()
                out_affs = out_affs[:3]
                # gt_affs = gt_affs[:, :3]
                show_affs_whole(iters, out_affs, gt_affs, cfg.valid_path)

                ##############
                # segmentation
                ##############
                # segmentation
                if cfg.TRAIN.if_seg:
                    if iters > 1000:
                        fragments = watershed(out_affs, 'maxima_distance')
                        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                        seg_waterz = list(waterz.agglomerate(out_affs, [0.50],
                                    fragments=fragments,
                                    scoring_function=sf,
                                    discretize_queue=256))[0]
           
                        # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
   
                        arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
                        voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
                        voi_sum_waterz = voi_split + voi_merge

                        seg_lmc = seg_waterz
                        arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
                        voi_split, voi_merge = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
                        voi_sum_lmc = voi_split + voi_merge
                    else:
                        voi_sum_waterz = 0.0
                        arand_waterz = 0.0
                        voi_sum_lmc = 0.0
                        arand_lmc = 0.0
                        print('model-%d, segmentation failed!' % iters)
                else:
                    voi_sum_waterz = 0.0
                    arand_waterz = 0.0
                    voi_sum_lmc = 0.0
                    arand_lmc = 0.0
                ##############

                # MSE
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
                whole_bce = np.sum(bce) / np.size(gt_affs)
                out_affs[out_affs <= 0.5] = 0
                out_affs[out_affs > 0.5] = 1
                # whole_f1 = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())
                whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                writer.add_scalar('valid/f1_score', whole_f1, iters)
                writer.add_scalar('valid/voi_waterz', voi_sum_waterz, iters)
                writer.add_scalar('valid/arand_waterz', arand_waterz, iters)
                writer.add_scalar('valid/voi_lmc', voi_sum_lmc, iters)
                writer.add_scalar('valid/arand_lmc', arand_lmc, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                                (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_3d_cremiC_data100', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('/data/ydchen/VLP/wafer4/config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp
    if cfg.DATA.shift_channels is None:
        assert cfg.MODEL.output_nc == 3, "output_nc must be 3"
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == cfg.DATA.shift_channels, "output_nc must be equal to shift_channels"
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg)
        cuda_count = torch.cuda.device_count()
        if cuda_count > 1:
            if cfg.TRAIN.batch_size == 1:
                print('a single GPU ... ', end='', flush=True)
            elif cfg.TRAIN.batch_size % cuda_count == 0:
                print('%d GPUs ... ' % cuda_count, end='', flush=True)
                model = nn.DataParallel(model)
            else:
                raise AttributeError(
                    'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
        else:
            print('a single GPU ... ', end='', flush=True)
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
        #                              eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6, amsgrad=False)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')

'''
我的思路其实是这样的，我的目的是进行test time adaptation，在CREMIA训练好的模型在CREMIB 测试。具体想通过local和global的visual prompt去调整整个网络，
这里我的分割网络是Unet，训练好Unet后，我使用相同结构的Unet当作特征提取器抽取local和global的visual prompt，global是抽取整个输入图像的特征，
local是抽取通过上面box得到网络最关注的区域，然后inference的时候固定分割网络的权重，改变这两个特征提取器的权重，将他们学习到的可变矩阵加到CREMIB的输入数据，
实现TTA，怎么实现呢，这个方法有什么可以改进的吗
'''