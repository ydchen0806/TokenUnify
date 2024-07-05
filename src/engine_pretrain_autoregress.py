# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import h5py
import numpy as np
import util_mae.misc as misc
import util_mae.lr_sched as lr_sched
import os

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(args.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        augsamples, gtsamples, next_token_gt = samples
        augsamples = augsamples.to(device, non_blocking=True)
        gtsamples = gtsamples.to(device, non_blocking=True)
        next_token_gt = next_token_gt.to(device, non_blocking=True)
        # samples = samples.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss = model(augsamples, gtsamples, next_token_gt) # model.forward(samples)

        loss_value = loss.item()
        # if data_iter_step == 0:
        #     recons = model.module.recons_visualize(samples, epoch)
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # print(f"sample shape is {samples.shape}, sample type is {samples.dtype}, sample max is {samples.max()}, sample min is {samples.min()}")
            torch.cuda.empty_cache()
            continue
            # sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # add gradient clipping
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # s_loss_reduce = misc.all_reduce_mean(s_loss)
        # mse_loss_reduce = misc.all_reduce_mean(mse_loss)
        # tvloss_reduce = misc.all_reduce_mean(tvloss)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('train_s_loss', s_loss_reduce, epoch_1000x)
            # log_writer.add_scalar('train_mse_loss', mse_loss_reduce, epoch_1000x)
            # log_writer.add_scalar('train_tvloss', tvloss_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        if data_iter_step == 0 and args.model != 'mala' and args.model != 'superhuman' and args.model != 'unetr'\
            and args.model != 'segmamba':
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            psnr = model.module.recons_visualize(augsamples, gtsamples, epoch_1000x, save_dir = args.visual_dir)
            if log_writer is not None:
                log_writer.add_scalar('train_psnr', psnr, epoch_1000x)
        # save visual



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_total(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler

        samples = samples.to(device, non_blocking=True)
        
        pred_imgs, raw_imgs = model.module.recons_visualize(samples, epoch, save_dir = None)

        current_rank = torch.distributed.get_rank()
        save_name = f'{current_rank}_{data_iter_step}_{epoch}'
        save_img2hdf5(pred_imgs, raw_imgs, args.hdf_save_dir, save_name)

    # gather the stats from all processes


def save_img2hdf5(pred_imgs, raw_imgs, hdf5_path, hdf5_key):
    # hdf5_path = os.path.join(hdf5_path, 'pred_raw.hdf5')
    os.makedirs(hdf5_path, exist_ok=True)
    hdf_name = os.path.join(hdf5_path, f'pred_raw_rank{misc.get_rank()}.hdf5')
    if (not os.path.exists(hdf_name)):
        with h5py.File(hdf_name, 'w') as f:
            f.create_group('pred')
            f.create_group('raw')
            f.create_group('psnr')

    with h5py.File(hdf_name, 'a') as f:
        if not 'pred' in f.keys():
            f.create_group('pred')
        if not 'raw' in f.keys():
            f.create_group('raw')
        if not 'psnr' in f.keys():
            f.create_group('psnr')
        pred_group = f['pred']
        raw_group = f['raw']
        psnr_group = f['psnr']
        if len(pred_imgs.shape) == 3:
            pred_group.create_dataset(hdf5_key, data=pred_imgs)
            raw_group.create_dataset(hdf5_key, data=raw_imgs)
            psnr_group.create_dataset(hdf5_key, data=calc_psnr(pred_imgs, raw_imgs))
        elif len(pred_imgs.shape) == 4:
            for i in range(pred_imgs.shape[0]):
                pred_group.create_dataset(f'{hdf5_key}_{i}', data=pred_imgs[i])
                raw_group.create_dataset(f'{hdf5_key}_{i}', data=raw_imgs[i])
                psnr_group.create_dataset(f'{hdf5_key}_{i}', data=calc_psnr(pred_imgs[i], raw_imgs[i]))
        else:
            raise NotImplementedError
        f.flush()



def calc_psnr(pred_img, raw_img):
    mse = np.mean((pred_img - raw_img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    psnr = 10 * math.log10(PIXEL_MAX / math.sqrt(mse))
    print(f'psnr is {psnr}')
    return psnr


