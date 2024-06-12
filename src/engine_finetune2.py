# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch.utils
import torch.utils.data
import waterz
import h5py
from utils.fragment import watershed, randomlabel, relabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import torch
# monai sliding window infer
from monai.inferers import sliding_window_inference, SlidingWindowInferer
from monai.transforms import SpatialPad
from timm.data import Mixup
from timm.utils import accuracy
from PIL import Image
import util_mamba.misc as misc
import util_mamba.lr_sched as lr_sched
import cv2
import time
import os
from provider_valid import Provider_valid
from tqdm import tqdm

from thop import profile
from thop import clever_format
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None,
                    dataset=None,
                    visual_dir=None,
                    val_provider=None,
                    cfg=None,
                    ):
    """
    start training for one epoch
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(args.log_dir))

    if val_provider is not None:
        val_loader = torch.utils.data.DataLoader(val_provider, batch_size=1)
    else: 
        val_loader = None

    for data_iter_step, (samples, targets, weightmap) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if bool(args.use_lr_scheduler) is True:
            # we use a per iteration (instead of per epoch) lr scheduler 
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        else:
            pass

        samples = samples.to(device, non_blocking=True)  # size (batch_size,1,16,256,256) batch_size=6
        targets = targets.to(device, non_blocking=True)
        weightmap = weightmap.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast(enabled=args.use_amp):
        with torch.cuda.amp.autocast():
            macs, params = profile(model, inputs=(samples,))
            macs, params = clever_format([macs, params], "%.3f")
            print(f"thop: MACs: {macs}")
            print(f"thop: Parameters: {params}")  
            # flops = FlopCountAnalysis(model, samples)
            # print("=====fvcore=====")
            # print("Flops: ", flops.total())
            # print(parameter_count_table(model))
            outputs = model(samples)
            loss = criterion(outputs, targets, weightmap)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            # continue

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # if epoch % 2 == 0 and misc.is_main_process():
    if epoch > 39 and epoch % 10 == 0 and misc.is_main_process():
    # if epoch % 1 == 5 and misc.is_main_process():
            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    loss_all = []
                    print('the number of sub-volume:', len(val_provider))
                    losses_valid = []
                    t1 = time.time()
                    pbar = tqdm(total=len(val_provider))
                    for k, data in enumerate(val_loader, 0):
                        inputs, target, weightmap = data
                        # inputs = inputs.cuda()
                        # target = target.cuda()
                        # weightmap = weightmap.cuda()
                        inputs = inputs.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        weightmap = weightmap.to(device, non_blocking=True)

                        pred = model(inputs)
                        tmp_loss = torch.nn.functional.mse_loss(pred, target)
                        losses_valid.append(tmp_loss.item())
                        val_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
                        pbar.update(1)
                    pbar.close()
                    cost_time = time.time() - t1
                    print("Inference time=%.6f" % cost_time)
                    valid_mse_loss = sum(losses_valid) / len(losses_valid)
                    pred_data = val_provider.get_results()  # (3,100,1024,1024)
                    valid_affs = val_provider.get_gt_affs()  # (3,100,1024,1024)
                    valid_label = val_provider.get_gt_lb()
                    valid_data = val_provider.get_raw_data()
                    valid_data = torch.from_numpy(valid_data)
                    valid_data = valid_data.to(device, non_blocking=True)
                    valid_data = valid_data / 255
                    valid_data = valid_data.unsqueeze(0).unsqueeze(0)
                    val_provider.reset_output()
                    valid_label = valid_label.astype(np.uint32)
                    fragments = watershed(pred_data, 'maxima_distance')
                    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                    seg_waterz = list(waterz.agglomerate(pred_data, [0.50],
                                fragments=fragments,
                                scoring_function=sf,
                                discretize_queue=256))[0]
                    seg_waterz = relabel(seg_waterz).astype(np.uint64)
                    arand_waterz = adapted_rand_ref(valid_label, seg_waterz, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(valid_label, seg_waterz, ignore_labels=(0))
                    voi_sum_waterz = voi_split + voi_merge
                    epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                    print(f"Validation MSE Loss: {valid_mse_loss}, ARAND: {arand_waterz}, VOI: {voi_sum_waterz}, VOI_MERGE: {voi_merge}, VOI_SPLIT: {voi_split}")
            else:
                # valid_data, valid_label, valid_affs = dataset.valid_provide()
                if cfg.DATA.valid_dataset is not None:
                    valid_data, valid_label, valid_affs = dataset.valid_provide(valid_dataset=cfg.DATA.valid_dataset)
                else:
                    valid_data, valid_label, valid_affs = dataset.valid_provide()
                valid_data = valid_data / 255
                t1 = time.time()
                with torch.no_grad():
                    model.eval()
                    # IPZZ-016
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
                    pred_data = sliding_window_inference(valid_data, args.crop_size, args.batch_size, model, overlap=args.overlap, mode="gaussian") # parameters: data_raw_valid, model_input_size, valid batch size, model, 0.25
                    cost_time = time.time() - t1
                    print("Inference time=%.6f" % cost_time)
                    assert pred_data.shape == valid_affs.shape, f"pred_data shape: {pred_data.shape}, valid_affs shape: {valid_affs.shape}"
                    # pred_data = sliding_window_inference(valid_data, (16,256,256), 64, model)
                    valid_mse_loss = torch.nn.functional.mse_loss(pred_data, valid_affs)
                    pred_data = pred_data.squeeze(0)
                    pred_data = pred_data.cpu().numpy()
                    fragments = watershed(pred_data, 'maxima_distance')
                    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                    seg_waterz = list(waterz.agglomerate(pred_data, [0.50],
                                fragments=fragments,
                                scoring_function=sf,
                                discretize_queue=256))[0]
                    arand_waterz = adapted_rand_ref(valid_label, seg_waterz, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(valid_label, seg_waterz, ignore_labels=(0))
                    voi_sum_waterz = voi_split + voi_merge
                    epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                    print(f"Validation MSE Loss: {valid_mse_loss}, ARAND: {arand_waterz}, VOI: {voi_sum_waterz}, VOI_MERGE: {voi_merge}, VOI_SPLIT: {voi_split}")
                    valid_affs = valid_affs.squeeze(0)
                    valid_affs = valid_affs.cpu().numpy()
            log_writer.add_scalar('valid/mse_loss', valid_mse_loss, epoch_1000x)
            log_writer.add_scalar('valid/arand', arand_waterz, epoch_1000x)
            log_writer.add_scalar('valid/voi', voi_sum_waterz, epoch_1000x)
            log_writer.add_scalar('valid/voi_split', voi_split, epoch_1000x)
            log_writer.add_scalar('valid/voi_merge', voi_merge, epoch_1000x)
            log_writer.add_scalar('valid/inference_time', cost_time, epoch_1000x)

            if visual_dir is not None:
                # visualize seg_waterz, pred_data, valid_affs, valid_label, data
                # print('show affs...')
                # pred_data = (pred_data * 255).astype(np.uint8)
                # valid_affs = (valid_affs * 255).astype(np.uint8)
                # for i in range(pred_data.shape[1]):
                #     cat1 = np.concatenate([pred_data[0,i], pred_data[1,i], pred_data[2,i]], axis=1)
                #     cat2 = np.concatenate([valid_affs[0,i], valid_affs[1,i], valid_affs[2,i]], axis=1)
                #     im_cat = np.concatenate([cat1, cat2], axis=0)
                #     cv2.imwrite(os.path.join(visual_dir, 'affs'+str(i).zfill(4)+'.png'), im_cat)

                # print('show seg...')
                # seg_waterz[valid_label==0] = 0
                # color_seg = draw_fragments_3d(seg_waterz)
                # color_gt = draw_fragments_3d(valid_label)
                # for i in range(color_seg.shape[0]):
                #     im_cat = np.concatenate([color_seg[i], color_gt[i]], axis=1)
                #     cv2.imwrite(os.path.join(visual_dir, 'seg'+str(i).zfill(4)+'.png'), im_cat)
                # print('show done.')
                waterz_seg_color = draw_fragments_3d(seg_waterz)
                label_color = draw_fragments_3d(valid_label)
                label_img = Image.fromarray(label_color[0].astype(np.uint8))
                waterz_img = Image.fromarray(waterz_seg_color[0].astype(np.uint8))
                h, w = label_img.size
                white_line = Image.new('RGB', (w, 16), (255, 255, 255))
                if not isinstance(valid_data, np.ndarray):
                    valid_data = valid_data.detach().cpu().numpy()
                if not isinstance(valid_affs, np.ndarray):
                    valid_affs = valid_affs.detach().cpu().numpy()
                    valid_affs = valid_affs.squeeze(0)
                    valid_affs = valid_affs[:,0]
                
                # if not isinstance(valid_label, np.ndarray):
                #     valid_label = valid_label.cpu().numpy()
                if not isinstance(pred_data, np.ndarray):
                    pred_data = pred_data.detach().cpu().numpy()
                    pred_data = pred_data.squeeze(0)
                    pred_data = pred_data[:,0]
                pred_aff_img = Image.fromarray((pred_data[:,0]*255).astype(np.uint8).transpose(1,2,0))
                data_img = Image.fromarray((valid_data[0,0,0]*255).astype(np.uint8))
                affs_img = Image.fromarray((valid_affs[:,0]*255).astype(np.uint8).transpose(1,2,0))
                visual_img = Image.new('RGB', (w*5 + 16*4, h), (255, 255, 255))
                visual_img.paste(label_img, (0, 0))
                visual_img.paste(waterz_img, (w+16, 0))
                visual_img.paste(white_line, (w*2+16, 0))
                visual_img.paste(pred_aff_img, (w*2+16*2, 0))
                visual_img.paste(affs_img, (w*3+16*3, 0))
                visual_img.paste(white_line, (0, h))
                visual_img.paste(data_img, (w*4+16*4, 0))
                visual_img.save(visual_dir + f'/epoch_{epoch}_iter.png')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}