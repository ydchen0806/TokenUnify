# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from model_superhuman2 import UNet_PNI, UNet_PNI_Noskip
import torch
import torch.backends.cudnn as cudnn
from unet3d_mala import UNet3D_MALA
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset_hdf import hdfDataset
import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from dataloader.data_provider_pretraining import Train as EMDataset
import sys
sys.path.append('/data/ydchen/VLP/EM_Mamba/mambamae_EM')
import util_mae.misc as misc
from util_mae.misc import NativeScalerWithGradNormCount as NativeScaler
# from dataset_hdf import hdfDataset
import models_mae
from segmamba import SegMamba
from model_unetr import UNETR
from omegaconf import OmegaConf
from engine_pretrain import train_one_epoch
# from segmamba import SegMamba


def get_args_parser():
    parser = argparse.ArgumentParser('TokenUnify pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mala', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--fill_mode', default=0, type=int,
                        help='fill mode for the holes in the image')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.4, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--use_amp', type=bool, default=False,
                        help='use mixed precision training')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--hdf_path', default='/img_video/img/COCOunlabeled2017.hdf5', type=str,
                        help='dataset path')
    parser.add_argument('--EM_cfg_path', default='/data/ydchen/VLP/EM_Mamba/mambamae_EM/config/pretraining_all.yaml', type=str,
                        help='EM cfg dataset path')
    parser.add_argument('--output_dir', default='/h3cstore_ns/EM_pretrain/mamba_pretrain_MAE',
                        help='path where to save, empty for no saving')
    parser.add_argument('--visual_dir', default='/h3cstore_ns/EM_pretrain/mamba_pretrain_MAE/visual',
                        help='path where to save visual images')
    parser.add_argument('--log_dir', default='/h3cstore_ns/EM_pretrain/mamba_pretrain_MAE/tensorboard_log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pretrain_path', default='', type=str,
                        help='path to pretrained model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    cfg = OmegaConf.load(args.EM_cfg_path)
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_train = EMDataset(cfg,args = args)
    print(f'total len of dataset: {len(dataset_train)}')

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    # model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    if args.model == 'segmamba':
        model = SegMamba(in_chans=1, out_chans=1)
    if args.model == 'superhuman':
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                            out_planes=1,
                            filters=cfg.MODEL.filters,
                            upsample_mode=cfg.MODEL.upsample_mode,
                            decode_ratio=cfg.MODEL.decode_ratio,
                            merge_mode=cfg.MODEL.merge_mode,
                            pad_mode=cfg.MODEL.pad_mode,
                            bn_mode=cfg.MODEL.bn_mode,
                            relu_mode=cfg.MODEL.relu_mode,
                            init_mode=cfg.MODEL.init_mode)
    if args.model == 'mala':
        model = UNet3D_MALA(output_nc=1, if_sigmoid=cfg.MODEL.if_sigmoid,
                    init_mode=cfg.MODEL.init_mode_mala)
    if args.model == 'unetr':
        model = UNETR(
                in_channels=cfg.MODEL.input_nc,
                out_channels=1,
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
                dropout_rate=0.1)
    if args.model == 'segmamba':
        model = SegMamba(in_chans=1, out_chans=1)
        
    if args.pretrain_path:
        print("Loading pretrained model from %s" % args.pretrain_path)
        weights = torch.load(args.pretrain_path, map_location='cpu')
        # model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu')['model'], strict=False)
        model.load_state_dict(weights, strict=False)
        # freeze_prams = [param_name for param_name, _ in weights.items()]
    model.to(device)
    # model -> model.module
    # model : inference, model.inference -> model.module.inference
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs) and misc.is_main_process():
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
