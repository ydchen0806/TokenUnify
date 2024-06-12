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
import timm.optim.optim_factory as optim_factory
from pathlib import Path

import torch.utils
import torch.utils.data
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
from segmamba_variant import SegMamba_linear
from segmamba_deep import SegMamba_deep
# from segmamba_ar import SegMamba
# from SwinUMamba import SwinUMamba

from engine_finetune2 import train_one_epoch, evaluate
from model_superhuman2 import UNet_PNI
from model_unetr import UNETR
from unet3d_mala import UNet3D_MALA
# from model_unetr2 import UNETR
# from model_unetr2_variant import UNETR

import models_mamba

from provider_valid import Provider_valid

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='segmamba_base_model', type=str, metavar='MODEL',
                        help='Name of model to train')
    # parser.add_argument('--crop_size', default='', type=lambda s: list(map(int, s.split(','))),
                        # help='images crop size')  # 传入--crop_size=16,160,160
    parser.add_argument('--crop_size', default=[16,160,160], type=list,
                        help='images crop size')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--pretrain_path', default='', type=str,
                        help='path to pretrain model')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--overlap', type=float, default=0.25, metavar='PCT',)
    parser.add_argument('--use_monai', type=int, default=1,
                        help='mode for the model inference: 1 for monai and 0 for non-monai')
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--use_lr_scheduler', type=int, default=1,
                        help='decay the learning rate with half-cycle consine after warmup: 1 for use and 0 for unused')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/h3cstore_ns/hyshi/EM_mamba_new/result/EM_1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--visual_dir', default='/h3cstore_ns/hyshi/EM_mamba_new/result/EM_1/visual',
                        help='path where to save visual images')
    parser.add_argument('--log_dir', default='/h3cstore_ns/hyshi/EM_mamba_new/result/EM_1/tensorboard_log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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
    parser.add_argument('--auto_mode', default=0, type=int,
                        help='mode for autoregress pretraining')
    parser.add_argument('--use_amp', default=False, type=bool,
                        help="mode for training")
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

    cfg_file = 'seg_all_3d_ac4_data80'
    # cfg_file = 'seg_3d_ac3_data100'
    # with open('/data/ydchen/VLP/wafer4/config/' + cfg_file + '.yaml', 'r') as f:
    with open('/h3cstore_ns/hyshi/configs/' + cfg_file + '.yaml', 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    if cfg.DATA.shift_channels is not None:
        cfg.shift = shift_func(cfg.DATA.shift_channels)
    else:
        cfg.shift = None
    args.crop_size = cfg.MODEL.crop_size

    # define the model
    model = models_mamba.__dict__[args.model]()
    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        for k in list(checkpoint['model'].keys()):
            if k.startswith('module.'):
                checkpoint['model'][k[7:]] = checkpoint['model'].pop(k)
            if k in model.state_dict() and checkpoint['model'][k].shape != model.state_dict()[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Load pre-trained checkpoint from: %s" % args.pretrain_path)
    # if args.finetune and not args.eval:
    #     checkpoint = torch.load(args.finetune, map_location='cpu')

    #     print("Load pre-trained checkpoint from: %s" % args.finetune)
    #     checkpoint_model = checkpoint['model']
    #     state_dict = model.state_dict()
    #     for k in ['head.weight', 'head.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]

    #     # interpolate position embedding
    #     interpolate_pos_embed(model, checkpoint_model)

    #     # load pre-trained model
    #     msg = model.load_state_dict(checkpoint_model, strict=False)
    #     print(msg)

    #     if args.global_pool:
    #         assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    #     else:
    #         assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    #     # manually initialize fc layer
    #     trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    dataset_train = Trainset(cfg, args.crop_size)  # [16,256,256] [16,160,160] [32,320,320]

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        # if args.dist_eval:
        #     if len(dataset_val) % num_tasks != 0:
        #         print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #               'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #               'equal num of samples per-process.')
        #     sampler_val = torch.utils.data.DistributedSampler(
        #         dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        # else:
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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

    if bool(args.use_monai) is not True:
        valid_provider = Provider_valid(cfg, test_split=cfg.DATA.test_split)
        # val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
    else:
        valid_provider = None


    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        if cfg.MODEL.model_type == 'unetr' or cfg.MODEL.model_type == 'segmamba_linear':
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    # if mixup_fn is not None:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion = WeightedMSE()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # for key in model.state_dict():
    #     print(key)

    # if args.eval:
    #     test_stats = evaluate(data_loader_val, model, device)
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #     exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, 
            log_writer=log_writer,
            args=args,
            dataset = dataset_train,
            visual_dir = args.visual_dir,
            val_provider = valid_provider,  # 20240408
            cfg=cfg,
        )
        if args.output_dir and (epoch % 10 == 0 or epoch == args.epochs - 1) and misc.is_main_process():
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # test_stats = evaluate(data_loader_val, model, device)
        # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # max_accuracy = max(max_accuracy, test_stats["acc1"])
        # print(f'Max accuracy: {max_accuracy:.2f}%')

        # if log_writer is not None:
        #     log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
        #     log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
        #     log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        # **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

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
        Path(args.visual_dir).mkdir(parents=True, exist_ok=True)

    main(args)
