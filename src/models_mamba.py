# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_local.mamba_ssm_local import Mamba
import torch.nn.functional as F 

import os
import numpy as np
from PIL import Image

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba_type="v2",
                bimamba_type="v3",
                # nslices=num_slices,
        )
        # self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # x_flat = self.dropout(x_flat)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        # self.proj = nn.Conv3d(in_channles, in_channles, (1,3,3), 1, (0,1,1))  # anisotropy v2
        self.norm = nn.InstanceNorm3d(in_channles)
        # self.norm = LayerNorm(in_channles, eps=1e-6, data_format="channels_first")
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        # self.proj2 = nn.Conv3d(in_channles, in_channles, (1,3,3), 1, (0,1,1))  # anisotropy v2
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()


        # self.proj11 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)

    def forward(self, x):
        ## x is b, c, d, w, h
        b, c, d, w, h = x.shape
        # y = self.proj11(x)
        # y = torch.sigmoid(y)

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        # x = x + x * y
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[32, 64, 128, 256],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            #   nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),  # raw
              nn.Conv3d(in_chans, dims[0], kernel_size=(1,7,7), stride=2, padding=(0,3,3)),   # anisotropy v1
            #   nn.Conv3d(in_chans, dims[0], kernel_size=(1,5,5), stride=2, padding=(0,2,2)),   # anisotropy v2
              )
        self.downsample_layers.append(stem)
        for i in range(len(self.depths) - 1):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                # nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),  #  raw
                nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=2),  # anisotropy v1
                # nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,3,3), stride=2, padding=(0,1,1)),  # anisotropy v2
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        num_slices_list = [32, 16, 8, 4]
        cur = 0
        for i in range(len(self.depths)):
            gsc = GSC(dims[i])

            # stage = nn.Sequential(
            #     *[MambaLayer(dim=dims[i]) for j in range(depths[i])]
            # )

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i]) for j in range(depths[i])]
                # *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += self.depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        # norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(self.depths)):
            layer = nn.InstanceNorm3d(dims[i_layer])
            # layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(len(self.depths)):  # x.shape: torch.Size([6,1,16,160,160])
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        # add res
        # for i in range(4):
        #     x = self.downsample_layers[i](x)
        #     x_gsc = self.gscs[i](x)
        #     norm_layer = getattr(self, f'norm{i}')
        #     x_gsc_norm = norm_layer(x_gsc)
        #     x_tom = self.stages[i](x_gsc_norm)
        #     x_mid = x_gsc + x_tom
        #     x_tom_norm = norm_layer(x_mid)
        #     x_tom_norm_mlp = self.mlps[i](x_tom_norm)
        #     x_out = x_mid + x_tom_norm_mlp
        #     outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=3,
        # depths=[3, 3, 3, 3],
        depths=[2, 2, 2, 2],
        # feat_size=[208, 416, 832, 1664],  # 1183.89M
        # feat_size=[192, 384, 768, 1536],
        feat_size=[32, 64, 128, 256],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        # hidden_size: int = 3072, # 1008.91M
        hidden_size: int = 512,
        # kernel_size = (1,3,3),
        kernel_size = (1,5,5),
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            # kernel_size=3,  # raw
            # kernel_size=(1,3,3),  # anisotropy v1
            # kernel_size=(1,5,5),  # 20240426
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            # kernel_size=3,  # raw
            # kernel_size=(1,3,3),  # anisotropy v1
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            # kernel_size=3,  # raw 
            # kernel_size=(1,3,3),  # anisotropy v1 
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            # kernel_size=3,  # raw 
            # kernel_size=(1,3,3),  # anisotroy v1
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            # kernel_size=3,  # raw 
            # kernel_size=(1,3,3),  # anisotropy v1
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            # kernel_size=3,  # raw
            # kernel_size=[1,3,3],  # anisotropy v1
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            # upsample_kernel_size=[1,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            # kernel_size=3,  # raw 
            # kernel_size=(1,3,3),  # ansiotropy v1
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            # kernel_size=3,  # raw 
            # kernel_size=(1,3,3),  # anisotropy v1
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            # kernel_size=3,  # raw 
            # kernel_size=(1,3,3),  # anisotropy v1
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            # kernel_size=3,  # raw 
            # kernel_size=(1,3,3),  # anisotropy v1
            # kernel_size=(1,5,5),  # 20240426
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=32, out_channels=self.out_chans)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)
        self.sigmoid = nn.Sigmoid()

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in, x_gt = None):
        outs = self.vit(x_in)  # x_in.shape: [6,1,16,160,160]
        # x_in = F.pad(x_in, (2,2,2,2,0,0), mode='constant', value=0)  # 20240426 适用encoder1（1,5,5）
        enc1 = self.encoder1(x_in)
        x2 = outs[0]  # [6,32,8,80,80]
        enc2 = self.encoder2(x2)
        x3 = outs[1]  # [6,64,4,40,40]
        enc3 = self.encoder3(x3)
        x4 = outs[2]  #  [6,128,2,20,20]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])  # outs[3]:[6,256,1,10,10]
        dec3 = self.decoder5(enc_hidden, enc4)
        # dec3 = self.decoder5(enc_hidden)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # dec0 = F.pad(dec0, (2,2,2,2,0,0), mode='constant', value=0)  # 202402426 适用decoder1（1,5,5）
        out = self.decoder1(dec0)
        out = self.out(out)
        out = self.sigmoid(out)
        if x_gt is not None:
            loss = F.mse_loss(out, x_gt)
            return loss
        else:
            return out
        
    def recons_visualize(self, x_in, gt, epochs, save_dir = None):
        with torch.no_grad():
            out = self.forward(x_in)
            data_range = gt.max() - gt.min()
            # psnr = 10 * torch.log10(1 / F.mse_loss(out, gt))
            psnr = 10 * torch.log10(data_range**2 / F.mse_loss(out, gt))
            if save_dir:
                pred_imgs = out
                raw_imgs = gt
                aug_imgs = x_in
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'pred_epoch_{epochs}.png')
                pred_imgs = pred_imgs[0,0,10].cpu().numpy().squeeze()
                raw_imgs = raw_imgs[0,0,10].cpu().numpy().squeeze()
                aug_imgs = aug_imgs[0,0,10].cpu().numpy().squeeze()
                white_line = np.ones((raw_imgs.shape[0], 20))
                # concat_imgs = np.concatenate((raw_imgs, white_line, pred_imgs), axis=1)
                concat_imgs = np.concatenate((aug_imgs, white_line, raw_imgs, white_line, pred_imgs), axis=1)
                concat_imgs = Image.fromarray((concat_imgs * 255).astype('uint8'))
                concat_imgs = concat_imgs.convert('L')
                concat_imgs.save(save_path)

                # pred_imgs = Image.fromarray((pred_imgs * 255).astype('uint8'))
                # pred_imgs.save(save_path)
                # save_path = os.path.join(save_dir, f'raw_epoch_{epochs}.png')
                # raw_imgs = raw_imgs.cpu().numpy().squeeze()
                # raw_imgs = Image.fromarray((raw_imgs * 255).astype('uint8'))
                # raw_imgs.save(save_path)

                return psnr


def segmamba_base_model(**kwargs):  # 28.3M
    model = SegMamba(
        depths=[2,2,2,2], feat_size=[32,64,128,256], hidden_size=512, kernel_size=(1,5,5), **kwargs
    )
    return model

def segmamba_small_model(**kwargs):  # 112.53M
    model = SegMamba(
        depths=[2,2,2,2], feat_size=[64,128,256,512], hidden_size=1024, kernel_size=(1,5,5), **kwargs
    )
    return model

def segmamba_mid_model(**kwargs):  # 206.58M
    model = SegMamba(
        depths=[2,2,2,2], feat_size=[96,192,384,768], hidden_size=1024, kernel_size=(1,5,5), **kwargs
    )
    return model

def segmamba_large_model(**kwargs):  # 506.57M
    model = SegMamba(
        depths=[2,2,2,2], feat_size=[144,288,576,1104], hidden_size=2048, kernel_size=(1,5,5), **kwargs
    )
    return model

def segmamba_huge_model(**kwargs):  # 1008.91M
    model = SegMamba(
        depths=[2,2,2,2], feat_size=[192,384,768,1536], hidden_size=3072, kernel_size=(1,5,5), **kwargs
    )
    return model



# __dict__ = {
#     "segmamba_base": segmamba_base_model(),
#     "segmamba_huge": segmamba_huge_model(),
# }
# segmamba_base = segmamba_base_model()
# segmamba_mid = segmamba_mid_model()
# segmamba_huge = segmamba_huge_model()
