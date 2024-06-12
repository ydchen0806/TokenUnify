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

from monai.networks.blocks import UnetrPrUpBlock
from einops.layers.torch import Rearrange
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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
        self.drop_out = nn.Dropout(0.1)  # 20240428
    
    def forward(self, x):
        x_mamba = self.mamba(x)
        x_mamba = self.drop_out(x_mamba)  # 20240428
        out = self.norm(x_mamba)
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, dropout = 0.):
        super().__init__()
        # self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        # self.act = nn.GELU()
        # self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[32, 64, 128, 256],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 image_size=[160,160], image_patch_size=[16,16], dim=512, frames=32, frame_patch_size=4,
                 channels=3, emb_dropout=0.1, dropout=0.1):
        super().__init__()

        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(image_patch_size)
        image_height, image_width = 160, 160
        patch_height, patch_width = 16, 16

        # num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        # patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
        Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
        nn.Linear(1024, dim),
        )
        self.dropout = nn.Dropout(emb_dropout)

        # self.stages = nn.ModuleList()
        # self.gscs = nn.ModuleList()
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # num_slices_list = [32, 16, 8, 4]

        # self.out_indices = out_indices


        self.layers = nn.ModuleList([])
        for _ in range(sum(depths)):
            self.layers.append(nn.ModuleList([
                MambaLayer(dim=512),
                MlpChannel(512,1024)
            ]))

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward_features(self, x):
        outs = []
    
        for mamba_encoder, ff_encoder in self.layers:
            x = mamba_encoder(x) + x
            x = ff_encoder(x) + x
            outs.append(x)
        return outs


    # def forward(self, x):
    #     x = self.forward_features(x)
    #     return x

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = self.dropout(x)
        # x = self.proj_feat(x, 512, (8, 10, 10))
        outs = self.forward_features(x)
        return outs

class SegMamba_linear(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        # feat_size=[48, 96, 192, 384],
        feat_size=[32, 64, 128, 256],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        # hidden_size: int = 768,
        hidden_size: int = 512,
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
            image_size=[160,160],
            image_patch_size=[16,16],

        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            # kernel_size=3,  # raw
            kernel_size=(1,3,3),  # anisotropy v1
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=self.feat_size[1],
            num_layer=2,
            kernel_size=(1,3,3),
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=self.feat_size[2],
            num_layer=1,
            kernel_size=(1,3,3),
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=self.feat_size[3],
            num_layer=0,
            kernel_size=(1,3,3),
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=(1,3,3),
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=(1,3,3),
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=(1,3,3),
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=(1,3,3),
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            # kernel_size=3,  # raw 
            kernel_size=(1,3,3),  # anisotropy v1
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=32, out_channels=self.out_chans)
        self.sigmoid = nn.Sigmoid()
        self.dtrans = nn.Conv3d(self.feat_size[1],self.feat_size[1],kernel_size=(3, 3, 3), \
            stride=(int(16/4), 1, 1), padding=(1, 1, 1), bias=False)


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x_in, x_gt = None):
        outs = self.vit(x_in)  # x_in.shape: [6,1,16,160,160]
        enc1 = self.encoder1(x_in)
        x2 = outs[1]  # [6,32,8,80,80]
        enc2 = self.encoder2(self.proj_feat(x2, 512, (8, 10, 10)))
        x3 = outs[3]  # [6,64,4,40,40]
        enc3 = self.encoder3(self.proj_feat(x3, 512, (8, 10, 10)))
        x4 = outs[5]  #  [6,128,2,20,20]
        enc4 = self.encoder4(self.proj_feat(x4, 512, (8, 10, 10)))
        # enc_hidden = self.encoder5(outs[3])  # outs[3]:[6,256,1,10,10]
        dec4 = self.proj_feat(outs[7], 512, (8, 10, 10))
        dec3 = self.decoder5(dec4, enc4)
        # dec3 = self.decoder5(enc_hidden)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec1 = self.dtrans(dec1)
        dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0)
        # out = self.out(out)
        out = self.out(dec0)
        out = self.sigmoid(out)
        if x_gt is not None:
            loss = F.mse_loss(out, x_gt)
            return loss
        else:
            return out
