# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union
# from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock
# # from model.unetr_block import UnetrUpBlock
# from monai.networks.blocks.unetr_block import UnetrUpBlock
# from monai.networks.blocks.dynunet_block import UnetOutBlock
import sys
sys.path.append('/braindat/lab/chenyd/code/Miccai23')
# from monai.networks.nets import ViT
# from model.vit_3d import ViT
from model_vit_3d import ViT
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        img_size: Tuple = (32, 160, 160),
        patch_size: Tuple = (4, 16, 16),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 8,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        kernel_size: Union[Sequence[int], int] = 3,
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.1,
        skip_connection: bool = False,
        show_feature: bool = True,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.img_size = img_size
        self.num_layers = 12
        self.show_feature = show_feature
        self.skip = skip_connection
        self.patch_size = patch_size
        self.feat_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.mha = nn.MultiheadAttention(16 ** 3, 4 ,batch_first=True)
        self.adapool = nn.AdaptiveAvgPool3d((16,16,16))
        self.vit = ViT(
                        image_size = img_size[1:],          # image size
                        frames = img_size[0],               # number of frames
                        image_patch_size = patch_size[1:],     # image patch size
                        frame_patch_size = patch_size[0],      # frame patch size
                        channels=1,
                        num_classes = 1000,
                        dim = hidden_size,
                        depth = 12, # 12
                        heads = num_heads,
                        mlp_dim = mlp_dim,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock( #(1,768,8,10,10)
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=kernel_size,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=kernel_size,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=kernel_size,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.dtrans = nn.Conv3d(feature_size * 2,feature_size * 2,kernel_size=(3, 3, 3), \
            stride=(int(self.patch_size[1]/self.patch_size[0]), 1, 1), padding=(1, 1, 1), bias=False)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        # x, hidden_states_out, _ = self.vit(x_in) # L1
        x, hidden_states_out = self.vit(x_in) # L1
        enc1 = self.encoder1(x_in) # L2
        x2 = hidden_states_out[3] # 3
        # print('x2 shape: ', x2.shape)
        # print('hidden_size: ', self.hidden_size, 'feat_size: ', self.feat_size)
        # print('patch size: ', self.patch_size[0], self.patch_size[1], self.patch_size[2],'img size: ', self.img_size[0], self.img_size[1], self.img_size[2])
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6] # 6
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9] # 9
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        if self.patch_size[0] != self.patch_size[1]:
            dec1 = self.dtrans(dec1)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out) # Ln
        if self.show_feature:
            return dec3,dec2,[dec3, dec2, dec1, out], torch.sigmoid(logits)
        else:
            return torch.sigmoid(logits)
    
if __name__ == "__main__":
    import yaml
    from attrdict import AttrDict
    #torch.cuda.empty_cache()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    cfg_file = 'pretraining_all.yaml'
    with open('/braindat/lab/chenyd/code/Miccai23/config/' + cfg_file, 'r') as f:
            cfg = AttrDict(yaml.safe_load(f))
    unetr = UNETR(
                        in_channels=cfg.MODEL.input_nc,
                        out_channels=cfg.MODEL.output_nc,
                        img_size=cfg.MODEL.unetr_size,
                        patch_size=cfg.MODEL.patch_size,
                        feature_size=64,
                        hidden_size=768,
                        mlp_dim=2048,
                        num_heads=8,
                        pos_embed='perceptron',
                        norm_name='instance',
                        conv_block=True,
                        res_block=True,
                        kernel_size=cfg.MODEL.kernel_size,
                        skip_connection=False,
                        show_feature=True,
                        dropout_rate=0.1).to(device)
    # 参数量测试
    print('参数量(M): ', sum(param.numel() for param in unetr.parameters()) / 1e6)
    x = torch.randn(1, 1,32,160,160).to(device)
    _,_,feature,out = unetr(x)
    print(out.shape)
    for i in feature:
        print(i.shape)
    torch.cuda.empty_cache()
    print(unetr)