import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # self.layers = nn.ModuleList([])
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        #         # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        #     ]))
        self.dim = dim
        self.attn = Attention(self.dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # y = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            y.append(x)
        return x,y

        x_norm = self.norm(x)
        x_attn = self.attn(x_norm)
        # return x_attn

        # B, C = x.shape[:2]
        # assert C == self.dim
        # n_tokens = x.shape[2:].numel()
        # img_dims = x.shape[2:]
        # x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm = self.norm(x_flat)
        # x_attn = self.attn(x_norm)
        # out = x_attn.transpose(-1, -2).reshape(B, C, *img_dims)
        # return out

class MlpChannel(nn.Module):  # copy from segmamba 
    def __init__(self,hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size,  frames, image_patch_size, frame_patch_size, channels = 3, 
                 num_classes, dim, depth = [3, 3, 3, 3], heads, mlp_dim, dropout = 0., emb_dropout = 0.,
                 pool = 'cls', dim_head = 64, feature_size = [32, 64, 128, 256], out_indices = [0, 1, 2, 3]):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.Linear(patch_dim, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # 20240411

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

        # align at the architecture of segmamba 
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(channels, feature_size[0], kernel_size=(1,5,5), stride=2, padding=(0,2,2)),   # anisotropy v2
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(feature_size[i]),
                nn.Conv3d(feature_size[i], feature_size[i+1], kernel_size=(1,3,3), stride=2, padding=(0,1,1)),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Transformer(dim=feature_size[i], depth=depth[i], heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout) for j in range(depth[i])]
            )
            self.stages.append(stage)
            cur += depth[i]
        
        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(feature_size[i_layer])
            layer_name = f'nore{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(feature_size[i_layer], 2 * feature_size[i_layer]))

        self.rearrange = Rearrange('b c f h w -> b (f h w) c')

    def forward_features(self, x):
        outs = []
        # x = self.to_patch_embedding(x)
        # x = self.proj_feat(x, 512, [8,80,80])
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.rearrange(x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return tuple(outs)

        
    def forward(self, x):  # x.shape[1,1,16,160,160]
        # x = self.to_patch_embedding(x)  # x.shape[1,400,512]
        # x = x + self.pos_embedding
        # x = self.dropout(x)
        x = self.forward_features(x)
        return x

    def proj_feat(self, x, hidden_size, feat_size):
            x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            return x

    # def forward(self, video):  
    #     x = self.to_patch_embedding(video)  
    #     b, n, _ = x.shape

    #     # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
    #     # x = torch.cat((cls_tokens, x), dim=1)
    #     # x += self.pos_embedding[:, :(n + 1)]
    #     x = self.dropout(x)

    #     x,hidden_states_out = self.transformer(x)
        
    #     # return x, hidden_states_out

    #     # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

    #     # x = self.to_latent(x)
    #     return self.to_patch_embedding(video),hidden_states_out,self.mlp_head(x)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    #model = ViT(image_size = 160, image_patch_size = 16, frames = 32, frame_patch_size = 16, num_classes = 1000, channels=1,dim = 768, depth = 6, heads = 8, mlp_dim = 3072, dropout = 0.1, emb_dropout = 0.1)
    model = ViT(
    image_size = 160,          # image size
    frames = 32,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 4,      # frame patch size
    channels=1,
    num_classes = 1000,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)
    x = torch.randn(1, 1, 32,160,160)
    model = model.cuda()
    x = x.cuda()
    patch_embedding,hidden_states,mlp_heads = model(x) # (1, 1000)
    print(patch_embedding.shape,hidden_states[0].shape,len(hidden_states),mlp_heads.shape)
    for i in (hidden_states):
        print(i.shape)
