import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveScale(nn.Module):
    """Adaptive scale module."""
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        scale = torch.sigmoid(self.fc(x.mean(dim=1)))  # Assuming BxNxC -> BxC
        return scale

class MultiScaleAttention(nn.Module):
    """ Multi-scale Attention Module with Frequency Emphasis and Adaptive Scaling """
    def __init__(self, embed_dim, num_heads=12, max_scale=3, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.max_scale = max_scale
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(embed_dim)
        self.adaptive_scale = AdaptiveScale(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        adaptive_scale = self.adaptive_scale(x) * self.max_scale

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        # multi-scale attention with frequency emphasis
        attns = []
        for i in range(int(adaptive_scale)):
            scale = 2 ** i
            q_s = q[:, :, :, ::scale]
            k_s = k[:, :, :, ::scale]
            v_s = v[:, :, :, ::scale]

            attn = (q_s @ k_s.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attns.append(attn)

            attn = attn @ v_s
            attn = attn.transpose(1, 2).reshape(B, -1, C)
            attns.append(attn)

        attn = torch.cat(attns, dim=1)

        # projection
        x = self.proj(attn)
        x = self.proj_drop(x)
        x = x + x
        x = self.norm(x)
        return x

class MultiScaleAttentionHighFre(nn.Module):
    """ Multi-scale Attention Module with Enhanced Focus on High-Frequency Information """
    def __init__(self, embed_dim, num_heads=12, max_scale=3, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.max_scale = max_scale
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(embed_dim)
        self.adaptive_scale = AdaptiveScale(embed_dim)        

        self.freq_amplification_factor = 2  # 高频放大系数
        self.freq_threshold = 0.5  # 频率阈值

    def forward(self, x):
        B, N, C = x.shape
        adaptive_scale = self.adaptive_scale(x) * self.max_scale

 
        freq = torch.fft.fft(x, dim=1)
        freq_amp = torch.abs(freq) 

        freq_weights = torch.sigmoid((freq_amp - self.freq_threshold) * self.freq_amplification_factor)
        freq_weights = freq_weights.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attns = []
        for i in range(int(adaptive_scale)):
            scale = 2 ** i
            q_s = q[:, :, :, ::scale]
            k_s = k[:, :, :, ::scale]
            v_s = v[:, :, :, ::scale]

            attn = (q_s @ k_s.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

      
            attn = attn * freq_weights.unsqueeze(1).unsqueeze(2)

            attn = self.attn_drop(attn)
            attns.append(attn)

            attn = attn @ v_s
            attn = attn.transpose(1, 2).reshape(B, -1, C)
            attns.append(attn)

        attn = torch.cat(attns, dim=1)

        # projection
        x = self.proj(attn)
        x = self.proj_drop(x)
        x = x + x
        x = self.norm(x)
        return x

         