# -*- coding: utf-8 -*-
# @Author: Chen Renjie
# @Date:   2021-04-10 19:49:35
# @Last Modified by:   Chen Renjie
# @Last Modified time: 2021-05-30 00:44:05

"""ResNet for CIFAR10
Change conv1 kernel size from 7 to 3
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class MSA(nn.Module):
    def __init__(self, window_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B_, N, C = x.shape
        num_heads = self.num_heads
        window_size = self.window_size

        qkv = self.qkv(x).reshape(B_, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(window_size * window_size, window_size * window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ConvformerLayer(nn.Module):
    def __init__(self, 
        resolution, window_size, embed_dim, num_heads, 
        mlp_ratio=2., qkv_bias=False, qk_scale=None, 
        drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
    ):
        super(ConvformerLayer, self).__init__()

        k, p, s = window_size, 0, window_size
        l = resolution
        self.k, self.p, self.s = k, p, s
        self.num_wins = ((l - k + 2 * p ) // s + 1)**2

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MSA(window_size, embed_dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_ratio)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.unfold = nn.Unfold(kernel_size=k, padding=p, stride=s)
        self.fold = nn.Fold(output_size=l, kernel_size=k, padding=p, stride=s)
        
        scaler = torch.ones(1, embed_dim, resolution, resolution)
        scaler = self.fold(self.unfold(scaler))

        self.register_buffer("scaler", scaler)

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.k
        num_wins = self.num_wins
        
        x = self.unfold(x)
        x = x.permute(0, 2, 1).reshape(B*num_wins, C, k*k).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).reshape(B, num_wins, C*k*k).permute(0, 2, 1)
        x = self.fold(x)
        x = x / self.scaler
        return x
        
class Convformer(nn.Module):
    def __init__(
        self, upscale, img_size=64, window_size=4, embed_dim=96, num_feat=64, 
        depths=8, num_heads=6, mlp_ratio=2.0, drop_path_rate = 0.1,
    ):
        super(Convformer, self).__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_feat = num_feat
        
        self.preconv = nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.layers = nn.Sequential(*[
            ConvformerLayer(img_size, window_size, embed_dim, num_heads, drop_path_ratio=dpr[i], mlp_ratio=mlp_ratio)
            for i in range(depths)
        ])

        self.postconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.upsample = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(embed_dim, num_feat, kernel_size=3, stride=1, padding=1, bias=True)),
            ("act", nn.LeakyReLU(inplace=True)),
            ("expand", nn.Conv2d(num_feat, num_feat * self.upscale**2, 3, 1, 1)),
            ("pixelshuffle", nn.PixelShuffle(self.upscale)),
            ("conv2", nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1, bias=True))
        ]))


    def forward(self, x):
        x = self.preconv(x)
        x = self.postconv(self.layers(x)) + x
        x = self.upsample(x)
        return x

def convformerTiny(upscale, img_size=64, window_size=4):
    return Convformer(upscale, img_size, window_size, num_feat=32, embed_dim=48, depths=8, num_heads=6, drop_path_rate=0.1)

def convformerSmall(upscale, img_size=64, window_size=4):
    return Convformer(upscale, img_size, window_size, num_feat=64, embed_dim=96, depths=10, num_heads=6, drop_path_rate=0.1)

def convformerBase(upscale, img_size=64, window_size=4):
    return Convformer(upscale, img_size, window_size, num_feat=128, embed_dim=144, depths=12, num_heads=6, drop_path_rate=0.1)


# unit test
if __name__ == "__main__":
    import time
    upscale = 4
    x = torch.rand(1, 3, 64, 64)
    for arch in [convformerTiny, convformerSmall, convformerBase]:
        model = arch(upscale).eval()
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            ts = time.time()
            out = model(x)
            t = time.time() - ts
        
        print(f"{arch.__name__} shape: {tuple(out.shape)} params: {params/1e6:.2f}M Infer: {t:.3f}s")