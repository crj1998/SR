import math, random

try:
    from supernet.nn import Linear, Conv2d, LayerNorm, Identity
except:
    from nn import Linear, Conv2d, LayerNorm, Identity

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from thop.vision.basic_hooks import count_convNd, count_linear

def calculate_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.DoubleTensor([2 * input_size])

def count_normalization(m, x, y):
    # TODO: add test cases
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if hasattr(m, "affine"):
        flops *= 2
    m.total_ops += flops




def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.sampled_embed_dim = self.max_embed_dim = embed_dim
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else Identity()

    def set_sample_config(self, embed_dim=None):
        self.sampled_embed_dim = embed_dim or self.sampled_embed_dim
        self.proj.set_sample_config(self.sampled_embed_dim, self.sampled_embed_dim)
        self.norm.set_sample_config(self.sampled_embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x



class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim
        self.sampled_embed_dim = self.max_embed_dim = embed_dim

    def set_sample_config(self, embed_dim=None):
        self.sampled_embed_dim = embed_dim or self.sampled_embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.sampled_embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.max_in_features = self.sampled_in_features = in_features
        self.max_hidden_features = self.sampled_hidden_features = hidden_features
        self.max_out_features = self.sampled_out_features = out_features
        self.fc1 = Linear(in_features, hidden_features, bias=True, scale=False)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, bias=True, scale=False)
        self.drop = nn.Dropout(drop)

    def set_sample_config(self, in_features=None, hidden_features=None, out_features=None):
        self.sampled_in_features = in_features or self.sampled_in_features
        self.sampled_hidden_features = hidden_features or self.sampled_hidden_features
        self.sampled_out_features = out_features or self.sampled_out_features
        self.fc1.set_sample_config(self.sampled_in_features, self.sampled_hidden_features)
        self.fc2.set_sample_config(self.sampled_hidden_features, self.sampled_out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, embed_dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = 16 # dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.max_embed_dim = self.sampled_embed_dim = embed_dim
        self.max_num_heads = self.sampled_num_heads = num_heads

        self.max_num_heads = num_heads
        self.sampled_num_heads = num_heads
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing ="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = Linear(embed_dim, self.head_dim*self.num_heads, bias=qkv_bias, scale=False)
        self.k = Linear(embed_dim, self.head_dim*self.num_heads, bias=qkv_bias, scale=False)
        self.v = Linear(embed_dim, self.head_dim*self.num_heads, bias=qkv_bias, scale=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(self.head_dim*num_heads, embed_dim, bias=True, scale=False)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def set_sample_config(self, embed_dim=None, num_heads=None):
        self.sampled_embed_dim = embed_dim or self.sampled_embed_dim
        self.sampled_num_heads = num_heads or self.sampled_num_heads
        self.q.set_sample_config(self.sampled_embed_dim, self.head_dim*self.sampled_num_heads)
        self.k.set_sample_config(self.sampled_embed_dim, self.head_dim*self.sampled_num_heads)
        self.v.set_sample_config(self.sampled_embed_dim, self.head_dim*self.sampled_num_heads)
        self.proj.set_sample_config(self.head_dim*self.sampled_num_heads, self.sampled_embed_dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        num_heads = self.sampled_num_heads
        head_dim = self.head_dim
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B_, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B_, N, num_heads, head_dim).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1), :self.sampled_num_heads].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, num_heads*head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.max_embed_dim = self.sampled_embed_dim = embed_dim
        self.max_num_heads = self.sampled_num_heads = num_heads
        self.max_mlp_ratio = self.sampled_mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(embed_dim)
        self.attn = WindowAttention(
            embed_dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None

        self.register_buffer("attn_mask", attn_mask)

    def set_sample_config(self, embed_dim=None, num_heads=None, mlp_ratio=None):
        self.sampled_embed_dim = embed_dim or self.sampled_embed_dim
        self.sampled_num_heads = num_heads or self.sampled_num_heads
        self.sampled_mlp_ratio = mlp_ratio or self.sampled_mlp_ratio
        self.norm1.set_sample_config(self.sampled_embed_dim)
        self.norm2.set_sample_config(self.sampled_embed_dim)
        self.attn.set_sample_config(self.sampled_embed_dim, self.sampled_num_heads)
        self.mlp.set_sample_config(self.sampled_embed_dim, int(self.sampled_embed_dim*self.sampled_mlp_ratio), self.sampled_embed_dim) 


    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = torch.where(attn_mask == 0, 0.0, -100.0)

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class BasicBlock(nn.Module):
    def __init__(self, embed_dim, resolution, patch_size, depth=2, num_heads=4, window_size=4, mlp_ratio=2, qkv_bias=False, qk_scale=None, 
        drop=0.0, attn_drop=0.0):
        super().__init__()

        self.max_embed_dim = self.sampled_embed_dim = embed_dim
        self.max_num_heads = self.sampled_num_heads = num_heads
        self.max_mlp_ratio = self.sampled_mlp_ratio = mlp_ratio

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                embed_dim=embed_dim, input_resolution=resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=0.0,
                norm_layer=LayerNorm
            ) for i in range(depth)
        ])

        self.conv = Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(resolution, patch_size, embed_dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(resolution, patch_size, embed_dim)

    def set_sample_config(self, embed_dim=None, num_heads=None, mlp_ratio=None):
        self.sampled_embed_dim = embed_dim or self.sampled_embed_dim
        self.sampled_num_heads = num_heads or self.sampled_num_heads
        self.sampled_mlp_ratio = mlp_ratio or self.sampled_mlp_ratio
        for blk in self.blocks:
            blk.set_sample_config(self.sampled_embed_dim, self.sampled_num_heads, self.sampled_mlp_ratio)
        self.conv.set_sample_config(self.sampled_embed_dim, self.sampled_embed_dim)
        self.patch_embed.set_sample_config(self.sampled_embed_dim)
        self.patch_unembed.set_sample_config(self.sampled_embed_dim)


    def forward(self, x, x_size):
        shortcut = x
        for block in self.blocks:
            x = block(x, x_size)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x + shortcut

class SwinIR(nn.Module):
    def __init__(self, 
        search_layers=[0, 1, 2, 3, 4], search_num_heads=[2, 3, 4], search_mlp_ratio=[1.0, 1.5, 2.0], 
        search_embed_dim=[16, 24, 32], search_num_feat=[16, 24, 32], 
        upscale=2, img_size=64, window_size=4, patch_size=1, 
        qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer=LayerNorm, patch_norm=True, in_chans=3, **kwargs):
        super(SwinIR, self).__init__()
        self.in_chans = in_chans
        self.upscale = upscale
        self.window_size = window_size
        self.patch_norm = patch_norm
        
        max_embed_dim = max(search_embed_dim)
        max_num_feat = max(search_num_feat)
        max_num_heads = max(search_num_heads)
        max_mlp_ratio = max(search_mlp_ratio)

        self.sampled_layer_num = self.max_layer_num = max(search_layers)

        self.search_space = {
            "layers": search_layers, 
            "embed_dim": search_embed_dim, 
            "num_feat": search_num_feat,
            "num_heads": search_num_heads,
            "mlp_ratio": search_mlp_ratio
        }


        ################################### 1, shallow feature extraction ###################################
        self.shallow_feature_extractor = Conv2d(in_chans, max_embed_dim, 3, 1, 1)

        ################################### 2, deep feature extraction ######################################
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size, patch_size, max_embed_dim, norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patches_resolution = self.patch_embed.patches_resolution
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, max_embed_dim)

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.max_layer_num):
            layer = BasicBlock(
                embed_dim=max_embed_dim,
                resolution=(patches_resolution[0], patches_resolution[1]),
                depth=2,
                num_heads=max_num_heads,
                window_size=window_size,
                mlp_ratio=max_mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                patch_size=patch_size
            )
            self.layers.append(layer)

        ################################ 3, high quality image reconstruction ################################
        self.upsample = nn.Sequential(OrderedDict([
            ("conv1", Conv2d(max_embed_dim, max_num_feat, 3, 1, 1)),
            ("act", nn.LeakyReLU(inplace=True)),
            ("conv2", Conv2d(max_num_feat, max_num_feat * upscale**2, 3, 1, 1)),
            ("pixelshuffle", nn.PixelShuffle(upscale)),
            ("conv3", Conv2d(max_num_feat, in_chans, 3, 1, 1))
        ]))

        self.shortcut = nn.Upsample(scale_factor=upscale, mode="bilinear", align_corners=False)
        self.apply(self._init_weights)

        self.sample("max")

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_sample_config(self, config):
        self.config = config
        self.sampled_layer_num = config["layers"]
        self.sampled_embed_dim = config["embed_dim"]
        self.sampled_num_feat = config["num_feat"]

        self.shallow_feature_extractor.set_sample_config(self.in_chans, self.sampled_embed_dim)
        self.patch_embed.set_sample_config(self.sampled_embed_dim)
        self.patch_unembed.set_sample_config(self.sampled_embed_dim)

        for i, layer in enumerate(self.layers):
            if i >= self.sampled_layer_num:
                break
            layer.set_sample_config(config["embed_dim"], config[f"num_heads_{i}"], config[f"mlp_ratio_{i}"])

        self.upsample.conv1.set_sample_config(self.sampled_embed_dim, self.sampled_num_feat)
        self.upsample.conv2.set_sample_config(self.sampled_num_feat, self.sampled_num_feat * self.upscale**2)
        self.upsample.conv3.set_sample_config(self.sampled_num_feat, self.in_chans)

    def sample(self, t="random"):
        if t == "max":
            config = {k: max(self.search_space[k]) for k in ["layers", "embed_dim", "num_feat"]}
            config.update({f"num_heads_{i}": max(self.search_space["num_heads"]) for i in range(self.max_layer_num)})
            config.update({f"mlp_ratio_{i}": max(self.search_space["mlp_ratio"]) for i in range(self.max_layer_num)})
        elif t == "min":
            config = {k: min(self.search_space[k]) for k in ["layers", "embed_dim", "num_feat"]}
            config.update({f"num_heads_{i}": min(self.search_space["num_heads"]) for i in range(self.max_layer_num)})
            config.update({f"mlp_ratio_{i}": min(self.search_space["mlp_ratio"]) for i in range(self.max_layer_num)})
        elif t.upper() == "MAGIC-T":
            config = {k: v for k, v in self.config.items()}
            config.update({k: random.choice(self.search_space[k]) for k in ["embed_dim", "num_feat"]})
            layers = min(max(0, config["layers"] + random.choice([-1, 0, 1])), self.max_layer_num)
            if layers == config["layers"]:
                if layers == 0:
                    config["layers"] = 1
                else:
                    i = random.randint(0, layers-1)
                    config[f"num_heads_{i}"] = random.choice(self.search_space["num_heads"])
                    config[f"mlp_ratio_{i}"] = random.choice(self.search_space["mlp_ratio"])
            else:
                config["layers"] = layers
        else:
            config = {k: random.choice(self.search_space[k]) for k in ["layers", "embed_dim", "num_feat"]}
            config.update({f"num_heads_{i}": random.choice(self.search_space["num_heads"]) for i in range(self.max_layer_num)})
            config.update({f"mlp_ratio_{i}": random.choice(self.search_space["mlp_ratio"]) for i in range(self.max_layer_num)})
        self.set_sample_config(config)
        return config

    def check_image_size(self, x):
        _, _, h, w = x.size()
        ws = self.window_size
        mod_pad_h = (ws - h % ws) % ws
        mod_pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        for i, layer in enumerate(self.layers):
            if i >= self.sampled_layer_num:
                break
            x = layer(x, x_size)
        x = self.patch_unembed(x, x_size)
        
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.shallow_feature_extractor(x)

        if self.sampled_layer_num > 0:
            x = self.forward_features(x) + x

        x = self.upsample(x)

        x = x[:, :, :H*self.upscale, :W*self.upscale]
        return x

    def forward_layer(self, x):
        x = self.check_image_size(x)

        x = self.shallow_feature_extractor(x)

        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)

        latent_features = [x.clone()]
        for i, layer in enumerate(self.layers):
            x = layer(x, x_size)
            latent_features.append(x.clone())
        
        return latent_features

    def get_custom_ops(self):
        custom_ops = {
            Conv2d: count_convNd,
            Linear: count_linear,
            LayerNorm: count_normalization
        }
        return custom_ops

if __name__ == '__main__':
    from thop import profile
    torch.manual_seed(42)
    random.seed(42)
    upscale = 2
    window_size = 4
    h, w = 64, 64
    H = h if h%window_size==0 else h + window_size - h%window_size
    W = w if w%window_size==0 else w + window_size - w%window_size

    x = torch.rand((1, 3, h, w))

    dim = 32
    model = SwinIR(
        search_embed_dim=[16, 24, 32], search_num_feat=[16, 24, 32], 
        search_layers=[0, 1, 2, 3, 4], search_num_heads=[2, 3, 4], search_mlp_ratio=[1.0, 1.5, 2.0], 
        upscale=upscale, img_size=(H, W), patch_size=1, window_size=window_size
    )



    with torch.no_grad():
        for _ in range(10):
            model.sample()
            macs, params = profile(model, inputs=(x, ), custom_ops=model.get_custom_ops(), verbose=False)
            y = model(x)
            print(f"SwinIR depth@ {macs/1e6:9.2f} /{params/1e3:6.2f}")
            print(tuple(x.shape), tuple(y.shape), y.mean(dim=(1, 2, 3)))
            print(20*"===")
