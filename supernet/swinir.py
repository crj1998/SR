import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict

try:
    from supernet.nn import Linear, Conv2d, LayerNorm, Identity
except:
    from nn import Linear, Conv2d, LayerNorm, Identity



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
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = Identity()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else Identity()

    def set_sample_config(self, embed_dim):
        self.norm.set_sample_config(embed_dim)

    def get_params(self):
        params = 0
        params += self.norm.get_params()
        return params


    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
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

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.sampled_in_features = in_features
        self.sampled_hidden_features = hidden_features
        self.sampled_out_features = out_features

    def set_sample_config(self, in_features, hidden_features=None, out_features=None):
        self.sampled_in_features = in_features
        self.sampled_hidden_features = hidden_features or self.sampled_hidden_features
        self.sampled_out_features = out_features or self.sampled_out_features
        self.fc1.set_sample_config(self.sampled_in_features, self.sampled_hidden_features)
        self.fc2.set_sample_config(self.sampled_hidden_features, self.sampled_out_features)

    def get_params(self):
        params = 0
        params += self.fc1.get_params()
        params += self.fc2.get_params()

        return params

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
        # self.head_dim = embed_dim // num_heads
        self.head_dim = 16
        self.scale = qk_scale or self.head_dim ** -0.5

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

        self.q = Linear(embed_dim, self.head_dim*self.num_heads, bias=qkv_bias)
        self.k = Linear(embed_dim, self.head_dim*self.num_heads, bias=qkv_bias)
        self.v = Linear(embed_dim, self.head_dim*self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = Linear(self.head_dim*num_heads, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def set_sample_config(self, embed_dim):
        self.q.set_sample_config(embed_dim, self.head_dim*self.num_heads)
        self.k.set_sample_config(embed_dim, self.head_dim*self.num_heads)
        self.v.set_sample_config(embed_dim, self.head_dim*self.num_heads)
        self.proj.set_sample_config(self.head_dim*self.num_heads, embed_dim)

    def get_params(self):
        params = 0
        params += self.q.get_params()
        params += self.k.get_params()
        params += self.v.get_params()
        params += self.proj.get_params()

        return params


    def forward(self, x, mask=None):
        B_, N, C = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        window_size = self.window_size
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B_, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B_, N, num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(window_size[0] * window_size[1], window_size[0] * window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, head_dim*num_heads)
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
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(embed_dim)
        self.attn = WindowAttention(
            embed_dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim, act_layer=act_layer, drop=drop)

        attn_mask = self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        window_size, shift_size = self.window_size, self.shift_size
        h_slices = [slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)]
        w_slices = [slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)]

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = torch.where(attn_mask == 0, 0.0, -100.0)

        return attn_mask

    def set_sample_config(self, embed_dim):
        self.norm1.set_sample_config(embed_dim)
        self.attn.set_sample_config(embed_dim)
        self.norm2.set_sample_config(embed_dim)
        self.mlp.set_sample_config(embed_dim, int(embed_dim * self.mlp_ratio), embed_dim)

    def get_params(self):
        params = 0
        params += self.norm1.get_params()
        params += self.attn.get_params()
        params += self.norm2.get_params()
        params += self.mlp.get_params()

        return params

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


class BasicLayer(nn.Module):
    def __init__(self, embed_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNorm, downsample=None):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.input_resolution = input_resolution
        
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                embed_dim=embed_dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        # patch merging layer
        self.downsample = downsample(input_resolution, dim=embed_dim, norm_layer=norm_layer) if downsample is not None else Identity()

    def set_sample_config(self, embed_dim):
        for blk in self.blocks:
            blk.set_sample_config(embed_dim)

    def get_params(self):
        params = 0
        for blk in self.blocks:
            params += blk.get_params()
        return params

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        x = self.downsample(x)
        return x



class RSTB(nn.Module):
    def __init__(self, embed_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNorm, downsample=None, 
                 img_size=224, patch_size=4
        ):
        super(RSTB, self).__init__()

        self.dim = embed_dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            embed_dim=embed_dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
        )

        self.conv = Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, embed_dim)

    def set_sample_config(self, embed_dim):
        self.residual_group.set_sample_config(embed_dim)
        self.conv.set_sample_config(embed_dim, embed_dim)

    
    def get_params(self):
        params = 0
        params += self.residual_group.get_params()
        params += self.conv.get_params()
        return params

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class AbsPosEmbed(nn.Module):
    def __init__(self, length: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.sampled_embed_dim = embed_dim

    def set_sample_config(self, embed_dim):
        self.sampled_embed_dim = embed_dim

    def sample_parameters(self):
        pos_embed = self.pos_embed[..., :self.sampled_embed_dim]
        return pos_embed

    def get_params(self):
        pos_embed = self.sample_parameters()
        params = pos_embed.numel()
        return params

    def forward(self, x):
        pos_embed = self.sample_parameters()
        return x + pos_embed



class SwinIR(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, num_feat = 64,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNorm, ape=False, patch_norm=True, upscale=2, 
                 **kwargs):
        super(SwinIR, self).__init__()

        self.upscale = upscale
        self.window_size = window_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        ################################### 1, shallow feature extraction ###################################
        self.shallow_feature_extractor = Conv2d(in_chans, embed_dim, 3, 1, 1)

        ################################### 2, deep feature extraction ######################################
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim, norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patches_resolution = self.patch_embed.patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, embed_dim)

        # absolute position embedding
        self.pos_embed = AbsPosEmbed(num_patches, embed_dim) if ape else Identity()
        self.pos_drop = nn.Dropout(drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                embed_dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample = None,
                img_size=img_size,
                patch_size=patch_size
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = Conv2d(embed_dim, embed_dim, 3, 1, 1)


        ################################ 3, high quality image reconstruction ################################
        self.upsample = nn.Sequential(OrderedDict([
            ("conv1", Conv2d(embed_dim, num_feat, 3, 1, 1)),
            ("act", nn.LeakyReLU(inplace=True)),
            ("expand", Conv2d(num_feat, num_feat * self.upscale**2, 3, 1, 1)),
            ("pixelshuffle", nn.PixelShuffle(self.upscale)),
            ("conv2", Conv2d(num_feat, in_chans, 3, 1, 1))
        ]))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_sample_config(self, embed_dim, num_feat):
        self.shallow_feature_extractor.set_sample_config(3, embed_dim)
        self.patch_embed.set_sample_config(embed_dim)
        self.pos_embed.set_sample_config(embed_dim)
        for layer in self.layers:
            layer.set_sample_config(embed_dim)
        self.norm.set_sample_config(embed_dim)
        self.conv_after_body.set_sample_config(embed_dim, embed_dim)

        self.upsample.conv1.set_sample_config(embed_dim, num_feat)
        self.upsample.expand.set_sample_config(num_feat, num_feat * self.upscale**2)
        self.upsample.conv2.set_sample_config(num_feat, 3)


    def get_params(self):
        params = 0
        params += self.shallow_feature_extractor.get_params()
        params += self.patch_embed.get_params()
        params += self.pos_embed.get_params()
        for layer in self.layers:
            params += layer.get_params()
        params += self.norm.get_params()
        params += self.conv_after_body.get_params()
        params += self.upsample.conv1.get_params()
        params += self.upsample.expand.get_params()
        params += self.upsample.conv2.get_params()
        return params

    def check_image_size(self, x):
        _, _, h, w = x.size()
        ws = self.window_size
        # mod_pad_h = 0 if h % ws == 0 else ws - h % ws
        mod_pad_h = (ws - h % ws) % ws
        mod_pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        # print(tuple(x.shape), f"\t\t Patch Embed")
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
            # print(tuple(x.shape), f"\t\t Stage")

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        # print(tuple(x.shape), f"\t\t Patch unEmbed")
        return x

    def forward(self, x):
        # self.sample_parameters()
        H, W = x.shape[2:]
        # print(tuple(x.shape), f"\t\t Input")
        x = self.check_image_size(x)
        # print(tuple(x.shape), f"\t\t Input padded")

        x = self.shallow_feature_extractor(x)
        # print(tuple(x.shape), f"\t\t Shallow feature")

        x = self.conv_after_body(self.forward_features(x)) + x
        # print(tuple(x.shape), f"\t\t deep feature")

        x = self.upsample(x)
        # for layer in self.upsample:
        #     x = layer(x)
        #     print(tuple(x.shape), f"\t\t upsample")

        x = x[:, :, :H*self.upscale, :W*self.upscale]
        # print(tuple(x.shape), f"\t\t Output unpadded")
        return x



if __name__ == '__main__':
    torch.manual_seed(42)
    upscale = 2
    window_size = 4
    height, width = 64, 64
    # height = (height // window_size + 1) * window_size
    # width = (width // window_size + 1) * window_size

    model = SwinIR(
        upscale=upscale, img_size=(height, width), patch_size=1, num_feat=64, embed_dim=64,
        window_size=window_size, depths=[2, 2, 2, 2], num_heads=[2, 2, 2, 2], mlp_ratio=2
    )


    x = torch.rand((5, 3, height, width))
    x[0] *= 0
    x[1] *= 0.25
    x[2] *= 0.5
    x[3] *= 0.75
    x[4] *= 1.0
    import random
    cand = [(16, 16), (16, 32), (16, 64), (32, 16), (32, 32), (32, 64), (64, 16), (64, 32), (64, 64)]
    print(20*"==")
    with torch.no_grad():
        for _ in range(9):
            # embed_dim, num_fead = random.choice([16, 32, 64]), random.choice([16, 32, 64])
            embed_dim, num_fead = cand[_]
            model.set_sample_config(embed_dim, num_fead)
            print(embed_dim, num_fead, model.get_params()/1e6)
            y = model(x)
            print(tuple(x.shape), tuple(y.shape), y.mean(dim=(1, 2, 3)))
        # assert ((y.mean(dim=(1, 2, 3)) - torch.Tensor([0.0447, 0.0438, 0.0429, 0.0423, 0.0419])).abs() > 0.0001).sum().item() == 0