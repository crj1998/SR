import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = F.relu(self.body(x) + x)
        return out


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):

        m = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: 
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None: 
            m.append(act)
        super(BasicBlock, self).__init__(*m)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)

class Block(nn.Module):
    def __init__(self, nf, group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(nf, nf, group=group)
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class CARN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, scale=4, multi_scale=False, group=4):
        super(CARN, self).__init__()
        self.scale = scale
        rgb_range = 1
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.b1 = Block(nf, group=group)
        self.b2 = Block(nf, group=group)
        self.b3 = Block(nf, group=group)
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0)
        
        self.upsample = UpsampleBlock(nf, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out


if __name__ == "__main__":
    model = CARN(3, 3, 36, scale=2, group=4)
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    print(out.shape)
