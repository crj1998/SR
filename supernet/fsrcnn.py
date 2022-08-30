"""Realize the model definition function."""
import os
from math import sqrt
from collections import OrderedDict
try:
    from supernet.nn import Conv2d, PReLU, ConvTranspose2d, PixelShuffle
except:
    from nn import Conv2d, PReLU, ConvTranspose2d, PixelShuffle

import torch
import torch.nn as nn
# from torch.nn import Conv2d, PReLU, ConvTranspose2d, PixelShuffle


class MapLayer(nn.Module):
    def __init__(self, channels, residual=False):
        super().__init__()
        self.channels = channels
        self.residual = residual
        self.conv = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.prelu = PReLU(channels)
    
    def set_sample_config(self, channels):
        self.conv.set_sample_config(channels, channels)
        self.prelu.set_sample_config(channels)

    def get_params(self):
        params = self.conv.get_params() + self.prelu.get_params()
        return params

    def forward(self, x):
        if self.residual:
            return self.prelu(self.conv(x)) + x
        else:
            return self.prelu(self.conv(x))

class FSRCNN(nn.Module):
    def __init__(self, 
        upscale: int = 2, 
        channels: int = 3,
        d: int = 56,
        s: int = 12,
        m: int = 4,
        upsample: bool = False
    ) -> None:
        super(FSRCNN, self).__init__()
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(OrderedDict([
            ("conv", Conv2d(channels, d, kernel_size=5, stride=1, padding=2)),
            ("prelu", PReLU(d))
        ]))

        # Shrinking layer.
        self.shrink = nn.Sequential(OrderedDict([
            ("conv", Conv2d(d, s, kernel_size=1, stride=1, padding=0)),
            ("prelu", PReLU(s))
        ]))
        
        # Mapping layer.
        self.map = nn.Sequential(*[MapLayer(s, False) for _ in range(m)])

        # Expanding layer.
        self.expand = nn.Sequential(OrderedDict([
            ("conv", Conv2d(s, d, kernel_size=1, stride=1, padding=0)),
            ("prelu", PReLU(d))
        ]))

        # Deconvolution layer.
        # self.deconv = ConvTranspose2d(d, channels, kernel_size=9, stride=upscale, padding=4, output_padding=upscale-1)
        self.deconv = nn.Sequential(OrderedDict([
            ("conv1", Conv2d(d, d//2*upscale**2, kernel_size=3, stride=1, padding=1)),
            ("pixelshuffle", PixelShuffle(upscale)),
            ("conv2", nn.Conv2d(d//2, d//2, kernel_size=3, stride=1, padding=1)),
            ("prelu", PReLU(d)),
            ("conv3", nn.Conv2d(d//2, channels, kernel_size=3, stride=1, padding=1)),
        ]))

        if upsample:
            self.upsample = nn.Upsample(scale_factor=upscale, mode="bicubic", align_corners=False)
        else:
            self.upsample = None

        # Initialize model weights.
        self.apply(self._init_weights)

    def set_sample_config(self, upscale):
        # self.feature_extraction.conv.set_sample_config(3, d)
        # self.feature_extraction.prelu.set_sample_config(d)

        # self.shrink.conv.set_sample_config(d, s)
        # self.shrink.prelu.set_sample_config(s)

        # for module in self.map:
        #     module.set_sample_config(s)
        
        # self.expand.conv.set_sample_config(s, d)
        # self.expand.prelu.set_sample_config(d)

        # self.deconv.set_sample_config(d, 3)
        d = 56
        self.deconv.conv1.set_sample_config(d, d//2*upscale**2)
        self.deconv.pixelshuffle.set_sample_config(upscale)

    def get_params(self):
        params = 0
        params += self.feature_extraction.conv.get_params()
        params += self.feature_extraction.prelu.get_params()

        params += self.shrink.conv.get_params()
        params += self.shrink.prelu.get_params()

        for module in self.map:
            params += module.get_params()
        
        params += self.expand.conv.get_params()
        params += self.expand.prelu.get_params()

        params += self.deconv.get_params()

        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            upsample = self.upsample(x)
        x = self.feature_extraction(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)

        if self.upsample is None:
            return x
        else:
            return x + upsample

    # The filter weight of each layer is a Gaussian distribution with zero mean and standard deviation initialized by random extraction 0.001 (deviation is 0).
    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, Conv2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
            nn.init.zeros_(m.bias.data)
        if isinstance(m, ConvTranspose2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
            nn.init.zeros_(m.bias.data)

if __name__ == "__main__":
    from thop import profile
    from thop.vision.basic_hooks import count_convNd, count_prelu
    model = FSRCNN(2, 3, 56, 12, 4)
    x = torch.randn(1, 3, 64, 64)
    custom_ops = {
        Conv2d: count_convNd,
        ConvTranspose2d: count_convNd,
        PReLU: count_prelu,
    }
    # model.set_sample_config(7, 12)
    for d in range(3, 57):
        model.set_sample_config(d)
        # print(d, model.get_params()/10**3)
        # model = FSRCNN(2, 3, d, 12, 4)
        flops, params = profile(model, inputs=(x, ), custom_ops=custom_ops, verbose=False)
        print(d, flops/10**6, params/10**3)
    with torch.no_grad():
        out = model(x)
    print(out.shape)
