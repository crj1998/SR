"""Realize the model definition function."""
from math import sqrt

import torch
import torch.nn as nn

class MapLayer(nn.Module):
    def __init__(self, channels, residual=False):
        super().__init__()
        self.channels = channels
        self.residual = residual
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU(channels)
    
    def forward(self, x):
        if self.residual:
            return self.prelu(self.conv(x)) + x
        else:
            return self.prelu(self.conv(x))

class HFM(nn.Module):
    def __init__(self, 
        scale: int = 2, 
    ) -> None:
        super(HFM, self).__init__()
        self.downsample = nn.AvgPool2d(scale, scale)
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
    
    def forward(self, x):
        return x - self.upsample(self.downsample(x))


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
        # self.hfm = HFM(upscale)
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(channels, d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(d)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1, stride=1, padding=0),
            nn.PReLU(s)
        )

        # Mapping layer.
        self.map = nn.Sequential(*[MapLayer(s, False) for _ in range(m)])

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, stride=1, padding=0),
            nn.PReLU(d)
        )

        # Deconvolution layer.
        self.deconv = nn.Sequential(
            nn.Conv2d(d, d//2*upscale**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(d//2, d//2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d//2),
            nn.Conv2d(d//2, d//2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d//2),
            nn.Conv2d(d//2, channels, kernel_size=3, stride=1, padding=1),
        )

        # self.deconv = nn.ConvTranspose2d(d, channels, kernel_size=9, stride=upscale, padding=4, output_padding=upscale-1)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=upscale, mode="bicubic", align_corners=False)
        else:
            self.upsample = None

        # Initialize model weights.
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            upsample = self.upsample(x.detach())
        # x = self.hfm(x)
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
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
            nn.init.zeros_(m.bias.data)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
            nn.init.zeros_(m.bias.data)

if __name__ == "__main__":
    model = FSRCNN(2, 3, 56, 12, 4, False)
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    print(out.shape)
