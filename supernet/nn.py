import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Identity):
    def __init__(self):
        super(Identity, self).__init__()
    
    def set_sample_config(self, *args, **kwargs):
        return 

    def get_params(self):
        params = 0
        return params
        

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=False):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        self.sample_in_features = in_features
        self.sample_out_features = out_features
        # self.scale = nn.Parameter(torch.ones(in_features)) if scale else None
        self.scale = torch.linspace(0.5, 1.5, in_features) if scale else None

    def set_sample_config(self, in_features, out_features):
        self.sample_in_features = in_features
        self.sample_out_features = out_features

    def sample_parameters(self):
        weight = self.weight[:self.sample_out_features, :self.sample_in_features]
        bias = self.bias[:self.sample_out_features] if self.bias is not None else None
        return weight, bias

    def get_params(self):
        weight, bias = self.sample_parameters()
        params = weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias = self.sample_parameters()
        if self.scale is not None:
            x = x * self.scale[..., :x.size(-1)] / self.scale[..., :x.size(-1)].sum() * x.size(-1)
        return F.linear(x, weight, bias)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.sample_in_channels = in_channels
        self.sample_out_channels = out_channels

    def set_sample_config(self, in_channels, out_channels):
        self.sample_in_channels = in_channels
        self.sample_out_channels = out_channels
    
    def sample_parameters(self):
        weight = self.weight[:self.sample_out_channels, :self.sample_in_channels]
        bias = self.bias[:self.sample_out_channels] if self.bias is not None else None
        return weight, bias
    
    def get_params(self):
        weight, bias = self.sample_parameters()
        params = weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias = self.sample_parameters()
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class PReLU(nn.PReLU):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.sample_num_parameters = num_parameters

    def set_sample_config(self, num_parameters):
        self.sample_num_parameters = num_parameters

    def sample_parameters(self):
        weight = self.weight[:self.sample_num_parameters]
        return weight

    def get_params(self):
        weight = self.sample_parameters()
        params = weight.numel()
        return params

    def forward(self, x):
        weight = self.sample_parameters()
        return F.prelu(x, weight)


class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

        self.sample_in_channels = in_channels
        self.sample_out_channels = out_channels

    def set_sample_config(self, in_channels, out_channels):
        self.sample_in_channels = in_channels
        self.sample_out_channels = out_channels
    
    def sample_parameters(self):
        weight = self.weight[:self.sample_in_channels, :self.sample_out_channels]
        bias = self.bias[:self.sample_out_channels] if self.bias is not None else None
        return weight, bias

    def get_params(self):
        weight, bias = self.sample_parameters()
        params = weight.numel()
        params += 0 if bias is None else bias.numel()
        return params
        
    def forward(self, x):
        weight, bias = self.sample_parameters()
        return F.conv_transpose2d(x, weight, bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.sampled_normalized_shape = self.normalized_shape
    
    def set_sample_config(self, normalized_shape):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )  
        self.sampled_normalized_shape = tuple([min(i, j) for i, j in zip(normalized_shape, self.normalized_shape)])

    def sample_parameters(self):
        indices = [slice(0, i) for i in self.sampled_normalized_shape]
        weight = self.weight[indices] if self.weight is not None else None
        bias = self.bias[indices] if self.bias is not None else None
        return weight, bias

    def get_params(self):
        weight, bias = self.sample_parameters()
        params = 0 if weight is None else weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias = self.sample_parameters()
        return F.layer_norm(x, self.sampled_normalized_shape, weight, bias, eps=self.eps)




class PixelShuffle(nn.PixelShuffle):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__(upscale_factor)
        self.sampled_upscale_factor = upscale_factor

    def set_sample_config(self, upscale_factor):
        self.sampled_upscale_factor = upscale_factor

    def sample_parameters(self):
        return self.sampled_upscale_factor

    def get_params(self):
        # self.sample_parameters()
        params = 0
        return params

    def forward(self, x):
        upscale_factor = self.sample_parameters()
        return F.pixel_shuffle(x, upscale_factor)


class PixelUpscale(nn.Conv2d):
    def __init__(self, upscale_factor, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PixelUpscale, self).__init__(in_channels, out_channels*upscale_factor**2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        self.set_sample_config(upscale_factor, in_channels, out_channels)

    def set_sample_config(self, upscale_factor, in_channels, out_channels):
        self.sampled_upscale_factor = upscale_factor
        self.sampled_in_channels = in_channels
        self.sampled_out_channels = out_channels
        

    def sample_parameters(self):
        upscale_factor = self.sampled_upscale_factor
        ij = torch.arange(self.sampled_out_channels) * self.sampled_upscale_factor**2
        ij = ij.reshape(-1, 1)
        offset = self.upscale_factor * torch.arange(upscale_factor).reshape(-1, 1) + torch.arange(upscale_factor).reshape(1, -1)
        offset = offset.reshape(1, -1)
        indices = (ij + offset).reshape(-1)
        weight = self.weight[indices, :self.sampled_in_channels]
        bias = self.bias[indices] if self.bias is not None else None
        return weight, bias, upscale_factor

    def get_params(self):
        weight, bias, _ = self.sample_parameters()
        params = weight.numel()
        params += 0 if bias is None else bias.numel()
        return params

    def forward(self, x):
        weight, bias, upscale_factor = self.sample_parameters()
        x = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        x = F.pixel_shuffle(x, upscale_factor)
        return x
    


if __name__ == "__main__":
    model = PixelUpscale(2, 3, 8)
    x = torch.rand(4, 3, 8, 8)
    with torch.no_grad():
        y = model(x)
        print(x.shape, y.shape)
