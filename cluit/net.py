import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ecaNet import eca_layer
# from .AdaIN2d import AdaIN2d
# from .spadeIN import SPADEIN
from .spadeIN import DSPADEIN


def conv1x1(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), **kwargs)


def conv3x3(in_channels, out_channels, stride=1, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1, **kwargs)


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, downsample=False, normalize=False, inplace=False, **conv_kwargs):
        super().__init__()
        self.register_buffer("gain", torch.rsqrt(torch.as_tensor(2.0)))
        activation = nn.LeakyReLU(0.2, inplace=inplace)
        residual = []
        if normalize:
            residual.append(nn.InstanceNorm2d(dim_in))
        residual.append(activation)
        residual.append(conv3x3(dim_in, dim_in, **conv_kwargs))
        if downsample:
            residual.append(nn.AvgPool2d(kernel_size=2))
        if normalize:
            residual.append(nn.InstanceNorm2d(dim_in))
        residual.append(eca_layer(dim_in, k_size=3))
        residual.append(activation)
        residual.append(conv3x3(dim_in, dim_out, **conv_kwargs))
        self.residual = nn.Sequential(*residual)
        shortcut = []
        if dim_in != dim_out:
            shortcut.append(conv1x1(dim_in, dim_out, bias=False))
        if downsample:
            shortcut.append(nn.AvgPool2d(kernel_size=2))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        x = self.shortcut(x) + self.residual(x)
        return self.gain * x


class AdainResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = upsample
        self.register_buffer("gain", torch.rsqrt(torch.as_tensor(2.0)))
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        # self.norm1 = AdaIN2d(in_channels, style_dim)
        # self.norm2 = AdaIN2d(out_channels, style_dim)
        # self.norm1 = SPADEIN(in_channels, style_dim)
        # self.norm2 = SPADEIN(out_channels, style_dim)
        self.norm1 = DSPADEIN(in_channels, style_dim)
        self.norm2 = DSPADEIN(out_channels, style_dim)
        shortcut = []
        if upsample:
            shortcut.append(nn.UpsamplingNearest2d(scale_factor=2))
        if in_channels != out_channels:
            shortcut.append(conv1x1(in_channels, out_channels, bias=False))
        self.shortcut = nn.Sequential(*shortcut)

    def _residual(self, x, style_code):
        x = self.norm1(x, style_code)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, style_code)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x, style_code):
        out = self._residual(x, style_code) + self.shortcut(x)
        return self.gain * out


class ContentEncoder(nn.Module):
    def __init__(self, inout, dim_in, repeat_num, max_conv_dim):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [conv1x1(inout, dim_in)]
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.model.append(ResBlock(dim_in, dim_out, downsample=True, normalize=True, inplace=True))
            dim_in = dim_out
        for _ in range(2):
            self.model.append(ResBlock(dim_out, dim_out, downsample=False, normalize=True, inplace=True))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, inout, image_size, style_dim, dim_in=64, max_conv_dim=512):
        super().__init__()
        repeat_num = int(np.log2(image_size)) - 4
        activation = nn.LeakyReLU(0.2, inplace=True)
        self.content_encoder = ContentEncoder(inout, dim_in, repeat_num, max_conv_dim)
        self.to_rgb = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True), activation, conv1x1(dim_in, inout))
        self.decoder = nn.ModuleList()
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.decoder.insert(0, AdainResBlock(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out
        for _ in range(2):
            self.decoder.insert(0, AdainResBlock(dim_out, dim_out, style_dim, upsample=False))

    def forward(self, *args, command="_forward", **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward(self, x, style):
        return self.decode(self.encode(x), style)

    def encode(self, x):
        return self.content_encoder(x)

    def decode(self, x, style):
        for block in self.decoder:
            x = block(x, style)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, image_size, feature_dim=256, dim_in=64, max_conv_dim=512):
        super().__init__()
        activation = nn.LeakyReLU(0.2, inplace=False)
        repeat_num = int(np.log2(image_size)) - 2
        blocks = [conv1x1(3, dim_in)]
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlock(dim_in, dim_out, downsample=True, normalize=False, inplace=False))
            dim_in = dim_out
        blocks.append(activation)
        self.shared = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, (4, 4), (1, 1), 0),
            activation,
            nn.Conv2d(dim_out, 1, (1, 1), (1, 1), 0))
        self.projector = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, (4, 4), (1, 1), 0),
            activation,
            nn.Conv2d(dim_out, feature_dim, (1, 1), (1, 1), 0))

    def forward(self, x, logit=True, project=True):
        x = self.shared(x)
        if logit:
            logits = self.classifier(x).view(x.size(0), -1)
        else:
            logits = None
        if project:
            projections = self.projector(x).view(x.size(0), -1)
            projections = F.normalize(projections)
        else:
            projections = None
        return logits, projections


class StyleEncoder(nn.Module):
    def __init__(self, image_size, style_dim, max_conv_dim=512):
        super(StyleEncoder, self).__init__()
        dim_in = 64
        activation = nn.LeakyReLU(0.2)
        repeat_num = int(np.log2(image_size)) - 2
        self.model = []
        self.model += [conv1x1(3, dim_in)]
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.model.append(ResBlock(dim_in, dim_out, downsample=True, normalize=False, inplace=False))
            dim_in = dim_out
        self.feat_dim = dim_out
        self.model.append(activation)
        self.model.append(nn.Conv2d(dim_out, dim_out, (4, 4), (1, 1), 0))
        self.model.append(activation)
        self.model.append(nn.Flatten(start_dim=1))
        self.model.append(nn.Linear(dim_out, style_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
