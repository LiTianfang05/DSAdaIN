import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import Generator


class REBNCONV_e(nn.Module):
    def __init__(self, in_ch, out_ch, dirate):
        super(REBNCONV_e, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, (3, 3), padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.InstanceNorm2d(out_ch)
        self.relu_s1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout


class REBNCONV(nn.Module):
    def __init__(self, inout, style_dim):
        super(REBNCONV, self).__init__()
        self.generate = Generator(inout, image_size=256, style_dim=style_dim)

    def forward(self, x, s):
        out = self.generate(x, s)
        return out


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src


class UNET(nn.Module):
    def __init__(self, style_dim, in_ch=3, out_ch=3):
        super(UNET, self).__init__()
        self.rebnconvin = REBNCONV_e(in_ch, out_ch, dirate=1)
        self.net = REBNCONV(out_ch, style_dim)
        self.outconv = nn.Conv2d(2 * out_ch, out_ch, (1, 1))

    def forward(self, x, s):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1d = self.net(hxin, s)
        out = self.outconv(torch.cat((hx1d, hxin), 1))
        return out
