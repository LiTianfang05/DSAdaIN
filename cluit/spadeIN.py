import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SPADEIN(nn.Module):
    def __init__(self, feature_size, style_size):
        super(SPADEIN, self).__init__()
        self.norm = nn.InstanceNorm2d(feature_size)
        self.conv = nn.Sequential(spectral_norm(nn.Conv2d(style_size, 128, (3, 3), (1, 1), 1)),
                                  nn.ReLU(inplace=True))
        self.conv_gamma = spectral_norm(nn.Conv2d(128, feature_size, (3, 3), (1, 1), 1))
        self.conv_beta = spectral_norm(nn.Conv2d(128, feature_size, (3, 3), (1, 1), 1))

    def forward(self, x, s):
        s = s.view(s.size(0), s.size(1), 1, 1)
        s = self.conv(s)
        return self.norm(x) * self.conv_gamma(s) + self.conv_beta(s)


class DSPADEIN(nn.Module):
    def __init__(self, feature_size, style_size):
        super(DSPADEIN, self).__init__()
        self.norm = nn.InstanceNorm2d(feature_size)
        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(style_size, 128, (3, 3), (1, 1), 1)),
                                   nn.ReLU(inplace=True))
        self.conv1_gamma = spectral_norm(nn.Conv2d(128, feature_size, (3, 3), (1, 1), 1))
        self.conv1_beta = spectral_norm(nn.Conv2d(128, feature_size, (3, 3), (1, 1), 1))
        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(feature_size, 128, (3, 3), (1, 1), 1)),
                                   nn.ReLU(inplace=True))
        self.conv2_gamma = spectral_norm(nn.Conv2d(128, feature_size, (3, 3), (1, 1), 1))
        self.conv2_beta = spectral_norm(nn.Conv2d(128, feature_size, (3, 3), (1, 1), 1))
        self.fuse = FuseUnit(feature_size)

    def forward(self, x, s):
        s = s.view(s.size(0), s.size(1), 1, 1).expand(s.size(0), s.size(1), x.size(2), x.size(3))
        s = self.conv1(s)
        s_gamma = self.conv1_gamma(s)
        s_beta = self.conv1_beta(s)
        c = self.conv2(x)
        c_gamma = self.conv2_gamma(c)
        c_beta = self.conv2_beta(c)
        gamma = self.fuse(s_gamma, c_gamma)
        beta = self.fuse(s_beta, c_beta)
        out = self.norm(x) * gamma + beta
        return out


class FuseUnit(nn.Module):
    def __init__(self, channels):
        super(FuseUnit, self).__init__()
        self.proj1 = nn.Conv2d(2*channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))
        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride=(1, 1))
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride=(1, 1))
        self.fuse5x = nn.Conv2d(channels, 1, (5, 5), stride=(1, 1))
        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x = nn.ReflectionPad2d((2, 2, 2, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        Fcat = self.proj1(torch.cat((F1, F2), dim=1))
        F1 = self.proj2(F1)
        F2 = self.proj3(F2)
        fusion1 = self.sigmoid(self.fuse1x(Fcat))
        fusion3 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5 = self.sigmoid(self.fuse5x(self.pad5x(Fcat)))
        fusion = (fusion1 + fusion3 + fusion5) / 3
        return torch.clamp(fusion, min=0, max=1.0)*F1 + torch.clamp(1 - fusion, min=0, max=1.0)*F2
