import torch
import torch.nn as nn


class AdaIN2d(nn.Module):
    def __init__(self, in_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_features)
        self.fc = nn.Linear(style_dim, in_features*2)
        self.register_buffer("one", torch.ones(1))

    def forward(self, x, style_code):
        h = self.fc(style_code)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (self.one + gamma) * self.norm(x) + beta
