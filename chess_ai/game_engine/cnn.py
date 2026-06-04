import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x): 
        return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            Mish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = Mish()
        self.se = SEBlock(channels) if use_se else None

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se: 
            out = self.se(out)
        out += residual
        return self.act(out)

class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        input_channels = 120
        self.num_filters = 256
        self.num_blocks = 20
        filters = self.num_filters
        num_res_blocks = self.num_blocks
        se_start_idx = 10  # SE blocks from layer 10 onwards

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            Mish()
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(filters, use_se=(i >= se_start_idx)) 
             for i in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
            nn.Flatten(),
            nn.Linear(32*8*8, 4672)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            Mish(),
            nn.Flatten(),
            nn.Linear(64, 256),
            Mish(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.input_conv(x)
        for block in self.res_blocks: 
            x = block(x)
        return self.policy_head(x), self.value_head(x)
