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
            nn.Linear(channels // reduction, channels, bias=True),   # bias enables identity-init on grown blocks (Net2Net)
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ValueHead(nn.Module):
    """Global-pooling WDL value head (KataGo/Lc0 style). Replaces the old 1-channel
    bottleneck (Conv 320→1 → 64 spatial cells) that could physically only see 64 numbers.
    A 1x1 conv projects the trunk to `vc` value-feature planes, then GLOBAL avg+max pooling
    over the board gives the dense layers a whole-board read of every channel (avg = overall
    level, max = 'is there any square with this feature' → survives localized tactics)."""
    def __init__(self, in_ch: int, vc: int = 32, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, vc, 1, bias=False),
            nn.BatchNorm2d(vc),
            Mish(),
        )
        self.fc = nn.Sequential(
            nn.Linear(vc * 2, hidden),   # vc*2: avg-pool ‖ max-pool concat
            Mish(),
            nn.Linear(hidden, 3),        # WDL logits
        )

    def forward(self, x):
        y = self.conv(x)                 # (B, vc, 8, 8)
        avg = y.mean(dim=(2, 3))         # (B, vc)
        mx = y.amax(dim=(2, 3))          # (B, vc)
        return self.fc(torch.cat([avg, mx], dim=1))

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
        self.num_filters = 320   # Net2Net widen 256→320 (function-preserving); see tools/grow_net2net.py
        self.num_blocks = 20
        filters = self.num_filters
        num_res_blocks = self.num_blocks
        se_start_idx = 0   # SE on ALL blocks (was 10); new blocks 0-9 grown identity-init

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

        # value_head_gp: renamed from the old `value_head` on purpose — old checkpoints carry
        # value_head.* keys of a DIFFERENT shape, and load_state_dict(strict=False) still RAISES on
        # a shape mismatch for a shared key. Renaming makes the old value_head.* keys "unexpected"
        # (silently ignored) and the new value_head_gp.* keys "missing" (fresh init) → every existing
        # strict=False load degrades gracefully instead of crashing. Migrate via tools/grow_value_head.py.
        self.value_head_gp = ValueHead(filters)

        # Auxiliary heads (training-only; ignored at play time). They feed the shared trunk dense,
        # forward-looking signal to sharpen the value head — see local/plans/auxiliary-targets.md.
        self.material_head = nn.Sequential(   # → final material margin (tanh, [-1,1])
            nn.Conv2d(filters, 1, 1, bias=False),
            nn.BatchNorm2d(1), Mish(), nn.Flatten(), nn.Linear(64, 1)
        )
        self.plies_head = nn.Sequential(      # → plies remaining (sigmoid, [0,1])
            nn.Conv2d(filters, 1, 1, bias=False),
            nn.BatchNorm2d(1), Mish(), nn.Flatten(), nn.Linear(64, 1)
        )
        self.reply_head = nn.Sequential(      # → opponent's reply move (logits over 4672)
            nn.Conv2d(filters, 32, 1, bias=False),
            nn.BatchNorm2d(32), Mish(), nn.Flatten(), nn.Linear(32*8*8, 4672)
        )

    def forward(self, x, with_aux=False):
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        policy, value = self.policy_head(x), self.value_head_gp(x)
        if not with_aux:
            return policy, value          # inference path — unchanged 2-tuple
        material = torch.tanh(self.material_head(x)).squeeze(-1)   # (B,)
        plies    = torch.sigmoid(self.plies_head(x)).squeeze(-1)   # (B,)
        reply    = self.reply_head(x)                              # (B, 4672)
        return policy, value, material, plies, reply
