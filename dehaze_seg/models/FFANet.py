#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFA-Net dehazing module adapted for this project.

This implementation follows the public FFA-Net core structure:
- Feature Attention block = convolutional residual block + channel attention + pixel attention
- three serial groups, each with multiple FA blocks and group residual learning
- feature-fusion attention over the three group outputs
- pixel attention after group fusion
- global residual output: J = I + R

The only project-specific adaptation is the return signature:
    forward(I) -> (dehazed, residual, color_gain, sides)

No color-gain branch is used here, so it remains a fair baseline against
ColorAwareUNet's explicit color-gain design.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class PALayer(nn.Module):
    """Pixel attention used by FFA-Net for spatially uneven haze."""
    def __init__(self, channel):
        super().__init__()
        mid = max(1, channel // 8)
        self.pa = nn.Sequential(
            nn.Conv2d(channel, mid, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.pa(x)


class CALayer(nn.Module):
    """Channel attention used inside each FFA block."""
    def __init__(self, channel):
        super().__init__()
        mid = max(1, channel // 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, mid, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.ca(self.avg_pool(x))


class FFABlock(nn.Module):
    """
    FFA-Net basic block.

    Matches the public implementation pattern:
      conv + ReLU -> local residual -> conv -> CA -> PA -> block residual
    """
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        return res + x


class Group(nn.Module):
    """Serial FFA blocks followed by group-level residual learning."""
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        body = [FFABlock(conv, dim, kernel_size) for _ in range(blocks)]
        body.append(conv(dim, dim, kernel_size))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x


class FFANet(nn.Module):
    """
    FFA-Net baseline.

    Compatibility notes:
    - n_down is accepted but ignored; FFA-Net keeps full resolution.
    - n_ffab_deep is the number of FFA blocks per group. The paper/public
      implementation commonly uses 19.
    - residual_scale/refine_scale arguments from older local experiments are
      accepted but intentionally ignored to avoid smoothing the official-style
      residual path.
    """
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        n_down=0,
        n_ffab_deep=19,
        groups=3,
        residual_scale=None,
        min_blocks_per_group=None,
        block_res_scale=None,
        group_res_scale=None,
        refine_scale=None,
        conv=default_conv,
    ):
        super().__init__()
        if int(groups) != 3:
            raise ValueError("FFA-Net feature-fusion attention expects groups=3.")

        self.in_ch = int(in_ch)
        self.dim = int(base_ch)
        self.gps = int(groups)
        self.blocks = int(n_ffab_deep)
        kernel_size = 3

        self.pre = nn.Sequential(conv(self.in_ch, self.dim, kernel_size))
        self.g1 = Group(conv, self.dim, kernel_size, blocks=self.blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=self.blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=self.blocks)

        fuse_mid = max(1, self.dim // 16)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, fuse_mid, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_mid, self.dim * self.gps, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.palayer = PALayer(self.dim)
        self.post = nn.Sequential(
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, self.in_ch, kernel_size),
        )

        # Start close to identity for this training wrapper. The official
        # topology is unchanged; this only avoids early clamp saturation when
        # the shared script clamps dehazed images to [0, 1] before the loss.
        nn.init.constant_(self.post[-1].weight, 0.0)
        if self.post[-1].bias is not None:
            nn.init.constant_(self.post[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        b, c, _, _ = x.shape

        feat = self.pre(x)
        res1 = self.g1(feat)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        group_cat = torch.cat([res1, res2, res3], dim=1)
        weights = self.ca(group_cat)
        weights = weights.view(b, self.gps, self.dim, 1, 1)
        fused = (
            weights[:, 0, ...] * res1 +
            weights[:, 1, ...] * res2 +
            weights[:, 2, ...] * res3
        )
        fused = self.palayer(fused)

        residual = self.post(fused)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=1.0, neginf=-1.0)
        dehazed = x + residual

        color_gain = torch.ones(b, c, 1, 1, device=x.device, dtype=x.dtype)
        sides = [None, None, None]
        return dehazed, residual, color_gain, sides


if __name__ == "__main__":
    net = FFANet(in_ch=3, base_ch=64, n_down=0, n_ffab_deep=19, groups=3)
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out, res, gain, sides = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "sides", sides)
