#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSD-style dehazing baseline adapted for this project.

Core PSD-style idea kept:
- a restoration backbone predicts a clean image J
- a transmission branch predicts t(x)
- an atmospheric-light branch predicts A
- the hazy image can be reconstructed by I_hat = J * t + A * (1 - t)

Project adaptation:
- forward returns (J, residual, color_gain, sides, I_recon)
- no color-gain mechanism is applied; color_gain is a compatibility placeholder
- output is residual-from-input instead of direct tanh-to-image, which avoids
  gray/blurred predictions at startup and preserves high-frequency content
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _safe_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


class ReflectConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        _init_conv(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_safe_pad2d(x, self.pad))


class PSDResidualBlock(nn.Module):
    def __init__(self, ch: int, res_scale: float = 0.20):
        super().__init__()
        self.res_scale = float(res_scale)
        self.body = nn.Sequential(
            ReflectConv2d(ch, ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(ch, ch, kernel_size=3),
        )
        # Start as identity so early epochs do not collapse to smooth output.
        nn.init.constant_(self.body[-1].conv.weight, 0.0)
        if self.body[-1].conv.bias is not None:
            nn.init.constant_(self.body[-1].conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.res_scale * self.body(x), inplace=True)


class PSDConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int = 2):
        super().__init__()
        layers = [
            ReflectConv2d(in_ch, out_ch, kernel_size=3),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(1, int(blocks))):
            layers.append(PSDResidualBlock(out_ch, res_scale=0.20))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            ReflectConv2d(in_ch, out_ch, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            PSDResidualBlock(out_ch, res_scale=0.20),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            ReflectConv2d(in_ch, out_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            PSDResidualBlock(out_ch, res_scale=0.20),
        )

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="nearest")
        return self.proj(x)


class PSDBackboneUNet(nn.Module):
    """
    High-resolution U-Net backbone.

    The previous implementation downsampled to H/8 and used ConvTranspose2d,
    which was the main source of smooth outputs.  This version uses only two
    downsampling stages and nearest+conv upsampling to preserve details.
    """
    def __init__(self, in_ch=3, base_ch=32, feat_ch=64):
        super().__init__()
        c = int(base_ch)

        self.enc0 = PSDConvBlock(in_ch, c, blocks=1)
        self.down1 = DownBlock(c, c * 2)
        self.enc1 = PSDConvBlock(c * 2, c * 2, blocks=1)
        self.down2 = DownBlock(c * 2, c * 4)
        self.enc2 = PSDConvBlock(c * 4, c * 4, blocks=2)

        self.up1 = UpBlock(c * 4, c * 2)
        self.fuse1 = PSDConvBlock(c * 4, c * 2, blocks=1)
        self.up0 = UpBlock(c * 2, c)
        self.fuse0 = PSDConvBlock(c * 2, c, blocks=1)

        self.feat_conv = ReflectConv2d(c + in_ch, feat_ch, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))

        u1 = self.up1(e2, size=e1.shape[-2:])
        u1 = self.fuse1(torch.cat([u1, e1], dim=1))
        u0 = self.up0(u1, size=e0.shape[-2:])
        u0 = self.fuse0(torch.cat([u0, e0], dim=1))

        return self.feat_conv(torch.cat([u0, x], dim=1))


class PSDDehazeNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=32,
        feat_ch=64,
        residual_scale=0.45,
        refine_scale=0.15,
        t_min=0.10,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.residual_scale = float(residual_scale)
        self.refine_scale = float(refine_scale)
        self.t_min = float(t_min)

        self.backbone = PSDBackboneUNet(in_ch=in_ch, base_ch=base_ch, feat_ch=feat_ch)

        self.t_branch = nn.Sequential(
            ReflectConv2d(feat_ch, feat_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(feat_ch, 1, kernel_size=3),
        )
        self.j_branch = nn.Sequential(
            ReflectConv2d(feat_ch, feat_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(feat_ch, in_ch, kernel_size=3),
        )
        self.detail_refine = nn.Sequential(
            ReflectConv2d(in_ch + feat_ch, feat_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(feat_ch, feat_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(feat_ch, in_ch, kernel_size=3),
        )

        c_a = max(16, base_ch)
        self.a_net = nn.Sequential(
            ReflectConv2d(in_ch, c_a, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            ReflectConv2d(c_a, c_a, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_a, in_ch, kernel_size=1),
        )

        self._init_identity_heads()

    def _init_identity_heads(self):
        # Transmission starts high, close to identity reconstruction.
        last_t = self.t_branch[-1].conv
        nn.init.constant_(last_t.weight, 0.0)
        if last_t.bias is not None:
            nn.init.constant_(last_t.bias, 3.0)

        # Image residual heads start at zero, so J starts as the input image.
        last_j = self.j_branch[-1].conv
        nn.init.constant_(last_j.weight, 0.0)
        if last_j.bias is not None:
            nn.init.constant_(last_j.bias, 0.0)
        last_detail = self.detail_refine[-1].conv
        nn.init.constant_(last_detail.weight, 0.0)
        if last_detail.bias is not None:
            nn.init.constant_(last_detail.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], torch.Tensor]:
        b, c, h, w = x.shape

        feat = self.backbone(x)

        t_unit = torch.sigmoid(self.t_branch(feat))
        t_hat = self.t_min + (1.0 - self.t_min) * t_unit

        residual_main = torch.tanh(self.j_branch(feat)) * self.residual_scale
        residual_detail = torch.tanh(self.detail_refine(torch.cat([x, feat], dim=1))) * self.refine_scale
        residual = torch.nan_to_num(
            residual_main + residual_detail,
            nan=0.0,
            posinf=self.residual_scale + self.refine_scale,
            neginf=-(self.residual_scale + self.refine_scale),
        )
        j_hat = x + residual

        a_hat = torch.sigmoid(self.a_net(x))
        i_recon = j_hat * t_hat + a_hat * (1.0 - t_hat)

        color_gain = torch.ones(b, c, 1, 1, device=x.device, dtype=x.dtype)
        sides = [
            t_hat.expand(-1, c, -1, -1),
            a_hat.expand(-1, -1, h, w),
            torch.clamp(i_recon, 0.0, 1.0),
        ]

        return j_hat, residual, color_gain, sides, i_recon


if __name__ == "__main__":
    net = PSDDehazeNet(in_ch=3, base_ch=32, feat_ch=64)
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out, res, gain, sides, rec = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "rec", rec.shape)
    print("sides", [s.shape if s is not None else None for s in sides])
