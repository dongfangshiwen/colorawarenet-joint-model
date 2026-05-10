#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSBDN-DFF dehazing backbone.

Adapted from the public CVPR 2020 MSBDN-DFF architecture:
"Multi-Scale Boosted Dehazing Network with Dense Feature Fusion".

The module keeps the 5-scale encoder-decoder and MDC/DFF blocks, and adapts
output to this repository's shared return signature:
    forward(x) -> (out, residual, color_gain, sides)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_VERSION = "msbdn_dff_project_v1"


def _safe_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


def _init_conv(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _activation(name: Optional[str]) -> Optional[nn.Module]:
    if name == "relu":
        return nn.ReLU(True)
    if name == "prelu":
        return nn.PReLU()
    if name == "lrelu":
        return nn.LeakyReLU(0.2, True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in ("no", None):
        return None
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, activation: str = "prelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.act = _activation(activation)
        _init_conv(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.act is not None:
            out = self.act(out)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2, padding: int = 1, activation: str = "prelu"):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding)
        self.act = _activation(activation)
        _init_conv(self.deconv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.deconv(x)
        if self.act is not None:
            out = self.act(out)
        return out


class EncoderMDCBlock(nn.Module):
    """Encoder dense feature fusion block from MSBDN-DFF."""
    def __init__(self, num_filter: int, num_ft: int, mode: str = "iter2"):
        super().__init__()
        self.mode = mode
        self.num_ft = int(num_ft) - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(DeconvBlock(num_filter // (2 ** i), num_filter // (2 ** (i + 1))))
            self.down_convs.append(ConvBlock(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), kernel_size=4, stride=2, padding=1))

    def forward(self, ft_l: torch.Tensor, ft_h_list: List[torch.Tensor]) -> torch.Tensor:
        if self.mode != "iter2":
            raise ValueError("This MSBDN adapter keeps the official iter2 DFF mode.")
        ft_fusion = ft_l
        for i, ft_h in enumerate(ft_h_list):
            ft = ft_fusion
            steps = self.num_ft - i
            for j in range(steps):
                ft = self.up_convs[j](ft)
                if j == steps - 1 and ft.shape[-2:] != ft_h.shape[-2:]:
                    ft = F.interpolate(ft, size=ft_h.shape[-2:], mode="bilinear", align_corners=False)
            ft = ft - ft_h
            for j in range(steps):
                ft = self.down_convs[steps - j - 1](ft)
                if j == steps - 1 and ft.shape[-2:] != ft_fusion.shape[-2:]:
                    ft = F.interpolate(ft, size=ft_fusion.shape[-2:], mode="bilinear", align_corners=False)
            ft_fusion = ft_fusion + ft
        return ft_fusion


class DecoderMDCBlock(nn.Module):
    """Decoder dense feature fusion block from MSBDN-DFF."""
    def __init__(self, num_filter: int, num_ft: int, mode: str = "iter2"):
        super().__init__()
        self.mode = mode
        self.num_ft = int(num_ft) - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(ConvBlock(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size=4, stride=2, padding=1))
            self.up_convs.append(DeconvBlock(num_filter * (2 ** (i + 1)), num_filter * (2 ** i)))

    def forward(self, ft_h: torch.Tensor, ft_l_list: List[torch.Tensor]) -> torch.Tensor:
        if self.mode != "iter2":
            raise ValueError("This MSBDN adapter keeps the official iter2 DFF mode.")
        ft_fusion = ft_h
        for i, ft_l in enumerate(ft_l_list):
            ft = ft_fusion
            steps = self.num_ft - i
            for j in range(steps):
                ft = self.down_convs[j](ft)
                if j == steps - 1 and ft.shape[-2:] != ft_l.shape[-2:]:
                    ft = F.interpolate(ft, size=ft_l.shape[-2:], mode="bilinear", align_corners=False)
            ft = ft - ft_l
            for j in range(steps):
                ft = self.up_convs[steps - j - 1](ft)
                if j == steps - 1 and ft.shape[-2:] != ft_fusion.shape[-2:]:
                    ft = F.interpolate(ft, size=ft_fusion.shape[-2:], mode="bilinear", align_corners=False)
            ft_fusion = ft_fusion + ft
        return ft_fusion


class ConvLayer(nn.Module):
    """Reflection-padded convolution used by the public MSBDN implementation."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        _init_conv(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_safe_pad2d(x, self.pad))


class UpsampleConvLayer(nn.Module):
    """Transposed-convolution upsample layer used by MSBDN-DFF."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride)
        _init_conv(self.deconv)

    def forward(self, x: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        out = self.deconv(x)
        if size is not None and out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, scale: float = 0.1):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * self.scale
        return x + out


def _make_blocks(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*[ResidualBlock(channels) for _ in range(int(count))])


class MSBDN(nn.Module):
    """Project-adapted MSBDN-DFF backbone."""
    def __init__(self, in_ch: int = 3, base_ch: int = 32, nblocks: int = 3, blocks_per_scale: int = 3, output_eps: float = 1e-4, **unused):
        super().__init__()
        self.in_ch = int(in_ch)
        self.base_ch = max(8, int(base_ch))
        self.nblocks = max(1, int(nblocks))
        self.blocks_per_scale = max(1, int(blocks_per_scale))
        self.output_eps = float(output_eps)

        c = self.base_ch
        c1, c2, c4, c8, c16 = c, c * 2, c * 4, c * 8, c * 16

        self.conv_input = ConvLayer(self.in_ch, c1, kernel_size=11, stride=1)
        self.dense0 = _make_blocks(c1, self.blocks_per_scale)
        self.conv2x = ConvLayer(c1, c2, kernel_size=3, stride=2)
        self.fusion1 = EncoderMDCBlock(c2, 2)
        self.dense1 = _make_blocks(c2, self.blocks_per_scale)
        self.conv4x = ConvLayer(c2, c4, kernel_size=3, stride=2)
        self.fusion2 = EncoderMDCBlock(c4, 3)
        self.dense2 = _make_blocks(c4, self.blocks_per_scale)
        self.conv8x = ConvLayer(c4, c8, kernel_size=3, stride=2)
        self.fusion3 = EncoderMDCBlock(c8, 4)
        self.dense3 = _make_blocks(c8, self.blocks_per_scale)
        self.conv16x = ConvLayer(c8, c16, kernel_size=3, stride=2)
        self.fusion4 = EncoderMDCBlock(c16, 5)
        self.dehaze = _make_blocks(c16, self.nblocks)
        self.convd16x = UpsampleConvLayer(c16, c8, kernel_size=3, stride=2)
        self.dense_4 = _make_blocks(c8, self.blocks_per_scale)
        self.fusion_4 = DecoderMDCBlock(c8, 2)
        self.convd8x = UpsampleConvLayer(c8, c4, kernel_size=3, stride=2)
        self.dense_3 = _make_blocks(c4, self.blocks_per_scale)
        self.fusion_3 = DecoderMDCBlock(c4, 3)
        self.convd4x = UpsampleConvLayer(c4, c2, kernel_size=3, stride=2)
        self.dense_2 = _make_blocks(c2, self.blocks_per_scale)
        self.fusion_2 = DecoderMDCBlock(c2, 4)
        self.convd2x = UpsampleConvLayer(c2, c1, kernel_size=3, stride=2)
        self.dense_1 = _make_blocks(c1, self.blocks_per_scale)
        self.fusion_1 = DecoderMDCBlock(c1, 5)
        self.conv_output = ConvLayer(c1, self.in_ch, kernel_size=3, stride=1)

        print(f"[MSBDN] {ARCH_VERSION} base_ch={self.base_ch} nblocks={self.nblocks} blocks_per_scale={self.blocks_per_scale}")

    def _straight_through_clamp(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, ref)
        clipped = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        return x + (clipped - x).detach()

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        res1x = self.conv_input(x)
        feature_mem = [res1x]
        x1 = self.dense0(res1x) + res1x
        res2x = self.conv2x(x1)
        res2x = self.fusion1(res2x, feature_mem)
        feature_mem.append(res2x)
        res2x = self.dense1(res2x) + res2x
        res4x = self.conv4x(res2x)
        res4x = self.fusion2(res4x, feature_mem)
        feature_mem.append(res4x)
        res4x = self.dense2(res4x) + res4x
        res8x = self.conv8x(res4x)
        res8x = self.fusion3(res8x, feature_mem)
        feature_mem.append(res8x)
        res8x = self.dense3(res8x) + res8x
        res16x = self.conv16x(res8x)
        res16x = self.fusion4(res16x, feature_mem)
        res_dehaze = res16x
        in_ft = res16x * 2.0
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        feature_mem_up = [res16x]
        up16 = self.convd16x(res16x, size=res8x.shape[-2:])
        res8x = up16 + res8x
        res8x = self.dense_4(res8x) + res8x - up16
        res8x = self.fusion_4(res8x, feature_mem_up)
        feature_mem_up.append(res8x)
        up8 = self.convd8x(res8x, size=res4x.shape[-2:])
        res4x = up8 + res4x
        res4x = self.dense_3(res4x) + res4x - up8
        res4x = self.fusion_3(res4x, feature_mem_up)
        feature_mem_up.append(res4x)
        up4 = self.convd4x(res4x, size=res2x.shape[-2:])
        res2x = up4 + res2x
        res2x = self.dense_2(res2x) + res2x - up4
        res2x = self.fusion_2(res2x, feature_mem_up)
        feature_mem_up.append(res2x)
        up2 = self.convd2x(res2x, size=x1.shape[-2:])
        x1 = up2 + x1
        x1 = self.dense_1(x1) + x1 - up2
        x1 = self.fusion_1(x1, feature_mem_up)
        return self.conv_output(x1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        if x.shape[1] != self.in_ch:
            raise ValueError(f"MSBDN expected {self.in_ch} channels, got {x.shape[1]}.")
        raw = self._forward_core(x)
        out = self._straight_through_clamp(raw, x)
        residual = out - x
        color_gain = (out.mean(dim=(2, 3), keepdim=True) + 1e-6) / (x.mean(dim=(2, 3), keepdim=True) + 1e-6)
        color_gain = torch.clamp(color_gain, 0.2, 3.0)
        sides = [None, None, None]
        return out, residual, color_gain, sides


if __name__ == "__main__":
    net = MSBDN(in_ch=3, base_ch=16, nblocks=3)
    x = torch.rand(1, 3, 127, 129)
    with torch.no_grad():
        out, res, gain, sides = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "sides", sides)