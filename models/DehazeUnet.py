#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dehaze-UNet backbone.

This version is aligned more closely with the Dehaze-UNet paper:
- shallow 3x3 convolution
- DOWN is a 2x2 stride-2 convolution
- UP is 1x1 convolution + PixelShuffle, with phase-tied sub-pixels to avoid
  trained sub-pixel phase ghosts
- LAYER uses BN, two parallel Conv-BN-ReLU-Conv-ReLU-BN branches, point-product
  fusion, and depthwise 1x1 projection
- ASMFUN starts with grouped dilated 5x5 convolution, then estimates feature
  space t/A by global pooling and 1x1 convolutions
- no extra RGB escape/refinement head
- skip fusion is a 1x1 average-initialized layer instead of hard addition; this
  keeps the U-Net skip path but avoids doubled edges after PixelShuffle
- no direct hazy RGB detail copy.  The decoder predicts low/mid-frequency
  dehazing, and a full-resolution luminance detail adapter restores aligned
  texture without passing through a down/up path.

The return signature is adapted for the shared training wrapper:
    forward(x) -> (out, residual, color_gain, sides)

The public constructor is intentionally small.  Capacity is controlled by
`base_ch`; anti-ghost and sharpening constants are fixed inside the module so
the training script does not need to juggle many overlapping knobs.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_VERSION = "dehazeunet_aligned_luma_detail_v9"


def _init_conv(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _init_adapter_conv(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _safe_pad_to_multiple(x: torch.Tensor, multiple: int = 4):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    mode = "reflect" if h > pad_h and w > pad_w else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (h, w)


def _safe_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


def _local_mean(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return x
    k = radius * 2 + 1
    return F.avg_pool2d(_safe_pad2d(x, radius), k, stride=1, padding=0)


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _finite_feature(x: torch.Tensor, limit: float = 8.0) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit)
    return torch.clamp(x, -limit, limit)


def _init_average_fuse(conv: nn.Conv2d, channels: int) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()
        for i in range(channels):
            conv.weight[i, i, 0, 0] = 0.5
            conv.weight[i, i + channels, 0, 0] = 0.5


class Layer(nn.Module):
    """
    LAYER module from Dehaze-UNet.

    The two branches aggregate haze-related features and are fused by point-wise
    multiplication before a depthwise 1x1 projection.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(True),
            nn.BatchNorm2d(channels),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(True),
            nn.BatchNorm2d(channels),
        )
        self.proj = nn.Conv2d(channels, channels, 1, groups=channels)

        for m in self.modules():
            _init_conv(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        mid = self.conv_block1(x_norm) * self.conv_block2(x_norm)
        out = x_norm + self.proj(mid)
        return _finite_feature(out)


class Down(nn.Module):
    """DOWN in Dehaze-UNet: 2x2 stride-2 convolution."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 2, stride=2)
        _init_conv(self.down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """UP in Dehaze-UNet: phase-tied 1x1 convolution + PixelShuffle."""
    def __init__(self, in_channels: int, out_channels: int, anti_alias_strength: float = 0.08):
        super().__init__()
        self.anti_alias_strength = float(max(0.0, anti_alias_strength))
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.shuffle = nn.PixelShuffle(2)
        _init_conv(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tie the four sub-pixel phases during the whole training process, not
        # only at initialization. This keeps PixelShuffle paper-style upsampling
        # while preventing phase-specific shifted edges.
        out = self.shuffle(self.proj(x).repeat_interleave(4, dim=1))
        if self.anti_alias_strength > 0.0:
            out = out + self.anti_alias_strength * (_local_mean(out, radius=1) - out)
        return _finite_feature(out)


class SkipFuse(nn.Module):
    """Average-initialized 1x1 fusion for decoder and encoder skip features."""
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Conv2d(channels * 2, channels, 1)
        _init_average_fuse(self.fuse, channels)

    def forward(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return _finite_feature(self.fuse(torch.cat([up, skip], dim=1)))


class ASMFUN(nn.Module):
    """
    ASMFUN embeds the atmospheric scattering model in feature space.

    The paper first applies a grouped dilated 5x5 convolution, then estimates
    abstract t/A feature vectors through global pooling and 1x1 convolutions.
    The paper uses:
        OUT = (INPUT + A * (t - 1)) * t
    We keep that feature-space equation, with bounded identity-centered t/A for
    stable paired training.
    """
    def __init__(self, channels: int, t_scale: float = 0.20, a_scale: float = 0.20):
        super().__init__()
        self.t_scale = float(t_scale)
        self.a_scale = float(a_scale)
        self.group_dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            stride=1,
            padding=4,
            dilation=2,
            groups=channels,
            padding_mode="reflect",
        )
        self.shallow = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        hidden = channels * 2
        self.mlp_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.mlp_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, channels, 1),
        )
        for m in self.modules():
            _init_conv(m)
        self._init_identity()

    def _init_identity(self) -> None:
        for seq in (self.mlp_a, self.mlp_t):
            last = seq[-1]
            nn.init.constant_(last.weight, 0.0)
            if last.bias is not None:
                nn.init.constant_(last.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shallow(self.group_dilated(x))
        t = 1.0 + self.t_scale * torch.tanh(self.mlp_t(feat))
        a = self.a_scale * torch.tanh(self.mlp_a(feat))
        out = (x + a * (t - torch.ones_like(t))) * t
        return _finite_feature(out)


class LumaDetailRefineHead(nn.Module):
    """
    Full-resolution luminance detail adapter.

    The U-Net decoder is still the main predictor.  This head learns only a
    bounded luminance ratio from spatially aligned tensors, so it can recover
    table texture and thin edges without drawing an independent RGB layer.
    """
    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 32,
        scale: float = 0.35,
        ratio_limit: float = 0.30,
        output_eps: float = 1e-4,
    ):
        super().__init__()
        self.scale = float(scale)
        self.ratio_limit = float(max(0.0, min(ratio_limit, 0.5)))
        self.output_eps = float(output_eps)
        feat_ch = in_ch * 3 + 2
        self.body = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(True),
            nn.Conv2d(hidden, 1, 3, 1, 1, padding_mode="reflect"),
        )
        self.apply(_init_adapter_conv)
        final = self.body[-1]
        nn.init.constant_(final.weight, 0.0)
        if final.bias is not None:
            nn.init.constant_(final.bias, 0.0)

    def forward(self, hazy: torch.Tensor, pred: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        pred_y = _rgb_to_luma(pred).clamp(self.output_eps, 1.0 - self.output_eps)
        hazy_detail = _rgb_to_luma(hazy) - _local_mean(_rgb_to_luma(hazy), radius=2)
        pred_detail = pred_y - _local_mean(pred_y, radius=1)
        feat = torch.cat([hazy, pred, residual, hazy_detail, pred_detail], dim=1)
        raw = self.body(feat)
        raw = raw - _local_mean(raw, radius=3)
        log_ratio = self.scale * torch.tanh(raw)
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, 1.0 - self.ratio_limit, 1.0 + self.ratio_limit)
        return pred * ratio


class DehazeUNet(nn.Module):
    """
    Lightweight paper-core Dehaze-UNet.

    `base_ch` is honored. The training script passes base_ch=32, which keeps
    this model lightweight instead of silently expanding it to 96 channels.
    """
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        residual_scale: float = 0.45,
        output_eps: float = 1e-4,
        **unused,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.requested_base_ch = int(base_ch)
        self.base_ch = max(32, int(base_ch))
        self.residual_scale = float(max(0.05, min(abs(residual_scale), 1.0)))
        self.up_antialias_strength = 0.0
        self.input_luma_detail_scale = 0.18
        self.input_luma_detail_radius = 2
        self.input_luma_detail_ratio_limit = 0.18
        self.input_luma_detail_gate_threshold = 0.006
        self.input_luma_detail_gate_slope = 60.0
        self.pred_sharp_scale = 0.14
        self.pred_sharp_radius = 1
        self.pred_sharp_ratio_limit = 0.25
        self.output_eps = float(output_eps)

        c = self.base_ch
        self.inconv = nn.Sequential(
            nn.Conv2d(self.in_ch, c, 3, stride=1, padding=1, padding_mode="reflect"),
            nn.ReLU(True),
        )

        self.down1 = Down(c, c * 2)
        self.fuse1 = nn.Conv2d(c * 2, c * 2, 1)
        self.layer1 = Layer(c * 2)

        self.down2 = Down(c * 2, c * 4)
        self.fuse2 = nn.Conv2d(c * 4, c * 4, 1)
        self.layer2 = Layer(c * 4)
        self.layer3 = Layer(c * 4)
        self.asm1 = ASMFUN(c * 4)
        self.layer4 = Layer(c * 4)

        self.up1 = Up(c * 4, c * 2, anti_alias_strength=self.up_antialias_strength)
        self.skip1 = SkipFuse(c * 2)
        self.asm2 = ASMFUN(c * 2)
        self.layer5 = Layer(c * 2)
        self.up2 = Up(c * 2, c, anti_alias_strength=self.up_antialias_strength)
        self.skip0 = SkipFuse(c)

        self.outconv = nn.Conv2d(c, self.in_ch, 1)
        self.detail_refine = LumaDetailRefineHead(
            in_ch=self.in_ch,
            hidden=max(32, c),
            scale=0.35,
            ratio_limit=0.30,
            output_eps=self.output_eps,
        )

        for m in [self.inconv[0], self.fuse1, self.fuse2, self.outconv]:
            _init_conv(m)
        nn.init.normal_(self.outconv.weight, mean=0.0, std=1e-3)
        if self.outconv.bias is not None:
            nn.init.constant_(self.outconv.bias, 0.0)

        print(
            f"[DehazeUNet] {ARCH_VERSION} base_ch={self.base_ch} "
            f"requested_base_ch={self.requested_base_ch} "
            f"residual_scale={self.residual_scale:.3f} "
            f"up_antialias={self.up_antialias_strength:.2f} "
            f"luma_detail={self.input_luma_detail_scale:.2f} "
            f"detail_refine=True pred_sharp={self.pred_sharp_scale:.2f}"
        )

    def _straight_through_clamp(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, ref)
        clipped = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        return x + (clipped - x).detach()

    def _preserve_luma_detail(self, hazy: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.input_luma_detail_scale <= 0.0:
            return pred
        hazy_y = _rgb_to_luma(hazy)
        pred_y = _rgb_to_luma(pred).clamp(self.output_eps, 1.0 - self.output_eps)
        hazy_detail = hazy_y - _local_mean(hazy_y, radius=self.input_luma_detail_radius)
        pred_detail = pred_y - _local_mean(pred_y, radius=1)
        edge_strength = hazy_detail.abs()
        edge_gate = torch.sigmoid(
            self.input_luma_detail_gate_slope * (
                edge_strength - self.input_luma_detail_gate_threshold
            )
        )
        align_score = hazy_detail * pred_detail / (
            hazy_detail.abs() * pred_detail.abs() + 1e-6
        )
        agreement = torch.sigmoid(3.0 * align_score)
        target_y = torch.clamp(
            pred_y + self.input_luma_detail_scale * edge_gate * agreement * hazy_detail,
            self.output_eps,
            1.0 - self.output_eps,
        )
        ratio = target_y / (pred_y + 1e-6)
        ratio = torch.clamp(
            ratio,
            1.0 - self.input_luma_detail_ratio_limit,
            1.0 + self.input_luma_detail_ratio_limit,
        )
        return self._straight_through_clamp(pred * ratio, pred)

    def _sharpen_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        if self.pred_sharp_scale <= 0.0:
            return pred
        pred_y = _rgb_to_luma(pred).clamp(self.output_eps, 1.0 - self.output_eps)
        detail_y = pred_y - _local_mean(pred_y, radius=self.pred_sharp_radius)
        target_y = torch.clamp(
            pred_y + self.pred_sharp_scale * detail_y,
            self.output_eps,
            1.0 - self.output_eps,
        )
        ratio = target_y / (pred_y + 1e-6)
        ratio = torch.clamp(
            ratio,
            1.0 - self.pred_sharp_ratio_limit,
            1.0 + self.pred_sharp_ratio_limit,
        )
        return self._straight_through_clamp(pred * ratio, pred)

    def _forward_padded(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.inconv(x)

        d1 = self.down1(f0)
        d1 = self.fuse1(d1)
        d1 = self.layer1(d1)

        d2 = self.down2(d1)
        d2 = self.fuse2(d2)
        d2 = self.layer2(d2)
        d2 = self.layer3(d2)
        d2 = self.asm1(d2)
        d2 = self.layer4(d2)

        u1 = self.up1(d2)
        u1 = self.skip1(u1, d1)
        u1 = self.asm2(u1)
        u1 = self.layer5(u1)

        u0 = self.up2(u1)
        u0 = self.skip0(u0, f0)

        residual = torch.tanh(self.outconv(u0))
        out = x + self.residual_scale * residual
        out = self.detail_refine(x, out, residual)
        out = self._preserve_luma_detail(x, out)
        out = self._sharpen_prediction(out)
        return torch.where(torch.isfinite(out), out, x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        b, ch, _, _ = x.shape
        x_pad, original_hw = _safe_pad_to_multiple(x, multiple=4)
        out = self._forward_padded(x_pad)
        out = out[..., :original_hw[0], :original_hw[1]]
        out = torch.where(torch.isfinite(out), out, x)
        out = torch.clamp(out, self.output_eps, 1.0 - self.output_eps)
        residual = out - x

        color_gain = torch.ones(b, ch, 1, 1, device=x.device, dtype=x.dtype)
        sides = [None, None, None]
        return out, residual, color_gain, sides


if __name__ == "__main__":
    net = DehazeUNet(in_ch=3)
    x = torch.rand(2, 3, 127, 129)
    with torch.no_grad():
        out, res, gain, sides = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "sides", sides)
