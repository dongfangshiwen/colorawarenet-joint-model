#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeHamer-style dehazing module adapted for this project.

Paper idea followed:
- transmission-aware 3D position embedding: horizontal, vertical, haze-density
- Transformer branch for global context
- CNN encoder branch for local details
- feature modulation: transformer-conditioned gamma/beta modulates CNN features
- CNN decoder with multi-scale residual blocks for detail reconstruction

This is not the official DeHamer implementation.  The official paper uses a
three-stage Swin Transformer.  Here we use a compact Transformer encoder on a
patch-reduced grid so it can run inside the existing training script.

Forward signature:
    forward(I) -> (dehazed, residual, color_gain, sides)
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0.0)


def _safe_pad2d(x, pad):
    if pad <= 0:
        return x
    # Reflect padding avoids the bright frame produced by zero padding.  Fall
    # back to replicate for very small feature maps where reflect is invalid.
    mode = 'reflect' if x.shape[-2] > pad and x.shape[-1] > pad else 'replicate'
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


class ReflectConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, bias=True):
        super().__init__()
        self.pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        kaiming_init(self.conv)

    def forward(self, x):
        return self.conv(_safe_pad2d(x, self.pad))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, norm=False):
        super().__init__()
        self.conv1 = ReflectConv2d(in_ch, out_ch, kernel_size=k, stride=stride, bias=True)
        self.conv2 = ReflectConv2d(out_ch, out_ch, kernel_size=3, stride=1, bias=True)
        self.norm1 = nn.GroupNorm(max(1, min(8, out_ch // 4)), out_ch) if norm else None
        self.norm2 = nn.GroupNorm(max(1, min(8, out_ch // 4)), out_ch) if norm else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        return self.act(x)


class PyramidPooling(nn.Module):
    """Lightweight PPM used by the CNN encoder branch."""
    def __init__(self, ch, bins=(1, 2, 4), strength=0.10):
        super().__init__()
        self.strength = float(strength)
        mid = max(1, ch // 4)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(ch, mid, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            )
            for b in bins
        ])
        self.fuse = nn.Conv2d(ch + mid * len(bins), ch, kernel_size=1, bias=True)
        for m in self.modules():
            kaiming_init(m)
        nn.init.constant_(self.fuse.weight, 0.0)
        if self.fuse.bias is not None:
            nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            feats.append(y)
        return x + self.strength * self.fuse(torch.cat(feats, dim=1))


class MultiScaleResidualBlock(nn.Module):
    """Parallel receptive-field residual block for detail-preserving decoding."""
    def __init__(self, ch, res_scale=0.20):
        super().__init__()
        self.res_scale = float(res_scale)
        self.b1 = ReflectConv2d(ch, ch, kernel_size=3, dilation=1, bias=True)
        self.b2 = ReflectConv2d(ch, ch, kernel_size=3, dilation=2, bias=True)
        self.b3 = ReflectConv2d(ch, ch, kernel_size=3, dilation=3, bias=True)
        self.fuse = nn.Conv2d(ch * 3, ch, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        for m in self.modules():
            kaiming_init(m)
        nn.init.constant_(self.fuse.weight, 0.0)
        nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x):
        y1 = self.act(self.b1(x))
        y2 = self.act(self.b2(x))
        y3 = self.act(self.b3(x))
        y = self.fuse(torch.cat([y1, y2, y3], dim=1))
        return self.act(x + self.res_scale * y)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = ReflectConv2d(in_ch, out_ch, kernel_size=3, bias=True)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='nearest')
        return F.relu(self.proj(x), inplace=True)


def dark_channel_prior(x, kernel_size=15):
    """DCP(I), used as haze-density coordinate for 3D position embedding."""
    pad = kernel_size // 2
    min_rgb = x.min(dim=1, keepdim=True)[0]
    # Min filter via negative max-pooling.  Use explicit reflect padding;
    # max_pool2d's zero padding creates an artificial low dark-channel frame.
    min_rgb = _safe_pad2d(min_rgb, pad)
    return -F.max_pool2d(-min_rgb, kernel_size=kernel_size, stride=1, padding=0)


class Sinusoidal3DPositionEmbedding(nn.Module):
    """
    3D sinusoidal position embedding over x, y and haze density d.
    The paper uses dimensions for horizontal, vertical and density positions.
    This implementation supports any trans_dim by splitting channels across
    the three coordinates.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    @staticmethod
    def _encode(pos, dim):
        if dim <= 0:
            return pos.new_zeros(pos.shape[0], 0, pos.shape[2], pos.shape[3])
        half = max(1, dim // 2)
        freq = torch.arange(half, device=pos.device, dtype=pos.dtype)
        freq = torch.exp(-math.log(10000.0) * freq / max(1, half - 1))
        values = pos * freq.view(1, half, 1, 1)
        enc = torch.cat([torch.sin(values), torch.cos(values)], dim=1)
        if enc.shape[1] < dim:
            enc = torch.cat([enc, torch.sin(values[:, :1])], dim=1)
        return enc[:, :dim]

    def forward(self, density):
        b, _, h, w = density.shape
        dtype = density.dtype
        device = density.device
        xs = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w)
        ys = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1)
        x_pos = xs.expand(b, 1, h, w) * 31.0
        y_pos = ys.expand(b, 1, h, w) * 31.0
        d_pos = density.clamp(0.0, 1.0) * 31.0

        d1 = self.dim // 3
        d2 = self.dim // 3
        d3 = self.dim - d1 - d2
        return torch.cat([
            self._encode(x_pos, d1),
            self._encode(y_pos, d2),
            self._encode(d_pos, d3),
        ], dim=1)


class FeatureModulation(nn.Module):
    """
    Transformer-conditioned feature modulation.

    DeHamer's paper addresses Transformer/CNN feature inconsistency by learning
    modulation matrices conditioned on Transformer features.  In this compact
    implementation we keep only coefficient modulation by default.  The bias
    term shifts brightness directly and was the main cause of white contours
    around high-contrast borders.

    Fm = (1 + s * tanh(gamma(Ft))) * Fc
    """
    def __init__(self, trans_dim, target_ch, strength=0.10, bias_strength=0.0):
        super().__init__()
        self.strength = float(strength)
        self.bias_strength = float(bias_strength)
        hidden = max(target_ch, trans_dim // 2)
        self.gamma = nn.Sequential(
            ReflectConv2d(trans_dim, hidden, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(hidden, target_ch, kernel_size=3),
        )
        self.beta = None
        if self.bias_strength > 0:
            self.beta = nn.Sequential(
                ReflectConv2d(trans_dim, hidden, kernel_size=3),
                nn.ReLU(inplace=True),
                ReflectConv2d(hidden, target_ch, kernel_size=3),
            )
        for m in self.modules():
            kaiming_init(m)
        nn.init.constant_(self.gamma[-1].conv.weight, 0.0)
        nn.init.constant_(self.gamma[-1].conv.bias, 0.0)
        if self.beta is not None:
            nn.init.constant_(self.beta[-1].conv.weight, 0.0)
            nn.init.constant_(self.beta[-1].conv.bias, 0.0)

    def forward(self, tfeat, cfeat):
        if tfeat.shape[-2:] != cfeat.shape[-2:]:
            tfeat = F.interpolate(tfeat, size=cfeat.shape[-2:], mode='nearest')
        coeff = 1.0 + self.strength * torch.tanh(self.gamma(tfeat))
        bias = 0.0
        if self.beta is not None:
            bias = self.bias_strength * torch.tanh(self.beta(tfeat))
        out = coeff * cfeat + bias
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def border_residual_weight(x: torch.Tensor, border: int = 12, min_weight: float = 0.25) -> torch.Tensor:
    """
    A deterministic confidence prior for restoration boundaries.

    Compact encoder-decoder dehazers often over-correct the outermost pixels
    because they see less context there.  This attenuates only the residual near
    the image frame; the input image itself is still passed through unchanged.
    """
    if border <= 0:
        return torch.ones(x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    h, w = x.shape[-2:]
    yy = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
    xx = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
    dist_y = torch.minimum(yy, (h - 1) - yy)
    dist_x = torch.minimum(xx, (w - 1) - xx)
    dist = torch.minimum(dist_y, dist_x)
    ramp = torch.clamp(dist / float(border), 0.0, 1.0)
    return min_weight + (1.0 - min_weight) * ramp


class DehamerNet(nn.Module):
    def __init__(self,
                 in_ch: int = 3,
                 base_ch: int = 32,
                 trans_dim: int = 64,
                 nheads: int = 4,
                 n_layers: int = 2,
                 patch_size: int = 8,
                 dropout: float = 0.0,
                 use_learnable_pos: bool = True,
                 gain_scale: float = 0.25,
                 residual_scale: float = 0.35):
        super().__init__()
        assert patch_size >= 1 and isinstance(patch_size, int)
        if trans_dim % nheads != 0:
            raise ValueError("trans_dim must be divisible by nheads")

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.trans_dim = trans_dim
        self.patch_size = patch_size
        self.gain_scale = float(gain_scale)
        self.residual_scale = float(residual_scale)
        self.use_learnable_pos = bool(use_learnable_pos)

        # CNN encoder: three stages, preserving local details.
        self.enc1 = ConvBlock(in_ch, base_ch)
        # Keep the highest-resolution feature purely local.  Applying PPM at
        # full resolution introduces coarse overlays that look like separated
        # image layers around strong borders.
        self.ppm1 = nn.Identity()
        self.down1 = ReflectConv2d(base_ch, base_ch * 2, kernel_size=3, stride=2)
        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.ppm2 = PyramidPooling(base_ch * 2, strength=0.08)
        self.down2 = ReflectConv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2)
        self.enc3 = ConvBlock(base_ch * 4, base_ch * 4)
        self.ppm3 = PyramidPooling(base_ch * 4, strength=0.08)

        # Transformer branch on a patch-reduced grid for memory control.
        self.patch_embed = nn.Conv2d(in_ch, trans_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        kaiming_init(self.patch_embed)
        self.pos3d = Sinusoidal3DPositionEmbedding(trans_dim)
        if self.use_learnable_pos:
            self.pos_refine = nn.Conv2d(trans_dim, trans_dim, kernel_size=1)
            kaiming_init(self.pos_refine)
        else:
            self.pos_refine = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=nheads,
            dim_feedforward=trans_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Transformer-conditioned modulation at three CNN stages.
        # Keep the highest-resolution CNN feature unmodulated.  The Transformer
        # tokens are coarse in this compact implementation; forcing them onto
        # f1 creates layer-inconsistent smoothing.  Middle/deep stages still
        # follow DeHamer's coefficient/bias feature modulation.
        self.mod2 = FeatureModulation(trans_dim, base_ch * 2, strength=0.025, bias_strength=0.0)
        self.mod3 = FeatureModulation(trans_dim, base_ch * 4, strength=0.050, bias_strength=0.0)

        # CNN decoder.  Transformer features are used only as conditions; the
        # decoder consumes modulated CNN features to retain texture details.
        self.mrb3 = MultiScaleResidualBlock(base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec2 = ConvBlock(base_ch * 6, base_ch * 2)
        self.mrb2 = MultiScaleResidualBlock(base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)
        self.dec1 = ConvBlock(base_ch * 3, base_ch)
        self.mrb1 = MultiScaleResidualBlock(base_ch)
        self.output_refine = nn.Sequential(
            ConvBlock(base_ch + in_ch, base_ch),
            MultiScaleResidualBlock(base_ch),
            MultiScaleResidualBlock(base_ch),
        )
        self.shallow_detail = nn.Sequential(
            ReflectConv2d(in_ch, base_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(base_ch, base_ch, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.out_conv = ReflectConv2d(base_ch, in_ch, kernel_size=3)
        nn.init.constant_(self.out_conv.conv.weight, 0.0)
        if self.out_conv.conv.bias is not None:
            nn.init.constant_(self.out_conv.conv.bias, 0.0)
        self.res_gate = nn.Sequential(
            ReflectConv2d(base_ch + in_ch, base_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflectConv2d(base_ch, in_ch, kernel_size=3),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.res_gate[-2].conv.weight, 0.0)
        if self.res_gate[-2].conv.bias is not None:
            nn.init.constant_(self.res_gate[-2].conv.bias, -1.2)

        self.side1 = nn.Conv2d(base_ch, in_ch, kernel_size=1)
        self.side2 = nn.Conv2d(base_ch * 2, in_ch, kernel_size=1)
        self.side3 = nn.Conv2d(base_ch * 4, in_ch, kernel_size=1)
        kaiming_init(self.side1)
        kaiming_init(self.side2)
        kaiming_init(self.side3)

    def _pad_to_patch(self, x):
        h, w = x.shape[-2:]
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        if ph == 0 and pw == 0:
            return x, h, w
        return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w

    def _transformer_features(self, x):
        x_pad, h0, w0 = self._pad_to_patch(x)
        out_dtype = x.dtype
        # Keep Transformer math in FP32.  Under AMP, multi-head attention and
        # LayerNorm can occasionally overflow after several epochs.
        with torch.cuda.amp.autocast(enabled=False):
            x_pad_f = x_pad.float()
            token_map = self.patch_embed(x_pad_f)
            hp, wp = token_map.shape[-2:]

            density = dark_channel_prior(x_pad_f)
            density = F.interpolate(density, size=(hp, wp), mode='bilinear', align_corners=False)
            pe = self.pos3d(density)
            if self.pos_refine is not None:
                pe = self.pos_refine(pe)
            token_map = token_map + pe

            tokens = token_map.flatten(2).transpose(1, 2)
            tokens = self.transformer(tokens.float())
            tfeat = tokens.transpose(1, 2).view(x.shape[0], self.trans_dim, hp, wp)
            tfeat = torch.nan_to_num(tfeat, nan=0.0, posinf=1.0, neginf=-1.0)
        return tfeat.to(out_dtype)

    def forward(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        # Keep the whole DeHamer branch in FP32 even when the outer training
        # loop uses AMP.  This preserves --amp for the experiment while avoiding
        # NaNs from attention, normalization and position encoding.
        with torch.cuda.amp.autocast(enabled=False):
            I = I.float()
            b, c, h, w = I.shape

            # CNN local branch.
            f1 = self.ppm1(self.enc1(I))
            f2 = self.ppm2(self.enc2(F.relu(self.down1(f1), inplace=True)))
            f3 = self.ppm3(self.enc3(F.relu(self.down2(f2), inplace=True)))

            # Transformer global branch with transmission-aware 3D PE.
            tfeat = self._transformer_features(I)

            # Modulation at corresponding CNN scales.
            m1 = f1
            m2 = self.mod2(tfeat, f2)
            m3 = self.mod3(tfeat, f3)

            # Decoder with CNN features and modulated features.
            d3 = self.mrb3(m3)
            u2 = self.up2(d3, size=f2.shape[-2:])
            d2 = self.dec2(torch.cat([u2, m2, f2], dim=1))
            d2 = self.mrb2(d2)
            u1 = self.up1(d2, size=f1.shape[-2:])
            d1 = self.dec1(torch.cat([u1, m1, f1], dim=1))
            d1 = self.mrb1(d1)

            shallow = self.shallow_detail(I)
            detail_feat = self.output_refine(torch.cat([d1 + 0.20 * shallow, I], dim=1))
            raw_residual = torch.tanh(self.out_conv(detail_feat)) * self.residual_scale
            gate = self.res_gate(torch.cat([detail_feat, I], dim=1))
            border_w = border_residual_weight(I, border=12, min_weight=0.25)
            residual = raw_residual * gate * border_w
            residual = torch.nan_to_num(residual, nan=0.0, posinf=self.residual_scale, neginf=-self.residual_scale)
            # The shared wrapper clamps before computing the loss/inference
            # image. Returning unclamped here prevents border residuals from
            # becoming saturated white strips inside the model itself.
            dehazed = I + residual

            s1 = self.side1(d1)
            s2 = F.interpolate(self.side2(d2), size=(h, w), mode='bilinear', align_corners=False)
            s3 = F.interpolate(self.side3(d3), size=(h, w), mode='bilinear', align_corners=False)
            sides = [s1, s2, s3]

            # Placeholder for the unified training wrapper; DeHamer itself does
            # not define a color-gain mechanism.
            color_gain = torch.ones(b, c, 1, 1, device=I.device, dtype=I.dtype)
            return dehazed, residual, color_gain, sides


if __name__ == "__main__":
    net = DehamerNet(in_ch=3, base_ch=16, trans_dim=64, nheads=4, n_layers=2, patch_size=8, use_learnable_pos=True)
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out, res, gain, sides = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "sides:", [s.shape if isinstance(s, torch.Tensor) else None for s in sides])
