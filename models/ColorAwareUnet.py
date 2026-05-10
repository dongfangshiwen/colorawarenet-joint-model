#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/ColorAwareUnet.py

- 修复你贴的第三段代码中 `gain_only_amplify` 未定义的问题
- 支持 gain_form: 'tanh' / 'amp'（amplify-only）
- 支持 disable_gain（gain_scale=0 或直接关闭）
- 可选 gain_min clamp（避免 tanh 过度压色导致偏灰）
- 输出保持兼容：out, residual, color_gain, sides
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm='inst'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        if norm == 'inst':
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == 'group':
            g = min(16, max(1, out_ch // 2))
            self.norm = nn.GroupNorm(max(1, g), out_ch)
        elif norm is None or norm == 'none':
            self.norm = None
        else:
            self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='inst'):
        super().__init__()
        self.net = nn.Sequential(
            ConvRelu(in_ch, out_ch, norm=norm),
            ConvRelu(out_ch, out_ch, norm=norm)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, norm=None):
        super().__init__()
        self.conv1 = ConvRelu(ch, ch, norm=norm)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        if norm == 'inst':
            self.norm2 = nn.InstanceNorm2d(ch, affine=True)
        elif norm == 'group':
            g = min(16, max(1, ch // 2))
            self.norm2 = nn.GroupNorm(max(1, g), ch)
        elif norm is None or norm == 'none':
            self.norm2 = None
        else:
            self.norm2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.act(out + identity)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=None):
        super().__init__()
        self.net = nn.Sequential(
            ConvRelu(in_ch, out_ch, norm=norm),
            ResidualBlock(out_ch, norm=norm)
        )

    def forward(self, x):
        return self.net(x)


class UpSampleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.proj(x)


class ColorAwareUNet(nn.Module):
    """
    UNet 去雾器
    返回: out, residual, color_gain, side_outputs(list)

    color_gain:
      - gain_form='tanh': 1 + tanh(raw) * gain_scale  (可增可减)
      - gain_form='amp' : 1 + relu(raw) * gain_scale  (只增不减，更保色)
    可选 gain_min: 对 gain 做下界 clamp，避免 tanh 过度压色导致偏灰
    """
    def __init__(
        self,
        in_ch=3,
        base_ch=32,
        residual_scale=0.5,
        gain_scale=0.65,
        gain_form='tanh',      # 'tanh' or 'amp'
        gain_min=None,         # e.g. 0.8 (only for tanh or general clamp)
        disable_gain=False,    # if True, force color_gain=1
        gain_mode='global',    # 'local' or 'global'
        gain_smooth_kernel=15, # smooth local gain so it handles illumination, not texture
        norm='inst',
        output_clamp=True,
        refine_scale=0.25
    ):
        super().__init__()
        assert gain_form in ['tanh', 'amp'], "gain_form must be 'tanh' or 'amp'"
        assert gain_mode in ['local', 'global'], "gain_mode must be 'local' or 'global'"

        self.base_ch = base_ch
        self.residual_scale = float(residual_scale)
        self.gain_scale = float(gain_scale)
        self.gain_form = gain_form
        self.gain_min = gain_min if gain_min is None else float(gain_min)
        self.disable_gain = bool(disable_gain)
        self.gain_mode = gain_mode
        self.gain_smooth_kernel = int(gain_smooth_kernel)
        self.norm = norm
        self.output_clamp = bool(output_clamp)
        self.refine_scale = float(refine_scale)

        # Encoder
        self.enc1 = DownBlock(in_ch, base_ch, norm=norm)
        self.enc2 = DownBlock(base_ch, base_ch * 2, norm=norm)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, norm=norm)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8, norm=norm)

        self.pool = nn.MaxPool2d(2, 2)
        self.center = nn.Sequential(
            ConvRelu(base_ch * 8, base_ch * 8, norm=norm),
            ConvRelu(base_ch * 8, base_ch * 8, norm=norm)
        )

        # Decoder (no norm). Bilinear upsample + conv avoids transpose artifacts.
        self.up4 = UpSampleConv(base_ch * 8, base_ch * 8)
        self.dec4 = UpBlock(base_ch * 8 + base_ch * 8, base_ch * 4, norm=None)

        self.up3 = UpSampleConv(base_ch * 4, base_ch * 4)
        self.dec3 = UpBlock(base_ch * 4 + base_ch * 4, base_ch * 2, norm=None)

        self.up2 = UpSampleConv(base_ch * 2, base_ch * 2)
        self.dec2 = UpBlock(base_ch * 2 + base_ch * 2, base_ch, norm=None)

        self.up1 = UpSampleConv(base_ch, base_ch)
        self.dec1 = nn.Sequential(
            ConvRelu(base_ch + base_ch, base_ch, norm=None),
            ResidualBlock(base_ch, norm=None),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            ConvRelu(base_ch * 2 + 9, base_ch, norm=None),
            ResidualBlock(base_ch, norm=None),
            ResidualBlock(base_ch, norm=None),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        )

        # side outputs
        self.side4 = nn.Conv2d(base_ch * 4, 3, kernel_size=1)
        self.side3 = nn.Conv2d(base_ch * 2, 3, kernel_size=1)
        self.side2 = nn.Conv2d(base_ch, 3, kernel_size=1)

        # gain head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gain_fc = nn.Sequential(
            nn.Conv2d(base_ch * 8, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        self.local_gain_head = nn.Sequential(
            ConvRelu(base_ch + base_ch, base_ch, norm=None),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)

        # zero-init last conv to encourage identity start
        try:
            last = self.dec1[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.constant_(last.weight, 0.0)
                if last.bias is not None:
                    nn.init.constant_(last.bias, 0.0)
            last_refine = self.refine[-1]
            if isinstance(last_refine, nn.Conv2d):
                nn.init.constant_(last_refine.weight, 0.0)
                if last_refine.bias is not None:
                    nn.init.constant_(last_refine.bias, 0.0)
        except Exception:
            pass
        try:
            last_gain = self.gain_fc[-1]
            if isinstance(last_gain, nn.Conv2d):
                nn.init.constant_(last_gain.weight, 0.0)
                if last_gain.bias is not None:
                    nn.init.constant_(last_gain.bias, 0.0)
            last_local_gain = self.local_gain_head[-1]
            if isinstance(last_local_gain, nn.Conv2d):
                nn.init.constant_(last_local_gain.weight, 0.0)
                if last_local_gain.bias is not None:
                    nn.init.constant_(last_local_gain.bias, 0.0)
        except Exception:
            pass

    def _make_gain(self, raw):
        # raw: Bx3x1x1 or Bx3xHxW
        if self.disable_gain or self.gain_scale <= 0:
            gain = torch.ones_like(raw)
            return gain

        if self.gain_form == 'amp':
            # Keep amplify-only gain bounded; unbounded ReLU gain can explode under AMP.
            gain = 1.0 + torch.tanh(F.relu(raw)) * self.gain_scale
        else:
            gain = 1.0 + torch.tanh(raw) * self.gain_scale

        if self.gain_min is not None:
            gain = torch.clamp(gain, min=self.gain_min)

        return gain

    def _smooth_local_gain_raw(self, raw):
        if self.gain_smooth_kernel <= 1:
            return raw
        k = self.gain_smooth_kernel
        if k % 2 == 0:
            k += 1
        pad = k // 2
        raw = F.pad(raw, (pad, pad, pad, pad), mode='replicate')
        return F.avg_pool2d(raw, kernel_size=k, stride=1)

    @staticmethod
    def _align_to(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))

        # Decoder
        d4 = self.up4(c)
        d4 = self._align_to(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._align_to(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align_to(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align_to(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        residual = self.dec1(d1)

        # sides
        side4 = F.interpolate(self.side4(d4), size=x.shape[-2:], mode='bilinear', align_corners=False)
        side3 = F.interpolate(self.side3(d3), size=x.shape[-2:], mode='bilinear', align_corners=False)
        side2 = F.interpolate(self.side2(d2), size=x.shape[-2:], mode='bilinear', align_corners=False)
        sides = [side2, side3, side4]

        # Gain. Local gain is spatially adaptive; global gain is kept for ablation.
        if self.gain_mode == 'local':
            raw = self.local_gain_head(d1)
            raw = self._smooth_local_gain_raw(raw)
        else:
            g = self.global_pool(c)
            raw = self.gain_fc(g)
        color_gain = self._make_gain(raw)

        coarse = x * color_gain + residual * self.residual_scale
        refine = self.refine(torch.cat([x, coarse, residual, d1], dim=1))
        out = coarse + refine * self.refine_scale
        if self.output_clamp:
            out = torch.clamp(out, 0.0, 1.0)
        return out, residual, color_gain, sides
