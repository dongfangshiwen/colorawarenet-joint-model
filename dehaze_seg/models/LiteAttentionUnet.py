#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化 Attention U-Net 实现（适合分割任务）
- 轻量化手段：depthwise-separable conv, 可调 width_mult
- 注意力机制：Attention Gate（用于 skip connection） + 可选 SE（通道注意力）
- 输出：像素级 logits（B x num_classes x H x W），支持 aux 可选辅助头
- 兼容性：提供 imagenet_mean / imagenet_std 属性，可直接替换到 Dehaze wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 基础轻量模块
# -------------------------
class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise + pointwise"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1, bias=False):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, padding=pad, dilation=dilation, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """两个 depthwise-separable conv 串联的基础块"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1 = DepthwiseSeparableConv(in_ch, mid_ch)
        self.conv2 = DepthwiseSeparableConv(mid_ch, out_ch)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# -------------------------
# Squeeze-and-Excitation（可选）
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))


# -------------------------
# Attention Gate（来自 Attention U-Net）
# 用于 skip connection 的加权
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: channels of gating signal (decoder)
        F_l: channels of skip connection (encoder)
        F_int: intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: skip (from encoder), g: gating (from decoder)
        if x.shape[-2:] != g.shape[-2:]:
            x = F.interpolate(x, size=g.shape[-2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# -------------------------
# Encoder / Decoder
# -------------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch)
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False, attention=True):
        super().__init__()
        # in_ch: channels from decoder side (after concat), typically dec_ch + skip_ch
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch)
        self.attention = attention
        self.att_gate = None  # will be created in parent where channel sizes known

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if self.att_gate is not None:
                skip = self.att_gate(skip, x)
            # pad if needed
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


# -------------------------
# Lite Attention U-Net
# -------------------------
class LiteAttentionUNet(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.0, base_ch=32, use_se=False, attention=True, aux=False):
        super().__init__()
        self.num_classes = num_classes
        self.aux = aux
        self.attention = attention
        # channel scaling
        def mch(x):
            return max(8, int(x * width_mult))
        b = mch(base_ch)
        # encoder
        self.inc = ConvBlock(3, b)
        self.down1 = Down(b, mch(b*2), use_se=use_se)   # 1/2
        self.down2 = Down(mch(b*2), mch(b*4), use_se=use_se)  # 1/4
        self.down3 = Down(mch(b*4), mch(b*8), use_se=use_se)  # 1/8
        self.down4 = Down(mch(b*8), mch(b*16), use_se=use_se) # 1/16
        # bottleneck
        self.bottleneck = ConvBlock(mch(b*16), mch(b*16))
        # decoder (channels must match for concatenation)
        self.up3 = Up(mch(b*16) + mch(b*8), mch(b*8), use_se=use_se, attention=attention)
        self.up2 = Up(mch(b*8) + mch(b*4), mch(b*4), use_se=use_se, attention=attention)
        self.up1 = Up(mch(b*4) + mch(b*2), mch(b*2), use_se=use_se, attention=attention)
        self.up0 = Up(mch(b*2) + b, mch(b), use_se=use_se, attention=attention)
        # final conv
        self.outc = nn.Conv2d(mch(b), num_classes, kernel_size=1)

        # aux head
        if aux:
            self.aux_head = nn.Sequential(
                nn.Conv2d(mch(b*4), mch(b*2), 3, padding=1, bias=False),
                nn.BatchNorm2d(mch(b*2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(mch(b*2), num_classes, 1)
            )

        # create attention gates if needed (set after channels known)
        if attention:
            # AttentionGate(F_g, F_l, F_int) -- F_g should match gating (decoder) channels
            self.up3.att_gate = AttentionGate(F_g=mch(b*16), F_l=mch(b*8), F_int=max(8, mch(b*8)//4))
            self.up2.att_gate = AttentionGate(F_g=mch(b*8), F_l=mch(b*4), F_int=max(8, mch(b*4)//4))
            self.up1.att_gate = AttentionGate(F_g=mch(b*4), F_l=mch(b*2), F_int=max(8, mch(b*2)//4))
            self.up0.att_gate = AttentionGate(F_g=mch(b*2), F_l=b, F_int=max(8, b//4))

        # imagenet norm attrs for compatibility
        self.imagenet_mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        self.imagenet_std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

    def forward(self, x: torch.Tensor):
        # encoder
        x1 = self.inc(x)       # 1x
        x2 = self.down1(x1)    # 1/2
        x3 = self.down2(x2)    # 1/4
        x4 = self.down3(x3)    # 1/8
        x5 = self.down4(x4)    # 1/16
        # bottleneck
        b = self.bottleneck(x5)
        # decode (note pairing: b->up->x4, then progressively)
        d3 = self.up3(b, x4)
        d2 = self.up2(d3, x3)
        d1 = self.up1(d2, x2)
        d0 = self.up0(d1, x1)
        logits = self.outc(d0)
        # upsample to input spatial if necessary (here conv design preserves spatial by pooling)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if self.aux:
            aux = self.aux_head(d2)
            aux = F.interpolate(aux, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return {'out': logits, 'aux': aux}
        return logits


# -------------------------
# utils
# -------------------------
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
