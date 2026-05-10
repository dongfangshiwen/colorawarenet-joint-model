#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lightweight_segmentation.py

A small, self-contained lightweight segmentation model suitable for
training together with the dehazer or used standalone for fast inference.

Provides:
 - LightSegNet: depthwise-separable encoder + simple decoder UNet-like
 - DehazeLightSegWrapper: convenience wrapper that mirrors earlier
   DehazeSegModel API (returns dehazed, seg_out, sides)

Design goals:
 - small parameter count
 - easy to read and adapt
 - compatible with inputs in [0,1]

Usage:
  from lightweight_segmentation import LightSegNet, DehazeLightSegWrapper
  seg = LightSegNet(num_classes=4)
  model = DehazeLightSegWrapper(dehazer, seg)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Basic building blocks
# -------------------------
class DWConv(nn.Module):
    """Depthwise separable convolution: DW + PW"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class ConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Encoder (small)
# -------------------------
class LightEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=16, width_mult=1.0):
        super().__init__()
        b1 = int(base_ch * width_mult)
        b2 = int(base_ch * 2 * width_mult)
        b4 = int(base_ch * 4 * width_mult)
        b8 = int(base_ch * 8 * width_mult)

        self.stem = ConvBnAct(in_ch, b1, k=3, s=2, p=1)   # /2
        self.block1 = nn.Sequential(DWConv(b1, b1), DWConv(b1, b2, stride=2))  # /4
        self.block2 = nn.Sequential(DWConv(b2, b2), DWConv(b2, b4, stride=2))  # /8
        self.block3 = nn.Sequential(DWConv(b4, b4), DWConv(b4, b8, stride=2))  # /16

    def forward(self, x):
        s1 = self.stem(x)    # 1/2
        s2 = self.block1(s1) # 1/4
        s3 = self.block2(s2) # 1/8
        s4 = self.block3(s3) # 1/16
        return s1, s2, s3, s4

# -------------------------
# Decoder (light)
# -------------------------
class LightDecoder(nn.Module):
    def __init__(self, base_ch=16, width_mult=1.0, num_classes=3):
        super().__init__()
        b1 = int(base_ch * width_mult)
        b2 = int(base_ch * 2 * width_mult)
        b4 = int(base_ch * 4 * width_mult)
        b8 = int(base_ch * 8 * width_mult)

        # lateral convs to unify channels
        self.lat4 = ConvBnAct(b8, b4, k=1, s=1, p=0)
        self.lat3 = ConvBnAct(b4, b2, k=1, s=1, p=0)
        self.lat2 = ConvBnAct(b2, b1, k=1, s=1, p=0)

        # refine convs (after upsample+concat)
        self.up3 = ConvBnAct(b4 + b4, b4)
        self.up2 = ConvBnAct(b4 + b2, b2)
        self.up1 = ConvBnAct(b2 + b1, b1)

        # final heads
        self.classifier = nn.Sequential(
            ConvBnAct(b1, b1),
            nn.Conv2d(b1, num_classes, kernel_size=1)
        )

    def forward(self, s1, s2, s3, s4):
        # s4 -> up to s3
        l4 = self.lat4(s4)
        up3 = F.interpolate(l4, size=s3.shape[-2:], mode='bilinear', align_corners=False)
        r3 = self.up3(torch.cat([up3, s3], dim=1))

        l3 = self.lat3(r3)
        up2 = F.interpolate(l3, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        r2 = self.up2(torch.cat([up2, s2], dim=1))

        l2 = self.lat2(r2)
        up1 = F.interpolate(l2, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        r1 = self.up1(torch.cat([up1, s1], dim=1))

        out = self.classifier(r1)
        return out

# -------------------------
# Full LightSegNet
# -------------------------
class LightSegNet(nn.Module):
    def __init__(self, num_classes=21, in_ch=3, base_ch=16, width_mult=1.0):
        super().__init__()
        self.encoder = LightEncoder(in_ch=in_ch, base_ch=base_ch, width_mult=width_mult)
        self.decoder = LightDecoder(base_ch=base_ch, width_mult=width_mult, num_classes=num_classes)

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        logits = self.decoder(s1, s2, s3, s4)
        # upsample logits to input spatial size
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

# -------------------------
# Wrapper: match DehazeSegModel API
# -------------------------
class DehazeLightSegWrapper(nn.Module):
    """
    Wrap a dehazer and the LightSegNet so it behaves like the previous
    DehazeSegModel in your project (returns dehazed, seg_out, sides).

    If you prefer to do normalization externally, set imagenet_norm=False.
    """
    def __init__(self, dehazer, seg_net: LightSegNet, imagenet_norm=True):
        super().__init__()
        self.dehazer = dehazer
        self.seg = seg_net
        if imagenet_norm:
            self.register_buffer('imagenet_mean', torch.tensor([0.485,0.456,0.406], dtype=torch.float32).view(1,3,1,1))
            self.register_buffer('imagenet_std',  torch.tensor([0.229,0.224,0.225], dtype=torch.float32).view(1,3,1,1))
        else:
            self.imagenet_mean = None
            self.imagenet_std = None

    def forward(self, x):
        dz_out = self.dehazer(x)
        if isinstance(dz_out, (list, tuple)):
            dehazed = dz_out[0]
            sides = dz_out[3] if len(dz_out) > 3 else [None, None, None]
        else:
            dehazed = dz_out
            sides = [None, None, None]
        dehazed = torch.clamp(dehazed, 0.0, 1.0)
        if self.imagenet_mean is not None:
            seg_in = (dehazed - self.imagenet_mean) / self.imagenet_std
        else:
            seg_in = dehazed
        seg_out = self.seg(seg_in)
        return dehazed, seg_out, sides

# -------------------------
# Quick parameter counter utility
# -------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # quick smoke test
    m = LightSegNet(num_classes=4, base_ch=16, width_mult=1.0)
    x = torch.randn(2,3,256,256)
    logits = m(x)
    print('logits', logits.shape)
    print('params', count_params(m))

