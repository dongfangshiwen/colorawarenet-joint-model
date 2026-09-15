#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dark Channel Prior dehazer adapted to the unified training/inference wrapper.

This module follows the public DCP pipeline:
1. dark channel: min over RGB channels, then local min filter
2. atmospheric light A: top dark-channel pixels, choose the brightest RGB pixel
3. transmission: t(x) = 1 - omega * dark_channel(I / A)
4. edge-aware transmission refinement with guided filter
5. radiance recovery: J(x) = (I(x) - A) / max(t(x), t0) + A

ClassicalDCP is the parameter-free baseline (registry name 'dcp'). DCPDehaze
is the historical learned/bounded recovery extension; its
parameter names and forward calculation are preserved for old checkpoints.

Forward signature:
    forward(x) -> (out, residual, color_gain, sides)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb2gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _odd_size(size: int) -> int:
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


def _safe_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    h, w = x.shape[-2:]
    mode = "reflect" if h > pad and w > pad else "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


def _dark_channel(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    patch_size = _odd_size(patch_size)
    pad = patch_size // 2
    min_ch = x.min(dim=1, keepdim=True)[0]
    min_ch = _safe_pad(min_ch, pad)
    return -F.max_pool2d(-min_ch, kernel_size=patch_size, stride=1, padding=0)


def _local_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = 2 * int(radius) + 1
    mode_w = "reflect" if x.shape[-1] > radius else "replicate"
    mode_h = "reflect" if x.shape[-2] > radius else "replicate"
    x = F.avg_pool2d(F.pad(x, (radius, radius, 0, 0), mode=mode_w), kernel_size=(1, k), stride=1)
    x = F.avg_pool2d(F.pad(x, (0, 0, radius, radius), mode=mode_h), kernel_size=(k, 1), stride=1)
    return x


def _estimate_atmospheric_light(
    x: torch.Tensor,
    dark: torch.Tensor,
    top_percent: float = 0.001,
    bright_percent: float = 0.10,
) -> torch.Tensor:
    b, c, h, w = x.shape
    flat_size = h * w
    k = max(1, int(flat_size * float(top_percent)))
    a_list = []

    with torch.no_grad():
        for bi in range(b):
            dark_flat = dark[bi, 0].reshape(-1)
            _, top_idx = torch.topk(dark_flat, min(k, dark_flat.numel()), largest=True)
            x_flat = x[bi].reshape(c, -1)
            cand = x_flat[:, top_idx]
            intensity = cand.mean(dim=0)
            bright_k = max(1, min(cand.shape[1], int(round(cand.shape[1] * float(bright_percent)))))
            _, bright_idx = torch.topk(intensity, bright_k, largest=True)
            a_list.append(cand[:, bright_idx].mean(dim=1).view(c, 1, 1))

    return torch.stack(a_list, dim=0).clamp(1e-3, 1.0)


def _guided_filter(guidance: torch.Tensor, src: torch.Tensor, radius: int = 15, eps: float = 1e-3) -> torch.Tensor:
    """
    Guided filter in the form from He et al.'s guided filtering paper.
    Reflect padding avoids the border darkening caused by zero-padded avg_pool.
    """
    if radius <= 0:
        return src

    mean_i = _local_mean(guidance, radius)
    mean_p = _local_mean(src, radius)
    corr_i = _local_mean(guidance * guidance, radius)
    corr_ip = _local_mean(guidance * src, radius)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i

    mean_a = _local_mean(a, radius)
    mean_b = _local_mean(b, radius)
    return mean_a * guidance + mean_b


class ClassicalDCP(nn.Module):
    """Dark-channel recovery with optional guided transmission refinement.

    Atmospheric light is the brightest RGB pixel among the top dark-channel
    candidates. No learned correction, logit bounding, blending or sharpening.
    """
    def __init__(self, in_ch=3, patch_size=15, omega=.95, t0=.1,
                 top_percent=.001, guided=True, guided_radius=7, guided_eps=1e-4):
        super().__init__()
        if in_ch != 3 or not 0 < t0 <= 1 or not 0 < top_percent <= 1 or not 0 <= omega <= 1:
            raise ValueError("DCP requires RGB, 0 < t0/top_percent <= 1 and 0 <= omega <= 1")
        self.patch_size, self.omega, self.t0 = patch_size, omega, t0
        self.top_percent, self.guided = top_percent, guided
        self.guided_radius, self.guided_eps = guided_radius, guided_eps

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("DCP expects BCHW RGB input")
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            source = x.float().clamp(0, 1)
            dark = _dark_channel(source, self.patch_size)
            # bright_percent=0 chooses exactly one brightest candidate.
            atmosphere = _estimate_atmospheric_light(source, dark, self.top_percent, 0.)
            transmission = 1 - self.omega * _dark_channel(source / atmosphere, self.patch_size)
            if self.guided:
                transmission = _guided_filter(_rgb2gray(source), transmission,
                                              self.guided_radius, self.guided_eps)
            transmission = transmission.clamp(0, 1)
            restored = ((source - atmosphere) / transmission.clamp_min(self.t0) + atmosphere).clamp(0, 1)
        gain = source.new_ones((source.shape[0], 3, 1, 1))
        return restored.to(x.dtype), (restored-source).to(x.dtype), gain.to(x.dtype), [
            transmission.expand_as(source), dark.expand_as(source), atmosphere.expand_as(source)]


class ResidualRefineNet(nn.Module):
    """
    Full-resolution trainable detail refinement.

    It receives the hazy image, DCP recovery, local high-frequency detail, the
    transmission map, and the dark channel.  The output is a logit residual, not
    an RGB image, so the final prediction remains bounded without an external
    hard clamp.
    """
    def __init__(self, in_ch=3, hidden=32, residual_scale=0.25, highpass_radius: int = 2):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.highpass_radius = int(highpass_radius)
        self.conv1 = nn.Conv2d(in_ch * 3 + 2, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=1)
        self.conv3 = nn.Conv2d(hidden, in_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.conv3.weight, 0.0)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0.0)

    def forward(
        self,
        hazy: torch.Tensor,
        dcp_out: torch.Tensor,
        t: torch.Tensor,
        dark: torch.Tensor,
    ) -> torch.Tensor:
        detail = hazy - _local_mean(hazy, radius=1)
        x = torch.cat([hazy, dcp_out, detail, t, dark], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv_mid(x))
        raw = self.conv3(x)
        # Prevent the trainable branch from becoming a low-frequency smoothing
        # filter.  It can only add edge/detail logits.
        raw = raw - _local_mean(raw, radius=self.highpass_radius)
        edge_gate = torch.sigmoid(8.0 * (detail.abs().mean(dim=1, keepdim=True) - 0.015))
        return torch.tanh(raw) * edge_gate * self.residual_scale


class DCPDehaze(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        patch_size: int = 15,
        omega: float = 0.95,
        t0: float = 0.1,
        top_percent: float = 0.001,
        guided: bool = True,
        guided_radius: int = 7,
        guided_eps: float = 1e-4,
        use_learned_refine: bool = False,
        refine_scale: float = 0.25,
        force_trainable_refine: bool = True,
        atmospheric_bright_percent: float = 0.10,
        recovery_t_floor: float = 0.16,
        max_logit_residual: float = 1.50,
        residual_rgb_scale: float = 0.28,
        detail_preserve_scale: float = 0.10,
        output_blend: float = 0.85,
        input_detail_blend: float = 0.08,
        final_sharp_scale: float = 0.10,
        final_sharp_radius: int = 1,
        output_eps: float = 1e-4,
    ):
        super().__init__()
        assert in_ch == 3, "DCP expects 3-channel RGB input"
        self.patch_size = _odd_size(patch_size)
        self.omega = float(omega)
        self.t0 = float(t0)
        self.recovery_t_floor = max(float(recovery_t_floor), self.t0)
        self.top_percent = float(top_percent)
        self.atmospheric_bright_percent = float(atmospheric_bright_percent)
        self.guided = bool(guided)
        self.guided_radius = max(int(guided_radius), 1)
        self.guided_eps = float(guided_eps)
        self.requested_use_learned_refine = bool(use_learned_refine)
        self.force_trainable_refine = bool(force_trainable_refine)
        self.use_learned_refine = self.requested_use_learned_refine or self.force_trainable_refine
        self.max_logit_residual = float(max_logit_residual)
        self.residual_rgb_scale = float(residual_rgb_scale)
        self.detail_preserve_scale = float(detail_preserve_scale)
        self.output_blend = float(output_blend)
        self.input_detail_blend = float(input_detail_blend)
        self.final_sharp_scale = float(final_sharp_scale)
        self.final_sharp_radius = int(final_sharp_radius)
        self.output_eps = float(output_eps)

        self.refine_net = (
            ResidualRefineNet(in_ch=in_ch, hidden=32, residual_scale=refine_scale)
            if self.use_learned_refine else None
        )
        print(
            f"[DCPDehaze] patch={self.patch_size} omega={self.omega:.2f} "
            f"guided={self.guided} radius={self.guided_radius} refine={self.use_learned_refine} "
            f"detail={self.detail_preserve_scale:.2f} sharp={self.final_sharp_scale:.2f}"
        )

    def _bounded_recovery(self, x: torch.Tensor, j_raw: torch.Tensor) -> torch.Tensor:
        delta = j_raw - x
        residual_logits = self.max_logit_residual * delta / (delta.abs() + self.residual_rgb_scale)
        x_safe = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        recovered = torch.sigmoid(torch.logit(x_safe) + residual_logits)
        out = x_safe + self.output_blend * (recovered - x_safe)
        return torch.clamp(out, self.output_eps, 1.0 - self.output_eps)

    def _detail_logit_boost(self, x: torch.Tensor) -> torch.Tensor:
        detail = x - _local_mean(x, radius=1)
        return self.detail_preserve_scale * detail / (detail.abs() + 0.20)

    def _final_sharpen(self, hazy: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.final_sharp_scale <= 0 and self.input_detail_blend <= 0:
            return pred
        pred_safe = pred.clamp(self.output_eps, 1.0 - self.output_eps)
        pred_detail = pred_safe - _local_mean(pred_safe, radius=self.final_sharp_radius)
        input_detail = hazy - _local_mean(hazy, radius=1)
        detail_logits = (
            self.final_sharp_scale * pred_detail / (pred_detail.abs() + 0.20)
            + self.input_detail_blend * input_detail / (input_detail.abs() + 0.20)
        )
        out = torch.sigmoid(torch.logit(pred_safe) + detail_logits)
        return torch.clamp(out, self.output_eps, 1.0 - self.output_eps)

    def _core_dcp(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dark = _dark_channel(x, self.patch_size)
        a = _estimate_atmospheric_light(
            x,
            dark,
            top_percent=self.top_percent,
            bright_percent=self.atmospheric_bright_percent,
        )

        i_div_a = x / (a + 1e-6)
        dark_i_div_a = _dark_channel(i_div_a, self.patch_size)
        t_init = (1.0 - self.omega * dark_i_div_a).clamp(0.0, 1.0)

        if self.guided:
            t = _guided_filter(_rgb2gray(x), t_init, radius=self.guided_radius, eps=self.guided_eps)
            t = t.clamp(0.0, 1.0)
        else:
            t = t_init

        t_clip = t.clamp(min=self.recovery_t_floor)
        j_raw = (x - a) / (t_clip + 1e-6) + a
        j = self._bounded_recovery(x, j_raw)
        if self.detail_preserve_scale > 0:
            j = torch.sigmoid(torch.logit(j.clamp(self.output_eps, 1.0 - self.output_eps)) + self._detail_logit_boost(x))
        return j, t, a, dark

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        assert x.ndim == 4 and x.shape[1] == 3
        in_dtype = x.dtype

        # DCP contains divisions/top-k/filtering.  Keep it in FP32 even when the
        # outer training script uses --amp.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float().clamp(0.0, 1.0)
            j, t, a, dark = self._core_dcp(x_f)

            if self.refine_net is not None:
                delta_logits = self.refine_net(x_f, j, t, dark)
                j = torch.sigmoid(torch.logit(j.clamp(self.output_eps, 1.0 - self.output_eps)) + delta_logits)

            j = self._final_sharpen(x_f, j)
            j = torch.nan_to_num(j, nan=0.0, posinf=1.0, neginf=0.0)
            residual = j - x_f
            b, c, _, _ = x_f.shape
            color_gain = torch.ones(b, c, 1, 1, device=x.device, dtype=j.dtype)
            sides = [t.expand(-1, c, -1, -1), dark.expand(-1, c, -1, -1), a.expand(-1, -1, x_f.shape[-2], x_f.shape[-1])]

        return j.to(in_dtype), residual.to(in_dtype), color_gain.to(in_dtype), sides


if __name__ == "__main__":
    net = DCPDehaze(patch_size=15, omega=0.95, t0=0.1, top_percent=0.001, guided=True)
    x = torch.rand(2, 3, 128, 128)
    out, res, gain, sides = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "t", sides[0].shape)
