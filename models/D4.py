# utils/D4.py
# -*- coding: utf-8 -*-
"""
D4-style dehazing backbone.

Reference idea: "Self-Augmented Unpaired Image Dehazing via Density and Depth
Decomposition" (CVPR 2022).  The dehazing branch in the public D4 code estimates
the transmission map t and haze density beta, estimates atmospheric light A from
the hazy image, and restores the clean image by:

    J = (H - A) / t + A
    depth = log(t) / (-beta)

This project version keeps that core algorithm and adapts the return signature:
    forward(x) -> (J, residual, color_gain, sides)

The previous versions predicted transmission through a down/up decoder and then
mixed hazy-image detail back into the output.  That combination can create a
soft dehazed layer plus a slightly shifted hazy edge layer.  This version uses a
full-resolution transmission path by default and keeps only a low-frequency
trainable color correction after the physics recovery.

For paired training with the repository wrapper, the raw physics result is
converted to a bounded logit-space residual before it is returned.  This avoids
the common D4 failure mode where low t or over-large A sends J outside [0, 1],
the wrapper clamps it, and gradients disappear.

The final output is sharpened from its own luminance detail and receives a
gated luminance-detail transfer from the hazy input.  The transfer is applied as
a luminance ratio, not direct RGB addition, and is strongest only where the
prediction already has a same-polarity local edge.  This keeps geometry sharp
without recreating the separated hazy-color edge layer.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_VERSION = "d4_density_depth_texture_guard_v11"


def _init_conv(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _safe_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


def _local_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = radius * 2 + 1
    return F.avg_pool2d(_safe_pad2d(x, radius), k, stride=1, padding=0)


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _guided_filter(guidance: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
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


class ReflectConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, bias: bool = True):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
        _init_conv(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_safe_pad2d(x, self.pad))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            ReflectConv2d(ch, ch, 3),
            nn.ReLU(inplace=True),
            ReflectConv2d(ch, ch, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.body(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            ReflectConv2d(in_ch, out_ch, 3, stride=2),
            nn.ReLU(inplace=True),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpFuse(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ReflectConv2d(in_ch + skip_ch, out_ch, 3),
            nn.ReLU(inplace=True),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class DetailRefineHead(nn.Module):
    """
    Full-resolution residual head for paired training.

    D4's density-depth physics branch remains the main prediction.  This head
    learns a small bounded logit correction from the hazy image, physics output,
    local detail, transmission, and depth, so validation output can move instead
    of staying at the fixed analytic solution.  The correction is full
    resolution; it does not pass through an upsampling decoder, so learned edge
    corrections stay spatially aligned with the output.
    """
    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 48,
        scale: float = 0.55,
        high_scale: float = 0.80,
        low_scale: float = 0.08,
    ):
        super().__init__()
        self.scale = float(scale)
        self.high_scale = float(high_scale)
        self.low_scale = float(low_scale)
        feat_ch = in_ch * 3 + 2
        self.body = nn.Sequential(
            ReflectConv2d(feat_ch, hidden, 3),
            nn.ReLU(inplace=True),
            ResBlock(hidden),
            ReflectConv2d(hidden, hidden, 3),
            nn.ReLU(inplace=True),
            ReflectConv2d(hidden, in_ch, 3),
        )
        self.color = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, 1),
        )
        last = self.body[-1].conv
        nn.init.constant_(last.weight, 0.0)
        if last.bias is not None:
            nn.init.constant_(last.bias, 0.0)
        color_last = self.color[-1]
        nn.init.constant_(color_last.weight, 0.0)
        if color_last.bias is not None:
            nn.init.constant_(color_last.bias, 0.0)

    def forward(
        self,
        hazy: torch.Tensor,
        physics: torch.Tensor,
        detail: torch.Tensor,
        t: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hazy, physics, detail, t, depth], dim=1)
        raw = self.body(x)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=4.0, neginf=-4.0)
        high = raw - F.avg_pool2d(_safe_pad2d(raw, 1), 3, stride=1)
        low = F.avg_pool2d(_safe_pad2d(raw, 3), 7, stride=1)
        color = self.color(x)
        # D4's physics branch gives the structural prior.  Bias this trainable
        # branch toward aligned high-frequency correction; excessive low-pass
        # correction was the main source of soft D4 outputs in paired training.
        return self.scale * torch.tanh(self.high_scale * high + self.low_scale * low + color)


class D4DehazeNet(nn.Module):
    """
    D4 dehazing branch approximation with density-depth decomposition.

    `base_ch=32` is kept from the training script.  The network predicts a
    high-resolution transmission map with an identity-like initialization, so
    early training does not destroy texture details.
    """
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        t_min: float = 0.05,
        t_max: float = 0.95,
        beta_min: float = 0.04,
        beta_max: float = 0.20,
        use_dark_channel_A: bool = True,
        detail_preserve_scale: float = 0.0,
        recovery_t_floor: float = 0.18,
        atmospheric_top_percent: float = 0.001,
        max_logit_residual: float = 2.20,
        residual_rgb_scale: float = 0.24,
        refine_scale: float = 0.60,
        output_blend: float = 1.00,
        final_output_blend: float = 1.00,
        max_rgb_delta: float = 0.80,
        bad_output_mean: float = 0.02,
        guided_t_refine: bool = False,
        guided_t_radius: int = 2,
        guided_t_eps: float = 1e-3,
        use_decoder_t: bool = False,
        final_detail_preserve_scale: float = 0.30,
        final_detail_radius: int = 2,
        final_detail_gate_threshold: float = 0.006,
        final_detail_gate_slope: float = 60.0,
        final_detail_ratio_limit: float = 0.25,
        final_pred_sharp_scale: float = 0.45,
        final_pred_sharp_radius: int = 1,
        final_pred_sharp_ratio_limit: float = 0.35,
        output_eps: float = 1e-4,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.base_ch = int(base_ch)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.recovery_t_floor = max(float(recovery_t_floor), self.t_min)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.use_dark_channel_A = bool(use_dark_channel_A)
        self.detail_preserve_scale = float(detail_preserve_scale)
        self.atmospheric_top_percent = float(atmospheric_top_percent)
        self.max_logit_residual = float(max_logit_residual)
        self.residual_rgb_scale = float(residual_rgb_scale)
        self.output_blend = float(output_blend)
        self.final_output_blend = float(final_output_blend)
        self.max_rgb_delta = float(max_rgb_delta)
        self.bad_output_mean = float(bad_output_mean)
        self.guided_t_refine = bool(guided_t_refine)
        self.guided_t_radius = max(0, int(guided_t_radius))
        self.guided_t_eps = float(guided_t_eps)
        self.use_decoder_t = bool(use_decoder_t)
        self.final_detail_preserve_scale = float(max(0.0, final_detail_preserve_scale))
        self.final_detail_radius = max(1, int(final_detail_radius))
        self.final_detail_gate_threshold = float(max(final_detail_gate_threshold, 0.0))
        self.final_detail_gate_slope = float(max(final_detail_gate_slope, 1.0))
        self.final_detail_ratio_limit = float(max(0.0, min(final_detail_ratio_limit, 0.5)))
        self.final_pred_sharp_scale = float(max(0.0, final_pred_sharp_scale))
        self.final_pred_sharp_radius = max(1, int(final_pred_sharp_radius))
        self.final_pred_sharp_ratio_limit = float(max(0.0, min(final_pred_sharp_ratio_limit, 0.5)))
        self.output_eps = float(output_eps)
        self.eps = 1e-6
        print(
            f"[D4DehazeNet] {ARCH_VERSION} base_ch={self.base_ch} "
            f"t_min={self.t_min:.3f} t_max={self.t_max:.3f} "
            f"recovery_t_floor={self.recovery_t_floor:.3f} refine_scale={refine_scale:.2f} "
            f"output_blend={self.output_blend:.2f} final_blend={self.final_output_blend:.2f} "
            f"decoder_t={self.use_decoder_t} guided_t={self.guided_t_refine} r={self.guided_t_radius} "
            f"luma_detail={self.final_detail_preserve_scale:.2f} "
            f"pred_sharp={self.final_pred_sharp_scale:.2f}"
        )

        c = self.base_ch
        self.stem = nn.Sequential(
            ReflectConv2d(self.in_ch, c, 3),
            nn.ReLU(inplace=True),
            ResBlock(c),
        )
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 4)

        if self.use_decoder_t:
            self.up2 = UpFuse(c * 4, c * 4, c * 4)
            self.up1 = UpFuse(c * 4, c * 2, c * 2)
            self.up0 = UpFuse(c * 2, c, c)
        else:
            self.up2 = None
            self.up1 = None
            self.up0 = None

        self.t_head = nn.Sequential(
            ReflectConv2d(c, c, 3),
            nn.ReLU(inplace=True),
            ReflectConv2d(c, 1, 3),
        )

        self.beta_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 4, c, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
        )
        self.refine_head = DetailRefineHead(
            in_ch=self.in_ch,
            hidden=max(48, c),
            scale=refine_scale,
            high_scale=0.80,
            low_scale=0.08,
        )

        self._init_output_heads()

    def _init_output_heads(self) -> None:
        # Transmission starts near 0.9, close to identity restoration but not
        # saturated at t_max, so gradients still reach the whole estimator.  A
        # tiny non-zero spatial kernel avoids the global-bias-only first epoch
        # that can otherwise drive t into a clamp dead zone.
        last_t = self.t_head[-1].conv
        nn.init.normal_(last_t.weight, mean=0.0, std=1e-3)
        if last_t.bias is not None:
            # Start with meaningful haze removal instead of an almost-identity
            # t map.  The bounded output path keeps this from over-shooting.
            nn.init.constant_(last_t.bias, 1.20)

        last_beta = self.beta_head[-1]
        nn.init.constant_(last_beta.weight, 0.0)
        if last_beta.bias is not None:
            nn.init.constant_(last_beta.bias, 0.0)

    def _estimate_atmospheric_light(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dark_channel_A:
            return x.flatten(2).amax(dim=2).view(x.shape[0], x.shape[1], 1, 1)

        # Robust atmospheric light estimate from the brightest dark-channel
        # pixels.  A single channel-wise maximum is too sensitive to indoor
        # lamps/windows and often makes the physical recovery collapse to dark,
        # low-saturation output after the training wrapper clamps J.
        b, c, h, w = x.shape
        with torch.no_grad():
            dark = x.min(dim=1).values.flatten(1)
            k = max(1, int(self.atmospheric_top_percent * h * w))
            idx = torch.topk(dark, k=k, dim=1, largest=True, sorted=False).indices
            flat = x.flatten(2).transpose(1, 2)
            gathered = torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, c))
            gray = gathered.mean(dim=2)
            bright_k = max(1, int(0.25 * k))
            bright_idx = torch.topk(gray, k=bright_k, dim=1, largest=True, sorted=False).indices
            bright = torch.gather(gathered, 1, bright_idx.unsqueeze(-1).expand(-1, -1, c))
            return bright.mean(dim=1).view(b, c, 1, 1)

    def _local_high_frequency(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        smooth = F.avg_pool2d(_safe_pad2d(x, kernel_size // 2), kernel_size, stride=1, padding=0)
        return x - smooth

    def _local_luma_high_frequency(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        y = _rgb_to_luma(x)
        smooth = F.avg_pool2d(_safe_pad2d(y, kernel_size // 2), kernel_size, stride=1, padding=0)
        return y - smooth

    def _bounded_physics_output(self, x: torch.Tensor, j_physics: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        j_physics = torch.where(torch.isfinite(j_physics), j_physics, x_safe)
        delta_rgb = torch.clamp(j_physics - x_safe, -self.max_rgb_delta, self.max_rgb_delta)
        residual_logits = self.max_logit_residual * delta_rgb / (
            delta_rgb.abs() + self.residual_rgb_scale
        )
        physics = torch.sigmoid(torch.logit(x_safe) + residual_logits)
        physics = torch.where(torch.isfinite(physics), physics, x_safe)
        out = x_safe + self.output_blend * (physics - x_safe)
        return self._straight_through_clamp(out, x_safe)

    def _straight_through_clamp(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, ref)
        clipped = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        return x + (clipped - x).detach()

    def _preserve_input_detail(self, hazy: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.final_detail_preserve_scale <= 0.0:
            return pred
        hazy_y = _rgb_to_luma(hazy)
        pred_y = _rgb_to_luma(pred).clamp(self.output_eps, 1.0 - self.output_eps)
        hazy_detail = hazy_y - _local_mean(hazy_y, self.final_detail_radius)
        pred_detail = pred_y - _local_mean(pred_y, self.final_detail_radius)
        edge_strength = hazy_detail.abs()
        edge_gate = torch.sigmoid(
            self.final_detail_gate_slope * (edge_strength - self.final_detail_gate_threshold)
        )
        # Transfer hazy luminance texture only where the dehazed prediction has
        # a compatible local edge.  This keeps fine geometry while avoiding the
        # visible second layer that direct hazy-detail copying can create.
        align_score = hazy_detail * pred_detail / (
            hazy_detail.abs() * pred_detail.abs() + self.eps
        )
        agreement = torch.sigmoid(3.0 * align_score)
        detail_y = hazy_detail * edge_gate * (0.35 + 0.65 * agreement)
        target_y = torch.clamp(
            pred_y + self.final_detail_preserve_scale * detail_y,
            self.output_eps,
            1.0 - self.output_eps,
        )
        ratio = target_y / (pred_y + self.eps)
        ratio = torch.clamp(
            ratio,
            1.0 - self.final_detail_ratio_limit,
            1.0 + self.final_detail_ratio_limit,
        )
        out = pred * ratio
        return self._straight_through_clamp(out, pred)

    def _sharpen_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        if self.final_pred_sharp_scale <= 0.0:
            return pred
        pred_y = _rgb_to_luma(pred).clamp(self.output_eps, 1.0 - self.output_eps)
        detail_y = pred_y - _local_mean(pred_y, self.final_pred_sharp_radius)
        target_y = torch.clamp(
            pred_y + self.final_pred_sharp_scale * detail_y,
            self.output_eps,
            1.0 - self.output_eps,
        )
        ratio = target_y / (pred_y + self.eps)
        ratio = torch.clamp(
            ratio,
            1.0 - self.final_pred_sharp_ratio_limit,
            1.0 + self.final_pred_sharp_ratio_limit,
        )
        out = pred * ratio
        return self._straight_through_clamp(out, pred)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        h, w = x.shape[-2:]
        x_safe = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)

        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)

        if self.use_decoder_t:
            d2 = self.up2(f3, f2)
            d1 = self.up1(d2, f1)
            t_feat = self.up0(d1, f0)
        else:
            # Full-resolution transmission avoids decoder upsampling phase
            # offsets, which are visible as halo/double contours after the
            # point-wise atmospheric recovery.
            t_feat = f0

        t_unit = torch.sigmoid(self.t_head(t_feat))
        t_unit = torch.where(torch.isfinite(t_unit), t_unit, t_unit.new_full((), 0.95))
        t = self.t_min + (self.t_max - self.t_min) * t_unit
        if t.shape[-2:] != (h, w):
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        if self.guided_t_refine:
            t = _guided_filter(_rgb_to_luma(x_safe), t, self.guided_t_radius, self.guided_t_eps)
            t = torch.clamp(t, self.t_min, self.t_max)
            t = torch.where(torch.isfinite(t), t, t_unit.new_full((), 0.85))

        beta_unit = torch.sigmoid(self.beta_head(f3))
        beta_unit = torch.where(torch.isfinite(beta_unit), beta_unit, beta_unit.new_full((), 0.5))
        beta = self.beta_min + (self.beta_max - self.beta_min) * beta_unit
        depth = torch.log(t + self.eps) / (-(beta + self.eps))
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        A = self._estimate_atmospheric_light(x)
        t_recover = torch.clamp(t, min=self.recovery_t_floor, max=self.t_max)
        J_physics = (x_safe - A) / (t_recover + self.eps) + A

        # Keep geometry detail as luminance information. Direct RGB detail from
        # the hazy image creates colored layer separation after dehazing.
        detail = self._local_luma_high_frequency(x).expand(-1, self.in_ch, -1, -1)
        if self.detail_preserve_scale > 0:
            J_physics = J_physics + self.detail_preserve_scale * detail

        J = self._bounded_physics_output(x, J_physics)
        refine_logits = self.refine_head(x, J, detail, t, depth)
        refine_logits = torch.nan_to_num(refine_logits, nan=0.0, posinf=2.0, neginf=-2.0)
        refine_logits = torch.clamp(refine_logits, -2.0, 2.0)
        J_refined = torch.sigmoid(torch.logit(J.clamp(self.output_eps, 1.0 - self.output_eps)) + refine_logits)
        J = torch.where(torch.isfinite(J_refined), J_refined, J)
        J = x_safe + self.final_output_blend * (J - x_safe)
        bad = J.flatten(1).mean(dim=1).view(-1, 1, 1, 1) < self.bad_output_mean
        J = torch.where(bad, x_safe, J)
        J = self._preserve_input_detail(x_safe, J)
        J = self._sharpen_prediction(J)
        J = torch.clamp(J, self.output_eps, 1.0 - self.output_eps)
        residual = J - x
        color_gain = J.mean(dim=[2, 3], keepdim=True) / (x.mean(dim=[2, 3], keepdim=True) + 1e-6)
        color_gain = torch.clamp(color_gain, 0.2, 3.0)
        sides = [
            t.expand(-1, self.in_ch, -1, -1),
            depth.expand(-1, self.in_ch, -1, -1),
            A.expand(-1, -1, h, w),
        ]
        return J, residual, color_gain, sides


if __name__ == "__main__":
    net = D4DehazeNet(in_ch=3, base_ch=32)
    x = torch.rand(2, 3, 127, 129)
    y, res, gain, sides = net(x)
    print("out", y.shape, "res", res.shape, "gain", gain.shape, "sides", [s.shape for s in sides])
