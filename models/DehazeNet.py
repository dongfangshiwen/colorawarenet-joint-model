#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-aligned DehazeNet backbone.

The original DehazeNet estimates a transmission map and then recovers the
haze-free image with the atmospheric scattering model.  This implementation
keeps that core path:

    centered RGB -> Maxout feature extraction -> 3/5/7 multi-scale mapping
    -> local extremum -> BReLU transmission -> guided filtering
    -> J = (I - A) / t + A

The public training wrapper in this repository expects a unified return
signature:

    forward(x) -> (out, residual, color_gain, sides)

DehazeNet was originally trained with transmission supervision.  This repository
trains through paired RGB reconstruction, so the paper core is wrapped with
stable bounded transmission/output parameterizations.  The default core keeps
the paper operations but uses a wider feature/mapping width, and an optional
zero-initialized full-resolution logit refinement head is used after the
atmospheric-scattering recovery.  The refinement starts as an exact no-op, so
the first prediction is still the paper physics output.  Since a single
transmission map cannot reconstruct all RGB high-frequency texture by itself,
the final image also uses a gated luminance-detail preservation step.  It
modulates luminance instead of adding hazy RGB edges directly, avoiding the
shifted/ghost layer that direct detail copying can create.

The public constructor is intentionally small.  Most stabilization constants
are fixed inside the module because exposing every clamp/gate value as an
argument made the backbone hard to reason about without adding useful capacity.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_VERSION = "dehazenet_trainable_adapter_v16"


def _init_conv(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        # The paper initializes convolution weights with N(0, 0.001).
        nn.init.normal_(m.weight, mean=0.0, std=1e-3)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _init_adapter_conv(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _pad2d(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    if left == right == top == bottom == 0:
        return x
    h, w = x.shape[-2:]
    mode = "reflect" if h > max(top, bottom) and w > max(left, right) else "replicate"
    return F.pad(x, pad, mode=mode)


def _same_pad(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    return _pad2d(x, (radius, radius, radius, radius))


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Separable mean box filter.  It is equivalent to a square average filter but
    is much cheaper for the official DehazeNet guided-filter radius.
    """
    if radius <= 0:
        return x
    k = radius * 2 + 1
    x = _pad2d(x, (radius, radius, 0, 0))
    x = F.avg_pool2d(x, kernel_size=(1, k), stride=1)
    x = _pad2d(x, (0, 0, radius, radius))
    x = F.avg_pool2d(x, kernel_size=(k, 1), stride=1)
    return x


def _local_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    return _box_filter(x, radius)


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _dark_channel(x: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    patch_size = max(1, int(patch_size))
    if patch_size % 2 == 0:
        patch_size += 1
    pad = patch_size // 2
    min_ch = x.min(dim=1, keepdim=True).values
    return -F.max_pool2d(-_same_pad(min_ch, pad), kernel_size=patch_size, stride=1)


class SameConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, bias: bool = True):
        super().__init__()
        total_pad = kernel_size - 1
        left = total_pad // 2
        right = total_pad - left
        self.pad = (left, right, left, right)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=bias)
        _init_conv(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_pad2d(x, self.pad))


class BReLU(nn.Module):
    """Bilateral ReLU used by DehazeNet to constrain transmission to [0, 1]."""
    def __init__(self, lower: float = 0.0, upper: float = 1.0, straight_through: bool = True):
        super().__init__()
        self.lower = float(lower)
        self.upper = float(upper)
        self.straight_through = bool(straight_through)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(x, self.lower, self.upper)
        if not self.straight_through:
            return clipped
        return x + (clipped - x).detach()


class MaxoutConv(nn.Module):
    """
    Official-style Maxout feature extraction.

    The public DehazeNet code first produces 16 feature channels and then groups
    them into 4 Maxout responses.  In paper mode this grouping is fixed even if
    the caller passes a different maxout_k through the shared training script.
    """
    def __init__(self, in_ch: int, out_ch: int = 4, maxout_k: int = 4, kernel_size: int = 5):
        super().__init__()
        self.out_ch = int(out_ch)
        self.maxout_k = int(maxout_k)
        self.conv = SameConv2d(in_ch, self.out_ch * self.maxout_k, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        b, _, h, w = y.shape
        y = y.view(b, self.out_ch, self.maxout_k, h, w)
        return y.max(dim=2).values


class GuidedFilter(nn.Module):
    """
    Differentiable guided filter for refining the transmission map.

    DehazeNet's public MATLAB code uses guided filtering on t before recovery.
    This implementation keeps the same role while using separable box filters so
    the official radius is practical inside a PyTorch training loop.
    """
    def __init__(self, radius: int = 15, eps: float = 1e-3):
        super().__init__()
        self.radius = int(radius)
        self.eps = float(eps)

    def forward(self, guidance: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        if self.radius <= 0:
            return src

        mean_i = _box_filter(guidance, self.radius)
        mean_p = _box_filter(src, self.radius)
        corr_i = _box_filter(guidance * guidance, self.radius)
        corr_ip = _box_filter(guidance * src, self.radius)

        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p

        a = cov_ip / (var_i + self.eps)
        b = mean_p - a * mean_i

        mean_a = _box_filter(a, self.radius)
        mean_b = _box_filter(b, self.radius)
        return mean_a * guidance + mean_b


class OutputRefineHead(nn.Module):
    """
    Full-resolution paired-training adapter after the DehazeNet physics core.

    The DehazeNet paper core predicts transmission and restores radiance with
    the atmospheric scattering model.  That path stays intact.  This head only
    learns a bounded logit correction from full-resolution, spatially aligned
    tensors, so it can recover texture without a down/up decoder or shifted
    copied RGB edges.  The last layers are zero-initialized; at initialization
    the correction is exactly zero.
    """
    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 48,
        scale: float = 0.55,
        high_scale: float = 0.90,
        low_scale: float = 0.05,
    ):
        super().__init__()
        self.scale = float(scale)
        self.high_scale = float(high_scale)
        self.low_scale = float(low_scale)
        feat_ch = in_ch * 4 + 1
        self.body = nn.Sequential(
            SameConv2d(feat_ch, hidden, 3),
            nn.ReLU(inplace=True),
            SameConv2d(hidden, hidden, 3),
            nn.ReLU(inplace=True),
            SameConv2d(hidden, hidden, 3),
            nn.ReLU(inplace=True),
            SameConv2d(hidden, in_ch, 3),
        )
        self.color = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, 1),
        )
        self.apply(_init_adapter_conv)
        final = self.body[-1].conv
        nn.init.constant_(final.weight, 0.0)
        if final.bias is not None:
            nn.init.constant_(final.bias, 0.0)
        color_final = self.color[-1]
        nn.init.constant_(color_final.weight, 0.0)
        if color_final.bias is not None:
            nn.init.constant_(color_final.bias, 0.0)

    def forward(
        self,
        hazy: torch.Tensor,
        physics: torch.Tensor,
        detail: torch.Tensor,
        t: torch.Tensor,
        atmospheric: torch.Tensor,
    ) -> torch.Tensor:
        a_map = atmospheric.expand(-1, -1, hazy.shape[-2], hazy.shape[-1])
        feat = torch.cat([hazy, physics, detail, t, a_map], dim=1)
        raw = self.body(feat)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=4.0, neginf=-4.0)
        high = raw - _local_mean(raw, 1)
        low = _local_mean(raw, 3)
        color = self.color(feat)
        return self.scale * torch.tanh(self.high_scale * high + self.low_scale * low + color)


class AtmosphericRefineHead(nn.Module):
    """
    Global atmospheric-light adapter for paired RGB training.

    The dark-channel atmospheric estimate remains the initial value.  This head
    starts as an exact no-op and learns only a small global correction, which is
    enough to fix many indoor color/brightness errors without replacing the
    DehazeNet transmission core.
    """
    def __init__(self, in_ch: int = 3, hidden: int = 48, scale: float = 0.12):
        super().__init__()
        self.scale = float(scale)
        feat_ch = in_ch + 2
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, 1),
        )
        self.apply(_init_adapter_conv)
        final = self.net[-1]
        nn.init.constant_(final.weight, 0.0)
        if final.bias is not None:
            nn.init.constant_(final.bias, 0.0)

    def forward(self, hazy: torch.Tensor, t: torch.Tensor, atmospheric: torch.Tensor) -> torch.Tensor:
        dark = _dark_channel(hazy, patch_size=15)
        feat = torch.cat([hazy, dark, t], dim=1)
        delta = self.scale * torch.tanh(self.net(feat))
        return torch.clamp(atmospheric + delta, 1e-6, 1.0)


class DehazeNet(nn.Module):
    """
    DehazeNet with official core operations.

    Parameters still accept old project arguments for compatibility.  The core
    transmission algorithm remains DehazeNet-style; the optional refinement head
    is a paired-training adapter after the physics recovery, not a replacement
    for the paper path.
    """
    def __init__(
        self,
        in_ch: int = 3,
        maxout_k: int = 4,
        guided_filter_refine: bool = True,
        t0: float = 0.10,
        use_image_refine: bool = False,
        refine_scale: float = 0.0,
        feature_ch: int = 24,
        mapping_ch: int = 96,
        output_refine_hidden: int = 128,
        **unused,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.requested_maxout_k = int(maxout_k)
        self.maxout_k = 4
        self.feature_ch = max(4, int(feature_ch))
        self.mapping_ch = max(16, int(mapping_ch))
        self.t0 = float(t0)
        self.recovery_t_floor = self.t0
        self.low_trans_percent = 0.001
        self.local_extremum_kernel = 7
        self.init_transmission = 0.75
        self.final_detail_preserve_scale = 0.45
        self.final_detail_radius = 2
        self.final_detail_gate_threshold = 0.006
        self.final_detail_gate_slope = 60.0
        self.final_detail_ratio_limit = 0.35
        self.transmission_upper = 1.0
        self.atmospheric_bright_percent = 0.10
        self.max_logit_residual = 2.0
        self.residual_rgb_scale = 0.30
        self.output_blend = 1.0
        self.pred_sharp_scale = 0.55
        self.pred_sharp_radius = 1
        self.pred_sharp_ratio_limit = 0.35
        self.straight_through_output = True
        self.output_eps = 1e-6

        self.requested_guided_filter_refine = bool(guided_filter_refine)
        # Guided filtering is a paper/post-processing step, but this repository
        # trains from RGB reconstruction loss. Honor the caller so the training
        # script can disable the large-radius smoothing path.
        self.guided_filter_refine = bool(guided_filter_refine)
        self.guided_radius = 15
        self.guided_filter = GuidedFilter(radius=self.guided_radius, eps=1e-3)

        self.requested_use_image_refine = bool(use_image_refine)
        self.requested_refine_scale = float(refine_scale)
        # The original shared script passes use_image_refine=True.  Keep the
        # paper core as the main algorithm, but use that flag to enable a
        # zero-initialized full-resolution correction after physics recovery.
        self.use_image_refine = bool(use_image_refine)
        self.output_refine_scale = (
            max(float(refine_scale), 0.85)
            if self.use_image_refine else 0.0
        )

        self.feature = MaxoutConv(self.in_ch, out_ch=self.feature_ch, maxout_k=self.maxout_k, kernel_size=5)
        self.map3 = SameConv2d(self.feature_ch, self.mapping_ch, 3)
        self.map5 = SameConv2d(self.feature_ch, self.mapping_ch, 5)
        self.map7 = SameConv2d(self.feature_ch, self.mapping_ch, 7)
        # Table I uses a 6x6 convolution for non-linear regression.
        self.regress_t = SameConv2d(self.mapping_ch * 3, 1, 6)
        self.brelu = BReLU(0.0, self.transmission_upper, straight_through=True)
        self.output_refine = (
            OutputRefineHead(
                in_ch=self.in_ch,
                hidden=max(16, int(output_refine_hidden)),
                scale=self.output_refine_scale,
                high_scale=0.75,
                low_scale=0.25,
            )
            if self.use_image_refine else None
        )
        self.atmospheric_refine = AtmosphericRefineHead(
            in_ch=self.in_ch,
            hidden=max(32, int(output_refine_hidden) // 2),
            scale=0.12,
        )

        nn.init.normal_(self.regress_t.conv.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.regress_t.conv.bias, self.init_transmission)
        print(
            f"[DehazeNet] {ARCH_VERSION} maxout_k={self.maxout_k} "
            f"requested_maxout_k={self.requested_maxout_k} "
            f"feature_ch={self.feature_ch} mapping_ch={self.mapping_ch} "
            f"refine_hidden={max(16, int(output_refine_hidden))} "
            f"requested_guided={self.requested_guided_filter_refine} guided={self.guided_filter_refine} "
            f"guided_radius={self.guided_radius} "
            f"t_floor={self.recovery_t_floor:.3f} local_ext={self.local_extremum_kernel} "
            f"init_t={self.init_transmission:.2f} "
            f"t_upper={self.transmission_upper:.2f} "
            f"rgb_detail={self.final_detail_preserve_scale:.2f} "
            f"detail_ratio={self.final_detail_ratio_limit:.2f} "
            f"refine={self.use_image_refine} refine_scale={self.output_refine_scale:.2f} "
            f"pred_sharp={self.pred_sharp_scale:.2f} "
            f"A_adapter=True"
        )

    def _local_extremum(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.local_extremum_kernel // 2
        x = _same_pad(x, pad)
        return F.max_pool2d(x, kernel_size=self.local_extremum_kernel, stride=1)

    def _estimate_transmission(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.feature(x)
        f2 = torch.cat([self.map3(f1), self.map5(f1), self.map7(f1)], dim=1)
        f3 = self._local_extremum(f2)
        raw_t = self.regress_t(f3)
        t = self.brelu(raw_t)
        return torch.where(torch.isfinite(t), t, t.new_full((), self.init_transmission))

    def _estimate_atmospheric_light(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        low_k = max(1, min(n, int(round(n * self.low_trans_percent))))

        with torch.no_grad():
            score = _dark_channel(x, patch_size=15).flatten(1)
            top_idx = torch.topk(score, k=low_k, dim=1, largest=True).indices

            img_flat = x.flatten(2)
            cand = torch.gather(img_flat, 2, top_idx.unsqueeze(1).expand(-1, c, -1))
            cand_gray = cand.mean(dim=1)
            bright_k = max(1, min(low_k, int(round(low_k * self.atmospheric_bright_percent))))
            bright_idx = torch.topk(cand_gray, k=bright_k, dim=1, largest=True).indices
            a = torch.gather(cand, 2, bright_idx.unsqueeze(1).expand(-1, c, -1))
        return a.mean(dim=2).view(b, c, 1, 1).clamp(self.output_eps, 1.0)

    def _recover_raw(self, x: torch.Tensor, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        t_safe = torch.clamp(t, min=self.recovery_t_floor, max=1.0)
        j = (x - a * (1.0 - t_safe)) / t_safe
        return torch.where(torch.isfinite(j), j, x)

    def _straight_through_clamp(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, ref)
        clipped = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        if not self.straight_through_output:
            return clipped
        return x + (clipped - x).detach()

    def _bounded_recovery(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        raw = torch.where(torch.isfinite(raw), raw, x_safe)
        delta = raw - x_safe
        residual_logits = self.max_logit_residual * delta / (delta.abs() + self.residual_rgb_scale)
        recovered = torch.sigmoid(torch.logit(x_safe) + residual_logits)
        out = x_safe + self.output_blend * (recovered - x_safe)
        return self._straight_through_clamp(out, x_safe)

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
        align_score = hazy_detail * pred_detail / (
            hazy_detail.abs() * pred_detail.abs() + 1e-6
        )
        agreement = torch.sigmoid(3.0 * align_score)
        detail_y = hazy_detail * edge_gate * (0.35 + 0.65 * agreement)
        target_y = torch.clamp(
            pred_y + self.final_detail_preserve_scale * detail_y,
            self.output_eps,
            1.0 - self.output_eps,
        )
        ratio = target_y / (pred_y + 1e-6)
        ratio = torch.clamp(
            ratio,
            1.0 - self.final_detail_ratio_limit,
            1.0 + self.final_detail_ratio_limit,
        )
        out = pred * ratio
        return self._straight_through_clamp(out, pred)

    def _sharpen_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        if self.pred_sharp_scale <= 0.0:
            return pred
        pred_y = _rgb_to_luma(pred).clamp(self.output_eps, 1.0 - self.output_eps)
        detail_y = pred_y - _local_mean(pred_y, self.pred_sharp_radius)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        b, ch, h, w = x.shape
        if ch != self.in_ch:
            raise ValueError(f"DehazeNet expected {self.in_ch} channels, got {ch}.")

        t_raw = self._estimate_transmission(x)
        guidance = x.mean(dim=1, keepdim=True)
        if self.guided_filter_refine:
            t_refined = self.guided_filter(guidance, t_raw)
        else:
            t_refined = t_raw
        t_clipped = torch.clamp(t_refined, 0.0, self.transmission_upper)
        t_refined = t_refined + (t_clipped - t_refined).detach()
        t_refined = torch.where(torch.isfinite(t_refined), t_refined, t_raw)

        atmospheric = self._estimate_atmospheric_light(x, t_refined)
        atmospheric = self.atmospheric_refine(x, t_refined, atmospheric)
        raw = self._recover_raw(x, t_refined, atmospheric)
        out = self._bounded_recovery(x, raw)
        if self.output_refine is not None:
            pred_detail = out - _local_mean(out, 1)
            refine_logits = self.output_refine(x, out, pred_detail, t_refined, atmospheric)
            refine_logits = torch.nan_to_num(refine_logits, nan=0.0, posinf=2.0, neginf=-2.0)
            refine_logits = torch.clamp(refine_logits, -2.0, 2.0)
            out_refined = torch.sigmoid(
                torch.logit(out.clamp(self.output_eps, 1.0 - self.output_eps)) + refine_logits
            )
            out = self._straight_through_clamp(out_refined, out)
        out = self._preserve_input_detail(x, out)
        out = self._sharpen_prediction(out)
        residual = out - x

        color_gain = (out.mean(dim=(2, 3), keepdim=True) + 1e-6) / (
            x.mean(dim=(2, 3), keepdim=True) + 1e-6
        )
        color_gain = torch.clamp(color_gain, 0.2, 3.0)

        sides = [
            t_raw.expand(b, 3, h, w),
            t_refined.expand(b, 3, h, w),
            atmospheric.expand(b, ch, h, w),
        ]
        return out, residual, color_gain, sides


if __name__ == "__main__":
    net = DehazeNet(in_ch=3, maxout_k=2, guided_filter_refine=False, use_image_refine=True)
    x = torch.rand(2, 3, 127, 129)
    out, res, gain, sides = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape)
    print("sides", [None if s is None else s.shape for s in sides])
