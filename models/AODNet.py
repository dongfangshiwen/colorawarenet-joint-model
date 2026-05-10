# utils/aodnet_paper.py
# -*- coding: utf-8 -*-
"""
AOD-Net backbone.

This implementation keeps the core algorithm from the ICCV 2017 paper:
- K-estimation module with five convolution layers.
- Kernel sizes are 1x1, 3x3, 5x5, 7x7, and 3x3.
- The paper uses three filters in each intermediate K-estimation layer.
- Clean image reconstruction follows:
      J(x) = K(x) * I(x) - K(x) + b

The network returns (J, residual, color_gain, sides) for the shared training
wrapper.  The default path returns the direct native AOD-Net reconstruction
with the final ReLU used by common public implementations.  There is no PONO,
decoder, or trainable refinement head in the default/native path.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_VERSION = "aodnet_native_core_v10"


def _init_conv(m: nn.Module) -> None:
    # Keep PyTorch's default Conv2d initialization, matching the common public
    # AOD-Net implementation.  Earlier tiny Gaussian weights made K nearly
    # spatially constant, which is a direct cause of soft outputs.
    return None


def _pad_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    if pad <= 0:
        return x
    mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


def _local_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    kernel_size = radius * 2 + 1
    return F.avg_pool2d(_pad_same(x, kernel_size), kernel_size, stride=1, padding=0)


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class PaperConv(nn.Module):
    """Public AOD-Net convolution followed by optional ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, act: bool = True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        _init_conv(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class AODNet(nn.Module):
    """
    AOD-Net with paper-aligned K-estimation and reconstruction.

    Constructor arguments from older project versions are accepted for
    compatibility.  In paper mode, `mid_ch`, `use_tanh_on_K`, `k_scale`, and
    refinement-related arguments do not change the K-estimation module because
    the paper uses three filters and ReLU-bounded K.
    """
    def __init__(
        self,
        in_ch: int = 3,
        mid_ch: int = 8,
        use_tanh_on_K: bool = True,
        b: float = 1.0,
        k_scale: float = 0.8,
        k_min: float = 0.0,
        k_max: float = 5.0,
        use_paper_core: bool = True,
        detail_preserve_scale: float = 0.0,
        detail_radius: int = 2,
        detail_norm: float = 0.20,
        detail_gate_threshold: float = 0.006,
        detail_gate_slope: float = 60.0,
        pred_sharp_scale: float = 0.0,
        pred_sharp_radius: int = 1,
        pred_sharp_ratio_limit: float = 0.35,
        use_stable_k: bool = False,
        stable_k_range: float = 1.10,
        max_logit_residual: float = 2.20,
        residual_rgb_scale: float = 0.24,
        output_blend: float = 0.90,
        final_output_blend: float = 0.95,
        max_rgb_delta: float = 0.55,
        bad_output_mean: float = 0.02,
        k_init: float = 1.0,
        paper_final_relu: bool = True,
        straight_through_output: bool = True,
        output_eps: float = 1e-6,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.requested_mid_ch = int(mid_ch)
        self.use_tanh_on_K = bool(use_tanh_on_K)
        self.b = float(b)
        self.k_scale = float(k_scale)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.use_paper_core = bool(use_paper_core)
        self.use_stable_k = bool(use_stable_k)
        self.detail_preserve_scale = float(max(0.0, detail_preserve_scale))
        self.detail_radius = max(1, int(detail_radius))
        self.detail_norm = float(max(detail_norm, 1e-3))
        self.detail_gate_threshold = float(max(detail_gate_threshold, 0.0))
        self.detail_gate_slope = float(max(detail_gate_slope, 1.0))
        self.pred_sharp_scale = float(max(0.0, pred_sharp_scale))
        self.pred_sharp_radius = max(1, int(pred_sharp_radius))
        self.pred_sharp_ratio_limit = float(max(0.0, min(pred_sharp_ratio_limit, 0.5)))
        self.requested_max_logit_residual = float(max_logit_residual)
        self.requested_residual_rgb_scale = float(residual_rgb_scale)
        self.requested_output_blend = float(output_blend)
        self.requested_final_output_blend = float(final_output_blend)
        self.requested_max_rgb_delta = float(max_rgb_delta)
        self.requested_bad_output_mean = float(bad_output_mean)
        self.k_init = float(k_init)
        self.paper_final_relu = bool(paper_final_relu)
        self.straight_through_output = bool(straight_through_output)
        self.output_eps = float(output_eps)

        k_ch = 3 if self.use_paper_core else int(mid_ch)

        # AOD-Net K-estimation module.
        # Public/paper implementations use dense intermediate concatenation:
        # f3 sees [f1, f2], f4 sees [f2, f3], f5 sees [f1, f2, f3, f4].
        self.conv1 = PaperConv(self.in_ch, k_ch, kernel_size=1)
        self.conv2 = PaperConv(k_ch, k_ch, kernel_size=3)
        self.conv3 = PaperConv(k_ch * 2, k_ch, kernel_size=5)
        self.conv4 = PaperConv(k_ch * 2, k_ch, kernel_size=7)
        self.conv5 = PaperConv(k_ch * 4, self.in_ch, kernel_size=3, act=False)

        # Near-identity bias without shrinking the public default conv weights:
        # if K=1 and b=1, then J=I, while non-tiny weights still let K learn
        # spatial structure early.
        if self.conv5.conv.bias is not None:
            nn.init.constant_(self.conv5.conv.bias, self.k_init)
        print(
            f"[AODNet] {ARCH_VERSION} paper_core={self.use_paper_core} "
            f"k_ch={k_ch} requested_mid_ch={self.requested_mid_ch} "
            f"k_bounds=({self.k_min:.2f},{self.k_max:.2f}) k_init={self.k_init:.2f} "
            f"paper_relu={self.paper_final_relu} native_core=True "
            f"st_clamp={self.straight_through_output} "
            f"rgb_detail={self.detail_preserve_scale:.2f} "
            f"pred_sharp={self.pred_sharp_scale:.2f}"
        )

    def _estimate_k(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(torch.cat([f1, f2], dim=1))
        f4 = self.conv4(torch.cat([f2, f3], dim=1))
        raw_k = self.conv5(torch.cat([f1, f2, f3, f4], dim=1))

        if self.use_paper_core:
            # AOD-Net uses ReLU after the final K-estimation convolution.
            k = F.relu(raw_k)
        else:
            if self.use_tanh_on_K:
                k = 1.0 + torch.tanh(raw_k) * self.k_scale
            else:
                k = 1.0 + raw_k
        k = torch.where(torch.isfinite(k), k, torch.ones_like(k))
        # Forward-safe K bounds.  The detach term keeps gradients from becoming
        # zero when the training wrapper drives K briefly outside display range.
        k_safe = torch.clamp(k, min=self.k_min, max=self.k_max)
        k = k + (k_safe - k).detach()
        return raw_k, k

    def _straight_through_clamp(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, ref)
        if not self.straight_through_output:
            return torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        clipped = torch.clamp(x, self.output_eps, 1.0 - self.output_eps)
        return x + (clipped - x).detach()

    def _preserve_input_detail(self, hazy: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.detail_preserve_scale <= 0.0:
            return pred

        detail = hazy - _local_mean(hazy, self.detail_radius)
        edge_strength = detail.abs().mean(dim=1, keepdim=True)
        edge_gate = torch.sigmoid(self.detail_gate_slope * (edge_strength - self.detail_gate_threshold))
        out = pred + self.detail_preserve_scale * edge_gate * detail
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

    def forward(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        raw_k, k = self._estimate_k(I)

        bval = I.new_tensor(self.b)
        J_raw = k * I - k + bval
        J = F.relu(J_raw) if self.paper_final_relu else J_raw
        J = self._straight_through_clamp(J, I)
        J = self._preserve_input_detail(I, J)
        J = self._sharpen_prediction(J)
        residual = J - I

        color_gain = J.mean(dim=[2, 3], keepdim=True) / (I.mean(dim=[2, 3], keepdim=True) + 1e-6)
        color_gain = torch.clamp(color_gain, 0.2, 3.0)
        sides = [raw_k, k, None]
        return J, residual, color_gain, sides


if __name__ == "__main__":
    m = AODNet(in_ch=3, mid_ch=8)
    x = torch.rand(2, 3, 255, 257)
    out, res, gain, sides = m(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "sides", [None if s is None else s.shape for s in sides])
