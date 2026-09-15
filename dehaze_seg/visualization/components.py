#!/usr/bin/env python3
"""Visualize real ColorAwareUNet components for one dataset sample."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..data.common import find_by_stem, pil_to_rgb_tensor
from ..metrics import batch_psnr, batch_ssim
from ..utils import ensure_dir
from .gain import load_joint_model, mask_overlay, select_device


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/colorawareunet.pth")
    parser.add_argument("--data-root", "--data_root", default="datasets")
    parser.add_argument("--sample", default="140")
    parser.add_argument("--output-dir", "--output_dir", default="results/components")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resize", type=int, nargs=2, help="Override visualization H W")
    parser.add_argument("--legacy-profile", choices=["paper", "legacy-infer"])
    parser.add_argument("--model-config")
    return parser.parse_args(argv)


def tensor_rgb(x: torch.Tensor) -> np.ndarray:
    return x[0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def magnitude_map(x: torch.Tensor) -> np.ndarray:
    return x[0].detach().abs().mean(dim=0).cpu().numpy()


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray((array.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path)


def save_heatmap(path: Path, values: np.ndarray, title: str, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(values, cmap="inferno", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean absolute RGB contribution")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@torch.no_grad()
def infer_components(model, image_path: Path, clear_path: Path, resize, device: torch.device) -> dict:
    h, w = int(resize[0]), int(resize[1])
    hazy_pil = Image.open(image_path).convert("RGB").resize((w, h), Image.BICUBIC)
    clear_pil = Image.open(clear_path).convert("RGB").resize((w, h), Image.BICUBIC)
    x = pil_to_rgb_tensor(hazy_pil).unsqueeze(0).to(device)
    clear = pil_to_rgb_tensor(clear_pil).unsqueeze(0).to(device)

    captured: dict[str, torch.Tensor] = {}

    def capture_refinement(_module, _inputs, output):
        captured["refinement_raw"] = output.detach()

    handle = model.dehazer.refine.register_forward_hook(capture_refinement)
    try:
        dehazed, logits, aux = model(x)
    finally:
        handle.remove()

    residual_raw = aux.get("residual")
    gain = aux.get("color_gain")
    refinement_raw = captured.get("refinement_raw")
    if residual_raw is None or gain is None or refinement_raw is None:
        raise RuntimeError("Failed to capture residual, color gain, or refinement tensors")

    residual_scale = float(model.dehazer.residual_scale)
    refine_scale = float(model.dehazer.refine_scale)
    gain_contribution = x * (gain - 1.0)
    residual_contribution = residual_raw * residual_scale
    coarse = x * gain + residual_contribution
    refinement_contribution = refinement_raw * refine_scale
    preclip = coarse + refinement_contribution
    reconstructed = preclip.clamp(0.0, 1.0)
    reconstruction_error = float((reconstructed - dehazed).abs().max().item())

    pred_label = logits.argmax(dim=1, keepdim=True).float()
    pred_label = F.interpolate(pred_label, size=(h, w), mode="nearest")[0, 0].cpu().numpy().astype(np.uint8)

    hazy_np = tensor_rgb(x)
    clear_np = tensor_rgb(clear)
    gain_only_np = tensor_rgb((x * gain).clamp(0.0, 1.0))
    coarse_np = tensor_rgb(coarse.clamp(0.0, 1.0))
    dehazed_np = tensor_rgb(dehazed)
    overlay_np = mask_overlay(dehazed_np, pred_label)

    return {
        "hazy": hazy_np,
        "clear": clear_np,
        "gain_only": gain_only_np,
        "coarse": coarse_np,
        "dehazed": dehazed_np,
        "overlay": overlay_np,
        "label": pred_label,
        "gain": gain[0, :, 0, 0].detach().cpu().numpy(),
        "gain_contribution": gain_contribution.detach().cpu().numpy(),
        "residual_raw": residual_raw.detach().cpu().numpy(),
        "residual_contribution": residual_contribution.detach().cpu().numpy(),
        "refinement_raw": refinement_raw.detach().cpu().numpy(),
        "refinement_contribution": refinement_contribution.detach().cpu().numpy(),
        "coarse_tensor": coarse.detach().cpu().numpy(),
        "preclip": preclip.detach().cpu().numpy(),
        "final_tensor": dehazed.detach().cpu().numpy(),
        "gain_map": magnitude_map(gain_contribution),
        "residual_map": magnitude_map(residual_contribution),
        "refinement_map": magnitude_map(refinement_contribution),
        "psnr": batch_psnr(dehazed, clear),
        "ssim": batch_ssim(dehazed, clear),
        "residual_scale": residual_scale,
        "refine_scale": refine_scale,
        "reconstruction_error": reconstruction_error,
    }


def save_gain_deviation_plot(item: dict, path: Path) -> None:
    deviations = item["gain"] - 1.0
    colors = ["#D62728", "#2CA02C", "#1F66B3"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bars = ax.bar(["R", "G", "B"], deviations, color=colors, width=0.62)
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.1)
    lower = min(float(deviations.min()), 0.0)
    upper = max(float(deviations.max()), 0.0)
    margin = max((upper - lower) * 0.35, 0.01)
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_ylabel(r"Gain deviation from identity  $(g_c-1)$")
    ax.set_title("Predicted Global RGB Gain", fontsize=13, fontweight="semibold", pad=12)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for bar, gain, deviation in zip(bars, item["gain"], deviations):
        ax.annotate(
            f"$g={gain:.4f}$\n{deviation * 100:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, deviation),
            xytext=(0, 6 if deviation >= 0 else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if deviation >= 0 else "top",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_paper_panel(
    item: dict,
    sample: str,
    path: Path,
    independent_heatmap_scale: bool = False,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.patch.set_facecolor("white")

    image_panels = [
        (item["hazy"], "(a) Hazy input"),
        (item["gain_only"], "(b) Input x global RGB gain"),
        (item["coarse"], "(c) Coarse output"),
        (item["dehazed"], f"(d) Final dehazed output\nPSNR {item['psnr']:.2f} dB | SSIM {item['ssim']:.4f}"),
    ]
    for ax, (image, title) in zip(axes[0], image_panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.axis("off")

    colors = ["#D62728", "#2CA02C", "#1F66B3"]
    axes[1, 0].bar(["R", "G", "B"], item["gain"], color=colors, width=0.62)
    axes[1, 0].axhline(1.0, color="#222222", linestyle="--", linewidth=1.1)
    margin = max(0.02, float(np.ptp(item["gain"])) * 0.25)
    axes[1, 0].set_ylim(min(0.95, float(item["gain"].min()) - margin), float(item["gain"].max()) + margin)
    axes[1, 0].set_ylabel("Gain value")
    axes[1, 0].set_title("(e) Global RGB gain (3 x 1 x 1)", fontsize=12, fontweight="semibold")
    axes[1, 0].grid(axis="y", alpha=0.25, linestyle="--")
    for index, value in enumerate(item["gain"]):
        axes[1, 0].text(index, value, f"{value:.4f}", ha="center", va="bottom")

    maps = [item["gain_map"], item["residual_map"], item["refinement_map"]]
    map_titles = [
        "(f) Global-gain contribution",
        f"(g) Scaled residual contribution (x{item['residual_scale']:.2f})",
        f"(h) Scaled refinement contribution (x{item['refine_scale']:.2f})",
    ]
    for ax, values, title in zip(axes[1, 1:], maps, map_titles):
        if independent_heatmap_scale:
            vmax = max(float(np.percentile(values, 99.0)), 1e-6)
            title += "\n(independent 99th-percentile scale)"
        else:
            vmax = max(float(np.percentile(np.concatenate([m.ravel() for m in maps]), 99.0)), 1e-6)
        heat = ax.imshow(values, cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.axis("off")
        fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.03)

    axes[2, 0].imshow(item["clear"])
    axes[2, 0].set_title("(i) Clear ground truth", fontsize=12, fontweight="semibold")
    axes[2, 0].axis("off")
    axes[2, 1].imshow(item["overlay"])
    axes[2, 1].set_title("(j) Segmentation overlay", fontsize=12, fontweight="semibold")
    axes[2, 1].axis("off")

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.82,
        r"$g=1+\tanh(g_{raw})\,s_g$" + "\n\n"
        + r"$I_{coarse}=I_{hazy}\odot g+r\,s_r$" + "\n\n"
        + r"$I_{out}=\mathrm{clip}(I_{coarse}+\Delta I\,s_{ref},0,1)$",
        fontsize=15,
        va="top",
    )
    axes[2, 3].axis("off")
    axes[2, 3].text(
        0.02,
        0.82,
        "Measured component means\n\n"
        f"gain effect: {item['gain_map'].mean():.6f}\n"
        f"scaled residual: {item['residual_map'].mean():.6f}\n"
        f"scaled refinement: {item['refinement_map'].mean():.6f}\n\n"
        f"forward reconstruction max error: {item['reconstruction_error']:.2e}",
        fontsize=12,
        va="top",
        family="monospace",
    )

    scale_note = "Enhanced per-component scales" if independent_heatmap_scale else "Shared absolute scale"
    fig.suptitle(
        f"Real ColorAwareUNet Component Analysis: sample {sample} ({scale_note})",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    out_dir = ensure_dir(Path(args.output_dir) / args.sample)
    hazy_path = find_by_stem(data_root / "hazy", args.sample)
    clear_path = find_by_stem(data_root / "clear", args.sample)
    if hazy_path is None or clear_path is None:
        raise RuntimeError(f"Missing hazy/clear pair for sample {args.sample}")

    device = select_device(args.device)
    model, resize, checkpoint = load_joint_model(Path(args.checkpoint), device, args.legacy_profile, args.model_config)
    resize = tuple(args.resize) if args.resize else resize
    if min(resize) < 32:
        raise ValueError("Visualization resize must be at least 32 in each dimension")
    if model.dehazer.gain_mode != "global":
        raise RuntimeError(f"Expected global gain checkpoint, got {model.dehazer.gain_mode}")

    item = infer_components(model, hazy_path, clear_path, resize, device)
    save_rgb(out_dir / f"{args.sample}_gain_only.png", item["gain_only"])
    save_rgb(out_dir / f"{args.sample}_coarse.png", item["coarse"])
    save_rgb(out_dir / f"{args.sample}_dehazed.png", item["dehazed"])
    save_rgb(out_dir / f"{args.sample}_seg_overlay.png", item["overlay"])
    Image.fromarray((item["label"] > 0).astype(np.uint8) * 255, mode="L").save(
        out_dir / f"{args.sample}_seg_mask.png"
    )

    maps = {
        "global_gain_contribution": item["gain_map"],
        "scaled_residual_contribution": item["residual_map"],
        "scaled_refinement_contribution": item["refinement_map"],
    }
    for name, values in maps.items():
        np.save(out_dir / f"{args.sample}_{name}.npy", values)
        component_vmax = max(float(np.percentile(values, 99.0)), 1e-6)
        save_heatmap(
            out_dir / f"{args.sample}_{name}.png",
            values,
            name.replace("_", " ").title() + " (Independent Scale)",
            component_vmax,
        )

    tensor_names = [
        "gain_contribution",
        "residual_raw",
        "residual_contribution",
        "refinement_raw",
        "refinement_contribution",
        "coarse_tensor",
        "preclip",
        "final_tensor",
    ]
    for name in tensor_names:
        np.save(out_dir / f"{args.sample}_{name}.npy", item[name])

    panel_path = out_dir / f"{args.sample}_coloraware_components.png"
    save_paper_panel(item, args.sample, panel_path)
    save_paper_panel(
        item,
        args.sample,
        out_dir / f"{args.sample}_coloraware_components_enhanced.png",
        independent_heatmap_scale=True,
    )
    save_gain_deviation_plot(item, out_dir / f"{args.sample}_global_rgb_gain_deviation.png")

    metrics = {
        "sample": args.sample,
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "gain_mode": model.dehazer.gain_mode,
        "gain_r": float(item["gain"][0]),
        "gain_g": float(item["gain"][1]),
        "gain_b": float(item["gain"][2]),
        "residual_scale": item["residual_scale"],
        "refine_scale": item["refine_scale"],
        "gain_contribution_mean": float(item["gain_map"].mean()),
        "residual_contribution_mean": float(item["residual_map"].mean()),
        "refinement_contribution_mean": float(item["refinement_map"].mean()),
        "psnr": item["psnr"],
        "ssim": item["ssim"],
        "reconstruction_max_error": item["reconstruction_error"],
    }
    with (out_dir / f"{args.sample}_component_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"Device: {device}")
    print(f"Gain mode: {model.dehazer.gain_mode}, shape: 3x1x1")
    print(f"RGB gain: {item['gain'].tolist()}")
    print(f"PSNR: {item['psnr']:.4f} dB, SSIM: {item['ssim']:.6f}")
    print(f"Forward reconstruction max error: {item['reconstruction_error']:.3e}")
    print(f"Saved component analysis to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
