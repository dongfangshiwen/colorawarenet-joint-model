#!/usr/bin/env python3
"""Generate paper-ready ColorAwareUNet visualizations from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..data.common import IMG_EXTS, find_by_stem, pil_to_rgb_tensor
from ..metrics import batch_psnr, batch_ssim
from ..utils import ensure_dir


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/colorawareunet.pth")
    parser.add_argument("--data-root", "--data_root", default="datasets")
    parser.add_argument("--output-dir", "--output_dir", default="results/color_gain")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--samples",
        default="",
        help="Optional comma-separated stems. When omitted, samples are selected deterministically.",
    )
    parser.add_argument("--resize", type=int, nargs=2, help="Override visualization H W")
    parser.add_argument("--legacy-profile", choices=["paper", "legacy-infer"])
    parser.add_argument("--model-config")
    return parser.parse_args(argv)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def load_joint_model(checkpoint_path: Path, device: torch.device, profile=None, config_path=None):
    from ..engine.checkpoint import load_checkpoint
    model, config, checkpoint = load_checkpoint(checkpoint_path, device=device, profile=profile, config_path=config_path)
    if not config["joint"] or config["model"] != "coloraware":
        raise ValueError("Gain visualization requires a joint ColorAwareUNet checkpoint")
    resize = tuple(checkpoint.get("train_config", checkpoint.get("args", {})).get("resize", [512, 512]))
    return model, resize, checkpoint


def choose_images(hazy_dir: Path, count: int, seed: int, requested: str) -> list[Path]:
    images = sorted(
        p for p in hazy_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not images:
        raise RuntimeError(f"No images found in {hazy_dir}")

    by_stem = {p.stem: p for p in images}
    if requested.strip():
        stems = [s.strip() for s in requested.split(",") if s.strip()]
        missing = [s for s in stems if s not in by_stem]
        if missing:
            raise RuntimeError(f"Requested samples not found: {missing}")
        return [by_stem[s] for s in stems]

    rng = random.Random(seed)
    if count <= 0 or count >= len(images):
        return images
    return sorted(rng.sample(images, count), key=lambda p: p.name)


def mask_overlay(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    mask = (label > 0)[..., None].astype(np.float32)
    red = np.array([1.0, 0.20, 0.20], dtype=np.float32).reshape(1, 1, 3)
    return np.clip(image * (1.0 - 0.45 * mask) + red * (0.45 * mask), 0.0, 1.0)


@torch.no_grad()
def infer_sample(model, image_path: Path, clear_dir: Path, resize, device: torch.device) -> dict:
    hazy_pil = Image.open(image_path).convert("RGB")
    original_size = hazy_pil.size
    h, w = int(resize[0]), int(resize[1])
    model_pil = hazy_pil.resize((w, h), Image.BICUBIC)
    x = pil_to_rgb_tensor(model_pil).unsqueeze(0).to(device)

    dehazed, logits, aux = model(x)
    gain = aux.get("color_gain")
    if gain is None:
        raise RuntimeError("The model did not return color_gain")

    # This is an effect map, not a spatial gain map. Global gain itself is 3x1x1.
    contribution = torch.abs(x * (gain - 1.0)).mean(dim=1, keepdim=True)
    gain_values = gain[0].reshape(3, -1).mean(dim=1).detach().cpu().numpy()

    pred_label = logits.argmax(dim=1, keepdim=True).float()
    pred_label = F.interpolate(pred_label, size=(h, w), mode="nearest")[0, 0].cpu().numpy().astype(np.uint8)

    hazy = np.asarray(model_pil, dtype=np.float32) / 255.0
    dehazed_np = dehazed[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    overlay = mask_overlay(dehazed_np, pred_label)

    clear_path = find_by_stem(clear_dir, image_path.stem) if clear_dir.exists() else None
    clear_np = None
    psnr = float("nan")
    ssim = float("nan")
    if clear_path is not None:
        clear_pil = Image.open(clear_path).convert("RGB").resize((w, h), Image.BICUBIC)
        clear_tensor = pil_to_rgb_tensor(clear_pil).unsqueeze(0).to(device)
        clear_np = np.asarray(clear_pil, dtype=np.float32) / 255.0
        psnr = batch_psnr(dehazed, clear_tensor)
        ssim = batch_ssim(dehazed, clear_tensor)

    return {
        "stem": image_path.stem,
        "original_size": original_size,
        "hazy": hazy,
        "dehazed": dehazed_np,
        "clear": clear_np,
        "overlay": overlay,
        "label": pred_label,
        "gain": gain_values,
        "contribution": contribution[0, 0].cpu().numpy(),
        "psnr": psnr,
        "ssim": ssim,
    }


def save_panel(item: dict, out_path: Path, contribution_vmax: float, gain_ylim: tuple[float, float]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    axes[0, 0].imshow(item["hazy"])
    axes[0, 0].set_title("(a) Hazy Input")

    axes[0, 1].imshow(item["dehazed"])
    metric_text = "(b) Dehazed Output"
    if np.isfinite(item["psnr"]):
        metric_text += f"\nPSNR {item['psnr']:.2f} dB | SSIM {item['ssim']:.4f}"
    axes[0, 1].set_title(metric_text)

    if item["clear"] is not None:
        axes[0, 2].imshow(item["clear"])
        axes[0, 2].set_title("(c) Clear Ground Truth")
    else:
        axes[0, 2].text(0.5, 0.5, "Clear GT unavailable", ha="center", va="center")
        axes[0, 2].set_title("(c) Clear Ground Truth")

    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    axes[1, 0].bar(["R", "G", "B"], item["gain"], color=colors, width=0.62)
    axes[1, 0].axhline(1.0, color="#222222", linestyle="--", linewidth=1.2)
    axes[1, 0].set_ylim(*gain_ylim)
    axes[1, 0].set_ylabel("Gain value")
    axes[1, 0].set_title("(d) Global RGB Color Gain")
    axes[1, 0].grid(axis="y", alpha=0.25, linestyle="--")
    for i, value in enumerate(item["gain"]):
        axes[1, 0].text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    axes[1, 1].imshow(item["hazy"])
    heat = axes[1, 1].imshow(
        item["contribution"], cmap="inferno", alpha=0.60,
        vmin=0.0, vmax=contribution_vmax,
    )
    axes[1, 1].set_title("(e) Color-Gain Contribution Map")
    cbar = fig.colorbar(heat, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Mean absolute RGB contribution")

    axes[1, 2].imshow(item["overlay"])
    axes[1, 2].set_title("(f) Segmentation Overlay")

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
        ax.axis("off")

    fig.suptitle(f"Color-Gain-Guided Joint Inference: {item['stem']}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_standalone_outputs(item: dict, out_dir: Path, contribution_vmax: float) -> None:
    Image.fromarray((item["dehazed"] * 255).round().astype(np.uint8)).save(
        out_dir / f"{item['stem']}_dehazed.png"
    )
    Image.fromarray((item["overlay"] * 255).round().astype(np.uint8)).save(
        out_dir / f"{item['stem']}_seg_overlay.png"
    )
    Image.fromarray((item["label"] > 0).astype(np.uint8) * 255, mode="L").save(
        out_dir / f"{item['stem']}_seg_mask.png"
    )
    np.save(out_dir / f"{item['stem']}_color_gain_contribution.npy", item["contribution"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(item["hazy"])
    heat = ax.imshow(
        item["contribution"], cmap="inferno", alpha=0.60,
        vmin=0.0, vmax=contribution_vmax,
    )
    ax.axis("off")
    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean absolute RGB contribution")
    fig.tight_layout()
    fig.savefig(out_dir / f"{item['stem']}_color_gain_contribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    output_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    model, resize, checkpoint = load_joint_model(checkpoint_path, device, args.legacy_profile, args.model_config)
    resize = tuple(args.resize) if args.resize else resize
    if min(resize) < 32:
        raise ValueError("Visualization resize must be at least 32 in each dimension")

    paths = choose_images(data_root / "hazy", args.count, args.seed, args.samples)
    print("Selected samples:", ", ".join(p.stem for p in paths))
    items = [infer_sample(model, p, data_root / "clear", resize, device) for p in paths]

    all_contributions = np.concatenate([item["contribution"].reshape(-1) for item in items])
    contribution_vmax = max(float(np.percentile(all_contributions, 99.0)), 1e-6)
    all_gains = np.concatenate([item["gain"] for item in items])
    gain_margin = max(0.02, float(all_gains.max() - all_gains.min()) * 0.20)
    gain_ylim = (min(0.95, float(all_gains.min()) - gain_margin), float(all_gains.max()) + gain_margin)

    rows = []
    for item in items:
        save_panel(item, output_dir / f"{item['stem']}_paper_panel.png", contribution_vmax, gain_ylim)
        save_standalone_outputs(item, output_dir, contribution_vmax)
        rows.append({
            "sample": item["stem"],
            "gain_r": float(item["gain"][0]),
            "gain_g": float(item["gain"][1]),
            "gain_b": float(item["gain"][2]),
            "psnr": item["psnr"],
            "ssim": item["ssim"],
            "contribution_mean": float(item["contribution"].mean()),
            "contribution_max": float(item["contribution"].max()),
        })

    csv_path = output_dir / "color_gain_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(items)} paper panels in: {output_dir.resolve()}")
    print(f"Metrics: {csv_path.resolve()}")
    stats = checkpoint.get("stats", {})
    if stats:
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}, score stats available: {len(stats)}")


if __name__ == "__main__":
    main()
