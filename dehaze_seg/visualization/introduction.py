#!/usr/bin/env python3
"""Compose the paper Introduction figure with deterministic vector layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image


BLUE = "#245A9C"
ORANGE = "#D97706"
GREEN = "#2F7D43"
PURPLE = "#74639A"
TEXT = "#20242A"
MUTED = "#5A616B"
LIGHT = "#E7EBF0"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="007")
    parser.add_argument("--data-root", "--data_root", default="datasets")
    parser.add_argument("--visualization-dir", "--visualization_dir", default="results/color_gain")
    parser.add_argument("--output-dir", "--output_dir", default="results/figures")
    return parser.parse_args(argv)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def find_image(folder: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for sample {stem} under {folder}")


def load_gain_row(csv_path: Path, sample: str) -> dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sample"] == sample:
                return {key: float(value) for key, value in row.items() if key != "sample"}
    raise RuntimeError(f"Sample {sample} not found in {csv_path}")


def style_image_axis(ax, title: str, note: str = "") -> None:
    ax.set_title(title, fontsize=12, fontweight="semibold", color=TEXT, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if note:
        ax.text(
            0.5, -0.075, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5, color=MUTED,
        )


def add_box(ax, x, y, w, h, text, edge, fontsize=9.5, linewidth=1.25, face="#FFFFFF"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=linewidth, edgecolor=edge, facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=fontsize, color=TEXT, linespacing=1.25,
    )
    return patch


def add_arrow(ax, start, end, color="#343A40", linewidth=1.2):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=linewidth, color=color, transform=ax.transAxes,
        shrinkA=1, shrinkB=1,
    )
    ax.add_patch(arrow)


def draw_pipeline(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 0.13, 0.925, 0.74, 0.048, "Hazy input", BLUE, fontsize=9.5)
    add_box(ax, 0.13, 0.842, 0.74, 0.048, "ColorAwareUNet", BLUE, fontsize=9.5)
    add_arrow(ax, (0.50, 0.925), (0.50, 0.890))

    add_box(
        ax, 0.04, 0.665, 0.42, 0.125,
        "Global RGB gain  $g$\nfrom bottleneck $c$\n$g\\in\\mathbb{R}^{3\\times1\\times1}$",
        ORANGE, fontsize=9.0,
    )
    add_box(
        ax, 0.54, 0.665, 0.42, 0.125,
        "Residual prediction  $r$\nfrom top decoder $d_1$",
        BLUE, fontsize=9.0,
    )
    add_arrow(ax, (0.50, 0.842), (0.25, 0.790))
    add_arrow(ax, (0.50, 0.842), (0.75, 0.790))

    add_box(
        ax, 0.08, 0.525, 0.84, 0.082,
        "Coarse color-aware fusion\n$I_c=I_h\\odot g+s_r r$",
        ORANGE, fontsize=9.2,
    )
    add_arrow(ax, (0.25, 0.665), (0.38, 0.607))
    add_arrow(ax, (0.75, 0.665), (0.62, 0.607))

    add_box(
        ax, 0.08, 0.383, 0.84, 0.085,
        "Refinement head\n$\\Delta I=f_{ref}(I_h,I_c,r,d_1)$",
        BLUE, fontsize=9.2, face="#FBFCFE",
    )
    add_arrow(ax, (0.50, 0.525), (0.50, 0.468))

    add_box(
        ax, 0.08, 0.267, 0.84, 0.067,
        "Dehazed output\n$I_{out}=\\mathrm{clip}(I_c+s_{ref}\\Delta I,0,1)$",
        BLUE, fontsize=9.0,
    )
    add_arrow(ax, (0.50, 0.383), (0.50, 0.334))

    add_box(ax, 0.08, 0.172, 0.84, 0.050, "LiteAttentionUNet", GREEN, fontsize=9.4)
    add_arrow(ax, (0.50, 0.267), (0.50, 0.222))
    add_box(ax, 0.08, 0.092, 0.84, 0.048, "Semantic segmentation", GREEN, fontsize=9.4)
    add_arrow(ax, (0.50, 0.172), (0.50, 0.140))

    stage = FancyBboxPatch(
        (0.03, 0.008), 0.94, 0.044,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=0.9, edgecolor=PURPLE, facecolor="#F4F1F9",
        transform=ax.transAxes,
    )
    ax.add_patch(stage)
    ax.text(
        0.5, 0.030,
        "Stage-wise: dehaze pretraining  →  frozen-dehazer segmentation  →  joint fine-tuning",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=7.8, color="#51466D",
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    sample = args.sample
    data_root = Path(args.data_root)
    vis_dir = Path(args.visualization_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hazy = load_rgb(find_image(data_root / "hazy", sample))
    clear = load_rgb(find_image(data_root / "clear", sample))
    dehazed = load_rgb(vis_dir / f"{sample}_dehazed.png")
    overlay = load_rgb(vis_dir / f"{sample}_seg_overlay.png")
    contribution = np.load(vis_dir / f"{sample}_color_gain_contribution.npy")
    gains_row = load_gain_row(vis_dir / "color_gain_metrics.csv", sample)
    gains = np.array([gains_row["gain_r"], gains_row["gain_g"], gains_row["gain_b"]])
    deviations = gains - 1.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#7B838D",
        "axes.linewidth": 0.8,
    })

    fig = plt.figure(figsize=(20, 9.5), dpi=300, facecolor="white")
    outer = fig.add_gridspec(
        2, 12, height_ratios=[1.55, 0.82],
        left=0.025, right=0.985, top=0.915, bottom=0.075,
        wspace=0.65, hspace=0.34,
    )

    ax_hazy = fig.add_subplot(outer[0, 0:3])
    ax_pipe = fig.add_subplot(outer[0, 3:7])
    right_top = outer[0, 7:12].subgridspec(1, 2, wspace=0.10)
    ax_dehazed = fig.add_subplot(right_top[0, 0])
    ax_clear = fig.add_subplot(right_top[0, 1])

    ax_gain = fig.add_subplot(outer[1, 0:4])
    ax_heat = fig.add_subplot(outer[1, 4:8])
    ax_seg = fig.add_subplot(outer[1, 8:12])

    fig.text(0.145, 0.963, "Visual motivation", ha="center", va="center",
             fontsize=15, fontweight="semibold", color=BLUE)
    fig.text(0.430, 0.963, "Proposed joint pipeline", ha="center", va="center",
             fontsize=15, fontweight="semibold", color=BLUE)
    fig.text(0.790, 0.963, "Qualitative example", ha="center", va="center",
             fontsize=15, fontweight="semibold", color=BLUE)

    ax_hazy.imshow(hazy)
    style_image_axis(ax_hazy, "(a) Hazy input", "Reduced visibility and attenuated contrast")

    draw_pipeline(ax_pipe)

    ax_dehazed.imshow(dehazed)
    style_image_axis(ax_dehazed, "(b) Dehazed output")
    ax_clear.imshow(clear)
    style_image_axis(ax_clear, "(c) Clear ground truth")
    fig.text(0.790, 0.425, "Visual comparison with the clear reference",
             ha="center", fontsize=9.5, color=MUTED)

    colors = ["#D62728", "#2CA02C", "#1F66B3"]
    bars = ax_gain.bar(["R", "G", "B"], deviations, color=colors, width=0.58)
    ax_gain.set_ylim(0.0, 0.064)
    ax_gain.set_ylabel(r"Gain deviation  $(g_c-1)$", fontsize=9.5)
    ax_gain.set_title("(d) Predicted global RGB gain", fontsize=12, pad=8)
    ax_gain.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax_gain.set_axisbelow(True)
    ax_gain.spines["top"].set_visible(False)
    ax_gain.spines["right"].set_visible(False)
    ax_gain.text(0.99, 0.02, "identity: 0", transform=ax_gain.transAxes,
                 ha="right", va="bottom", fontsize=8.5, color=MUTED)
    for bar, gain, dev in zip(bars, gains, deviations):
        ax_gain.text(
            bar.get_x() + bar.get_width() / 2, dev + 0.0015,
            f"$g={gain:.3f}$\n{dev:.3f}",
            ha="center", va="bottom", fontsize=8.8, color=TEXT,
        )

    ax_heat.imshow(hazy)
    heat = ax_heat.imshow(contribution, cmap="inferno", alpha=0.66, vmin=0.0, vmax=0.04)
    style_image_axis(ax_heat, "(e) Color-gain contribution map")
    cbar = fig.colorbar(heat, ax=ax_heat, fraction=0.044, pad=0.025)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean absolute RGB contribution", fontsize=8.5)
    ax_heat.text(
        0.5, -0.075,
        r"$M(h,w)=\mathrm{mean}_c\,|I_{h,c}(h,w)(g_c-1)|$",
        transform=ax_heat.transAxes, ha="center", va="top", fontsize=9.5, color=TEXT,
    )

    ax_seg.imshow(overlay)
    style_image_axis(ax_seg, "(f) Segmentation overlay", "Road-region prediction after the joint pipeline")

    png_path = out_dir / f"introduction_figure_{sample}.png"
    pdf_path = out_dir / f"introduction_figure_{sample}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
