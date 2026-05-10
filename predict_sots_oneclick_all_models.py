#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click SOTS single-image inference for all dehaze models.

Usage example:
python predict_sots_oneclick_all_models.py \
  --hazy_path SOTS/indoor/hazy/1400_10.png \
  --ckpt_dir ckpt_stos_indoor/joint \
  --out_dir results/sots_oneclick
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

if "OMP_NUM_THREADS" in os.environ:
    try:
        if int(os.environ["OMP_NUM_THREADS"]) <= 0:
            raise ValueError
    except Exception:
        os.environ["OMP_NUM_THREADS"] = "1"
        print("[Warn] OMP_NUM_THREADS is invalid, fallback to 1")

import torch

from predict_dehaze_all import (
    MODEL_BUILDERS,
    MODEL_LABELS,
    canonical_model_name,
    draw_centered,
    extract_dehazed,
    find_ckpt_for_model,
    format_metric,
    load_checkpoint,
    load_font,
    pad_to_multiple,
    parse_name_list,
    psnr,
    resize_pil,
    ssim_metric,
    to_pil,
    to_tensor,
    unpad_tensor,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_gt_for_sots(gt_dir: Optional[Path], hazy_stem: str) -> Optional[Path]:
    if gt_dir is None or not gt_dir.exists():
        return None
    for ext in IMAGE_EXTS:
        p = gt_dir / f"{hazy_stem}{ext}"
        if p.exists():
            return p
    base = hazy_stem.split("_")[0]
    for ext in IMAGE_EXTS:
        p = gt_dir / f"{base}{ext}"
        if p.exists():
            return p
    for p in gt_dir.glob(f"{base}.*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def center_crop_tensor(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[..., top : top + target_h, left : left + target_w]


def align_pair_for_metric(a: torch.Tensor, b: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ah, aw = a.shape[-2], a.shape[-1]
    bh, bw = b.shape[-2], b.shape[-1]
    if (ah, aw) == (bh, bw):
        return a, b
    if mode == "none":
        raise RuntimeError(f"Metric size mismatch: pred=({ah},{aw}) vs gt=({bh},{bw})")
    if mode == "resize":
        b = torch.nn.functional.interpolate(b, size=(ah, aw), mode="bilinear", align_corners=False)
        return a, b
    th, tw = min(ah, bh), min(aw, bw)
    return center_crop_tensor(a, th, tw), center_crop_tensor(b, th, tw)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hazy_path", type=str, required=True, help="single hazy image path, e.g. SOTS/indoor/hazy/1400_10.png")
    p.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint folder containing model .pth files")
    p.add_argument("--sots_root", type=str, default="SOTS/indoor", help="SOTS root containing hazy/ and gt/ (or clear/)")
    p.add_argument("--gt_dir", type=str, default=None, help="override GT folder path")
    p.add_argument("--out_dir", type=str, default="results/sots_oneclick", help="output folder")
    p.add_argument("--models", type=str, default="all", help="comma-separated model names or 'all'")
    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    p.add_argument("--resize", type=int, nargs=2, default=None, help="H W")
    p.add_argument("--keep_aspect", action="store_true", help="used with --resize")
    p.add_argument("--pad_multiple", type=int, default=16, help="pad input to multiple before model forward")
    p.add_argument("--metric_align", type=str, default="crop", choices=["crop", "resize", "none"], help="how to align pred/gt when sizes mismatch")
    p.add_argument("--grid_name", type=str, default="oneclick_grid.png", help="grid image filename")
    p.add_argument("--font", type=str, default=None, help="ttf font path")
    p.add_argument("--strict_ckpt", action="store_true", help="raise error when checkpoint for a model is missing")
    return p


def main():
    args = build_parser().parse_args()

    hazy_path = Path(args.hazy_path)
    if not hazy_path.exists() or not hazy_path.is_file():
        raise FileNotFoundError(f"hazy_path not found: {hazy_path}")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    sots_root = Path(args.sots_root)
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        gt_dir = sots_root / "gt"
        if not gt_dir.exists():
            alt = sots_root / "clear"
            gt_dir = alt if alt.exists() else None

    model_names = parse_name_list(args.models, lower=True)
    if not model_names or args.models.lower() == "all":
        model_names = list(MODEL_BUILDERS.keys())
    else:
        model_names = [canonical_model_name(m) for m in model_names]
    unknown = [m for m in model_names if m not in MODEL_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[Info] device = {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_raw = out_dir / "outputs"
    out_raw.mkdir(parents=True, exist_ok=True)

    hazy_img = Image.open(str(hazy_path)).convert("RGB")
    size = (int(args.resize[1]), int(args.resize[0])) if args.resize else None
    if size:
        hazy_img = resize_pil(hazy_img, size, args.keep_aspect, (255, 255, 255))
    hazy_t = to_tensor(hazy_img).to(device)

    gt_img = None
    gt_t = None
    gt_path = find_gt_for_sots(gt_dir, hazy_path.stem)
    if gt_path is not None:
        gt_img = Image.open(str(gt_path)).convert("RGB")
        if size:
            gt_img = resize_pil(gt_img, size, args.keep_aspect, (255, 255, 255))
        gt_t = to_tensor(gt_img).to(device)
        print(f"[GT] matched: {gt_path}")
    else:
        print("[GT] not found, metrics/GT column will be skipped")

    models = {}
    for name in model_names:
        ckpt = find_ckpt_for_model(name, ckpt_dir)
        if ckpt is None:
            msg = f"[CKPT] {name}: not found in {ckpt_dir}"
            if args.strict_ckpt:
                raise FileNotFoundError(msg)
            print(msg + ", skipped")
            continue
        model = MODEL_BUILDERS[name]().to(device).eval()
        print(f"[CKPT] loading {name}: {ckpt}")
        load_checkpoint(model, ckpt, name)
        models[name] = model

    if not models:
        raise RuntimeError("No model was loaded. Check ckpt_dir or models.")

    row_images = [hazy_img]
    labels = ["Hazy"]
    metric_texts = [""]
    if gt_t is not None:
        hz_m, gt_m = align_pair_for_metric(hazy_t, gt_t, args.metric_align)
        metric_texts[0] = format_metric(psnr(hz_m, gt_m), ssim_metric(hz_m, gt_m))

    for name, model in models.items():
        with torch.no_grad():
            x_in, pads = pad_to_multiple(hazy_t, args.pad_multiple)
            out = model(x_in)
            y = extract_dehazed(out)
            y = unpad_tensor(y, pads).clamp(0.0, 1.0)

        img = to_pil(y)
        if size:
            img = resize_pil(img, size, args.keep_aspect, (255, 255, 255))
        row_images.append(img)
        labels.append(MODEL_LABELS.get(name, name))

        model_dir = out_raw / name
        model_dir.mkdir(parents=True, exist_ok=True)
        img.save(model_dir / f"{hazy_path.stem}.png")

        if gt_t is not None:
            y_m, gt_m = align_pair_for_metric(y, gt_t, args.metric_align)
            metric_texts.append(format_metric(psnr(y_m, gt_m), ssim_metric(y_m, gt_m)))
        else:
            metric_texts.append("")

    (out_raw / "hazy").mkdir(parents=True, exist_ok=True)
    hazy_img.save(out_raw / "hazy" / f"{hazy_path.stem}.png")

    if gt_img is not None:
        gt_vis = gt_img if gt_img.size == hazy_img.size else gt_img.resize(hazy_img.size, Image.BILINEAR)
        row_images.append(gt_vis)
        labels.append("GT")
        metric_texts.append("inf / 1")
        (out_raw / "gt").mkdir(parents=True, exist_ok=True)
        gt_img.save(out_raw / "gt" / f"{hazy_path.stem}.png")

    # Build single-row journal grid.
    bg = (255, 255, 255)
    metric_h, label_h, pad, margin = 40, 42, 8, 12
    cell_w, cell_h = row_images[0].size
    n_cols = len(row_images)
    total_w = margin * 2 + n_cols * cell_w + (n_cols - 1) * pad
    total_h = margin * 2 + metric_h + cell_h + label_h
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(args.font, size=20)
    metric_font = load_font(args.font, size=18)

    for col, img in enumerate(row_images):
        x0 = margin + col * (cell_w + pad)
        canvas.paste(img, (x0, margin + metric_h))
        draw_centered(draw, metric_texts[col], (x0 + cell_w // 2, margin + metric_h // 2), metric_font)
        draw_centered(draw, labels[col], (x0 + cell_w // 2, margin + metric_h + cell_h + label_h // 2), label_font)

    grid_path = out_dir / args.grid_name
    canvas.save(grid_path)
    print(f"[Done] grid saved: {grid_path}")
    print(f"[Done] per-model outputs saved under: {out_raw}")


if __name__ == "__main__":
    main()
