#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ColorAwareUNet inference script for SOTS-style datasets.

Features:
- single image or folder inference
- robust checkpoint loading for dehazer-only / joint-training checkpoints
- optional GT matching for SOTS naming (e.g. 1400_10 -> 1400)
- optional PSNR/SSIM report
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Guard invalid OMP settings to avoid libgomp runtime warnings.
if "OMP_NUM_THREADS" in os.environ:
    try:
        if int(os.environ["OMP_NUM_THREADS"]) <= 0:
            raise ValueError
    except Exception:
        os.environ["OMP_NUM_THREADS"] = "1"
        print("[Warn] OMP_NUM_THREADS is invalid, fallback to 1")

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils.ColorAwareUnet import ColorAwareUNet


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(hazy_path: Optional[Path], hazy_dir: Optional[Path]) -> List[Path]:
    if hazy_path is not None:
        if not hazy_path.exists() or not hazy_path.is_file():
            raise FileNotFoundError(f"hazy_path not found: {hazy_path}")
        return [hazy_path]
    if hazy_dir is None or not hazy_dir.exists():
        raise FileNotFoundError("hazy_dir not found")
    files = [p for p in hazy_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.stem)
    if not files:
        raise RuntimeError("No images found in hazy_dir")
    return files


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def _extract_dehazer_from_model_state(st: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    prefix_candidates = ["dehazer.", "model.dehazer.", "module.dehazer.", "module.model.dehazer."]
    for pref in prefix_candidates:
        sub = {k[len(pref):]: v for k, v in st.items() if k.startswith(pref)}
        if sub:
            return sub
    return None


def resolve_checkpoint_state(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "dehazer_state" in ckpt and isinstance(ckpt["dehazer_state"], dict):
            state = ckpt["dehazer_state"]
        elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            st = ckpt["model_state"]
            sub = _extract_dehazer_from_model_state(st)
            state = sub if sub else st
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            st = ckpt["state_dict"]
            sub = _extract_dehazer_from_model_state(st)
            state = sub if sub else st
        else:
            state = ckpt
    else:
        raise TypeError("Checkpoint object is not a dict")

    if not isinstance(state, dict):
        raise TypeError("Resolved state is not a dict")

    # Common wrapper prefixes from DDP / wrappers.
    for pref in ("module.", "model.", "dehazer."):
        state = _strip_prefix(state, pref)
    return state


def load_coloraware_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        meta = []
        for k in ("epoch", "psnr", "ssim", "miou"):
            if k in ckpt:
                v = ckpt[k]
                if isinstance(v, float):
                    meta.append(f"{k}={v:.4f}")
                else:
                    meta.append(f"{k}={v}")
        if meta:
            print(f"[CKPT] Meta: {', '.join(meta)}")
    state = resolve_checkpoint_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    print(f"[CKPT] Loaded params: {loaded}/{len(model.state_dict())}")
    if missing:
        print(f"[CKPT] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[CKPT] Unexpected keys: {len(unexpected)}")


def resize_pil(img: Image.Image, size: Optional[Tuple[int, int]], keep_aspect: bool) -> Image.Image:
    if size is None:
        return img
    if not keep_aspect:
        return img.resize(size, Image.BILINEAR)
    w, h = img.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    out = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    canvas.paste(out, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(img).unsqueeze(0).to(device)


def to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    return TF.to_pil_image(x)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    if multiple <= 1:
        return x, (0, 0, 0, 0)
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    mode = "reflect" if h >= 2 and w >= 2 else "replicate"
    return F.pad(x, (left, right, top, bottom), mode=mode), (left, right, top, bottom)


def unpad(x: torch.Tensor, pads: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pads
    h_end = x.shape[-2] - bottom if bottom > 0 else x.shape[-2]
    w_end = x.shape[-1] - right if right > 0 else x.shape[-1]
    return x[..., top:h_end, left:w_end]


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2)
    v = 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))
    return float(v.item())


def ssim_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    a_mean = a.mean(dim=[1, 2, 3], keepdim=True)
    b_mean = b.mean(dim=[1, 2, 3], keepdim=True)
    a_var = ((a - a_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    b_var = ((b - b_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    cov = ((a - a_mean) * (b - b_mean)).mean(dim=[1, 2, 3], keepdim=True)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * a_mean * b_mean + c1) * (2 * cov + c2)) / ((a_mean ** 2 + b_mean ** 2 + c1) * (a_var + b_var + c2))
    return float(ssim_map.mean().item())


def find_gt(gt_dir: Optional[Path], hazy_stem: str) -> Optional[Path]:
    if gt_dir is None or not gt_dir.exists():
        return None
    # 1) exact stem match
    for ext in IMAGE_EXTS:
        p = gt_dir / f"{hazy_stem}{ext}"
        if p.exists():
            return p
    # 2) SOTS common pattern: 1400_10 -> 1400
    base = hazy_stem.split("_")[0]
    if base:
        for ext in IMAGE_EXTS:
            p = gt_dir / f"{base}{ext}"
            if p.exists():
                return p
    # 3) fallback wildcard
    for p in gt_dir.glob(f"{base}.*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            return p
    return None


def extract_output(y):
    if isinstance(y, (tuple, list)) and len(y) > 0:
        return y[0]
    if isinstance(y, dict) and "out" in y:
        return y["out"]
    return y


def center_crop_tensor(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    if target_h > h or target_w > w:
        raise ValueError(f"target crop {(target_h, target_w)} exceeds input {(h, w)}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[..., top : top + target_h, left : left + target_w]


def align_pair_for_metric(
    pred_t: torch.Tensor,
    gt_t: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ph, pw = pred_t.shape[-2], pred_t.shape[-1]
    gh, gw = gt_t.shape[-2], gt_t.shape[-1]
    if ph == gh and pw == gw:
        return pred_t, gt_t
    if mode == "none":
        raise RuntimeError(
            f"Metric size mismatch: pred=({ph},{pw}) vs gt=({gh},{gw}). "
            "Use --resize, or set --metric_align crop/resize."
        )
    if mode == "resize":
        gt_t = F.interpolate(gt_t, size=(ph, pw), mode="bilinear", align_corners=False)
        return pred_t, gt_t
    # mode == "crop": center-crop both to common minimum region
    th, tw = min(ph, gh), min(pw, gw)
    return center_crop_tensor(pred_t, th, tw), center_crop_tensor(gt_t, th, tw)


def make_compare_strip(hazy: Image.Image, pred: Image.Image, gt: Optional[Image.Image]) -> Image.Image:
    # For stable layout, bring all panes to hazy resolution when needed.
    base_w, base_h = hazy.size
    pred_v = pred if pred.size == (base_w, base_h) else pred.resize((base_w, base_h), Image.BILINEAR)
    if gt is None:
        strip = Image.new("RGB", (base_w * 2, base_h), (255, 255, 255))
        strip.paste(hazy, (0, 0))
        strip.paste(pred_v, (base_w, 0))
        return strip
    gt_v = gt if gt.size == (base_w, base_h) else gt.resize((base_w, base_h), Image.BILINEAR)
    strip = Image.new("RGB", (base_w * 3, base_h), (255, 255, 255))
    strip.paste(hazy, (0, 0))
    strip.paste(pred_v, (base_w, 0))
    strip.paste(gt_v, (base_w * 2, 0))
    return strip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hazy_path", type=str, default=None, help="single hazy image path")
    p.add_argument("--hazy_dir", type=str, default="SOTS/indoor/hazy", help="hazy image folder")
    p.add_argument("--gt_dir", type=str, default="SOTS/indoor/gt", help="GT folder (optional)")
    p.add_argument("--ckpt", type=str, required=True, help="path to coloraware checkpoint")
    p.add_argument("--out_dir", type=str, default="results/sots_coloraware", help="output folder")
    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    p.add_argument("--resize", type=int, nargs=2, default=None, help="H W")
    p.add_argument("--keep_aspect", action="store_true", help="used with --resize")
    p.add_argument("--pad_multiple", type=int, default=16, help="pad to multiple before inference")
    p.add_argument("--metric_align", type=str, default="crop", choices=["crop", "resize", "none"],
                   help="when pred/gt size mismatch: center-crop both, resize gt to pred, or raise")
    p.add_argument("--save_compare", action="store_true", help="save hazy|pred|gt strip")
    p.add_argument("--limit", type=int, default=0, help="limit number of images")
    return p


def main():
    args = build_parser().parse_args()

    hazy_path = Path(args.hazy_path) if args.hazy_path else None
    hazy_dir = Path(args.hazy_dir) if args.hazy_dir else None
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[Info] device = {device}")

    model = ColorAwareUNet(in_ch=3, base_ch=32, residual_scale=0.5, gain_scale=0.65).to(device).eval()
    print(f"[CKPT] Loading: {ckpt_path}")
    load_coloraware_checkpoint(model, ckpt_path)

    files = list_images(hazy_path, hazy_dir)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    out_dir = Path(args.out_dir)
    out_pred = out_dir / "dehazed"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_cmp = out_dir / "compare"
    if args.save_compare:
        out_cmp.mkdir(parents=True, exist_ok=True)

    size = None
    if args.resize:
        size = (int(args.resize[1]), int(args.resize[0]))

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    input_psnr_vals: List[float] = []
    input_ssim_vals: List[float] = []
    size_warned = False

    for i, hp in enumerate(files, 1):
        hazy = Image.open(str(hp)).convert("RGB")
        hazy = resize_pil(hazy, size, args.keep_aspect)
        x = to_tensor(hazy, device)

        with torch.no_grad():
            x_in, pads = pad_to_multiple(x, args.pad_multiple)
            y = model(x_in)
            y = extract_output(y)
            y = unpad(y, pads).clamp(0.0, 1.0)

        pred = to_pil(y)
        pred_path = out_pred / f"{hp.stem}.png"
        pred.save(pred_path)

        gt_path = find_gt(gt_dir, hp.stem)
        metric_text = ""
        if gt_path is not None:
            gt = Image.open(str(gt_path)).convert("RGB")
            gt = resize_pil(gt, size, args.keep_aspect)
            gt_t = to_tensor(gt, device)

            x_m, gt_in_m = align_pair_for_metric(x, gt_t, args.metric_align)
            in_p = psnr(x_m, gt_in_m)
            in_s = ssim_metric(x_m, gt_in_m)
            input_psnr_vals.append(in_p)
            input_ssim_vals.append(in_s)

            y_m, gt_m = align_pair_for_metric(y, gt_t, args.metric_align)
            if (y.shape[-2:] != gt_t.shape[-2:]) and (not size_warned):
                size_warned = True
                print(
                    f"[Warn] pred/gt size mismatch detected (pred={tuple(y.shape[-2:])}, gt={tuple(gt_t.shape[-2:])}), "
                    f"metric_align={args.metric_align}"
                )
            p = psnr(y_m, gt_m)
            s = ssim_metric(y_m, gt_m)
            psnr_vals.append(p)
            ssim_vals.append(s)
            metric_text = f" | IN:{in_p:.2f}/{in_s:.4f} -> OUT:{p:.2f}/{s:.4f} (dPSNR={p - in_p:+.2f})"

            if args.save_compare:
                strip = make_compare_strip(hazy, pred, gt)
                strip.save(out_cmp / f"{hp.stem}.png")
        elif args.save_compare:
            strip = make_compare_strip(hazy, pred, None)
            strip.save(out_cmp / f"{hp.stem}.png")

        print(f"[{i}/{len(files)}] {hp.name} -> {pred_path}{metric_text}")

    print(f"[Done] saved dehazed images to: {out_pred}")
    if psnr_vals:
        print(f"[Metric] Avg PSNR={sum(psnr_vals)/len(psnr_vals):.2f}, Avg SSIM={sum(ssim_vals)/len(ssim_vals):.4f}")
        print(
            f"[Metric] Avg Input PSNR={sum(input_psnr_vals)/len(input_psnr_vals):.2f}, "
            f"Avg Input SSIM={sum(input_ssim_vals)/len(input_ssim_vals):.4f}, "
            f"Avg dPSNR={(sum(psnr_vals)-sum(input_psnr_vals))/len(psnr_vals):+.2f}"
        )
    else:
        print("[Metric] GT not found or unmatched, skipped PSNR/SSIM.")


if __name__ == "__main__":
    main()
