#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic dehaze inference + journal-style grid builder for all dehaze nets in utils/.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Some environments set a non-numeric OMP_NUM_THREADS; guard it to avoid libgomp warnings.
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
from utils.AODNet import AODNet
from utils.DehazeNet import DehazeNet
from utils.GridDehazeNet import GridDehazeNet
from utils.Dehamer import DehamerNet
from utils.FFANet import FFANet
from utils.C2PNet import C2PNet
from utils.MSBDN import MSBDN
from utils.DCP import DCPDehaze
from utils.DehazeUnet import DehazeUNet
from utils.D4 import D4DehazeNet
from utils.PSD import PSDDehazeNet


MODEL_BUILDERS = {
    "coloraware": lambda: ColorAwareUNet(in_ch=3, base_ch=32, residual_scale=0.5, gain_scale=0.65),
    "dehazenet": lambda: DehazeNet(in_ch=3, maxout_k=2, guided_filter_refine=False, t0=0.05),
    "aodnet": lambda: AODNet(in_ch=3, mid_ch=8, use_tanh_on_K=False, b=1.0),
    "grid": lambda: GridDehazeNet(in_ch=3, rows=3, cols=6, base_ch=32),
    "dehamer": lambda: DehamerNet(in_ch=3, base_ch=16, trans_dim=64, nheads=4, n_layers=2, patch_size=8, use_learnable_pos=True),
    "ffanet": lambda: FFANet(in_ch=3, base_ch=32, n_down=2, n_ffab_deep=2),
    "c2pnet": lambda: C2PNet(in_ch=3, base_ch=32, blocks_per_group=6, groups=3, use_pdu=True),
    "msbdn": lambda: MSBDN(in_ch=3, base_ch=32, nblocks=3),
    "dcp": lambda: DCPDehaze(in_ch=3, patch_size=15, omega=0.95, t0=0.1, top_percent=0.001,
                              guided=True, guided_radius=15, guided_eps=1e-3, use_learned_refine=True),
    "dehazeunet": lambda: DehazeUNet(in_ch=3, base_ch=32),
    "d4": lambda: D4DehazeNet(in_ch=3, base_ch=32),
    "psd": lambda: PSDDehazeNet(in_ch=3, base_ch=32, feat_ch=64),
}

MODEL_LABELS = {
    "coloraware": "ColorAwareUNet",
    "dehazenet": "DehazeNet",
    "aodnet": "AODNet",
    "grid": "GridDehazeNet",
    "dehamer": "Dehamer",
    "ffanet": "FFA-Net",
    "c2pnet": "C2PNet",
    "msbdn": "MSBDN",
    "dcp": "DCP",
    "dehazeunet": "DehazeUNet",
    "d4": "D4",
    "psd": "PSD",
}

MODEL_CKPT_ALIASES = {
    "coloraware": ["coloraware", "colorawareunet"],
    "dehazenet": ["dehazenet"],
    "aodnet": ["aodnet"],
    "grid": ["grid", "griddehazenet"],
    "dehamer": ["dehamer"],
    "ffanet": ["ffanet", "ffanetnet"],
    "c2pnet": ["c2pnet"],
    "msbdn": ["msbdn"],
    "dcp": ["dcp"],
    "dehazeunet": ["dehazeunet"],
    "d4": ["d4"],
    "psd": ["psd"],
}

MODEL_NAME_ALIASES = {
    "colorawareunet": "coloraware",
    "griddehazenet": "grid",
    "ffa-net": "ffanet",
}

CKPT_SUFFIXES = {"best", "last", "final", "latest", "epoch", "weights"}


def normalize_token(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def parse_name_list(s: Optional[str], lower: bool = False) -> Optional[List[str]]:
    if not s:
        return None
    out = [x.strip() for x in s.split(",") if x.strip()]
    if lower:
        out = [x.lower() for x in out]
    return out


def list_images(hazy_dir: Path, ids: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if ids:
        items = []
        for sid in ids:
            p = find_by_stem(hazy_dir, sid, exts)
            if p:
                items.append((sid, p))
        return items
    files = [p for p in hazy_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    files.sort(key=lambda p: p.stem)
    return [(p.stem, p) for p in files]


def find_by_stem(root: Path, stem: str, exts: Optional[set] = None) -> Optional[Path]:
    if exts is None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for ext in exts:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    for p in root.glob(f"{stem}.*"):
        if p.suffix.lower() in exts:
            return p
    return None


def resize_pil(img: Image.Image, size: Optional[Tuple[int, int]], keep_aspect: bool, bg: Tuple[int, int, int]) -> Image.Image:
    if size is None:
        return img
    if not keep_aspect:
        return img.resize(size, Image.BILINEAR)
    w, h = img.size
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", size, bg)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas.paste(resized, (x0, y0))
    return canvas


def to_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img).unsqueeze(0)
    return t


def to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0.0, 1.0)
    if t.ndim == 4:
        t = t[0]
    return TF.to_pil_image(t)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2)
    val = 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))
    return float(val.item())


def ssim_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    a_mean = a.mean(dim=[1, 2, 3], keepdim=True)
    b_mean = b.mean(dim=[1, 2, 3], keepdim=True)
    a_var = ((a - a_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    b_var = ((b - b_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    cov = ((a - a_mean) * (b - b_mean)).mean(dim=[1, 2, 3], keepdim=True)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * a_mean * b_mean + c1) * (2 * cov + c2)) / ((a_mean ** 2 + b_mean ** 2 + c1) * (a_var + b_var + c2))
    val = ssim_map.mean()
    return float(val.item())


def extract_dehazed(out):
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return out[0]
    if isinstance(out, dict):
        if "out" in out:
            return out["out"]
    return out


def _strip_prefix_once(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def _extract_prefixed_substate(
    state: Dict[str, torch.Tensor],
    prefixes: List[str],
) -> Optional[Dict[str, torch.Tensor]]:
    for pref in prefixes:
        pref_dot = f"{pref}."
        picked = {k[len(pref_dot):]: v for k, v in state.items() if k.startswith(pref_dot)}
        if picked:
            return picked
    return None


def _choose_best_state_for_model(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    model_name: str,
) -> Dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())

    aliases = [model_name]
    aliases += MODEL_CKPT_ALIASES.get(model_name, [])
    label = MODEL_LABELS.get(model_name)
    if label:
        aliases.append(label)
    alias_tokens = [normalize_token(x) for x in aliases if x]

    # Candidate 1: remove common wrappers (module/model/dehazer) globally.
    stripped = dict(state)
    for p in ("module.", "model.", "dehazer."):
        stripped = _strip_prefix_once(stripped, p)

    candidates = [state, stripped]

    # Candidate 2: keys nested under model-specific prefixes.
    pref_variants = []
    for a in aliases:
        low = a.lower()
        pref_variants.extend([low, f"module.{low}", f"model.{low}", f"dehazer.{low}"])
    sub = _extract_prefixed_substate(state, pref_variants)
    if sub:
        candidates.append(sub)

    # Candidate 3: segment-based extraction for keys like "dehazer.colorawareunet.enc1.conv.weight".
    segmented = {}
    for k, v in state.items():
        parts = k.split(".")
        for idx, part in enumerate(parts[:-1]):
            if normalize_token(part) in alias_tokens:
                nk = ".".join(parts[idx + 1 :])
                if nk:
                    segmented[nk] = v
                break
    if segmented:
        candidates.append(segmented)

    # Pick candidate with most matching keys and fewest extras.
    best = state
    best_score = (-1, float("inf"))
    for cand in candidates:
        keys = set(cand.keys())
        matched = len(model_keys & keys)
        extras = len(keys - model_keys)
        score = (matched, -extras)
        if score > best_score:
            best_score = score
            best = cand
    return best


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, model_name: str) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "dehazer_state" in ckpt:
        state = ckpt["dehazer_state"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if isinstance(state, dict):
        state = _choose_best_state_for_model(model, state, model_name)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[CKPT] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[CKPT] Unexpected keys: {len(unexpected)}")
        loaded = len(model.state_dict()) - len(missing)
        print(f"[CKPT] Loaded params: {loaded}/{len(model.state_dict())}")


def pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
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
    if mode == "reflect" and (h < 2 or w < 2):
        mode = "replicate"
    return F.pad(x, (left, right, top, bottom), mode=mode), (left, right, top, bottom)


def unpad_tensor(x: torch.Tensor, pads: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pads
    h_end = x.shape[-2] - bottom if bottom > 0 else x.shape[-2]
    w_end = x.shape[-1] - right if right > 0 else x.shape[-1]
    return x[..., top:h_end, left:w_end]


def load_font(path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if path:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    for name in ("Times New Roman.ttf", "times.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, text: str, center: Tuple[int, int], font, fill=(0, 0, 0)):
    if not text:
        return
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)
    x = center[0] - w // 2
    y = center[1] - h // 2
    draw.text((x, y), text, font=font, fill=fill)


def format_metric(psnr_val: Optional[float], ssim_val: Optional[float]) -> str:
    if psnr_val is None or ssim_val is None:
        return ""
    return f"{psnr_val:.2f} / {ssim_val:.4f}"


def resolve_ckpt_map(args) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if args.ckpt_map:
        with open(args.ckpt_map, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            mapping[k.lower()] = v
    if args.ckpt:
        for item in args.ckpt:
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            mapping[k.strip().lower()] = v.strip()
    return mapping


def canonical_model_name(name: str) -> str:
    key = name.strip().lower()
    key = MODEL_NAME_ALIASES.get(key, key)
    return key


def score_ckpt_name(file_stem: str, candidates: List[str]) -> int:
    stem = normalize_token(file_stem)
    best = -1
    for cand in candidates:
        c = normalize_token(cand)
        if not c:
            continue
        if stem == c:
            best = max(best, 100)
            continue
        if stem in {f"{c}best", f"best{c}", f"{c}last", f"last{c}", f"{c}final", f"final{c}"}:
            best = max(best, 95)
            continue
        if stem.startswith(c) or stem.endswith(c):
            # allow filenames like "colorawareunet_epoch10"
            tail = stem[len(c):] if stem.startswith(c) else stem[: -len(c)]
            if not tail:
                best = max(best, 100)
            elif any(tag in tail for tag in CKPT_SUFFIXES) or any(ch.isdigit() for ch in tail):
                best = max(best, 90)
            else:
                best = max(best, 80)
            continue
        if c in stem:
            best = max(best, 60)
    return best


def find_ckpt_for_model(name: str, ckpt_dir: Optional[Path]) -> Optional[Path]:
    if not ckpt_dir or not ckpt_dir.exists():
        return None
    candidates = [name]
    candidates += MODEL_CKPT_ALIASES.get(name, [])
    label = MODEL_LABELS.get(name)
    if label:
        candidates.append(label)

    files = [p for p in ckpt_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pth", ".pt", ".ckpt"}]
    if not files:
        return None

    ranked = []
    for p in files:
        score = score_ckpt_name(p.stem, candidates)
        if score >= 0:
            ranked.append((score, len(p.parts), p))
    if not ranked:
        return None
    ranked.sort(key=lambda x: (-x[0], x[1], x[2].name.lower()))
    return ranked[0][2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="root with hazy/clear subfolders")
    parser.add_argument("--hazy_path", type=str, default=None, help="single hazy image path")
    parser.add_argument("--hazy_dir", type=str, default=None, help="path to hazy images")
    parser.add_argument("--clear_dir", type=str, default=None, help="path to clear/GT images")
    parser.add_argument("--out_dir", type=str, default="results/journal", help="output folder")
    parser.add_argument("--grid_name", type=str, default="journal_grid.png", help="grid image name")
    parser.add_argument("--models", type=str, default="all", help="comma-separated model names or 'all'")
    parser.add_argument("--ckpt_map", type=str, default=None, help="json mapping: model -> checkpoint path")
    parser.add_argument("--ckpt", action="append", default=[], help="model=path (repeatable)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="folder to auto-search checkpoints")
    parser.add_argument("--ids", type=str, default=None, help="comma-separated image stems")
    parser.add_argument("--limit", type=int, default=0, help="limit number of images (0 = all)")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="H W resize for input/grid")
    parser.add_argument("--keep_aspect", action="store_true", help="keep aspect ratio with padding")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--save_individual", action="store_true", help="save per-model outputs")
    parser.add_argument("--pad_multiple", type=int, default=16, help="pad input to multiple before model forward (0/1 disables)")
    parser.add_argument("--font", type=str, default=None, help="ttf font path")
    parser.add_argument("--metric_h", type=int, default=40)
    parser.add_argument("--label_h", type=int, default=42)
    parser.add_argument("--pad", type=int, default=8)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--row_gap", type=int, default=16)
    args = parser.parse_args()

    single_hazy = Path(args.hazy_path) if args.hazy_path else None

    if single_hazy is not None:
        if not single_hazy.exists() or not single_hazy.is_file():
            raise FileNotFoundError(f"hazy_path not found: {single_hazy}")
        hazy_dir = single_hazy.parent
        clear_dir = Path(args.clear_dir) if args.clear_dir else None
    elif args.data_root and (args.hazy_dir is None):
        root = Path(args.data_root)
        hazy_dir = root / "hazy"
        clear_dir = root / "clear"
    else:
        hazy_dir = Path(args.hazy_dir) if args.hazy_dir else None
        clear_dir = Path(args.clear_dir) if args.clear_dir else None

    if hazy_dir is None or not hazy_dir.exists():
        raise FileNotFoundError("hazy_dir not found")

    if clear_dir and not clear_dir.exists():
        clear_dir = None

    model_names = parse_name_list(args.models, lower=True)
    if not model_names or args.models.lower() == "all":
        model_names = list(MODEL_BUILDERS.keys())
    else:
        model_names = [canonical_model_name(m) for m in model_names]

    unknown = [m for m in model_names if m not in MODEL_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    ckpt_map = resolve_ckpt_map(args)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if single_hazy is not None:
        items = [(single_hazy.stem, single_hazy)]
    else:
        ids = parse_name_list(args.ids, lower=False)
        items = list_images(hazy_dir, ids=ids)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No hazy images found")

    if clear_dir:
        missing = [stem for stem, _ in items if not find_by_stem(clear_dir, stem)]
        if missing:
            print(f"[Warn] missing GT for {len(missing)} images, disable GT/metrics")
            clear_dir = None

    models = {}
    for name in model_names:
        model = MODEL_BUILDERS[name]()
        ckpt_path = None
        if name in ckpt_map:
            ckpt_path = Path(ckpt_map[name])
        elif ckpt_dir:
            ckpt_path = find_ckpt_for_model(name, ckpt_dir)
        if ckpt_path and ckpt_path.exists():
            print(f"[CKPT] Loading {name}: {ckpt_path}")
            load_checkpoint(model, ckpt_path, name)
        else:
            print(f"[CKPT] {name}: no checkpoint, using random init")
        model.to(device).eval()
        models[name] = model

    labels = ["Hazy"]
    labels += [MODEL_LABELS.get(m, m) for m in model_names]
    if clear_dir:
        labels.append("GT")

    label_font = load_font(args.font, size=20)
    metric_font = load_font(args.font, size=18)

    bg = (255, 255, 255)

    sample0 = Image.open(str(items[0][1])).convert("RGB")
    size = None
    if args.resize:
        size = (int(args.resize[1]), int(args.resize[0]))
    if size:
        sample0 = resize_pil(sample0, size, args.keep_aspect, bg)
    cell_w, cell_h = sample0.size

    n_cols = len(labels)
    n_rows = len(items)
    row_h = args.metric_h + cell_h + args.label_h
    total_w = args.margin * 2 + n_cols * cell_w + (n_cols - 1) * args.pad
    total_h = args.margin * 2 + n_rows * row_h + (n_rows - 1) * args.row_gap
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    for row_idx, (stem, hazy_path) in enumerate(items):
        hazy_img = Image.open(str(hazy_path)).convert("RGB")
        clear_img = None
        if clear_dir:
            clear_path = find_by_stem(clear_dir, stem)
            if clear_path:
                clear_img = Image.open(str(clear_path)).convert("RGB")

        if size:
            hazy_img = resize_pil(hazy_img, size, args.keep_aspect, bg)
            if clear_img is not None:
                clear_img = resize_pil(clear_img, size, args.keep_aspect, bg)

        hazy_t = to_tensor(hazy_img).to(device)
        clear_t = to_tensor(clear_img).to(device) if clear_img is not None else None

        metric_texts = []
        if clear_t is not None:
            metric_texts.append(format_metric(psnr(hazy_t, clear_t), ssim_metric(hazy_t, clear_t)))
        else:
            metric_texts.append("")

        row_images = [hazy_img]

        for name in model_names:
            model = models[name]
            with torch.no_grad():
                model_in, pads = pad_to_multiple(hazy_t, args.pad_multiple)
                out = model(model_in)
                dehazed = extract_dehazed(out)
                dehazed = unpad_tensor(dehazed, pads)
                dehazed = torch.clamp(dehazed, 0.0, 1.0)
            if clear_t is not None:
                metric_texts.append(format_metric(psnr(dehazed, clear_t), ssim_metric(dehazed, clear_t)))
            else:
                metric_texts.append("")
            out_img = to_pil(dehazed)
            if size:
                out_img = resize_pil(out_img, size, args.keep_aspect, bg)
            row_images.append(out_img)
            if args.save_individual:
                od = out_dir / "outputs" / name
                od.mkdir(parents=True, exist_ok=True)
                out_img.save(od / f"{stem}.png")

        if clear_img is not None:
            row_images.append(clear_img)
            metric_texts.append("inf / 1")

        y0 = args.margin + row_idx * (row_h + args.row_gap)
        for col_idx, img in enumerate(row_images):
            x0 = args.margin + col_idx * (cell_w + args.pad)
            canvas.paste(img, (x0, y0 + args.metric_h))
            metric = metric_texts[col_idx] if col_idx < len(metric_texts) else ""
            draw_centered(draw, metric, (x0 + cell_w // 2, y0 + args.metric_h // 2), metric_font)
            label = labels[col_idx] if col_idx < len(labels) else ""
            draw_centered(draw, label, (x0 + cell_w // 2, y0 + args.metric_h + cell_h + args.label_h // 2), label_font)

    grid_path = out_dir / args.grid_name
    canvas.save(grid_path)
    print(f"[Done] grid saved: {grid_path}")


if __name__ == "__main__":
    main()
