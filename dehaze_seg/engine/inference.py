"""Single-image and folder prediction, paired evaluation, and comparison grids."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from ..data.common import IMG_EXTS, find_by_stem, pil_to_rgb_tensor, mask_to_label_tensor
from ..data.datasets import split_ids
from ..metrics import confusion_matrix, segmentation_metrics
from ..models.joint import JointDehazeSegModel, unpack_dehaze_output
from ..utils import ensure_dir, select_device, write_csv, write_json
from .checkpoint import load_checkpoint
from .train import image_metrics


def list_images(source):
    source = Path(source)
    images = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise ValueError(f"No images under {source}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Duplicate input stems would overwrite outputs")
    return images


def tensor_image(tensor):
    array = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8))


@torch.no_grad()
def infer_image(model, image, device, resize=None):
    original_size = (image.height, image.width)
    if resize:
        image = image.resize((resize[1], resize[0]), Image.Resampling.BILINEAR)
    x = pil_to_rgb_tensor(image).unsqueeze(0).to(device)
    h, w = x.shape[-2:]
    ph, pw = max(32, ((h + 15) // 16) * 16) - h, max(32, ((w + 15) // 16) * 16) - w
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    if isinstance(model, JointDehazeSegModel):
        restored, logits, aux = model(x)
    else:
        restored, aux = unpack_dehaze_output(model(x))
        logits = None
    restored = restored[..., :h, :w]
    if restored.shape[-2:] != original_size:
        restored = F.interpolate(restored, size=original_size, mode="bilinear", align_corners=False)
    if logits is not None:
        logits = logits[..., :h, :w]
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
    return restored, logits, aux


def align_pair(pred, target, mode="crop", label=False):
    if pred.shape[-2:] == target.shape[-2:]:
        return pred, target
    if mode == "none":
        raise ValueError(f"Size mismatch {pred.shape[-2:]} vs {target.shape[-2:]}; select --metric-align crop or resize")
    if mode == "resize":
        target = F.interpolate(target.float(), size=pred.shape[-2:], mode="nearest" if label else "bilinear",
                               **({} if label else {"align_corners": False}))
        return pred, target.long() if label else target
    h, w = min(pred.shape[-2], target.shape[-2]), min(pred.shape[-1], target.shape[-1])
    def crop(t):
        y, x = (t.shape[-2] - h) // 2, (t.shape[-1] - w) // 2
        return t[..., y:y+h, x:x+w]
    return crop(pred), crop(target)


def paired_paths(args):
    root = Path(args.data_root)
    if args.dataset == "hsts":
        return root / "synthetic" / "synthetic", root / "synthetic" / "original", None
    clear = root / "clear"
    if not clear.is_dir():
        clear = root / "gt"
    masks = root / "masks" if args.dataset == "paired-road" else None
    return root / "hazy", clear, masks


def reference_path(directory, stem, dataset):
    path = find_by_stem(directory, stem)
    if path is None and dataset.startswith("sots-"):
        path = find_by_stem(directory, stem.split("_")[0])
    return path


def select_split(images, args, checkpoint_path, checkpoint):
    split = args.split or ("val" if args.dataset == "paired-road" else "all")
    if split == "all":
        return images
    split_file = Path(args.split_file) if args.split_file else None
    if split_file is None:
        for folder in (Path(checkpoint_path).parent, Path(checkpoint_path).parent.parent):
            if (folder / "split.json").is_file():
                split_file = folder / "split.json"
                break
    if split_file:
        record = json.loads(split_file.read_text(encoding="utf-8"))
        if record.get("dataset", args.dataset) != args.dataset:
            raise ValueError("Saved split belongs to another dataset; pass a matching --split-file or --split all")
        ids = record[split]
    else:
        saved = checkpoint.get("train_config", checkpoint.get("args", {}))
        train, val = split_ids([p.stem for p in images], saved.get("val_ratio", .15), saved.get("seed", 42))
        ids = val if split == "val" else train
        print("Reconstructing split from saved seed/ratio; no split manifest found.")
    lookup = {p.stem: p for p in images}
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("The selected split must contain unique, nonempty sample IDs")
    missing = set(ids) - lookup.keys()
    if missing:
        raise ValueError(f"Missing images from saved split: {sorted(missing)[:10]}")
    return [lookup[i] for i in ids]


def save_prediction(folder, stem, hazy, pred, logits, clear=None):
    output = tensor_image(pred)
    output.save(ensure_dir(folder / "dehazed") / f"{stem}.png")
    panels = [hazy, output]
    if logits is not None:
        labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        Image.fromarray(labels).save(ensure_dir(folder / "masks") / f"{stem}.png")
        base = np.asarray(output, dtype=np.float32)
        overlay = np.where((labels > 0)[..., None], base * .55 + np.array([255, 64, 64]) * .45, base)
        overlay = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
        overlay.save(ensure_dir(folder / "overlays") / f"{stem}.png")
        panels.append(overlay)
    if clear is not None:
        panels.append(clear.resize(hazy.size, Image.Resampling.BILINEAR))
    strip = Image.new("RGB", (hazy.width * len(panels), hazy.height))
    for i, panel in enumerate(panels):
        strip.paste(panel, (i * hazy.width, 0))
    strip.save(ensure_dir(folder / "compare") / f"{stem}.png")


def run_inference(args, evaluate=False):
    device = select_device(args.device)
    if evaluate:
        source, clear_dir, mask_dir = paired_paths(args)
    else:
        source, clear_dir, mask_dir = Path(args.input), None, None
    all_images = list_images(source)
    summaries, output_folders = [], []
    for index, checkpoint_path in enumerate(args.checkpoint):
        model, config, checkpoint = load_checkpoint(checkpoint_path, device, args.legacy_profile,
                                                     args.model, args.model_config)
        images = select_split(all_images, args, checkpoint_path, checkpoint) if evaluate else all_images
        if args.limit:
            images = images[:args.limit]
        folder = Path(args.output)
        if len(args.checkpoint) > 1:
            folder /= f"{index+1:02d}_{config['model']}_{Path(checkpoint_path).stem}"
        rows, cm = [], None
        for path in images:
            with Image.open(path) as image:
                hazy = image.convert("RGB")
            pred, logits, aux = infer_image(model, hazy, device, args.resize)
            row, clear_image = {"sample": path.stem}, None
            gain = aux.get("color_gain")
            if gain is not None:
                values = gain[0].reshape(3, -1).mean(1).cpu().tolist()
                row.update(dict(zip(("gain_r", "gain_g", "gain_b"), values)))
            if evaluate:
                cp = reference_path(clear_dir, path.stem, args.dataset)
                if cp is None:
                    raise ValueError(f"Missing clear reference for {path.name}")
                with Image.open(cp) as image:
                    clear_image = image.convert("RGB")
                target = pil_to_rgb_tensor(clear_image)[None].to(device)
                p, t = align_pair(pred, target, args.metric_align)
                row.update(image_metrics(p, t))
                if logits is not None and mask_dir is not None:
                    mp = find_by_stem(mask_dir, path.stem)
                    if mp is None:
                        raise ValueError(f"Missing segmentation mask for {path.name}")
                    with Image.open(mp) as image:
                        target_mask = mask_to_label_tensor(image, logits.shape[1])[None, None].to(device)
                    p, t = align_pair(logits.argmax(1, keepdim=True), target_mask, args.metric_align, label=True)
                    current = confusion_matrix(p, t, logits.shape[1])
                    cm = current if cm is None else cm + current
                    row.update({k: v for k, v in segmentation_metrics(current).items() if isinstance(v, float)})
            if not evaluate or args.save_images:
                save_prediction(folder, path.stem, hazy, pred, logits, clear_image)
            rows.append(row)
        summary = {"model": config["model"], "checkpoint": str(checkpoint_path), "count": len(rows)}
        for key in rows[0]:
            if key != "sample":
                summary[key] = sum(r[key] for r in rows) / len(rows)
        if cm is not None:
            summary.update(segmentation_metrics(cm))
        write_csv(folder / "metrics.csv", rows)
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        output_folders.append((config["model"], folder, images))
        print(json.dumps(summary, ensure_ascii=False))
        del model
    if len(summaries) > 1:
        write_csv(Path(args.output) / "comparison.csv", summaries)
        if not evaluate or args.save_images:
            save_model_grid(output_folders, Path(args.output) / "comparison")
    return summaries


def save_model_grid(outputs, folder):
    ensure_dir(folder)
    common = set.intersection(*({p.stem for p in images} for _, _, images in outputs))
    for stem in sorted(common):
        path = next(p for p in outputs[0][2] if p.stem == stem)
        with Image.open(path) as image:
            source = image.convert("RGB")
        panels = [("Hazy", source)]
        for name, directory, _ in outputs:
            with Image.open(directory / "dehazed" / f"{stem}.png") as image:
                panels.append((name, image.convert("RGB")))
        w = min(512, source.width)
        h = round(source.height * w / source.width)
        canvas = Image.new("RGB", (w * len(panels), h + 26), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (name, image) in enumerate(panels):
            canvas.paste(image.resize((w, h)), (i*w, 26))
            draw.text((i*w+8, 6), name, fill="black")
        canvas.save(folder / f"{stem}.png")
