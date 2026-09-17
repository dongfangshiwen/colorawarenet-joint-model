"""Single-image and folder prediction, paired evaluation, and comparison grids."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from ..data.common import IMG_EXTS, find_by_stem, pil_to_rgb_tensor, mask_to_label_tensor
from ..data.datasets import split_ids
from ..data.splits import (paired_directories, clear_reference, canonical_root, checkpoint_split,
                           training_overlap, sample_records, file_digest)
from ..metrics import confusion_matrix, segmentation_metrics, segmentation_metric_row
from ..models.joint import JointDehazeSegModel, unpack_dehaze_output
from ..models.registry import build_model, model_config
from ..models.DCP import ClassicalDCP
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
def infer_image(model, image, device, resize=None, return_to_original=True):
    original_size = (image.height, image.width)
    if resize:
        image = image.resize((resize[1], resize[0]), Image.Resampling.BILINEAR)
    output_size = original_size if return_to_original else (image.height, image.width)
    x = pil_to_rgb_tensor(image).unsqueeze(0).to(device)
    h, w = x.shape[-2:]
    ph, pw = max(32, ((h + 15) // 16) * 16) - h, max(32, ((w + 15) // 16) * 16) - w
    dehazer = model.dehazer if isinstance(model, JointDehazeSegModel) else model
    if isinstance(dehazer, ClassicalDCP):
        # DCP estimates global atmospheric light: neural-network padding would
        # change its candidate population, and hence the classical prediction.
        restored, aux = unpack_dehaze_output(dehazer(x))
        logits = (model.segment(F.pad(restored, (0, pw, 0, ph), mode="replicate"))
                  if isinstance(model, JointDehazeSegModel) else None)
    elif isinstance(model, JointDehazeSegModel):
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        restored, logits, aux = model(x)
    else:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        restored, aux = unpack_dehaze_output(model(x))
        logits = None
    restored = restored[..., :h, :w]
    if restored.shape[-2:] != output_size:
        restored = F.interpolate(restored, size=output_size, mode="bilinear", align_corners=False)
    if logits is not None:
        logits = logits[..., :h, :w]
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
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
    return paired_directories(args.dataset, args.data_root)


def reference_path(directory, stem, dataset):
    return clear_reference(directory, stem, dataset)


def select_split(images, args, checkpoint_path, checkpoint):
    saved = checkpoint.get("train_config", checkpoint.get("args", {}))
    record = (json.loads(Path(args.split_file).read_text(encoding="utf-8")) if args.split_file
              else checkpoint_split(checkpoint_path, checkpoint))
    matching = record is not None and record.get("dataset", args.dataset) == args.dataset
    train_root = record.get("train_root", saved.get("data_root")) if matching else saved.get("data_root")
    val_root = record.get("val_root", train_root) if matching else saved.get("val_root", train_root)
    root = canonical_root(args.data_root)
    on_train = bool(train_root) and root == canonical_root(train_root)
    on_val = bool(val_root) and root == canonical_root(val_root)
    split = args.split or ("val" if args.split_file or args.dataset == "paired-road" or on_train or on_val else "all")
    if split == "all":
        return images
    if record is not None and not matching:
        raise ValueError("Saved split belongs to another dataset; supply a matching --split-file")
    if split == "val" and on_train and val_root and canonical_root(val_root) != root:
        raise ValueError(f"This checkpoint used a separate validation root. Evaluate --data-root {val_root}")
    if record is not None:
        ids = record[split]
    else:
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


def use_shared_segmenter(model, shared):
    """Replace any per-method segmenter with one frozen, normalized evaluator."""
    if not isinstance(shared, JointDehazeSegModel):
        raise ValueError("--segmenter-checkpoint must contain a joint model with a segmenter")
    shared.segmenter.requires_grad_(False)
    shared.segmenter.eval()
    dehazer = model.dehazer if isinstance(model, JointDehazeSegModel) else model
    combined = JointDehazeSegModel(dehazer, shared.segmenter, shared.imagenet_norm).to(
        next(shared.segmenter.parameters()).device).eval()
    combined.imagenet_mean.copy_(shared.imagenet_mean)
    combined.imagenet_std.copy_(shared.imagenet_std)
    return combined


def save_prediction(folder, stem, hazy, pred, logits, clear=None):
    # Saved presentation images retain native size even when scores use the
    # model's inference grid. Never use these upsampled masks for those scores.
    size = (hazy.height, hazy.width)
    if pred.shape[-2:] != size:
        pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
    if logits is not None and logits.shape[-2:] != size:
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
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
    checkpoints = args.checkpoint or []
    include_dcp = args.include_dcp or (not checkpoints and args.model == "dcp")
    if not checkpoints and not include_dcp:
        raise ValueError("Supply --checkpoint or use --model dcp for parameter-free inference")
    if not checkpoints and (args.legacy_profile or args.model_config):
        raise ValueError("Legacy configuration options require --checkpoint")
    if not args.segmenter_checkpoint and (args.segmenter_legacy_profile or args.segmenter_model or args.segmenter_model_config):
        raise ValueError("Segmenter configuration options require --segmenter-checkpoint")
    device = select_device(args.device)
    if evaluate:
        source, clear_dir, mask_dir = paired_paths(args)
    else:
        source, clear_dir, mask_dir = Path(args.input), None, None
    all_images = list_images(source)
    shared, shared_info, shared_checkpoint = None, None, {}
    if args.segmenter_checkpoint:
        shared, shared_config, shared_checkpoint = load_checkpoint(
            args.segmenter_checkpoint, device, args.segmenter_legacy_profile,
            args.segmenter_model, args.segmenter_model_config)
        if not isinstance(shared, JointDehazeSegModel):
            raise ValueError("--segmenter-checkpoint must contain a joint model with a segmenter")
        shared.requires_grad_(False)
        shared_info = dict(checkpoint=canonical_root(args.segmenter_checkpoint),
                           sha256=file_digest(args.segmenter_checkpoint),
                           config=shared_config["segmenter"], imagenet_norm=shared_config["imagenet_norm"],
                           frozen=True)
        # Only retain the shared evaluator, not a duplicate restoration network.
        shared.dehazer = torch.nn.Identity()
    images, eval_records = None, None
    summaries, output_folders = [], []
    sources = list(checkpoints) + ([None] if include_dcp else [])
    for index, checkpoint_path in enumerate(sources):
        if checkpoint_path is None:
            config, checkpoint = model_config("dcp", joint=False), {}
            model = build_model(config).to(device).eval()
        else:
            model, config, checkpoint = load_checkpoint(checkpoint_path, device, args.legacy_profile,
                                                         args.model, args.model_config)
        # Select once, so every method sees exactly the same ordered samples.
        if images is None:
            shared_split = checkpoint_split(args.segmenter_checkpoint, shared_checkpoint)
            shared_has_split = (evaluate and shared_split is not None and
                                shared_split.get("dataset", args.dataset) == args.dataset)
            use_shared_split = bool(evaluate and shared is not None and
                                    (args.dataset == "paired-road" or shared_has_split))
            split_source = args.segmenter_checkpoint if use_shared_split else checkpoint_path
            split_checkpoint = shared_checkpoint if use_shared_split else checkpoint
            images = select_split(all_images, args, split_source, split_checkpoint) if evaluate else all_images
            if args.limit:
                images = images[:args.limit]
            if evaluate:
                eval_records = sample_records(args.dataset, args.data_root, [p.stem for p in images])
        overlap = set()
        if evaluate:
            for source_path, source_checkpoint in ((checkpoint_path, checkpoint),
                                                    (args.segmenter_checkpoint, shared_checkpoint)):
                overlap.update(training_overlap(args.dataset, args.data_root, [p.stem for p in images],
                                                source_path, source_checkpoint, eval_records))
            if overlap and not args.allow_training_overlap:
                raise ValueError(f"Evaluation overlaps known training samples/scenes: {sorted(overlap)[:10]}. "
                                 "Use an independent root or saved --split val. "
                                 "For an explicitly labelled diagnostic only, use --allow-training-overlap.")
        if shared is not None:
            model = use_shared_segmenter(model, shared)
        folder = Path(args.output)
        if len(sources) > 1:
            folder /= f"{index+1:02d}_{config['model']}_{Path(checkpoint_path).stem if checkpoint_path else 'classical'}"
        implementation = ("historical-enhanced" if config.get("dehazer_type") == "legacy-dcp"
                          else "learned-refinement" if config.get("dehazer_type") == "learned-dcp"
                          else "classical" if config["model"] == "dcp" else "network")
        protocol = dict(samples=[p.stem for p in images], model_config=config,
                        checkpoint_sha256=file_digest(checkpoint_path) if checkpoint_path else None,
                        shared_segmenter=shared_info,
                        segmentation="shared-frozen" if shared else "checkpoint" if config["joint"] else "none")
        score_at_inference = evaluate and args.metric_resolution == "inference"
        if evaluate:
            protocol.update(dataset=args.dataset, data_root=canonical_root(args.data_root),
                            references=eval_records, known_training_overlap=sorted(overlap),
                            allow_training_overlap=args.allow_training_overlap,
                            metric_align=args.metric_align, resize=args.resize,
                            metric_resolution=args.metric_resolution,
                            metric_definitions=dict(prediction="argmax over logits", classes="all, including background",
                                miou="mean of class IoU", mdice="mean of class hard Dice; not soft Dice loss",
                                absent_class="zero when absent from both prediction and target", epsilon=1e-6,
                                per_image="one confusion matrix per image",
                                summary="sum confusion matrices, then compute class and macro metrics",
                                confusion_matrix="rows: target; columns: prediction"),
                            reference_resize="PIL bilinear RGB, nearest mask" if score_at_inference and args.resize else None,
                            saved_image_resolution="original; scores may use the inference grid",
                            split_requested=args.split or "auto")
        write_json(folder / "protocol.json", protocol)
        rows, cm, segmentation_rows = [], None, []
        for path in images:
            with Image.open(path) as image:
                hazy = image.convert("RGB")
            pred, logits, aux = infer_image(model, hazy, device, args.resize,
                                            return_to_original=not score_at_inference)
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
                reference = (clear_image.resize((args.resize[1], args.resize[0]), Image.Resampling.BILINEAR)
                             if score_at_inference and args.resize else clear_image)
                target = pil_to_rgb_tensor(reference)[None].to(device)
                p, t = align_pair(pred, target, args.metric_align)
                row.update(image_metrics(p, t))
                if logits is not None and mask_dir is not None:
                    mp = find_by_stem(mask_dir, path.stem)
                    if mp is None:
                        raise ValueError(f"Missing segmentation mask for {path.name}")
                    with Image.open(mp) as image:
                        if score_at_inference and args.resize:
                            image = image.resize((args.resize[1], args.resize[0]), Image.Resampling.NEAREST)
                        target_mask = mask_to_label_tensor(image, logits.shape[1])[None, None].to(device)
                    p, t = align_pair(logits.argmax(1, keepdim=True), target_mask, args.metric_align, label=True)
                    current = confusion_matrix(p, t, logits.shape[1])
                    cm = current if cm is None else cm + current
                    metrics = segmentation_metrics(current)
                    row.update(segmentation_metric_row(metrics))
                    segmentation_rows.append(dict(sample=path.stem, height=p.shape[-2], width=p.shape[-1],
                        confusion_matrix=current.tolist(), mask_sha256=file_digest(mp), **metrics))
            if not evaluate or args.save_images:
                save_prediction(folder, path.stem, hazy, pred, logits, clear_image)
            rows.append(row)
        summary = {"model": config["model"], "implementation": implementation,
                   "checkpoint": str(checkpoint_path) if checkpoint_path else None, "count": len(rows),
                   "segmentation": protocol["segmentation"],
                   "segmenter_checkpoint": shared_info["checkpoint"] if shared_info else None,
                   "segmenter_sha256": shared_info["sha256"] if shared_info else None,
                   "known_training_overlap": len(overlap)}
        for key in rows[0]:
            if key != "sample":
                summary[key] = sum(r[key] for r in rows) / len(rows)
        if cm is not None:
            metrics = segmentation_metrics(cm)
            summary.update(metrics)
            summary.update(segmentation_metric_row(metrics))
            summary.update(metric_resolution=args.metric_resolution, confusion_matrix=cm.tolist())
            write_json(folder / "segmentation_metrics.json", dict(
                definitions=protocol["metric_definitions"], metric_resolution=args.metric_resolution,
                checkpoint_sha256=protocol["checkpoint_sha256"], samples=segmentation_rows,
                aggregate=dict(confusion_matrix=cm.tolist(), **metrics)))
        write_csv(folder / "metrics.csv", rows)
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        label = f"dcp ({implementation})" if config["model"] == "dcp" else config["model"]
        output_folders.append((label, folder, images))
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
