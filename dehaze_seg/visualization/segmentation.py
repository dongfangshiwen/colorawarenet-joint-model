"""Regenerate a segmentation comparison from real checkpoints and hard masks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from ..data.common import find_by_stem, mask_to_label_tensor
from ..data.splits import checkpoint_split, file_digest, training_overlap
from ..engine.checkpoint import load_checkpoint, PROFILES
from ..engine.inference import infer_image, list_images, select_split, use_shared_segmenter
from ..metrics import confusion_matrix, segmentation_metrics, segmentation_metric_row
from ..models.joint import JointDehazeSegModel
from ..models.registry import build_model, model_config
from ..utils import ensure_dir, select_device, write_csv, write_json


LABELS = dict(coloraware="ColorAwareUNet (Ours)", dcp="DCP", c2pnet="C2PNet",
              ffanet="FFA-Net", grid="GridDehazeNet", psd="PSD")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Actual dehazer/joint experiment checkpoints")
    parser.add_argument("--segmenter-checkpoint", required=True, help="One frozen joint checkpoint's segmenter for all methods")
    parser.add_argument("--legacy-profile", choices=PROFILES)
    parser.add_argument("--segmenter-legacy-profile", choices=PROFILES)
    parser.add_argument("--include-dcp", action="store_true", help="Add the parameter-free classical DCP baseline")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--split-file", help="Saved validation split; defaults to the shared segmenter's split")
    parser.add_argument("--sample", help="Validation sample ID; defaults to the first saved validation ID")
    parser.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--roi", type=float, nargs=4, default=[.20, .56, .75, .90],
                        metavar=("X0", "Y0", "X1", "Y1"), help="Normalized crop, identical across all panels")
    parser.add_argument("--output", default="results/figure7")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    # Multiples of 16 avoid padding changing the training validation grid.
    if any(n < 32 or n % 16 for n in args.resize) or args.threads < 1:
        parser.error("resize must use multiples of 16 >= 32; threads must be positive")
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        parser.error("roi must satisfy 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1")
    args.dataset, args.split = "paired-road", "val"
    return args


def overlay(rgb, labels):
    return np.where(labels[..., None] > 0, rgb * .60 + np.array([1., .20, .20]) * .40, rgb)


def save_figure(panels, records, roi, aspect, output):
    """Metric labels come directly from the same records saved alongside masks."""
    n = len(panels)
    fig = plt.figure(figsize=(3.1*n, 4.25), dpi=300, facecolor="white")
    left, gap = .012, .010
    width = (1 - 2*left - (n-1)*gap)/n
    x0, y0, x1, y1 = roi
    for i, (label, rgb) in enumerate(panels):
        x = left + i*(width+gap)
        center = x + width/2
        fig.text(center, .965, label, ha="center", va="center", fontsize=14, fontweight="semibold")
        record = records[i]
        text = (f"{record['miou']:.4f} / {record['mdice']:.4f}" if record else "mIoU / mDice")
        fig.text(center, .895, text, ha="center", va="center", fontsize=14)
        h, w = rgb.shape[:2]
        for row in (0, 1):
            ax = fig.add_axes([x, .430 if row == 0 else .070, width, .395 if row == 0 else .295])
            if row == 0:
                ax.imshow(rgb, extent=(0, aspect, 1, 0), interpolation="nearest", aspect="equal")
                ax.add_patch(Rectangle((x0*aspect, y0), (x1-x0)*aspect, y1-y0,
                                      fill=False, edgecolor="#d92323", linewidth=1.3))
            else:
                crop = rgb[round(y0*h):round(y1*h), round(x0*w):round(x1*w)]
                crop_aspect = aspect*(x1-x0)/(y1-y0)
                ax.imshow(crop, extent=(0, crop_aspect, 1, 0), interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#d92323" if row else "#777777")
                spine.set_linewidth(1 if row else .65)
    fig.text(.5, .020, "Red shading: foreground mask     Red boxes: enlarged regions     Scores: full inference grid",
             ha="center", fontsize=10.5)
    fig.savefig(output / "figure7.png", dpi=300)
    fig.savefig(output / "figure7.pdf", dpi=300)
    plt.close(fig)


@torch.inference_mode()
def generate(args):
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    shared, shared_config, shared_checkpoint = load_checkpoint(
        args.segmenter_checkpoint, device, args.segmenter_legacy_profile)
    if not isinstance(shared, JointDehazeSegModel) or shared_config["segmenter"]["num_classes"] != 2:
        raise ValueError("Figure generation requires a binary joint segmentation checkpoint")
    shared.requires_grad_(False)
    shared.dehazer = torch.nn.Identity()
    saved = shared_checkpoint.get("train_config", shared_checkpoint.get("args", {}))
    if (not args.split_file and checkpoint_split(args.segmenter_checkpoint, shared_checkpoint) is None
            and not {"seed", "val_ratio"} <= saved.keys()):
        raise ValueError("No saved validation split or seed/ratio; supply an explicit --split-file")
    root = Path(args.data_root)
    selected = select_split(list_images(root/"hazy"), args, args.segmenter_checkpoint, shared_checkpoint)
    sample = args.sample or selected[0].stem
    if sample not in {p.stem for p in selected}:
        raise ValueError(f"{sample} is outside the selected validation split; choose a validation sample")
    overlap = training_overlap("paired-road", root, [sample], args.segmenter_checkpoint, shared_checkpoint)
    if overlap:
        raise ValueError(f"Sample {sample} overlaps shared-segmenter training references; choose another validation sample")
    paths = {name: find_by_stem(root/name, sample) for name in ("hazy", "clear", "masks")}
    if any(p is None for p in paths.values()):
        raise ValueError(f"Missing triplet for {sample}")
    size = (args.resize[1], args.resize[0])
    with Image.open(paths["hazy"]) as source:
        hazy = source.convert("RGB")
        aspect = hazy.width/hazy.height
        input_rgb = np.asarray(hazy.resize(size, Image.Resampling.BILINEAR))/255.
    with Image.open(paths["clear"]) as source:
        clear = np.asarray(source.convert("RGB").resize(size, Image.Resampling.BILINEAR))/255.
    with Image.open(paths["masks"]) as source:
        target = mask_to_label_tensor(source.resize(size, Image.Resampling.NEAREST), 2)[None]
    target_array = target[0].numpy().astype(np.uint8)
    output = ensure_dir(args.output)
    Image.fromarray(target_array).save(output / "target_mask.png")
    panels, plot_records, method_records, csv_rows = [("Hazy input", input_rgb)], [None], [], []
    sources = ([None] if args.include_dcp else []) + list(args.checkpoint)
    for i, checkpoint_path in enumerate(sources):
        if checkpoint_path is None:
            config = model_config("dcp", joint=False)
            model = build_model(config).to(device).eval()
            label, digest = "DCP (classical)", None
        else:
            model, config, checkpoint = load_checkpoint(checkpoint_path, device, args.legacy_profile)
            overlap = training_overlap("paired-road", root, [sample], checkpoint_path, checkpoint)
            if overlap:
                raise ValueError(f"Sample {sample} overlaps training data for {checkpoint_path}")
            label = LABELS[config["model"]]
            if config["model"] == "dcp":
                label += " (learned)" if config.get("dehazer_type") == "learned-dcp" else " (checkpoint)"
            digest = file_digest(checkpoint_path)
        model = use_shared_segmenter(model, shared)
        restored, logits, _ = infer_image(model, hazy, device, args.resize, return_to_original=False)
        labels = logits.argmax(1).cpu()
        cm = confusion_matrix(labels, target, 2)
        metrics = segmentation_metrics(cm)
        mask_name = f"{i+1:02d}_{config['model']}_mask.png"
        rgb_name = f"{i+1:02d}_{config['model']}_dehazed.png"
        Image.fromarray(labels[0].numpy().astype(np.uint8)).save(output/mask_name)
        rgb = restored[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(output/rgb_name)
        panels.append((label, overlay(rgb, labels[0].numpy())))
        plot_records.append(metrics)
        method_records.append(dict(label=label, model_config=config, checkpoint=str(checkpoint_path) if checkpoint_path else None,
            checkpoint_sha256=digest, prediction_mask=mask_name, dehazed_image=rgb_name,
            confusion_matrix=cm.tolist(), metrics=metrics))
        csv_rows.append(dict(sample=sample, method=label, **segmentation_metric_row(metrics)))
        del model, logits, restored
    gt_metrics = segmentation_metrics(confusion_matrix(target, target, 2))
    panels.append(("Ground truth", overlay(clear, target_array)))
    plot_records.append(gt_metrics)
    record = dict(sample=sample, dataset="paired-road", split="val", known_training_overlap=[],
        sample_selection="explicit ID" if args.sample else "first validation ID; not selected by score",
        resize=args.resize, metric_resolution="inference", display_aspect=aspect, roi=args.roi,
        metric_definition="Hard argmax; mean over background and foreground; full grid, not ROI; absent classes zero; eps=1e-6",
        target_mask="target_mask.png", sources={k:dict(file=p.name, sha256=file_digest(p)) for k,p in paths.items()},
        shared_segmenter=dict(checkpoint=str(args.segmenter_checkpoint), sha256=file_digest(args.segmenter_checkpoint),
                              model_config=shared_config, frozen=True), methods=method_records,
        software=dict(torch=torch.__version__, device=str(device)))
    write_json(output/"figure7_metrics.json", record)
    write_csv(output/"figure7_metrics.csv", csv_rows)
    save_figure(panels, plot_records, args.roi, aspect, output)
    caption = (f"Fig. 7. Road-region segmentation on validation sample {sample}. "
        "The dehazing methods use one shared, frozen LiteAttentionUNet. "
        f"Full-image mIoU and mDice are computed at {args.resize[0]} x {args.resize[1]} from hard labels, "
        "averaged over background and foreground. Red shading denotes the foreground; "
        "the bottom row enlarges the red boxes. Display panels preserve the source aspect ratio. "
        "These are single-image results; the DCP panel uses the classical parameter-free implementation.")
    (output/"caption.txt").write_text(caption+"\n", encoding="utf-8")
    print(json.dumps(dict(sample=sample, scores=csv_rows, figure=str(output/"figure7.png")), ensure_ascii=False))
    return record


def main(argv=None):
    return generate(parse_args(argv))
