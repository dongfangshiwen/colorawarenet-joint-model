"""Regenerate a segmentation comparison from real checkpoints and hard masks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
PAPER_MODELS = ("dcp", "ffanet", "grid", "psd", "coloraware", "c2pnet")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate the complete six-model Figure 7 from actual experiment weights")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Six complete joint checkpoints; shared-frozen also accepts restoration-only weights")
    parser.add_argument("--segmentation-protocol", choices=("joint", "shared-frozen"), default="joint",
                        help="joint keeps each checkpoint's segmenter; shared-frozen replaces them with one evaluator")
    parser.add_argument("--segmenter-checkpoint", help="Required only for shared-frozen comparison")
    parser.add_argument("--legacy-profile", choices=PROFILES)
    parser.add_argument("--segmenter-legacy-profile", choices=PROFILES)
    parser.add_argument("--include-dcp", action="store_true", help="Add classical DCP in shared-frozen mode only")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--split-file", help="Saved validation split; otherwise use the first joint checkpoint or shared evaluator")
    parser.add_argument("--sample", help="Validation sample ID; defaults to the first saved validation ID")
    parser.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--roi", type=float, nargs=4, default=[.20, .56, .75, .90],
                        metavar=("X0", "Y0", "X1", "Y1"), help="Normalized crop, identical across all panels")
    parser.add_argument("--output", default="results/figure7")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--metric-font", help="Path to a Times New Roman TTF/OTF file; otherwise use the installed font")
    args = parser.parse_args(argv)
    if args.segmentation_protocol == "shared-frozen" and not args.segmenter_checkpoint:
        parser.error("shared-frozen requires --segmenter-checkpoint")
    if args.segmentation_protocol == "joint" and (args.segmenter_checkpoint or args.segmenter_legacy_profile or args.include_dcp):
        parser.error("joint requires six complete joint checkpoints; shared-segmenter and classical DCP options require --segmentation-protocol shared-frozen")
    # Multiples of 16 avoid padding changing the training validation grid.
    if any(n < 32 or n % 16 for n in args.resize) or args.threads < 1:
        parser.error("resize must use multiples of 16 >= 32; threads must be positive")
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        parser.error("roi must satisfy 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1")
    args.dataset, args.split = "paired-road", "val"
    return args


def figure_sources(args):
    """Fail before inference/writing files when a paper comparison is incomplete."""
    sources = {"dcp": None} if args.include_dcp else {}
    for checkpoint_path in args.checkpoint:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Missing experiment checkpoint: {checkpoint_path}")
        # Strict CPU loading validates both the identifier and architecture. Drop
        # each model immediately so the preflight does not retain six networks.
        model, config, checkpoint = load_checkpoint(checkpoint_path, "cpu", args.legacy_profile)
        if args.segmentation_protocol == "joint" and (
                not isinstance(model, JointDehazeSegModel) or config["segmenter"]["num_classes"] != 2):
            raise ValueError(f"Joint Figure 7 requires a binary joint checkpoint: {checkpoint_path}")
        name = config["model"]
        del model, checkpoint
        if name in sources:
            raise ValueError(f"Duplicate {name} source; provide exactly one checkpoint per method, "
                             "and use either --include-dcp or a DCP checkpoint")
        sources[name] = checkpoint_path
    missing = set(PAPER_MODELS) - sources.keys()
    if missing:
        raise ValueError("Figure 7 requires all six methods. Missing: " + ", ".join(sorted(missing)) +
                         ". Supply their trained checkpoints; --include-dcp supplies classical DCP. "
                         "No partial Figure 7 was generated.")
    return [sources[name] for name in PAPER_MODELS]


def overlay(rgb, labels):
    return np.where(labels[..., None] > 0, rgb * .60 + np.array([1., .20, .20]) * .40, rgb)


def metric_font(path=None):
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Metric font file not found: {path}")
        return font_manager.FontProperties(fname=str(path))
    try:
        font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as exc:
        raise ValueError("Times New Roman is required for Figure 7 metrics. Install it or pass "
                         "--metric-font /path/to/times.ttf. No substitute font was used.") from exc
    return font_manager.FontProperties(fname=font_path)


def save_figure(panels, records, roi, aspect, output, score_font=None):
    """Metric labels come directly from the same records saved alongside masks."""
    n = len(panels)
    score_font = score_font or metric_font()
    fig = plt.figure(figsize=(2.1*n, 3.65), dpi=300, facecolor="white")
    left, gap = .012, .010
    width = (1 - 2*left - (n-1)*gap)/n
    x0, y0, x1, y1 = roi
    for i, (label, rgb) in enumerate(panels):
        x = left + i*(width+gap)
        center = x + width/2
        fig.text(center, .99, label.split(" (", 1)[0], ha="center", va="top",
                 fontsize=16, fontweight="semibold", linespacing=1.05)
        record = records[i]
        text = (f"{record['miou']:.4f} / {record['mdice']:.4f}" if record else "mIoU / mDice")
        fig.text(center, .80, text, ha="center", va="center", fontsize=16, fontproperties=score_font)
        h, w = rgb.shape[:2]
        for row in (0, 1):
            ax = fig.add_axes([x, .385 if row == 0 else .070, width, .36 if row == 0 else .265])
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
             ha="center", fontsize=12)
    fig.savefig(output / "figure7.png", dpi=300)
    fig.savefig(output / "figure7.pdf", dpi=300)
    plt.close(fig)


@torch.inference_mode()
def generate(args):
    torch.set_num_threads(args.threads)
    sources = figure_sources(args)
    score_font = metric_font(args.metric_font)
    device = select_device(args.device)
    use_shared = args.segmentation_protocol == "shared-frozen"
    anchor_path = args.segmenter_checkpoint if use_shared else sources[0]
    anchor, anchor_config, anchor_checkpoint = load_checkpoint(
        anchor_path, device, args.segmenter_legacy_profile if use_shared else args.legacy_profile)
    if not isinstance(anchor, JointDehazeSegModel) or anchor_config["segmenter"]["num_classes"] != 2:
        raise ValueError("Figure generation requires a binary joint segmentation checkpoint")
    shared_info = None
    if use_shared:
        shared = anchor.requires_grad_(False)
        shared.dehazer = torch.nn.Identity()
        shared_info = dict(checkpoint=str(anchor_path), sha256=file_digest(anchor_path),
                           model_config=anchor_config, frozen=True)
    else:
        del anchor
    saved = anchor_checkpoint.get("train_config", anchor_checkpoint.get("args", {}))
    if (not args.split_file and checkpoint_split(anchor_path, anchor_checkpoint) is None
            and not {"seed", "val_ratio"} <= saved.keys()):
        raise ValueError("No saved validation split or seed/ratio; supply an explicit --split-file")
    root = Path(args.data_root)
    selected = select_split(list_images(root/"hazy"), args, anchor_path, anchor_checkpoint)
    sample = args.sample or selected[0].stem
    if sample not in {p.stem for p in selected}:
        raise ValueError(f"{sample} is outside the selected validation split; choose a validation sample")
    overlap = training_overlap("paired-road", root, [sample], anchor_path, anchor_checkpoint)
    if overlap:
        raise ValueError(f"Sample {sample} overlaps checkpoint training references; choose another validation sample")
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
                label += {"learned-dcp": " (learned)", "legacy-dcp": " (historical)"}.get(
                    config.get("dehazer_type"), " (classical)")
            digest = file_digest(checkpoint_path)
        if use_shared:
            model = use_shared_segmenter(model, shared)
        restored, logits, _ = infer_image(model, hazy, device, args.resize, return_to_original=False)
        labels = logits.argmax(1).cpu()
        cm = confusion_matrix(labels, target, 2)
        metrics = segmentation_metrics(cm)
        if metrics["mdice"] + 1e-7 < metrics["miou"]:
            raise RuntimeError("Inconsistent hard-mask macro metrics: mDice must be >= mIoU")
        mask_name = f"{i+1:02d}_{config['model']}_mask.png"
        rgb_name = f"{i+1:02d}_{config['model']}_dehazed.png"
        Image.fromarray(labels[0].numpy().astype(np.uint8)).save(output/mask_name)
        rgb = restored[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(output/rgb_name)
        panels.append((label, overlay(rgb, labels[0].numpy())))
        plot_records.append(metrics)
        method_records.append(dict(label=label, model_config=config, checkpoint=str(checkpoint_path) if checkpoint_path else None,
            checkpoint_sha256=digest, segmentation_protocol=args.segmentation_protocol,
            prediction_mask=mask_name, dehazed_image=rgb_name,
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
        segmentation_protocol=args.segmentation_protocol, shared_segmenter=shared_info, methods=method_records,
        typography=dict(metric_font=score_font.get_name(), metric_font_sha256=file_digest(score_font.get_file()),
                        panel_titles="model names without implementation or authorship suffixes"),
        software=dict(torch=torch.__version__, device=str(device)))
    write_json(output/"figure7_metrics.json", record)
    write_csv(output/"figure7_metrics.csv", csv_rows)
    save_figure(panels, plot_records, args.roi, aspect, output, score_font)
    caption = (f"Fig. 7. Road-region segmentation on validation sample {sample}. "
        + ("Each method uses its complete joint checkpoint, including its own LiteAttentionUNet. "
           if not use_shared else "The dehazing methods use one shared, frozen LiteAttentionUNet. ") +
        f"Full-image mIoU and mDice are computed at {args.resize[0]} x {args.resize[1]} from hard labels, "
        "averaged over background and foreground. Red shading denotes the foreground; "
        "the bottom row enlarges the red boxes. Display panels preserve the source aspect ratio. "
        "These are single-image results. " + ("The DCP panel uses the classical parameter-free implementation."
            if args.include_dcp else "The DCP panel uses its checkpoint's recorded implementation."))
    (output/"caption.txt").write_text(caption+"\n", encoding="utf-8")
    print(json.dumps(dict(sample=sample, scores=csv_rows, figure=str(output/"figure7.png")), ensure_ascii=False))
    return record


def main(argv=None):
    return generate(parse_args(argv))
