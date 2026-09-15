from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..utils import ensure_dir

def _parse_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _row_float(row: Dict[str, str], keys: str | List[str] | Tuple[str, ...]) -> float:
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        v = _parse_float(row.get(key, ""))
        if not math.isnan(v):
            return v
    return float("nan")


def _series(rows: List[Dict[str, str]], keys: str | List[str] | Tuple[str, ...]) -> List[float]:
    return [_row_float(row, keys) for row in rows]


def _has_series(rows: List[Dict[str, str]], keys: str | List[str] | Tuple[str, ...]) -> bool:
    return any(not math.isnan(v) for v in _series(rows, keys))


def _epoch_axis(rows: List[Dict[str, str]]) -> List[float]:
    xs = _series(rows, "global_epoch")
    if any(not math.isnan(v) for v in xs):
        return [v if not math.isnan(v) else i + 1 for i, v in enumerate(xs)]
    if len({row.get("stage", "") for row in rows}) == 1:
        xs = _series(rows, "epoch")
        if all(not math.isnan(v) for v in xs):
            return xs
    return [float(i + 1) for i in range(len(rows))]


def _decorate_stages(ax, rows: List[Dict[str, str]], xs: List[float]) -> None:
    if not rows or not xs:
        return
    colors = {"dehaze": "#e7f0ff", "seg": "#e8f6e8", "joint": "#f4ecff"}
    start = 0
    current = rows[0].get("stage", "")
    for i, row in enumerate(rows + [{}]):
        stage = row.get("stage", None)
        if i == len(rows) or stage != current:
            x0 = xs[start] - 0.5
            x1 = xs[i - 1] + 0.5
            ax.axvspan(x0, x1, color=colors.get(current, "#f3f3f3"), alpha=0.35, linewidth=0)
            if current:
                ax.text((x0 + x1) / 2, 0.98, current, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8, color="#555")
            start = i
            current = stage


def _plot_lines(
    rows: List[Dict[str, str]],
    keys: List[str | List[str] | Tuple[str, ...]],
    labels: List[str],
    title: str,
    out_path: Path,
) -> None:
    if plt is None:
        return
    xs = _epoch_axis(rows)
    plt.figure(figsize=(9, 5))
    drawn = False
    for key, label in zip(keys, labels):
        ys = _series(rows, key)
        if all(math.isnan(y) for y in ys):
            continue
        plt.plot(xs, ys, linewidth=1.8, label=label)
        drawn = True
    if not drawn:
        plt.close()
        return
    plt.title(title)
    plt.xlabel("Epoch")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_metric_grid(
    rows: List[Dict[str, str]],
    panels: List[Tuple[List[str | List[str] | Tuple[str, ...]], List[str], str]],
    title: str,
    out_path: Path,
    ncols: int = 2,
    figsize: Tuple[int, int] = (16, 12),
) -> bool:
    if plt is None:
        return False
    valid = [(keys, labels, subtitle) for keys, labels, subtitle in panels if any(_has_series(rows, k) for k in keys)]
    if not valid:
        return False
    nrows = int(math.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    xs = _epoch_axis(rows)
    for ax, (keys, labels, subtitle) in zip(axes.ravel(), valid):
        for key, label in zip(keys, labels):
            ys = _series(rows, key)
            if all(math.isnan(v) for v in ys):
                continue
            ax.plot(xs, ys, linewidth=1.8, label=label)
        ax.set_title(subtitle)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=9)
    for ax in axes.ravel()[len(valid):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    ensure_dir(out_path.parent)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _class_metric_keys(rows: List[Dict[str, str]], prefix: str) -> List[str]:
    keys = set()
    for row in rows:
        keys.update(k for k in row.keys() if k.startswith(prefix))
    return sorted(keys, key=lambda x: int(x.rsplit("_", 1)[-1]))


def save_metric_plots_from_csv(metrics_csv: Path, out_dir: Path) -> int:
    if plt is None:
        print("[Plot] matplotlib unavailable, skip metric plots.")
        return 0
    if not metrics_csv.exists():
        print(f"[Plot] metrics csv not found: {metrics_csv}")
        return 0
    with open(metrics_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[Plot] metrics csv empty, skip.")
        return 0

    plot_dir = ensure_dir(out_dir / "plots")
    for stale_name in [
        "loss_total.png",
        "loss_dehaze.png",
        "loss_seg.png",
        "val_dehaze_terms.png",
        "val_psnr_ssim.png",
        "val_seg_metrics.png",
        "lr.png",
        "dehaze_metrics_curves.png",
        "segmentation_metrics_curves.png",
    ]:
        stale_path = plot_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    count = 0
    dehaze_panels = [
        (["train_dehaze", "val_dehaze"], ["Train Dehaze Loss", "Val Dehaze Loss"], "Dehaze Loss"),
        (["val_psnr"], ["Val PSNR"], "Val PSNR"),
        (["val_ssim"], ["Val SSIM"], "Val SSIM"),
        (["val_delta_sat", "val_crerr"], ["DeltaSat", "CRerr"], "Val Color Error"),
        (["val_l1", "val_mse"], ["L1", "MSE"], "Val Pixel Error"),
        (["val_ssim_loss", "val_perc", "val_grad"], ["SSIM Loss", "VGG Perc", "Gradient"], "Val Dehaze Terms"),
        (["val_sat_ref", "val_sat_pred"], ["Mean Sat GT", "Mean Sat Pred"], "Saturation Statistics"),
        (["lr"], ["Learning Rate"], "Learning Rate"),
    ]

    base_seg_panels = [
        (["train_seg", "val_seg"], ["Train Seg Loss", "Val Seg Loss"], "Segmentation Loss"),
        (["train_ce", "val_ce"], ["Train CE", "Val CE"], "Cross Entropy"),
        (["train_dice_loss", "val_dice_loss"], ["Train Dice Loss", "Val Dice Loss"], "Dice Loss"),
        (["train_miou", "val_miou"], ["Train mIoU", "Val mIoU"], "mIoU"),
        (["train_mdice", "val_mdice"], ["Train mDice", "Val mDice"], "mDice"),
        (["train_f1", "val_f1"], ["Train F1", "Val F1"], "F1"),
        (["train_pixel_acc", "val_pixel_acc"], ["Train PixelAcc", "Val PixelAcc"], "Pixel Accuracy"),
        (["val_precision", "val_recall"], ["Precision", "Recall"], "Precision / Recall"),
        (["lr"], ["Learning Rate"], "Learning Rate"),
    ]

    stage_titles = {
        "dehaze": "Stage 1 Dehaze Pretraining",
        "seg": "Stage 2 Segmentation Pretraining",
        "joint": "Stage 3 Joint Finetuning",
    }
    stage_specs = [
        ("dehaze", True, False),
        ("seg", False, True),
        ("joint", True, True),
    ]
    for stage, draw_dehaze, draw_seg in stage_specs:
        stage_rows = [row for row in rows if row.get("stage", "") == stage]
        if not stage_rows:
            continue
        stage_dir = ensure_dir(plot_dir / stage)
        title_prefix = stage_titles.get(stage, stage)

        if draw_dehaze and _plot_metric_grid(
            stage_rows,
            dehaze_panels,
            f"{title_prefix} - Dehazing Metrics",
            stage_dir / "dehaze_metrics_curves.png",
        ):
            count += 1

        if draw_seg:
            seg_panels = list(base_seg_panels)
            class_iou_keys = _class_metric_keys(stage_rows, "val_class_iou_")
            class_dice_keys = _class_metric_keys(stage_rows, "val_class_dice_")
            if class_iou_keys:
                seg_panels.append((class_iou_keys, [f"IoU C{i}" for i in range(len(class_iou_keys))], "Class-wise IoU"))
            if class_dice_keys:
                seg_panels.append((class_dice_keys, [f"Dice C{i}" for i in range(len(class_dice_keys))], "Class-wise Dice"))
            if _plot_metric_grid(
                stage_rows,
                seg_panels,
                f"{title_prefix} - Segmentation Metrics",
                stage_dir / "segmentation_metrics_curves.png",
                figsize=(16, 14),
            ):
                count += 1

    for stage, _, _ in stage_specs:
        stage_rows = [row for row in rows if row.get("stage", "") == stage]
        if not stage_rows:
            continue
        stage_dir = ensure_dir(plot_dir / stage)
        dehaze_line_specs = [
            (["train_total", "val_total"], ["train_total", "val_total"], "Total Loss", "loss_total.png"),
            (["train_dehaze", "val_dehaze"], ["train_dehaze", "val_dehaze"], "Dehaze Loss", "loss_dehaze.png"),
            (["val_l1", "val_ssim_loss", "val_perc", "val_grad"], ["val_l1", "val_ssim_loss", "val_perc", "val_grad"], "Val Dehaze Terms", "val_dehaze_terms.png"),
            (["val_psnr", "val_ssim"], ["val_PSNR", "val_SSIM"], "Validation PSNR/SSIM", "val_psnr_ssim.png"),
            (["lr"], ["lr"], "Learning Rate", "lr.png"),
        ]
        seg_line_specs = [
            (["train_total", "val_total"], ["train_total", "val_total"], "Total Loss", "loss_total.png"),
            (["train_seg", "val_seg"], ["train_seg", "val_seg"], "Seg Loss", "loss_seg.png"),
            (["val_miou", "val_f1", "val_pixel_acc"], ["val_mIoU", "val_F1", "val_PixelAcc"], "Validation Seg Metrics", "val_seg_metrics.png"),
            (["lr"], ["lr"], "Learning Rate", "lr.png"),
        ]
        if stage == "dehaze":
            line_specs = dehaze_line_specs
        elif stage == "seg":
            line_specs = seg_line_specs
        else:
            line_specs = dehaze_line_specs + [
                spec for spec in seg_line_specs if spec[3] not in {"loss_total.png", "lr.png"}
            ]
        for keys, labels, title, filename in line_specs:
            out_path = stage_dir / filename
            _plot_lines(stage_rows, keys, labels, title, out_path)
            if out_path.exists():
                count += 1
    return count
