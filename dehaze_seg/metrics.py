from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from .losses import ssim_loss_approx

@torch.no_grad()
def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).clamp_min(1e-10)
    return float((-10.0 * torch.log10(mse)).item())


@torch.no_grad()
def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((1.0 - 2.0 * ssim_loss_approx(pred, target)).clamp(0.0, 1.0).item())


@torch.no_grad()
def saturation_map(x: torch.Tensor) -> torch.Tensor:
    mx = x.max(dim=1).values
    mn = x.min(dim=1).values
    return (mx - mn) / (mx + 1e-6)


@torch.no_grad()
def chromaticity_ratio_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    ps = pred.sum(dim=1, keepdim=True) + 1e-6
    ts = target.sum(dim=1, keepdim=True) + 1e-6
    return float((pred / ps - target / ts).abs().mean().item())


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Centre-cropped evaluation masks are often non-contiguous tensor views.
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()
    keep = (target >= 0) & (target < num_classes)
    idx = target[keep] * num_classes + pred[keep].clamp(0, num_classes - 1)
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).cpu()


def segmentation_metrics(cm: torch.Tensor, eps: float = 1e-6) -> Dict[str, float | list]:
    """Hard-mask metrics; macro averages include every class, including background.

    Rows of ``cm`` are ground truth and columns are predictions. A class absent
    from both masks contributes zero. For dataset scores, sum confusion matrices
    before calling this function; do not average per-image macro scores.
    """
    cm = cm.float()
    tp = torch.diag(cm)
    row = cm.sum(dim=1)
    col = cm.sum(dim=0)
    total = cm.sum().clamp_min(eps)
    iou = tp / (row + col - tp + eps)
    dice = 2 * tp / (row + col + eps)
    precision = tp / (col + eps)
    recall = tp / (row + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "pixel_acc": float(tp.sum() / total),
        "miou": float(iou.mean()),
        "mdice": float(dice.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "class_iou": [float(v) for v in iou.tolist()],
        "class_dice": [float(v) for v in dice.tolist()],
    }


def segmentation_metric_row(metrics):
    """Unambiguous scalar CSV columns, keeping macro and per-class Dice separate."""
    row = {key: value for key, value in metrics.items() if isinstance(value, float)}
    for name in ("iou", "dice"):
        row.update({f"{name}_class_{i}": value for i, value in enumerate(metrics[f"class_{name}"])})
    return row


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
