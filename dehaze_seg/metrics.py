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
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    keep = (target >= 0) & (target < num_classes)
    idx = target[keep] * num_classes + pred[keep].clamp(0, num_classes - 1)
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).cpu()


def segmentation_metrics(cm: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
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
