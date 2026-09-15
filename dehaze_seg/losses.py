from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def ssim_loss_approx(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, 1, pad)
    mu_y = F.avg_pool2d(y, window_size, 1, pad)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return torch.clamp((1.0 - ssim.mean()) * 0.5, 0.0, 1.0)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device: torch.device, layer_ids=(3, 8, 15)):
        super().__init__()
        self.layer_ids = tuple(layer_ids)
        try:
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            vgg = torchvision.models.vgg16(weights=weights).features.to(device).eval()
        except Exception:
            raise RuntimeError(
                "Could not load ImageNet VGG16 weights. Populate the torch hub cache "
                "(TORCH_HOME/hub/checkpoints) with vgg16-397923af.pth, or explicitly "
                "set --lam-perc 0 to run an experiment without perceptual loss."
            ) from None
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = pred.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = pred.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (pred - mean) / std
        y = (target - mean) / std
        loss = pred.new_tensor(0.0)
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss = loss + F.l1_loss(x, y)
            if i >= max(self.layer_ids):
                break
        return loss


def gradient_l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dx_x = x[..., :, 1:] - x[..., :, :-1]
    dx_y = y[..., :, 1:] - y[..., :, :-1]
    dy_x = x[..., 1:, :] - x[..., :-1, :]
    dy_y = y[..., 1:, :] - y[..., :-1, :]
    return F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def dehaze_losses(pred, clear, config, perceptual=None):
    """Only L1, SSIM and perceptual contribute to the restoration objective."""
    l1 = F.l1_loss(pred, clear)
    ssim = ssim_loss_approx(pred, clear)
    if config.lam_perc > 0 and perceptual is None:
        raise ValueError("A perceptual module is required when lam_perc > 0")
    perc = perceptual(pred, clear) if config.lam_perc > 0 else pred.new_zeros(())
    return dict(dehaze=l1 + config.lam_ssim * ssim + config.lam_perc * perc,
                l1=l1, ssim_loss=ssim, perc=perc,
                mse=F.mse_loss(pred, clear).detach(),
                grad=gradient_l1_loss(pred, clear).detach())


def segmentation_losses(logits, target, config):
    ce = F.cross_entropy(logits, target)
    dice = dice_loss_from_logits(logits, target, logits.shape[1])
    return dict(seg=config.lam_ce * ce + config.lam_dice * dice, ce=ce, dice_loss=dice)
