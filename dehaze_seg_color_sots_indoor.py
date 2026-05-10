#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dehaze_seg_optimized.py

优化去雾+分割训练脚本（完整）
改动（主要与颜色保留/饱和度相关）：
 - 精简去雾损失为必要子集：L1 / SSIM / Perceptual
 - 增大 color gain 表达能力（默认 gain_scale=0.6）
 - 去除冗余损失，减少互相牵制
 - 在 validate 中输出平均饱和度统计
 - 新增：在 validate 中计算并返回 SSIM 指标，训练阶段保存/打印中包含 SSIM
"""
import os
import argparse
import random
import time
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm
from models.ColorAwareUnet import ColorAwareUNet
from models.LiteAttentionUnet import *
from models.AODNet import AODNet
from models.DehazeNet import DehazeNet
from models.GridDehazeNet import GridDehazeNet
from models.Dehamer import DehamerNet
from models.FFANet import FFANet
from models.C2PNet import C2PNet
from models.MSBDN import MSBDN
from models.DCP import DCPDehaze
from models.DehazeUnet import DehazeUNet
from models.D4 import D4DehazeNet
from models.PSD import PSDDehazeNet

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -------------------------
# Repro / utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
# Dataset (same as yours)
# -------------------------
class HazySegDataset(Dataset):
    def __init__(self, root, num_classes, ids=None,
                 img_exts=(".png", ".jpg", ".jpeg"),
                 resize=None, augment=True):
        """
        root 目录要求：
          root/
            hazy/   : 合成雾图，比如 1400_1.png ... 1400_10.png
            clear/  : 对应的无雾 GT 图，或者
            gt/     : 如果没有 clear/，则使用 gt/ 作为 GT
            mask/   : (可选) 分割标签；如果没有，会生成全 0 mask

        映射规则：
          - hazy:  用完整 stem，例如 "1400_1"
          - clear/mask:
              1) 先尝试同名: "1400_1.*"
              2) 找不到时，退回前缀: "1400.*"   (兼容 SOTS)
        """
        super().__init__()
        self.root = Path(root)

        # hazy 必须存在
        self.hazy_dir = self.root / "hazy"
        if not self.hazy_dir.exists():
            raise RuntimeError(f"hazy folder not found: {self.hazy_dir}")

        # clear/gt 二选一
        clear_dir = self.root / "clear"
        gt_dir = self.root / "gt"
        if clear_dir.exists():
            self.clear_dir = clear_dir
        elif gt_dir.exists():
            self.clear_dir = gt_dir
        else:
            raise RuntimeError(f"Neither 'clear' nor 'gt' folder found under {self.root}")

        # mask 可选
        mask_dir = self.root / "mask"
        self.mask_dir = mask_dir if mask_dir.exists() else None
        self.has_real_masks = self.mask_dir is not None

        self.num_classes = int(num_classes)
        self.resize = resize  # (H, W) or None
        self.augment = augment

        # 构建样本 id 列表：全部来自 hazy 目录
        if ids is None:
            ids = []
            for p in self.hazy_dir.iterdir():
                if p.is_file() and p.suffix.lower() in img_exts:
                    ids.append(p.stem)   # e.g. "1400_1"
            ids.sort()
            self.ids = ids
        else:
            self.ids = ids

    def __len__(self):
        return len(self.ids)

    def _convert_mask_to_labels(self, arr, num_classes):
        """
        将 mask 的 numpy 数组转换为 [0, num_classes-1] 的整数标签。
        兼容：
          - 2D 标注
          - one-hot / multi-channel
          - 0~255 的灰度图
          - 0~1 的浮点图
        """
        if arr.ndim == 3:
            h, w, c = arr.shape
            if c == num_classes:
                return np.argmax(arr, axis=2).astype(np.int64)
            if c == 3:
                arr2 = arr[..., 0]
                arr = arr2
            else:
                return np.argmax(arr, axis=2).astype(np.int64)

        arr_f = arr.astype(np.float32)
        vmin, vmax = float(arr_f.min()), float(arr_f.max())

        if np.issubdtype(arr.dtype, np.integer):
            # 已经是 0~(num_classes-1)
            if vmax <= (num_classes - 1):
                return arr.astype(np.int64)
            # 0~255 灰度，映射到类别索引
            if vmax <= 255:
                labels = np.round(arr_f * ((num_classes - 1) / 255.0)).astype(np.int64)
                labels = np.clip(labels, 0, num_classes - 1)
                return labels

        # 浮点 0~1
        if vmin >= 0.0 and vmax <= 1.0:
            labels = np.round(arr_f * (num_classes - 1)).astype(np.int64)
            labels = np.clip(labels, 0, num_classes - 1)
            return labels

        # 兜底：直接 round + clip
        labels = np.round(arr_f).astype(np.int64)
        labels = np.clip(labels, 0, num_classes - 1)
        return labels

    def _sync_transform(self, hazy_pil, clear_pil, mask_pil):
        """
        对 hazy / clear / mask 做同步的数据增强 + resize
        """
        # if self.augment:
        #     # 随机水平翻转
        #     if random.random() > 0.5:
        #         hazy_pil = TF.hflip(hazy_pil)
        #         clear_pil = TF.hflip(clear_pil)
        #         mask_pil = TF.hflip(mask_pil)

        #     # 随机裁剪（90% 范围）
        #     try:
        #         w, h = hazy_pil.size
        #         if w > 320 and h > 320 and random.random() > 0.6:
        #             crop_w = int(0.9 * w)
        #             crop_h = int(0.9 * h)
        #             left = random.randint(0, w - crop_w)
        #             top = random.randint(0, h - crop_h)
        #             box = (left, top, left + crop_w, top + crop_h)
        #             hazy_pil = hazy_pil.crop(box)
        #             clear_pil = clear_pil.crop(box)
        #             mask_pil = mask_pil.crop(box)
        #     except Exception:
        #         pass

        # 统一 resize
        if self.resize is not None:
            target_w, target_h = self.resize[1], self.resize[0]
            hazy_pil = hazy_pil.resize((target_w, target_h), Image.BILINEAR)
            clear_pil = clear_pil.resize((target_w, target_h), Image.BILINEAR)
            mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)
        else:
            # No global resize: align to GT/clear size to preserve supervision fidelity.
            ref_size = clear_pil.size
            if hazy_pil.size != ref_size:
                hazy_pil = hazy_pil.resize(ref_size, Image.BILINEAR)
            if mask_pil.size != ref_size:
                mask_pil = mask_pil.resize(ref_size, Image.NEAREST)

        return hazy_pil, clear_pil, mask_pil

    def __getitem__(self, idx):
        id0 = self.ids[idx]  # e.g. "1400_1" 或 "xxx"

        def find_file(dirp: Path, stem: str):
            if dirp is None or (not dirp.exists()):
                return None
            # 优先匹配常见扩展名
            for ext in (".png", ".jpg", ".jpeg"):
                p = dirp / f"{stem}{ext}"
                if p.exists():
                    return p
            # 兜底：任意后缀
            for p in dirp.glob(f"{stem}.*"):
                if p.is_file():
                    return p
            return None

        # 1) hazy 必须是完整 stem，例如 "1400_1"
        hazy_path = find_file(self.hazy_dir, id0)

        # 2) clear：先尝试同名；找不到时退回前缀（兼容 SOTS）
        clear_path = find_file(self.clear_dir, id0)
        if clear_path is None:
            # 对 "1400_1" -> "1400"
            base_id = id0.split("_")[0] if "_" in id0 else id0
            clear_path = find_file(self.clear_dir, base_id)

        # 3) mask：如果有 mask_dir，则同样的逻辑；否则 mask_path = None
        mask_path = None
        if self.mask_dir is not None:
            mask_path = find_file(self.mask_dir, id0)
            if mask_path is None:
                base_id = id0.split("_")[0] if "_" in id0 else id0
                mask_path = find_file(self.mask_dir, base_id)

        if hazy_path is None or clear_path is None:
            raise FileNotFoundError(f"Missing hazy or clear file for id {id0} (hazy: {hazy_path}, clear: {clear_path})")

        # ---------------- 读取图像 ----------------
        hazy_pil = Image.open(str(hazy_path)).convert("RGB")
        clear_pil = Image.open(str(clear_path)).convert("RGB")

        # mask: 若没有真实文件，就造一个全 0 mask
        if mask_path is None:
            # 用 GT 的大小造一张全 0 mask
            w, h = clear_pil.size
            mask_arr = np.zeros((h, w), dtype=np.uint8)
            mask_pil = Image.fromarray(mask_arr)
        else:
            if mask_path.suffix.lower() == ".npy":
                mask_arr = np.load(str(mask_path))
                if mask_arr.ndim == 2:
                    mask_pil = Image.fromarray(mask_arr.astype(np.uint8))
                else:
                    mask_pil = Image.fromarray(mask_arr[..., 0].astype(np.uint8))
            else:
                mask_pil = Image.open(str(mask_path))

        # ---------------- 同步增强 + resize ----------------
        hazy_pil, clear_pil, mask_pil = self._sync_transform(hazy_pil, clear_pil, mask_pil)

        # ---------------- 转 tensor ----------------
        hazy = TF.to_tensor(hazy_pil)   # Bx3xHxW
        clear = TF.to_tensor(clear_pil)

        # Use transformed mask directly to guarantee H/W is synced with hazy/clear.
        mask_arr = np.array(mask_pil)

        # 转类别标签
        mask_labels = self._convert_mask_to_labels(mask_arr, self.num_classes)
        mask_tensor = torch.from_numpy(mask_labels).long()  # HxW, long

        return hazy, clear, mask_tensor

# -------------------------
# Losses / metrics (including high-frequency + color)
# -------------------------
def l1_loss(a,b): return F.l1_loss(a,b)

def ssim_loss_approx(a, b, window_size=11):
    """
    Local-window SSIM loss.
    The previous implementation used image-level global statistics, which was too weak
    to constrain local structures and often allowed over-smooth outputs.
    """
    pad = window_size // 2
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = F.avg_pool2d(a, window_size, stride=1, padding=pad)
    mu_b = F.avg_pool2d(b, window_size, stride=1, padding=pad)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = F.avg_pool2d(a * a, window_size, stride=1, padding=pad) - mu_a2
    sigma_b2 = F.avg_pool2d(b * b, window_size, stride=1, padding=pad) - mu_b2
    sigma_ab = F.avg_pool2d(a * b, window_size, stride=1, padding=pad) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2) + 1e-6
    )
    return torch.clamp(1.0 - ssim_map, 0.0, 2.0).mean()

# Add SSIM metric wrapper (returns SSIM in [0,1])
def ssim_metric(a, b):
    # ssim_loss_approx returns (1 - ssim), so invert it
    with torch.no_grad():
        val = torch.clamp(1.0 - ssim_loss_approx(a, b), 0.0, 1.0)
        # ensure scalar float
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

def dice_loss(pred_logits, target, eps=1e-6):
    probs = F.softmax(pred_logits, dim=1)
    B, C, H, W = probs.shape
    target_one = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()
    inter = (probs * target_one).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + target_one.sum(dim=(2,3))
    dice = 1.0 - ((2*inter + eps) / (union + eps))
    return dice.mean()

def psnr(a,b, max_val=1.0):
    mse = torch.mean((a - b)**2)
    return 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))

def _sat_map(img):
    p_max, _ = img.max(dim=1, keepdim=True)
    p_min, _ = img.min(dim=1, keepdim=True)
    return (p_max - p_min) / (p_max + 1e-6)

def saturation_deviation(pred, target):
    """
    Delta saturation: mean absolute difference between saturation maps.
    Lower is better.
    """
    return float(torch.abs(_sat_map(pred) - _sat_map(target)).mean().item())

def chromaticity_ratio_error(pred, target):
    """
    RGB chromaticity ratio error:
      r = R / (R+G+B), g = G / sum, b = B / sum
    Lower is better and useful for color-preservation reporting.
    """
    p_sum = pred.sum(dim=1, keepdim=True) + 1e-6
    t_sum = target.sum(dim=1, keepdim=True) + 1e-6
    p_ratio = pred / p_sum
    t_ratio = target / t_sum
    return float(torch.abs(p_ratio - t_ratio).mean().item())

# VGG perceptual (robust loading)
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cpu', layer_ids=(3,8,15)):
        super().__init__()
        try:
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            vgg = torchvision.models.vgg16(weights=weights).features.to(device).eval()
        except Exception:
            try:
                vgg = torchvision.models.vgg16(pretrained=True).features.to(device).eval()
            except Exception:
                vgg = None
        if vgg is not None:
            for p in vgg.parameters():
                p.requires_grad = False
        self.vgg = vgg
        self.layer_ids = layer_ids

    def forward(self, a, b):
        if self.vgg is None:
            return torch.tensor(0.0, device=a.device)
        mean = torch.tensor([0.485,0.456,0.406], device=a.device).view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225], device=a.device).view(1,3,1,1)
        a_in = (a - mean) / std
        b_in = (b - mean) / std
        loss = 0.0
        x = a_in
        y = b_in
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss = loss + F.l1_loss(x, y)
        return loss

def compute_dehaze_losses(
    dehazed,
    clear,
    vgg_loss=None,
    lam_ssim=0.40,
    lam_perc=0.05,
):
    zero = clear.new_tensor(0.0)

    loss_l1 = l1_loss(dehazed, clear)
    loss_ssim = lam_ssim * ssim_loss_approx(dehazed, clear) if lam_ssim > 0 else zero
    loss_perc = lam_perc * vgg_loss(dehazed, clear) if (vgg_loss is not None and lam_perc > 0) else zero
    loss_de = loss_l1 + loss_ssim + loss_perc
    return {
        "loss_de": loss_de,
        "loss_l1": loss_l1,
        "loss_ssim": loss_ssim,
        "loss_perc": loss_perc,
    }

# -------------------------
# Metrics: mean saturation
# -------------------------
def mean_saturation(img):
    """
    img: Bx3xHxW in [0,1]
    return: scalar mean saturation (approx)
    """
    p_max, _ = img.max(dim=1)  # BxHxW
    p_min, _ = img.min(dim=1)
    p_chroma = p_max - p_min
    sat = p_chroma / (p_max + 1e-6)
    return float(sat.mean().item())

# -------------------------
# Train / Val loops (minimal effective dehaze losses)
# -------------------------
def _unpack_dehaze_output(dz_out):
    aux = {
        "residual": None,
        "color_gain": None,
    }
    if isinstance(dz_out, (list, tuple)):
        out = dz_out[0]
        aux["residual"] = dz_out[1] if len(dz_out) > 1 and torch.is_tensor(dz_out[1]) else None
        aux["color_gain"] = dz_out[2] if len(dz_out) > 2 and torch.is_tensor(dz_out[2]) else None
        sides = dz_out[3] if len(dz_out) > 3 else [None, None, None]
    elif isinstance(dz_out, dict):
        out = dz_out.get("out") or dz_out.get("dehazed") or dz_out.get("dehaze")
        sides = dz_out.get("sides", [None, None, None])
        aux["residual"] = dz_out.get("residual")
        aux["color_gain"] = dz_out.get("color_gain")
    else:
        out = dz_out
        sides = [None, None, None]
    return out, sides, aux

def _dehaze_forward(model, hazy, clamp_output=True):
    dz_out = model.dehazer(hazy)
    dehazed, sides, aux = _unpack_dehaze_output(dz_out)
    if clamp_output:
        dehazed = torch.clamp(dehazed, 0.0, 1.0)
    aux["residual_scale"] = float(getattr(model.dehazer, "residual_scale", 1.0))
    return dehazed, sides, aux

def _segment_forward(model, dehazed):
    seg_in = (dehazed - model.imagenet_mean) / model.imagenet_std
    seg_tmp = model.seg(seg_in)
    if isinstance(seg_tmp, dict):
        return seg_tmp.get("out", seg_tmp)
    return seg_tmp

def _forward_joint_with_aux(model, hazy, clamp_output=True, clamp_for_seg=True):
    dehazed, sides, aux = _dehaze_forward(model, hazy, clamp_output=clamp_output)
    seg_input = torch.clamp(dehazed, 0.0, 1.0) if clamp_for_seg else dehazed
    seg_out = _segment_forward(model, seg_input)
    return dehazed, seg_out, sides, aux

def train_epoch_dehaze_only(model, dataloader, optimizer, device, scaler, vgg_loss=None,
                            lam_recon=1.0, lam_ssim=0.40, lam_perc=0.05,
                            print_every=50):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='TrainDehaze')
    for i, (hazy, clear, _) in pbar:
        hazy = hazy.to(device); clear = clear.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            dehazed, _, _ = _dehaze_forward(model, hazy)
            loss_dict = compute_dehaze_losses(
                dehazed,
                clear,
                vgg_loss=vgg_loss,
                lam_ssim=lam_ssim,
                lam_perc=lam_perc,
            )
            loss_de = loss_dict["loss_de"]
            loss = lam_recon * loss_de

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()
        running_loss += loss.item()
        if i % print_every == 0:
            pbar.set_postfix({'loss': f'{running_loss/(i+1):.4f}'} )
    return running_loss / len(dataloader)

def train_epoch_seg_only(model, dataloader, optimizer, device, scaler, num_classes, print_every=50):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='TrainSeg')
    for i, (hazy, _, mask) in pbar:
        hazy = hazy.to(device); mask = mask.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            with torch.no_grad():
                dehazed, _, _ = _dehaze_forward(model, hazy)
            seg_out = _segment_forward(model, dehazed)
            loss_seg = F.cross_entropy(seg_out, mask) + dice_loss(seg_out, mask)
            loss = loss_seg
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()
        running_loss += loss.item()
        if i % print_every == 0:
            pbar.set_postfix({'loss': f'{running_loss/(i+1):.4f}'} )
    return running_loss / len(dataloader)

def train_epoch_joint(model, dataloader, optimizer, device, scaler, num_classes,
                      lam_dehaze, lam_seg, vgg_loss=None, lam_ssim=0.40, lam_perc=0.05,
                      print_every=50):
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_de": 0.0,
        "loss_seg": 0.0,
        "loss_l1": 0.0,
        "loss_ssim": 0.0,
        "loss_perc": 0.0,
    }
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='TrainJoint')
    for i, (hazy, clear, mask) in pbar:
        hazy = hazy.to(device); clear = clear.to(device); mask = mask.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            dehazed, seg_out, _, _ = _forward_joint_with_aux(model, hazy)
            loss_dict = compute_dehaze_losses(
                dehazed,
                clear,
                vgg_loss=vgg_loss,
                lam_ssim=lam_ssim,
                lam_perc=lam_perc,
            )
            loss_de = loss_dict["loss_de"]
            loss_l1 = loss_dict["loss_l1"]
            loss_ssim = loss_dict["loss_ssim"]
            loss_perc = loss_dict["loss_perc"]

            loss_seg = F.cross_entropy(seg_out, mask) + dice_loss(seg_out, mask)
            loss = lam_dehaze * loss_de + lam_seg * loss_seg

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()
        running["loss_total"] += float(loss.item())
        running["loss_de"] += float(loss_de.item())
        running["loss_seg"] += float(loss_seg.item())
        running["loss_l1"] += float(loss_l1.item())
        running["loss_ssim"] += float(loss_ssim.item())
        running["loss_perc"] += float(loss_perc.item())
        if i % print_every == 0:
            pbar.set_postfix({
                'loss': f'{running["loss_total"]/(i+1):.4f}',
                'de': f'{running["loss_de"]/(i+1):.4f}',
                'seg': f'{running["loss_seg"]/(i+1):.4f}',
            })
    denom = float(len(dataloader)) if len(dataloader) > 0 else 1.0
    return {k: (v / denom) for k, v in running.items()}

def validate(model, dataloader, device):
    model.eval()
    tot_psnr = 0.0
    tot_ssim = 0.0
    tot_sat_ref = 0.0
    tot_sat_pred = 0.0
    tot_delta_sat = 0.0
    tot_crerr = 0.0
    n_img = 0
    with torch.no_grad():
        for hazy, clear, mask in tqdm(dataloader, desc='Val'):
            hazy = hazy.to(device); clear = clear.to(device)
            dehazed, _, _ = _dehaze_forward(model, hazy)

            bsz = dehazed.shape[0]

            # Strict per-image PSNR (not batch-mean PSNR)
            mse = ((dehazed - clear) ** 2).flatten(1).mean(dim=1)
            psnr_each = 20.0 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
            tot_psnr += float(psnr_each.sum().item())

            # Strict per-image statistics for all validation metrics
            for bi in range(bsz):
                tot_ssim += ssim_metric(dehazed[bi:bi+1], clear[bi:bi+1])
                tot_sat_ref += mean_saturation(clear[bi:bi+1])
                tot_sat_pred += mean_saturation(dehazed[bi:bi+1])
                tot_delta_sat += saturation_deviation(dehazed[bi:bi+1], clear[bi:bi+1])
                tot_crerr += chromaticity_ratio_error(dehazed[bi:bi+1], clear[bi:bi+1])
            n_img += bsz
    if n_img == 0:
        return 0.0, 0.0, 0.0, 0.0
    avg_psnr = tot_psnr / n_img
    avg_ssim = tot_ssim / n_img
    avg_sat_ref = tot_sat_ref / n_img
    avg_sat_pred = tot_sat_pred / n_img
    avg_delta_sat = tot_delta_sat / n_img
    avg_crerr = tot_crerr / n_img
    # print saturation + ssim summary
    print(
        f"[ValMetric] PSNR:{avg_psnr:.3f} SSIM:{avg_ssim:.4f} "
        f"DeltaSat:{avg_delta_sat:.4f} CRerr:{avg_crerr:.4f} "
        f"mean_sat_ref:{avg_sat_ref:.4f} mean_sat_pred:{avg_sat_pred:.4f}"
    )
    return avg_psnr, avg_ssim, avg_delta_sat, avg_crerr


def metric_score(psnr_v: float, ssim_v: float, mode: str) -> float:
    if mode == "psnr":
        return float(psnr_v)
    if mode == "ssim":
        return float(ssim_v)
    # balanced
    return float(psnr_v + 20.0 * ssim_v)

# -------------------------
# Helpers: checkpoints, sliding window, unsharp, saturation boost
# -------------------------
def save_checkpoint(stage_dir, state, is_best=False):
    os.makedirs(stage_dir, exist_ok=True)
    last_path = os.path.join(stage_dir, 'last.pth')
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(stage_dir, 'best.pth')
        torch.save(state, best_path)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    if x.ndim == 4:
        x = x[0]
    x = x.detach().cpu().clamp(0.0, 1.0)
    return TF.to_pil_image(x)


def resize_pil(img: Image.Image, resize_hw):
    if resize_hw is None:
        return img
    h, w = int(resize_hw[0]), int(resize_hw[1])
    return img.resize((w, h), Image.BILINEAR)


def maybe_unsharp_pil(img: Image.Image) -> Image.Image:
    # Light unsharp for perceptual sharpness without strong halos.
    return img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

def maybe_unsharp_if_enabled(img: Image.Image, enabled: bool) -> Image.Image:
    return maybe_unsharp_pil(img) if enabled else img


def align_pil_to_size(img: Image.Image, size_hw):
    """
    Align PIL image to target size (W, H) for stable metric computation.
    """
    if img.size == size_hw:
        return img
    return img.resize(size_hw, Image.BILINEAR)


def find_sots_gt(gt_dir: Path, stem: str):
    if gt_dir is None or (not gt_dir.exists()):
        return None
    for ext in IMAGE_EXTS:
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    base = stem.split("_")[0] if "_" in stem else stem
    for ext in IMAGE_EXTS:
        p = gt_dir / f"{base}{ext}"
        if p.exists():
            return p
    for p in gt_dir.glob(f"{base}.*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def resolve_dehazer_state_dict(ck):
    if isinstance(ck, dict) and "dehazer_state" in ck:
        return ck["dehazer_state"]
    if isinstance(ck, dict) and "model_state" in ck and isinstance(ck["model_state"], dict):
        st = ck["model_state"]
        dz_state = {k.replace("dehazer.", "", 1): v for k, v in st.items() if k.startswith("dehazer.")}
        return dz_state if dz_state else st
    if isinstance(ck, dict) and "state_dict" in ck and isinstance(ck["state_dict"], dict):
        st = ck["state_dict"]
        dz_state = {k.replace("dehazer.", "", 1): v for k, v in st.items() if k.startswith("dehazer.")}
        return dz_state if dz_state else st
    return ck


def load_dehazer_checkpoint(dehazer: nn.Module, ckpt_path: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = resolve_dehazer_state_dict(ck)
    if not isinstance(state, dict):
        raise RuntimeError("Resolved dehazer state is not a dict.")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("dehazer.") for k in state):
        state = {k.replace("dehazer.", "", 1): v for k, v in state.items()}
    missing, unexpected = dehazer.load_state_dict(state, strict=False)
    loaded = len(dehazer.state_dict()) - len(missing)
    print(f"[InitCKPT] loaded dehazer params: {loaded}/{len(dehazer.state_dict())}")
    if missing:
        print(f"[InitCKPT] missing keys: {len(missing)}")
    if unexpected:
        print(f"[InitCKPT] unexpected keys: {len(unexpected)}")


def save_metric_plots(history, save_dir: str):
    if not history:
        return
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    csv_path = os.path.join(plot_dir, "metrics_history.csv")
    fields = [
        "step", "stage", "epoch", "train_loss",
        "val_psnr", "val_ssim", "val_delta_sat", "val_crerr",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in history:
            writer.writerow({k: r.get(k, "") for k in fields})

    if plt is None:
        print(f"[Warn] matplotlib unavailable, only CSV saved: {csv_path}")
        return

    steps = [r["step"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_psnr = [r["val_psnr"] for r in history]
    val_ssim = [r["val_ssim"] for r in history]
    val_delta_sat = [r["val_delta_sat"] for r in history]
    val_crerr = [r["val_crerr"] for r in history]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(steps, train_loss, color="tab:blue")
    axs[0, 0].set_title("Train Loss")
    axs[0, 1].plot(steps, val_psnr, color="tab:green")
    axs[0, 1].set_title("Val PSNR")
    axs[1, 0].plot(steps, val_ssim, color="tab:orange")
    axs[1, 0].set_title("Val SSIM")
    axs[1, 1].plot(steps, val_delta_sat, color="tab:red", label="DeltaSat")
    axs[1, 1].plot(steps, val_crerr, color="tab:purple", label="CRerr")
    axs[1, 1].set_title("Val Color Error")
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_path = os.path.join(plot_dir, "metrics_curves.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"[Done] metrics saved: {fig_path}")


def _pick_auto_ckpt(save_dir: str, prefer_stage: str = "auto"):
    if prefer_stage == "dehaze":
        candidates = [
            os.path.join(save_dir, "dehaze", "best.pth"),
            os.path.join(save_dir, "dehaze", "last.pth"),
            os.path.join(save_dir, "joint", "best.pth"),
            os.path.join(save_dir, "joint", "last.pth"),
            os.path.join(save_dir, "seg", "best.pth"),
            os.path.join(save_dir, "seg", "last.pth"),
        ]
    elif prefer_stage == "joint":
        candidates = [
            os.path.join(save_dir, "joint", "best.pth"),
            os.path.join(save_dir, "joint", "last.pth"),
            os.path.join(save_dir, "dehaze", "best.pth"),
            os.path.join(save_dir, "dehaze", "last.pth"),
            os.path.join(save_dir, "seg", "best.pth"),
            os.path.join(save_dir, "seg", "last.pth"),
        ]
    elif prefer_stage == "seg":
        candidates = [
            os.path.join(save_dir, "seg", "best.pth"),
            os.path.join(save_dir, "seg", "last.pth"),
            os.path.join(save_dir, "joint", "best.pth"),
            os.path.join(save_dir, "joint", "last.pth"),
            os.path.join(save_dir, "dehaze", "best.pth"),
            os.path.join(save_dir, "dehaze", "last.pth"),
        ]
    else:
        candidates = [
            os.path.join(save_dir, "joint", "best.pth"),
            os.path.join(save_dir, "dehaze", "best.pth"),
            os.path.join(save_dir, "joint", "last.pth"),
            os.path.join(save_dir, "dehaze", "last.pth"),
            os.path.join(save_dir, "seg", "best.pth"),
            os.path.join(save_dir, "seg", "last.pth"),
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def run_auto_infer(args, ckpt_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dehazer = build_dehazer_from_args(args).to(device).eval()
    load_dehazer_checkpoint(dehazer, ckpt_path)

    single_hazy = Path(args.auto_infer_hazy_path) if args.auto_infer_hazy_path else None
    hazy_dir = Path(args.auto_infer_hazy_dir) if args.auto_infer_hazy_dir else Path(args.data_root) / "hazy"
    gt_dir = Path(args.auto_infer_gt_dir) if args.auto_infer_gt_dir else None
    if gt_dir is None:
        cdir = Path(args.data_root) / "clear"
        gdir = Path(args.data_root) / "gt"
        if cdir.exists():
            gt_dir = cdir
        elif gdir.exists():
            gt_dir = gdir
    if single_hazy is not None:
        if not single_hazy.exists() or (not single_hazy.is_file()):
            print(f"[Warn] auto infer skipped, single hazy image not found: {single_hazy}")
            return
        files = [single_hazy]
    else:
        if not hazy_dir.exists():
            print(f"[Warn] auto infer skipped, hazy dir not found: {hazy_dir}")
            return
        files = [p for p in hazy_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        files.sort(key=lambda x: x.stem)
        if args.auto_infer_limit > 0:
            files = files[: args.auto_infer_limit]
    if not files:
        print("[Warn] auto infer skipped, no hazy images found.")
        return

    out_dir = Path(args.save_dir) / "auto_infer"
    out_pred = out_dir / "dehazed"
    out_cmp = out_dir / "compare"
    out_pred.mkdir(parents=True, exist_ok=True)
    if args.auto_infer_save_compare:
        out_cmp.mkdir(parents=True, exist_ok=True)

    resize_hw = tuple(args.resize) if args.resize else None
    psnr_list, ssim_list, delta_sat_list, crerr_list = [], [], [], []
    rows = []
    with torch.no_grad():
        for i, hp in enumerate(files, 1):
            hazy_img = Image.open(str(hp)).convert("RGB")
            hazy_img = resize_pil(hazy_img, resize_hw)
            gp = find_sots_gt(gt_dir, hp.stem) if gt_dir is not None else None
            gt_img = None
            if gp is not None:
                gt_img = Image.open(str(gp)).convert("RGB")
                gt_img = resize_pil(gt_img, resize_hw)
                # If dataset pairs have different native sizes (e.g., 620x460 vs 640x480),
                # prefer GT size for inference to preserve output fidelity.
                if resize_hw is None and hazy_img.size != gt_img.size:
                    hazy_img = hazy_img.resize(gt_img.size, Image.BILINEAR)
            hazy_t = TF.to_tensor(hazy_img).unsqueeze(0).to(device)
            out = dehazer(hazy_t)
            dehazed, _, _ = _unpack_dehaze_output(out)
            dehazed = torch.clamp(dehazed, 0.0, 1.0)
            pred_img = tensor_to_pil(dehazed)
            pred_img = maybe_unsharp_if_enabled(pred_img, args.auto_infer_unsharp)
            pred_path = out_pred / f"{hp.stem}.png"
            pred_img.save(pred_path)
            pred_eval_t = TF.to_tensor(pred_img).unsqueeze(0).to(device)

            msg = f"[AutoInfer {i}/{len(files)}] {hp.name} -> {pred_path}"
            if gt_img is not None:
                gt_img = align_pil_to_size(gt_img, pred_img.size)
                gt_t = TF.to_tensor(gt_img).unsqueeze(0).to(device)
                p = float(psnr(pred_eval_t, gt_t).item())
                s = float(ssim_metric(pred_eval_t, gt_t))
                ds = float(saturation_deviation(pred_eval_t, gt_t))
                cr = float(chromaticity_ratio_error(pred_eval_t, gt_t))
                psnr_list.append(p)
                ssim_list.append(s)
                delta_sat_list.append(ds)
                crerr_list.append(cr)
                rows.append({"name": hp.name, "psnr": p, "ssim": s, "delta_sat": ds, "crerr": cr})
                msg += f" | PSNR={p:.2f} SSIM={s:.4f} DeltaSat={ds:.4f} CRerr={cr:.4f}"
                if args.auto_infer_save_compare:
                    w, h = hazy_img.size
                    strip = Image.new("RGB", (w * 3, h), (255, 255, 255))
                    strip.paste(hazy_img, (0, 0))
                    strip.paste(pred_img, (w, 0))
                    strip.paste(gt_img, (w * 2, 0))
                    strip.save(out_cmp / f"{hp.stem}.png")
            print(msg)

    if rows:
        metric_csv = out_dir / "metrics.csv"
        with open(metric_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim", "delta_sat", "crerr"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Done] auto infer metrics: {metric_csv}")

        if plt is not None:
            idx = list(range(1, len(rows) + 1))
            pvals = [r["psnr"] for r in rows]
            svals = [r["ssim"] for r in rows]
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(idx, pvals, color="tab:green", label="PSNR")
            ax1.set_xlabel("Image Index")
            ax1.set_ylabel("PSNR", color="tab:green")
            ax1.tick_params(axis='y', labelcolor="tab:green")
            ax2 = ax1.twinx()
            ax2.plot(idx, svals, color="tab:orange", label="SSIM")
            ax2.set_ylabel("SSIM", color="tab:orange")
            ax2.tick_params(axis='y', labelcolor="tab:orange")
            fig.tight_layout()
            fig.savefig(out_dir / "metrics_plot.png", dpi=180)
            plt.close(fig)

    if psnr_list:
        print(
            f"[AutoInfer Metric] Avg PSNR={sum(psnr_list)/len(psnr_list):.2f} "
            f"Avg SSIM={sum(ssim_list)/len(ssim_list):.4f} "
            f"Avg DeltaSat={sum(delta_sat_list)/len(delta_sat_list):.4f} "
            f"Avg CRerr={sum(crerr_list)/len(crerr_list):.4f}"
        )
    print(f"[Done] auto infer outputs saved to: {out_dir}")


def _resolve_monitor_gt(args, hazy_path: Path):
    if args.train_monitor_gt_path:
        gp = Path(args.train_monitor_gt_path)
        return gp if gp.exists() else None
    gt_dir = Path(args.train_monitor_gt_dir) if args.train_monitor_gt_dir else None
    if gt_dir is None:
        cdir = Path(args.data_root) / "clear"
        gdir = Path(args.data_root) / "gt"
        if cdir.exists():
            gt_dir = cdir
        elif gdir.exists():
            gt_dir = gdir
    return find_sots_gt(gt_dir, hazy_path.stem) if gt_dir is not None else None


def run_train_monitor_infer(model, args, device, stage: str, epoch: int):
    if not args.train_monitor_hazy_path:
        return
    hp = Path(args.train_monitor_hazy_path)
    if not hp.exists() or (not hp.is_file()):
        return

    resize_hw = tuple(args.resize) if args.resize else None
    out_root = Path(args.save_dir) / "train_monitor"
    out_pred = out_root / stage
    out_pred.mkdir(parents=True, exist_ok=True)
    out_cmp = out_root / "compare"
    if args.train_monitor_save_compare:
        out_cmp.mkdir(parents=True, exist_ok=True)

    gt_path = _resolve_monitor_gt(args, hp)
    gt_img = None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        hazy_img = Image.open(str(hp)).convert("RGB")
        hazy_img = resize_pil(hazy_img, resize_hw)
        if gt_path is not None and gt_path.exists():
            gt_img = Image.open(str(gt_path)).convert("RGB")
            gt_img = resize_pil(gt_img, resize_hw)
            if resize_hw is None and hazy_img.size != gt_img.size:
                hazy_img = hazy_img.resize(gt_img.size, Image.BILINEAR)
            gt_img = align_pil_to_size(gt_img, hazy_img.size)
        x = TF.to_tensor(hazy_img).unsqueeze(0).to(device)
        dz_out = model.dehazer(x)
        dehazed, _, _ = _unpack_dehaze_output(dz_out)
        dehazed = torch.clamp(dehazed, 0.0, 1.0)
        pred_img = tensor_to_pil(dehazed)
        pred_img = maybe_unsharp_if_enabled(pred_img, args.train_monitor_unsharp)
    if was_training:
        model.train()

    pred_name = f"{hp.stem}_{stage}_epoch{epoch:03d}.png"
    pred_path = out_pred / pred_name
    pred_img.save(pred_path)
    pred_eval_t = TF.to_tensor(pred_img).unsqueeze(0).to(device)

    msg = f"[TrainMonitor] {stage} epoch={epoch} -> {pred_path}"
    psnr_val, ssim_val, delta_sat_val, crerr_val = None, None, None, None
    if gt_img is not None:
        gt_img = align_pil_to_size(gt_img, pred_img.size)
        gt_t = TF.to_tensor(gt_img).unsqueeze(0).to(device)
        psnr_val = float(psnr(pred_eval_t, gt_t).item())
        ssim_val = float(ssim_metric(pred_eval_t, gt_t))
        delta_sat_val = float(saturation_deviation(pred_eval_t, gt_t))
        crerr_val = float(chromaticity_ratio_error(pred_eval_t, gt_t))
        msg += (
            f" | PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} "
            f"DeltaSat={delta_sat_val:.4f} CRerr={crerr_val:.4f}"
        )
        if args.train_monitor_save_compare:
            w, h = hazy_img.size
            strip = Image.new("RGB", (w * 3, h), (255, 255, 255))
            strip.paste(hazy_img, (0, 0))
            strip.paste(pred_img, (w, 0))
            strip.paste(gt_img, (w * 2, 0))
            strip.save(out_cmp / pred_name)
    elif args.train_monitor_save_compare:
        w, h = hazy_img.size
        strip = Image.new("RGB", (w * 2, h), (255, 255, 255))
        strip.paste(hazy_img, (0, 0))
        strip.paste(pred_img, (w, 0))
        strip.save(out_cmp / pred_name)
    print(msg)

    csv_path = out_root / "monitor_metrics.csv"
    header = ["stage", "epoch", "image", "pred_path", "psnr", "ssim", "delta_sat", "crerr"]
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow({
            "stage": stage,
            "epoch": epoch,
            "image": hp.name,
            "pred_path": str(pred_path),
            "psnr": "" if psnr_val is None else f"{psnr_val:.6f}",
            "ssim": "" if ssim_val is None else f"{ssim_val:.6f}",
            "delta_sat": "" if delta_sat_val is None else f"{delta_sat_val:.6f}",
            "crerr": "" if crerr_val is None else f"{crerr_val:.6f}",
        })

def build_dehazer_from_args(args):
    if args.dehaze_backbone == 'coloraware':
        dehazer = ColorAwareUNet(
            in_ch=3,
            base_ch=args.color_base_ch,
            residual_scale=args.color_residual_scale,
            gain_scale=args.color_gain_scale,
            gain_form=args.color_gain_form,
            gain_min=args.color_gain_min,
            disable_gain=args.color_disable_gain,
            gain_mode=args.color_gain_mode,
            gain_smooth_kernel=args.color_gain_smooth_kernel,
            norm=args.color_norm,
            output_clamp=not args.color_no_output_clamp,
            refine_scale=args.color_refine_scale,
        )

    elif args.dehaze_backbone == 'dehazenet':
        dehazer = DehazeNet(
            in_ch=3,
            maxout_k=2,              # 固定为 2
            guided_filter_refine=False,
            t0=0.10,
            use_image_refine=True,
            refine_scale=0.12
        )

    elif args.dehaze_backbone == 'aodnet':
        dehazer = AODNet(
            in_ch=3,
            mid_ch=8,                # 固定为 8
            use_tanh_on_K=True,
            b=1.0,
            k_scale=0.8
        )

    elif args.dehaze_backbone == 'grid':
        dehazer = GridDehazeNet(
            in_ch=3,
            rows=3,                  # 固定为 3
            cols=6,                  # 固定为 6
            base_ch=32               # 固定为 32
        )

    elif args.dehaze_backbone == 'dehamer':
        dehazer = DehamerNet(
            in_ch=3,
            base_ch=16,              # 固定为 16（对应你之前的写法）
            trans_dim=64,
            nheads=4,
            n_layers=2,
            patch_size=8,
            use_learnable_pos=True,
            residual_scale=0.30
        )

    elif args.dehaze_backbone == 'ffanet':
        dehazer = FFANet(
            in_ch=3,
            base_ch=64,
            n_down=0,                # FFA-Net keeps full resolution
            n_ffab_deep=19,
            groups=3
        )

    elif args.dehaze_backbone == 'c2pnet':
        dehazer = C2PNet(
            in_ch=3,
            base_ch=32,              # 固定为 32
            blocks_per_group=6,      # 固定为 6
            groups=3,                # 固定为 3
            use_pdu=True
        )

    elif args.dehaze_backbone == 'msbdn':
        dehazer = MSBDN(
            in_ch=3,
            base_ch=32,              # 固定为 32
            nblocks=3                # 固定为 3
        )
        
    elif args.dehaze_backbone == 'dcp':
        dehazer = DCPDehaze(
            in_ch=3,
            patch_size=15,
            omega=0.95,
            t0=0.1,
            top_percent=0.001,
            guided=True,
            guided_radius=7,
            guided_eps=1e-4,
            use_learned_refine=False
        )
    elif args.dehaze_backbone == 'dehazeunet':
        dehazer = DehazeUNet(
            in_ch=3,
            base_ch=32,
            residual_scale=0.45
        )
    
    elif args.dehaze_backbone == 'd4':
        dehazer = D4DehazeNet(
            in_ch=3,
            base_ch=32   # 可以视显存调成 16/32
        )
    elif args.dehaze_backbone == 'psd':
        dehazer = PSDDehazeNet(
            in_ch=3,
            base_ch=32,   # 可按显存调 16/32
            feat_ch=64,
            residual_scale=0.45,
            refine_scale=0.15,
            t_min=0.10
        )
    else:
        raise ValueError(f"Unknown dehaze_backbone: {args.dehaze_backbone}")

    print(f"[Build] Using dehaze backbone: {args.dehaze_backbone}")
    return dehazer
    
# -------------------------
# Main (entry)
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='root with hazy/ clear/ mask subfolders')
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--resize', type=int, nargs=2, default=None, help='H W or none')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_optimized')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--lam_dehaze', type=float, default=1.0)
    parser.add_argument('--lam_seg', type=float, default=0.3)
    parser.add_argument('--joint_seg_warmup_epochs', type=int, default=10,
                        help='ramp lam_seg from start factor to target in first N joint epochs')
    parser.add_argument('--joint_seg_start_factor', type=float, default=0.0,
                        help='start factor for lam_seg warmup, final factor is 1.0')
    parser.add_argument('--joint_best_metric', type=str, default='balanced',
                        choices=['psnr', 'ssim', 'balanced'],
                        help='criterion to save joint/best.pth')
    parser.add_argument('--joint_scheduler_metric', type=str, default='balanced',
                        choices=['psnr', 'ssim', 'balanced'],
                        help='criterion for ReduceLROnPlateau in joint stage')
    parser.add_argument('--lam_ssim', type=float, default=0.40)
    parser.add_argument('--lam_perc', type=float, default=0.05)
    parser.add_argument('--color_gain_scale', type=float, default=0.30)
    parser.add_argument('--color_base_ch', type=int, default=32)
    parser.add_argument('--color_gain_form', type=str, default='tanh', choices=['tanh', 'amp'])
    parser.add_argument('--color_gain_min', type=float, default=0.95)
    parser.add_argument('--color_disable_gain', action='store_true')
    parser.add_argument('--color_residual_scale', type=float, default=0.50)
    parser.add_argument('--color_gain_mode', type=str, default='global', choices=['local', 'global'])
    parser.add_argument('--color_gain_smooth_kernel', type=int, default=15,
                        help='odd smoothing kernel for local color gain; <=1 disables smoothing')
    parser.add_argument('--color_norm', type=str, default='inst', choices=['none', 'group', 'inst', 'batch'])
    parser.add_argument('--color_no_output_clamp', action='store_true',
                        help='disable internal clamp in ColorAwareUNet output')
    parser.add_argument('--color_refine_scale', type=float, default=0.25,
                        help='full-resolution refinement scale for ColorAwareUNet')
    parser.add_argument('--pretrained_seg', action='store_true')
    parser.add_argument('--pretrain_dehaze_epochs', type=int, default=10)
    parser.add_argument('--pretrain_seg_epochs', type=int, default=0)
    parser.add_argument('--finetune_epochs', type=int, default=0)
    parser.add_argument('--dehaze_pretrained_path', type=str, default=None)
    parser.add_argument('--dehaze_backbone', type=str, default='coloraware', choices=['coloraware', 'dehazenet', 'aodnet', 'grid', 'dehamer', 'ffanet', 'c2pnet', 'msbdn', 'dcp', 'dehazeunet', 'd4', 'psd'], help='which dehaze sub-network to use')
    parser.add_argument('--no_save_metric_plots', action='store_true', help='disable saving training metric csv/plots')
    parser.add_argument('--no_auto_infer_after_train', action='store_true', help='disable auto inference after training')
    parser.add_argument('--auto_infer_ckpt', type=str, default=None, help='explicit checkpoint path for auto inference')
    parser.add_argument('--auto_infer_ckpt_stage', type=str, default='dehaze',
                        choices=['auto', 'dehaze', 'joint', 'seg'],
                        help='when auto picking ckpt, prefer which stage')
    parser.add_argument('--auto_infer_hazy_path', type=str, default=None, help='single hazy image for auto inference (highest priority)')
    parser.add_argument('--auto_infer_hazy_dir', type=str, default=None, help='hazy dir for auto inference (default: data_root/hazy)')
    parser.add_argument('--auto_infer_gt_dir', type=str, default=None, help='gt/clear dir for auto inference')
    parser.add_argument('--auto_infer_limit', type=int, default=0, help='limit auto inference image count (0=all)')
    parser.add_argument('--auto_infer_save_compare', action='store_true', help='save hazy|pred|gt compare strips in auto infer')
    parser.add_argument('--auto_infer_unsharp', action='store_true', help='apply unsharp mask before saving auto inference images')
    parser.add_argument('--train_monitor_hazy_path', type=str, default=None, help='single hazy image for online monitor during training')
    parser.add_argument('--train_monitor_gt_path', type=str, default=None, help='optional GT image path for online monitor metrics')
    parser.add_argument('--train_monitor_gt_dir', type=str, default=None, help='optional GT dir for online monitor auto matching')
    parser.add_argument('--train_monitor_every', type=int, default=5, help='run online monitor every N epochs')
    parser.add_argument('--train_monitor_save_compare', action='store_true', help='save hazy|pred|gt strip for online monitor')
    parser.add_argument('--train_monitor_unsharp', action='store_true', help='apply unsharp mask before saving train monitor images')
    args = parser.parse_args()
    if args.train_monitor_every < 1:
        args.train_monitor_every = 1
    args.joint_seg_start_factor = float(min(max(args.joint_seg_start_factor, 0.0), 1.0))
    if args.joint_seg_warmup_epochs < 0:
        args.joint_seg_warmup_epochs = 0
    if args.train_monitor_hazy_path and (not Path(args.train_monitor_hazy_path).exists()):
        print(f"[Warn] train_monitor_hazy_path not found: {args.train_monitor_hazy_path}, disable online monitor.")
        args.train_monitor_hazy_path = None

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    resize_hw = tuple(args.resize) if args.resize else None
    ds_meta = HazySegDataset(args.data_root, num_classes=args.num_classes, resize=resize_hw, augment=False)
    n = len(ds_meta)
    if n == 0:
        raise RuntimeError("Dataset empty. Check data_root with hazy/ clear/ mask subfolders.")
    n_val = max(1, int(0.08 * n))
    # n_val = max(1, int(0.30 * n))
    n_train = n - n_val
    ids = list(ds_meta.ids)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    train_ds = HazySegDataset(args.data_root, num_classes=args.num_classes, ids=train_ids, resize=resize_hw, augment=True)
    val_ds = HazySegDataset(args.data_root, num_classes=args.num_classes, ids=val_ids, resize=resize_hw, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"Dataset sizes: total={n} train={n_train} val={n_val}")
    if not val_ds.has_real_masks and (args.pretrain_seg_epochs > 0 or args.finetune_epochs > 0 or args.epochs > 0):
        print(
            "[Warn] No mask/ folder found. Segmentation/joint training will use dummy all-zero masks "
            "and is not valid for paper segmentation comparison. For SOTS dehazing evaluation, "
            "use --pretrain_seg_epochs 0 --finetune_epochs 0 --epochs 0."
        )

    dehaze_dir = os.path.join(args.save_dir, 'dehaze')
    seg_dir = os.path.join(args.save_dir, 'seg')
    joint_dir = os.path.join(args.save_dir, 'joint')
    os.makedirs(dehaze_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(joint_dir, exist_ok=True)

    # dehazer = ColorAwareUNet(in_ch=3, base_ch=32, residual_scale=0.5, gain_scale=0.65)
    dehazer = build_dehazer_from_args(args)
    seg_net = LiteAttentionUNet(num_classes=args.num_classes, width_mult=1.0, base_ch=32, use_se=True, attention=True, aux=False)
    model = DehazeLightSegWrapper(dehazer, seg_net, imagenet_norm=True).to(device)
    if args.dehaze_pretrained_path:
        try:
            ck = torch.load(args.dehaze_pretrained_path, map_location='cpu')
            if 'dehazer_state' in ck:
                missing, unexpected = model.dehazer.load_state_dict(ck['dehazer_state'], strict=False)
                print(f"Loaded dehaze_state from checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
            elif 'model_state' in ck and isinstance(ck['model_state'], dict):
                st = ck['model_state']
                dz_state = {k.replace('dehazer.', ''): v for k, v in st.items() if k.startswith('dehazer.')}
                if dz_state:
                    missing, unexpected = model.dehazer.load_state_dict(dz_state, strict=False)
                    print(f"Loaded dehazer keys from model_state. missing={len(missing)} unexpected={len(unexpected)}")
                else:
                    try:
                        missing, unexpected = model.dehazer.load_state_dict(ck['model_state'], strict=False)
                        print(f"Loaded model_state to dehazer. missing={len(missing)} unexpected={len(unexpected)}")
                    except Exception as e:
                        print("Failed to load dehaze weights:", e)
            else:
                try:
                    missing, unexpected = model.dehazer.load_state_dict(ck, strict=False)
                    print(f"Loaded direct dehaze weights. missing={len(missing)} unexpected={len(unexpected)}")
                except Exception as e:
                    print("Failed to load provided dehaze file:", e)
        except Exception as e:
            print("Error loading dehaze_pretrained_path:", e)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None
    vgg_loss = VGGPerceptualLoss(device=device)

    best_psnr = -1.0
    best_ssim = -1.0
    best_joint_score = -1e18
    history = []
    global_step = 0

    # Stage 1: pretrain dehaze only
    if args.pretrain_dehaze_epochs > 0:
        print("Stage 1: Pretraining dehaze for", args.pretrain_dehaze_epochs, "epochs")
        dehazer_params = [p for p in model.dehazer.parameters() if p.requires_grad]
        if len(dehazer_params) == 0:
            print("[Info] Dehazer has no trainable parameters; evaluate and save fixed baseline once.")
            val_psnr, val_ssim, val_delta_sat, val_crerr = validate(model, val_loader, device)
            global_step += 1
            history.append({
                "step": global_step,
                "stage": "dehaze",
                "epoch": 0,
                "train_loss": 0.0,
                "val_psnr": float(val_psnr),
                "val_ssim": float(val_ssim),
                "val_delta_sat": float(val_delta_sat),
                "val_crerr": float(val_crerr),
            })
            best_psnr = max(best_psnr, val_psnr)
            best_ssim = max(best_ssim, val_ssim)
            state = {
                'epoch': 0,
                'dehazer_state': model.dehazer.state_dict(),
                'model_state': model.state_dict(),
                'optimizer_state': None,
                'psnr': val_psnr,
                'ssim': val_ssim,
                'delta_sat': val_delta_sat,
                'crerr': val_crerr,
            }
            save_checkpoint(dehaze_dir, state, is_best=True)
        else:
            opt_d = torch.optim.Adam(dehazer_params, lr=args.lr)
            for epoch in range(1, args.pretrain_dehaze_epochs + 1):
                t0 = time.time()
                loss = train_epoch_dehaze_only(model, train_loader, opt_d, device, scaler, vgg_loss=vgg_loss, lam_recon=1.0,
                                               lam_ssim=args.lam_ssim, lam_perc=args.lam_perc)
                val_psnr, val_ssim, val_delta_sat, val_crerr = validate(model, val_loader, device)
                print(
                    f"[Dehaze Pretrain] Epoch {epoch} loss:{loss:.4f} "
                    f"VAL PSNR:{val_psnr:.3f} SSIM:{val_ssim:.4f} "
                    f"DeltaSat:{val_delta_sat:.4f} CRerr:{val_crerr:.4f} "
                    f"time:{time.time()-t0:.1f}s"
                )
                global_step += 1
                history.append({
                    "step": global_step,
                    "stage": "dehaze",
                    "epoch": epoch,
                    "train_loss": float(loss),
                    "val_psnr": float(val_psnr),
                    "val_ssim": float(val_ssim),
                    "val_delta_sat": float(val_delta_sat),
                    "val_crerr": float(val_crerr),
                })
                if args.train_monitor_hazy_path and (epoch % args.train_monitor_every == 0):
                    run_train_monitor_infer(model, args, device, stage="dehaze", epoch=epoch)

                state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': opt_d.state_dict(),
                    'psnr': val_psnr,
                    'ssim': val_ssim,
                    'delta_sat': val_delta_sat,
                    'crerr': val_crerr,
                }
                is_best = False
                if val_psnr > best_psnr or val_ssim > best_ssim:
                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                    if val_ssim > best_ssim:
                        best_ssim = val_ssim
                    is_best = True
                save_checkpoint(dehaze_dir, state, is_best=is_best)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    # Stage 2: freeze dehaze, train segmentation only
    if args.pretrain_seg_epochs > 0:
        print("Stage 2: Training segmentation (frozen dehaze) for", args.pretrain_seg_epochs, "epochs")
        for p in model.dehazer.parameters():
            p.requires_grad = False
        opt_s = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        for epoch in range(1, args.pretrain_seg_epochs + 1):
            t0 = time.time()
            loss = train_epoch_seg_only(model, train_loader, opt_s, device, scaler, args.num_classes)
            val_psnr, val_ssim, val_delta_sat, val_crerr = validate(model, val_loader, device)
            print(
                f"[Seg Pretrain] Epoch {epoch} loss:{loss:.4f} "
                f"VAL PSNR:{val_psnr:.3f} SSIM:{val_ssim:.4f} "
                f"DeltaSat:{val_delta_sat:.4f} CRerr:{val_crerr:.4f} "
                f"time:{time.time()-t0:.1f}s"
            )
            global_step += 1
            history.append({
                "step": global_step,
                "stage": "seg",
                "epoch": epoch,
                "train_loss": float(loss),
                "val_psnr": float(val_psnr),
                "val_ssim": float(val_ssim),
                "val_delta_sat": float(val_delta_sat),
                "val_crerr": float(val_crerr),
            })
            if args.train_monitor_hazy_path and (epoch % args.train_monitor_every == 0):
                run_train_monitor_infer(model, args, device, stage="seg", epoch=epoch)

            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': opt_s.state_dict(),
                'psnr': val_psnr,
                'ssim': val_ssim,
                'delta_sat': val_delta_sat,
                'crerr': val_crerr,
            }
            is_best = False
            if val_ssim > best_ssim or val_psnr > best_psnr:
                if val_ssim > best_ssim:
                    best_ssim = val_ssim
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                is_best = True
            save_checkpoint(seg_dir, state, is_best=is_best)

        for p in model.dehazer.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    # Stage 3: joint finetune
    finetune_epochs = args.finetune_epochs if args.finetune_epochs > 0 else args.epochs
    print("Stage 3: Joint finetune for", finetune_epochs, "epochs")
    print(
        f"[JointWarmup] seg_start_factor={args.joint_seg_start_factor:.3f} "
        f"seg_warmup_epochs={args.joint_seg_warmup_epochs}"
    )
    print(
        f"[JointMetric] scheduler={args.joint_scheduler_metric} "
        f"best={args.joint_best_metric}"
    )
    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        if args.joint_seg_warmup_epochs > 0:
            warm = min(1.0, float(epoch) / float(args.joint_seg_warmup_epochs))
        else:
            warm = 1.0
        seg_factor = args.joint_seg_start_factor + (1.0 - args.joint_seg_start_factor) * warm
        lam_seg_eff = args.lam_seg * seg_factor
        train_stats = train_epoch_joint(model, train_loader, optimizer, device, scaler, args.num_classes,
                                        args.lam_dehaze, lam_seg_eff, vgg_loss=vgg_loss,
                                        lam_ssim=args.lam_ssim, lam_perc=args.lam_perc)
        val_psnr, val_ssim, val_delta_sat, val_crerr = validate(model, val_loader, device)
        sched_score = metric_score(val_psnr, val_ssim, args.joint_scheduler_metric)
        scheduler.step(-sched_score)
        elapsed = time.time() - t0
        print(
            f"[Joint] Epoch {epoch}/{finetune_epochs} time:{elapsed:.1f}s "
            f"train_loss:{train_stats['loss_total']:.4f} "
            f"(lam_de:{args.lam_dehaze:.3f}, lam_seg_eff:{lam_seg_eff:.3f}) "
            f"VAL PSNR:{val_psnr:.3f} VAL SSIM:{val_ssim:.4f} "
            f"DeltaSat:{val_delta_sat:.4f} CRerr:{val_crerr:.4f}"
        )
        print(
            f"[JointLoss] de:{train_stats['loss_de']:.4f} seg:{train_stats['loss_seg']:.4f} "
            f"l1:{train_stats['loss_l1']:.4f} ssim:{train_stats['loss_ssim']:.4f} "
            f"perc:{train_stats['loss_perc']:.4f}"
        )
        global_step += 1
        history.append({
            "step": global_step,
            "stage": "joint",
            "epoch": epoch,
            "train_loss": float(train_stats["loss_total"]),
            "val_psnr": float(val_psnr),
            "val_ssim": float(val_ssim),
            "val_delta_sat": float(val_delta_sat),
            "val_crerr": float(val_crerr),
        })
        if args.train_monitor_hazy_path and (epoch % args.train_monitor_every == 0):
            run_train_monitor_infer(model, args, device, stage="joint", epoch=epoch)

        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'psnr': val_psnr,
            'ssim': val_ssim,
            'delta_sat': val_delta_sat,
            'crerr': val_crerr,
        }
        cur_score = metric_score(val_psnr, val_ssim, args.joint_best_metric)
        is_best = cur_score > best_joint_score
        if is_best:
            best_joint_score = cur_score
        if val_ssim > best_ssim:
            best_ssim = val_ssim
        if val_psnr > best_psnr:
            best_psnr = val_psnr
        save_checkpoint(joint_dir, state, is_best=is_best)

    print("Finished training. Best PSNR:", best_psnr, "Best SSIM:", best_ssim)
    print("Checkpoints saved in:", args.save_dir)
    if not args.no_save_metric_plots:
        save_metric_plots(history, args.save_dir)
    if not args.no_auto_infer_after_train:
        if args.auto_infer_ckpt:
            ckpt = args.auto_infer_ckpt if os.path.exists(args.auto_infer_ckpt) else None
            if ckpt is None:
                print(f"[Warn] Auto inference skipped: auto_infer_ckpt not found: {args.auto_infer_ckpt}")
        else:
            ckpt = _pick_auto_ckpt(args.save_dir, prefer_stage=args.auto_infer_ckpt_stage)
        if ckpt is None:
            print("[Warn] Auto inference skipped: no checkpoint found in save_dir.")
        else:
            print(f"[AutoInfer] ckpt={ckpt}")
            run_auto_infer(args, ckpt)

if __name__ == '__main__':
    main()
