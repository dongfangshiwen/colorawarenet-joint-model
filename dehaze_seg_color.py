#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dehaze_seg_optimized.py

浼樺寲鍘婚浘+鍒嗗壊璁粌鑴氭湰锛堝畬鏁达級
鏀瑰姩锛堜富瑕佷笌棰滆壊淇濈暀/楗卞拰搴︾浉鍏筹級锛?
 - 澧炲姞 chroma / saturation loss 鍜?color ratio loss锛堜繚鑹诧級
 - 澧炲ぇ color gain 琛ㄨ揪鑳藉姏锛堥粯璁?gain_scale=0.6锛?
 - 璋冩暣鑻ュ共鎹熷け鏉冮噸榛樿鍊间互鍑忓皯瀵硅壊褰╃殑杩囧害鎶戝埗
 - 鍦?validate 涓緭鍑哄钩鍧囬ケ鍜屽害缁熻
 - 鏂板锛氬湪 validate 涓绠楀苟杩斿洖 SSIM 鎸囨爣锛岃缁冮樁娈典繚瀛?鎵撳嵃涓寘鍚?SSIM
"""
import os
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm
from utils.ColorAwareUnet import ColorAwareUNet
from utils.LiteAttentionUnet import *
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

# -------------------------
# Repro / utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def safe_collate(batch):
    # batch: list of (hazy, clear, mask)
    hazy_list, clear_list, mask_list = zip(*batch)

    hazy = torch.stack([x.contiguous() for x in hazy_list], dim=0)
    clear = torch.stack([x.contiguous() for x in clear_list], dim=0)
    mask = torch.stack([x.contiguous() for x in mask_list], dim=0)

    return hazy, clear, mask

# -------------------------
# Dataset (same as yours)
# -------------------------
class HazySegDataset(Dataset):
    def __init__(
        self,
        root,
        num_classes,
        ids=None,
        img_exts=(".png", ".jpg", ".jpeg"),
        resize=None,
        augment=True,
        fallback_resize=(512, 512),
    ):
        self.root = Path(root)
        self.hazy_dir = self.root / 'hazy'
        self.clear_dir = self.root / 'clear'
        self.mask_dir = self.root / 'mask'
        self.num_classes = int(num_classes)
        self.resize = resize  # (H, W) or None
        self.augment = augment
        self.fallback_resize = fallback_resize
        if ids is None:
            ids = []
            for p in self.hazy_dir.iterdir():
                if p.suffix.lower() in img_exts:
                    ids.append(p.stem)
            ids.sort()
            self.ids = ids
        else:
            self.ids = ids

    def __len__(self):
        return len(self.ids)

    def _convert_mask_to_labels(self, arr, num_classes):
        if arr.ndim == 3:
            h,w,c = arr.shape
            if c == num_classes:
                return np.argmax(arr, axis=2).astype(np.int64)
            if c == 3:
                arr2 = arr[...,0]
                arr = arr2
            else:
                return np.argmax(arr, axis=2).astype(np.int64)

        arr_f = arr.astype(np.float32)
        vmin, vmax = float(arr_f.min()), float(arr_f.max())

        if np.issubdtype(arr.dtype, np.integer):
            if vmax <= (num_classes - 1):
                return arr.astype(np.int64)
            if vmax <= 255:
                labels = np.round(arr_f * ((num_classes - 1) / 255.0)).astype(np.int64)
                labels = np.clip(labels, 0, num_classes - 1)
                return labels

        if vmin >= 0.0 and vmax <= 1.0:
            labels = np.round(arr_f * (num_classes - 1)).astype(np.int64)
            labels = np.clip(labels, 0, num_classes - 1)
            return labels

        labels = np.round(arr_f).astype(np.int64)
        labels = np.clip(labels, 0, num_classes - 1)
        return labels

    # def _sync_transform(self, hazy_pil, clear_pil, mask_pil):
    #     if self.augment:
    #         if random.random() > 0.5:
    #             hazy_pil = TF.hflip(hazy_pil)
    #             clear_pil = TF.hflip(clear_pil)
    #             mask_pil = TF.hflip(mask_pil)
    #         try:
    #             w, h = hazy_pil.size
    #             if w > 320 and h > 320 and random.random() > 0.6:
    #                 crop_w = int(0.9 * w)
    #                 crop_h = int(0.9 * h)
    #                 left = random.randint(0, w - crop_w)
    #                 top = random.randint(0, h - crop_h)
    #                 hazy_pil = hazy_pil.crop((left, top, left + crop_w, top + crop_h))
    #                 clear_pil = clear_pil.crop((left, top, left + crop_w, top + crop_h))
    #                 mask_pil = mask_pil.crop((left, top, left + crop_w, top + crop_h))
    #         except Exception:
    #             pass

    #     if self.resize is not None:
    #         target_w, target_h = self.resize[1], self.resize[0]
    #         hazy_pil = hazy_pil.resize((target_w, target_h), Image.BILINEAR)
    #         clear_pil = clear_pil.resize((target_w, target_h), Image.BILINEAR)
    #         mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)

    #     return hazy_pil, clear_pil, mask_pil
    def _sync_transform(self, hazy_pil, clear_pil, mask_pil):
        # 1) augmentation锛堜笉鏀瑰彉灏哄涓€鑷存€э級
        if self.augment:
            if random.random() > 0.5:
                hazy_pil = TF.hflip(hazy_pil)
                clear_pil = TF.hflip(clear_pil)
                mask_pil = TF.hflip(mask_pil)
    
            # 鍙€夐殢鏈鸿鍓細瑁佸壀鍚庝笁鑰呭悓灏哄
            try:
                w, h = hazy_pil.size
                if w > 320 and h > 320 and random.random() > 0.6:
                    crop_w = int(0.9 * w)
                    crop_h = int(0.9 * h)
                    left = random.randint(0, w - crop_w)
                    top = random.randint(0, h - crop_h)
                    hazy_pil = hazy_pil.crop((left, top, left + crop_w, top + crop_h))
                    clear_pil = clear_pil.crop((left, top, left + crop_w, top + crop_h))
                    mask_pil = mask_pil.crop((left, top, left + crop_w, top + crop_h))
            except Exception:
                pass
    
        # 2) 寮哄埗 resize锛氫繚璇佹墍鏈夋牱鏈緭鍑哄悓灏哄
        #    濡傛灉浣犳病浼?--resize锛屽氨鐢ㄤ竴涓粯璁ゅ€硷紙寤鸿浣犳牴鎹樉瀛樻敼锛?
        if self.resize is not None:
            target_h, target_w = int(self.resize[0]), int(self.resize[1])
            hazy_pil = hazy_pil.resize((target_w, target_h), Image.BILINEAR)
            clear_pil = clear_pil.resize((target_w, target_h), Image.BILINEAR)
            mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)
        elif self.fallback_resize is not None:
            target_h, target_w = int(self.fallback_resize[0]), int(self.fallback_resize[1])
            hazy_pil = hazy_pil.resize((target_w, target_h), Image.BILINEAR)
            clear_pil = clear_pil.resize((target_w, target_h), Image.BILINEAR)
            mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)
    
        return hazy_pil, clear_pil, mask_pil

    def __getitem__(self, idx):
        id0 = self.ids[idx]
    
        def find_file(dirp, stem):
            # 浼樺厛甯歌鍚庣紑
            for ext in ('.png', '.jpg', '.jpeg'):
                p = dirp / f"{stem}{ext}"
                if p.exists():
                    return p
            # 鍏滃簳锛氫换鎰忓悗缂€
            for p in dirp.glob(f"{stem}.*"):
                return p
            return None
    
        hazy_path = find_file(self.hazy_dir, id0)
        clear_path = find_file(self.clear_dir, id0)
        mask_path = find_file(self.mask_dir, id0)
    
        if hazy_path is None or clear_path is None or mask_path is None:
            raise FileNotFoundError(f"Missing file for id {id0}")
    
        # --- load hazy/clear as PIL ---
        hazy_pil = Image.open(str(hazy_path)).convert('RGB')
        clear_pil = Image.open(str(clear_path)).convert('RGB')
    
        # --- load mask into PIL (IMPORTANT: later we only use this PIL after sync_transform) ---
        if mask_path.suffix.lower() == '.npy':
            mask_arr = np.load(str(mask_path))
            # 缁熶竴鎴?2D锛圚,W锛?
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            # 涓轰簡 transform锛屽厛杞?PIL锛堢敤 uint8锛?
            mask_pil = Image.fromarray(mask_arr.astype(np.uint8))
        else:
            # 鍙兘鏄疪GB/LA绛夛紝閮界粺涓€鎴愬崟閫氶亾鐏板害锛岄伩鍏嶅悗缁?np.array 鍑虹幇3缁?
            mask_pil = Image.open(str(mask_path)).convert('L')
    
        # --- sync augment & resize (must be applied to mask_pil too) ---
        hazy_pil, clear_pil, mask_pil = self._sync_transform(hazy_pil, clear_pil, mask_pil)
    
        # --- to tensor (make sure resizable storage) ---
        hazy = TF.to_tensor(hazy_pil).contiguous().clone()
        clear = TF.to_tensor(clear_pil).contiguous().clone()
    
        # --- mask: ONLY use the transformed mask_pil, do NOT reload npy here ---
        mask_arr = np.array(mask_pil)
        # 闃插尽锛氬鏋滃洜涓烘煇浜涘師鍥犲彉鎴?缁达紝鍙栫0閫氶亾
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
    
        mask_labels = self._convert_mask_to_labels(mask_arr, self.num_classes)
        mask_tensor = torch.as_tensor(mask_labels, dtype=torch.long).contiguous().clone()
    
        return hazy, clear, mask_tensor

    # def __getitem__(self, idx):
    #     id0 = self.ids[idx]
    #     def find_file(dirp, stem):
    #         for ext in ('.png','.jpg','.jpeg'):
    #             p = dirp / f"{stem}{ext}"
    #             if p.exists(): return p
    #         for p in dirp.glob(f"{stem}.*"):
    #             return p
    #         return None

    #     hazy_path = find_file(self.hazy_dir, id0)
    #     clear_path = find_file(self.clear_dir, id0)
    #     mask_path = find_file(self.mask_dir, id0)
        
    #     if hazy_path is None or clear_path is None or mask_path is None:
    #         raise FileNotFoundError(f"Missing file for id {id0}")

    #     hazy_pil = Image.open(str(hazy_path)).convert('RGB')
    #     clear_pil = Image.open(str(clear_path)).convert('RGB')

    #     if mask_path.suffix.lower() == '.npy':
    #         mask_arr = np.load(str(mask_path))
    #         if mask_arr.ndim == 2:
    #             mask_pil = Image.fromarray(mask_arr.astype(np.uint8))
    #         else:
    #             mask_pil = Image.fromarray(mask_arr[...,0].astype(np.uint8))
    #     else:
    #         mask_pil = Image.open(str(mask_path))

    #     hazy_pil, clear_pil, mask_pil = self._sync_transform(hazy_pil, clear_pil, mask_pil)

    #     hazy = TF.to_tensor(hazy_pil)
    #     clear = TF.to_tensor(clear_pil)

    #     if mask_path.suffix.lower() == '.npy':
    #         mask_arr = np.load(str(mask_path))
    #         if self.resize is not None:
    #             mask_arr = np.array(Image.fromarray(mask_arr).resize((self.resize[1], self.resize[0]), Image.NEAREST))
    #     else:
    #         mask_arr = np.array(mask_pil)

    #     mask_labels = self._convert_mask_to_labels(mask_arr, self.num_classes)
    #     mask_tensor = torch.from_numpy(mask_labels).long()

    #     return hazy, clear, mask_tensor

# -------------------------
# Losses / metrics (including high-frequency + color)
# -------------------------
def l1_loss(a,b): return F.l1_loss(a,b)

def ssim_loss_approx(a, b):
    a_mean = a.mean(dim=[1,2,3], keepdim=True)
    b_mean = b.mean(dim=[1,2,3], keepdim=True)
    a_var = ((a - a_mean)**2).mean(dim=[1,2,3], keepdim=True)
    b_var = ((b - b_mean)**2).mean(dim=[1,2,3], keepdim=True)
    cov = ((a - a_mean)*(b - b_mean)).mean(dim=[1,2,3], keepdim=True)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2*a_mean*b_mean + c1)*(2*cov + c2))/((a_mean**2 + b_mean**2 + c1)*(a_var + b_var + c2))
    return torch.clamp(1.0 - ssim_map.mean(), 0.0, 1.0)

# Add SSIM metric wrapper (returns SSIM in [0,1])
def ssim_metric(a, b):
    # ssim_loss_approx returns (1 - ssim), so invert it
    with torch.no_grad():
        val = 1.0 - ssim_loss_approx(a, b)
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

def dice_metric(pred_logits, target, num_classes, eps=1e-6, ignore_index=None):
    """
    pred_logits: BxCxHxW
    target:      BxHxW (long)
    return: mean dice over classes (macro dice), float
    """
    pred = pred_logits.argmax(dim=1)  # BxHxW

    dices = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_c = (pred == cls)
        targ_c = (target == cls)

        # 濡傛灉璇ョ被鍦?GT 鍜?Pred 閮戒笉瀛樺湪锛岄€氬父璺宠繃锛堜笉璁″叆鍧囧€硷級
        denom = pred_c.sum().item() + targ_c.sum().item()
        if denom == 0:
            continue

        inter = (pred_c & targ_c).sum().item()
        dice = (2.0 * inter + eps) / (denom + eps)
        dices.append(dice)

    if len(dices) == 0:
        return 0.0
    return float(sum(dices) / len(dices))
    
def psnr(a,b, max_val=1.0):
    mse = torch.mean((a - b)**2)
    return 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))

def compute_mIoU(pred_logits, target, num_classes):
    pred = pred_logits.argmax(dim=1)
    iou_list = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union == 0:
            continue
        iou_list.append(inter/union)
    if len(iou_list) == 0:
        return 0.0
    return float(sum(iou_list)/len(iou_list))

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

def color_mean_loss(pred, target):
    p_mean = pred.mean(dim=[2,3])  # Bx3
    t_mean = target.mean(dim=[2,3])
    return F.mse_loss(p_mean, t_mean)

def luminance_mean(img):
    r = img[:,0:1,:,:]; g = img[:,1:2,:,:]; b = img[:,2:3,:,:]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.mean(dim=[2,3])

def luminance_loss(pred, target):
    p = luminance_mean(pred)
    t = luminance_mean(target)
    return F.mse_loss(p, t)

def contrast_scalar(x):
    r = x[:,0:1,:,:]; g = x[:,1:2,:,:]; b = x[:,2:3,:,:]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    y_flat = y.view(y.shape[0], -1)
    return y_flat.std(dim=1).mean()

def contrast_loss(pred, target):
    return F.mse_loss(contrast_scalar(pred), contrast_scalar(target))

# high-frequency kernels
_sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
_sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
_lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32).view(1,1,3,3)

def gradient_loss(pred, target):
    device = pred.device
    kx = _sobel_x.to(device); ky = _sobel_y.to(device)
    gx_p = F.conv2d(pred, kx.repeat(3,1,1,1), padding=1, groups=3)
    gy_p = F.conv2d(pred, ky.repeat(3,1,1,1), padding=1, groups=3)
    gx_t = F.conv2d(target, kx.repeat(3,1,1,1), padding=1, groups=3)
    gy_t = F.conv2d(target, ky.repeat(3,1,1,1), padding=1, groups=3)
    gp_mag = torch.sqrt(gx_p**2 + gy_p**2 + 1e-8)
    gt_mag = torch.sqrt(gx_t**2 + gy_t**2 + 1e-8)
    return F.l1_loss(gp_mag, gt_mag)

def laplacian_map_loss(pred, target):
    device = pred.device
    k = _lap.to(device).repeat(3,1,1,1)
    lap_p = F.conv2d(pred, k, padding=1, groups=3)
    lap_t = F.conv2d(target, k, padding=1, groups=3)
    return F.l1_loss(lap_p, lap_t)

def variance_of_laplacian(x):
    device = x.device
    k = _lap.to(device).repeat(3,1,1,1)
    lap = F.conv2d(x, k, padding=1, groups=3)
    b = lap.view(lap.shape[0], -1)
    return b.var(dim=1).mean()

# -------------------------
# NEW: color-preserving losses (chroma, saturation, color ratio)
# -------------------------
_eps = 1e-6

def chroma_and_saturation(pred, target):
    """
    pred, target: Bx3xHxW in [0,1]
    returns: (chroma_loss, sat_loss)
    chroma = max(channel) - min(channel)
    sat approximated as chroma / (max + eps)
    """
    p_max, _ = pred.max(dim=1, keepdim=True)
    p_min, _ = pred.min(dim=1, keepdim=True)
    t_max, _ = target.max(dim=1, keepdim=True)
    t_min, _ = target.min(dim=1, keepdim=True)

    p_chroma = p_max - p_min
    t_chroma = t_max - t_min

    p_sat = p_chroma / (p_max + _eps)
    t_sat = t_chroma / (t_max + _eps)

    chroma_loss = F.l1_loss(p_chroma, t_chroma)
    sat_loss = F.l1_loss(p_sat, t_sat)
    return chroma_loss, sat_loss

def color_ratio_loss(pred, target):
    """
    Match per-pixel normalized RGB ratios: r/(r+g+b)
    This preserves chromaticity independently of brightness.
    """
    s_pred = pred.sum(dim=1, keepdim=True) + 1e-6
    s_tgt = target.sum(dim=1, keepdim=True) + 1e-6
    r_pred = pred[:,0:1,:,:] / s_pred
    g_pred = pred[:,1:2,:,:] / s_pred
    b_pred = pred[:,2:3,:,:] / s_pred
    r_t = target[:,0:1,:,:] / s_tgt
    g_t = target[:,1:2,:,:] / s_tgt
    b_t = target[:,2:3,:,:] / s_tgt
    return F.l1_loss(r_pred, r_t) + F.l1_loss(g_pred, g_t) + F.l1_loss(b_pred, b_t)

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
# Train / Val loops (integrate side outputs & new color losses)
# -------------------------
def _unpack_dehaze_output(dz_out):
    if isinstance(dz_out, (list, tuple)):
        out = dz_out[0]
        sides = dz_out[3] if len(dz_out) > 3 else [None, None, None]
    else:
        out = dz_out
        sides = [None, None, None]
    return out, sides

def train_epoch_dehaze_only(model, dataloader, optimizer, device, scaler, vgg_loss=None,
                            lam_recon=1.0, lam_perc=0.0, lam_color=0.12, lam_bright=0.2, lam_contrast=0.05,
                            lam_grad=0.10, lam_lap=0.10, lam_cratio=0.20, lam_chroma=0.05, lam_sat=0.05,
                            ms_weights=(0.6,0.4,0.2), print_every=50):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='TrainDehaze')
    for i, (hazy, clear, _) in pbar:
        hazy = hazy.to(device); clear = clear.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            dz_out = model.dehazer(hazy)
            dehazed, sides = _unpack_dehaze_output(dz_out)
            loss_l1 = l1_loss(dehazed, clear)
            loss_ssim = 0.7 * ssim_loss_approx(dehazed, clear)
            loss_perc = lam_perc * (vgg_loss(dehazed, clear) if vgg_loss is not None else 0.0)
            loss_color = lam_color * color_mean_loss(dehazed, clear)
            loss_bright = lam_bright * luminance_loss(dehazed, clear)
            loss_contrast = lam_contrast * contrast_loss(dehazed, clear)
            loss_grad = lam_grad * gradient_loss(dehazed, clear)
            loss_lap = lam_lap * laplacian_map_loss(dehazed, clear)
            chroma_l, sat_l = chroma_and_saturation(dehazed, clear)
            loss_chroma = lam_chroma * chroma_l
            loss_sat = lam_sat * sat_l
            loss_cratio = lam_cratio * color_ratio_loss(dehazed, clear)
            loss_ms = 0.0
            if sides is not None:
                for w, s in zip(ms_weights, sides):
                    if s is not None:
                        loss_ms = loss_ms + w * l1_loss(s, clear)

            loss_de = (
                loss_l1 + loss_ssim + loss_perc + loss_color + loss_bright + loss_contrast +
                loss_grad + loss_lap + loss_chroma + loss_sat + loss_cratio + loss_ms
            )
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
                dz_out = model.dehazer(hazy)
                dehazed, _ = _unpack_dehaze_output(dz_out)
                dehazed = torch.clamp(dehazed, 0.0, 1.0)
            seg_in = (dehazed - model.imagenet_mean) / model.imagenet_std
            # seg_out = model.seg(seg_in)['out']
            tmp = model.seg(seg_in)
            if isinstance(tmp, dict):
                seg_out = tmp.get('out')
            else:
                seg_out = tmp
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
                      lam_dehaze, lam_seg, vgg_loss=None, lam_perc=0.01, lam_color=0.12,
                      lam_bright=0.2, lam_contrast=0.05,
                      lam_grad=0.10, lam_lap=0.10, lam_cratio=0.20, lam_chroma=0.05, lam_sat=0.05,
                      ms_weights=(0.6,0.4,0.2), print_every=50):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='TrainJoint')
    for i, (hazy, clear, mask) in pbar:
        hazy = hazy.to(device); clear = clear.to(device); mask = mask.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            dehazed, seg_out, sides = model(hazy)
            loss_l1 = l1_loss(dehazed, clear)
            loss_ssim = 0.5 * ssim_loss_approx(dehazed, clear)
            loss_perc = lam_perc * (vgg_loss(dehazed, clear) if vgg_loss is not None else 0.0)
            loss_color = lam_color * color_mean_loss(dehazed, clear)
            loss_bright = lam_bright * luminance_loss(dehazed, clear)
            loss_contrast = lam_contrast * contrast_loss(dehazed, clear)
            loss_grad = lam_grad * gradient_loss(dehazed, clear)
            loss_lap = lam_lap * laplacian_map_loss(dehazed, clear)
            chroma_l, sat_l = chroma_and_saturation(dehazed, clear)
            loss_chroma = lam_chroma * chroma_l
            loss_sat = lam_sat * sat_l
            loss_cratio = lam_cratio * color_ratio_loss(dehazed, clear)
            loss_ms = 0.0
            if sides is not None:
                for w, s in zip(ms_weights, sides):
                    if s is not None:
                        loss_ms = loss_ms + w * l1_loss(s, clear)

            loss_de = (
                loss_l1 + loss_ssim + loss_perc + loss_color + loss_bright + loss_contrast +
                loss_grad + loss_lap + loss_chroma + loss_sat + loss_cratio + loss_ms
            )

            loss_seg = F.cross_entropy(seg_out, mask) + dice_loss(seg_out, mask)
            loss = lam_dehaze * loss_de + lam_seg * loss_seg

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

def validate(model, dataloader, device, num_classes, ignore_index=None):
    model.eval()
    tot_psnr = 0.0
    tot_miou = 0.0
    tot_ssim = 0.0
    tot_dice = 0.0
    tot_sat_ref = 0.0
    tot_sat_pred = 0.0
    n = 0
    with torch.no_grad():
        for hazy, clear, mask in tqdm(dataloader, desc='Val'):
            hazy = hazy.to(device); clear = clear.to(device); mask = mask.to(device)
            dehazed, seg_out, _ = model(hazy)

            tot_psnr += psnr(dehazed, clear).item()
            tot_miou += compute_mIoU(seg_out, mask, num_classes)
            tot_ssim += ssim_metric(dehazed, clear)

            # Dice metric
            tot_dice += dice_metric(seg_out, mask, num_classes=num_classes, ignore_index=ignore_index)

            tot_sat_ref += mean_saturation(clear)
            tot_sat_pred += mean_saturation(dehazed)
            n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_psnr = tot_psnr / n
    avg_miou = tot_miou / n
    avg_ssim = tot_ssim / n
    avg_dice = tot_dice / n
    avg_sat_ref = tot_sat_ref / n
    avg_sat_pred = tot_sat_pred / n

    print(f"[ValSat] mean_sat_ref:{avg_sat_ref:.4f} mean_sat_pred:{avg_sat_pred:.4f} "
          f"mean_ssim:{avg_ssim:.4f} mean_dice:{avg_dice:.4f}")

    return avg_psnr, avg_miou, avg_ssim, avg_dice
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


def split_ids(
    ids: List[str],
    val_ratio: float,
    seed: int,
    split_by_scene: bool = True,
) -> Tuple[List[str], List[str]]:
    if not ids:
        return [], []
    val_ratio = min(max(float(val_ratio), 0.01), 0.9)
    rng = random.Random(seed)
    if split_by_scene:
        groups: Dict[str, List[str]] = {}
        for sid in ids:
            key = sid.split("_")[0] if "_" in sid else sid
            groups.setdefault(key, []).append(sid)
        keys = list(groups.keys())
        rng.shuffle(keys)
        n_scene = len(keys)
        n_val_scene = max(1, int(round(n_scene * val_ratio)))
        n_val_scene = min(n_val_scene, n_scene - 1) if n_scene > 1 else 1
        val_keys = set(keys[:n_val_scene])
        train_ids, val_ids = [], []
        for k in keys:
            if k in val_keys:
                val_ids.extend(groups[k])
            else:
                train_ids.extend(groups[k])
        return train_ids, val_ids
    ids_copy = list(ids)
    rng.shuffle(ids_copy)
    n = len(ids_copy)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1) if n > 1 else 1
    val_ids = ids_copy[:n_val]
    train_ids = ids_copy[n_val:]
    return train_ids, val_ids


def resolve_dehazer_state_dict(ck):
    if isinstance(ck, dict) and 'dehazer_state' in ck:
        return ck['dehazer_state']
    if isinstance(ck, dict) and 'model_state' in ck and isinstance(ck['model_state'], dict):
        st = ck['model_state']
        dz_state = {k.replace('dehazer.', '', 1): v for k, v in st.items() if k.startswith('dehazer.')}
        return dz_state if dz_state else st
    if isinstance(ck, dict) and 'state_dict' in ck and isinstance(ck['state_dict'], dict):
        st = ck['state_dict']
        dz_state = {k.replace('dehazer.', '', 1): v for k, v in st.items() if k.startswith('dehazer.')}
        return dz_state if dz_state else st
    return ck


def load_dehazer_checkpoint(dehazer: nn.Module, ckpt_path: str) -> None:
    ck = torch.load(ckpt_path, map_location='cpu')
    state = resolve_dehazer_state_dict(ck)
    if isinstance(state, dict):
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
        if any(k.startswith('dehazer.') for k in state):
            state = {k.replace('dehazer.', '', 1): v for k, v in state.items()}
        missing, unexpected = dehazer.load_state_dict(state, strict=False)
        loaded = len(dehazer.state_dict()) - len(missing)
        print(f"[InitCKPT] loaded dehazer params: {loaded}/{len(dehazer.state_dict())}")
        if missing:
            print(f"[InitCKPT] missing keys: {len(missing)}")
        if unexpected:
            print(f"[InitCKPT] unexpected keys: {len(unexpected)}")
    else:
        raise RuntimeError("Resolved dehazer state is not a dict.")


def save_stage_checkpoints(
    stage_dir: str,
    state: Dict,
    metrics: Dict[str, float],
    best_map: Dict[str, float],
) -> List[str]:
    os.makedirs(stage_dir, exist_ok=True)
    save_checkpoint(stage_dir, state, is_best=False)
    improved = []
    for k, v in metrics.items():
        old = best_map.get(k, -1e18)
        if v > old:
            best_map[k] = v
            torch.save(state, os.path.join(stage_dir, f"best_{k}.pth"))
            improved.append(k)
    if improved:
        save_checkpoint(stage_dir, state, is_best=True)
    return improved

def build_dehazer_from_args(args):
    if args.dehaze_backbone == 'coloraware':
        dehazer = ColorAwareUNet(
            in_ch=3,
            base_ch=32,              # 鍥哄畾涓?32
            residual_scale=0.5,
            gain_scale=0.65,
            gain_form=args.color_gain_form,
            gain_min=args.color_gain_min,
            disable_gain=args.color_disable_gain,
        )

    elif args.dehaze_backbone == 'dehazenet':
        dehazer = DehazeNet(
            in_ch=3,
            maxout_k=2,              # 鍥哄畾涓?2
            guided_filter_refine=False,
            t0=0.05                  # 鍥哄畾涓?0.05
        )

    elif args.dehaze_backbone == 'aodnet':
        dehazer = AODNet(
            in_ch=3,
            mid_ch=8,                # 鍥哄畾涓?8
            use_tanh_on_K=False,
            b=1.0
        )

    elif args.dehaze_backbone == 'grid':
        dehazer = GridDehazeNet(
            in_ch=3,
            rows=3,                  # 鍥哄畾涓?3
            cols=6,                  # 鍥哄畾涓?6
            base_ch=32               # 鍥哄畾涓?32
        )

    elif args.dehaze_backbone == 'dehamer':
        dehazer = DehamerNet(
            in_ch=3,
            base_ch=16,              # 鍥哄畾涓?16锛堝搴斾綘涔嬪墠鐨勫啓娉曪級
            trans_dim=64,
            nheads=4,
            n_layers=2,
            patch_size=8,
            use_learnable_pos=True
        )

    elif args.dehaze_backbone == 'ffanet':
        dehazer = FFANet(
            in_ch=3,
            base_ch=32,              # 鍥哄畾涓?32
            n_down=2,                # 鍥哄畾涓?2
            n_ffab_deep=2            # 鍥哄畾涓?2
        )

    elif args.dehaze_backbone == 'c2pnet':
        dehazer = C2PNet(
            in_ch=3,
            base_ch=32,              # 鍥哄畾涓?32
            blocks_per_group=6,      # 鍥哄畾涓?6
            groups=3,                # 鍥哄畾涓?3
            use_pdu=True
        )

    elif args.dehaze_backbone == 'msbdn':
        dehazer = MSBDN(
            in_ch=3,
            base_ch=32,              # 鍥哄畾涓?32
            nblocks=3                # 鍥哄畾涓?3
        )
        
    elif args.dehaze_backbone == 'dcp':
        dehazer = DCPDehaze(
            in_ch=3,
            patch_size=15,
            omega=0.95,
            t0=0.1,
            top_percent=0.001,
            guided=True,
            guided_radius=15,
            guided_eps=1e-3,
            use_learned_refine=True  
        )
    elif args.dehaze_backbone == 'dehazeunet':
        dehazer = DehazeUNet(
            in_ch=3,
            base_ch=32
        )
    
    elif args.dehaze_backbone == 'd4':
        dehazer = D4DehazeNet(
            in_ch=3,
            base_ch=32   # 鍙互瑙嗘樉瀛樿皟鎴?16/32
        )
    elif args.dehaze_backbone == 'psd':
        dehazer = PSDDehazeNet(
            in_ch=3,
            base_ch=32,   # 鍙寜鏄惧瓨璋?16/32
            feat_ch=64
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
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--resize', type=int, nargs=2, default=None, help='H W or none')
    parser.add_argument('--fallback_resize', type=int, nargs=2, default=[512, 512],
                        help='used when --resize is not set; set 0 0 to disable fallback resize')
    parser.add_argument('--val_ratio', type=float, default=0.08, help='validation split ratio in (0,1)')
    parser.add_argument('--no_split_by_scene', action='store_true',
                        help='disable scene-level split and split by sample id directly')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_optimized')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--lam_dehaze', type=float, default=1.0)
    parser.add_argument('--lam_seg', type=float, default=1.0)
    parser.add_argument('--lam_perc', type=float, default=0.01)
    parser.add_argument('--lam_color', type=float, default=0.12)
    parser.add_argument('--lam_bright', type=float, default=0.2)
    parser.add_argument('--lam_contrast', type=float, default=0.05)
    parser.add_argument('--lam_grad', type=float, default=0.10)
    parser.add_argument('--lam_lap', type=float, default=0.10)
    parser.add_argument('--lam_cratio', type=float, default=0.20)
    parser.add_argument('--lam_chroma', type=float, default=0.05)
    parser.add_argument('--lam_sat', type=float, default=0.05)
    parser.add_argument('--color_gain_form', type=str, default='tanh', choices=['tanh', 'amp'])
    parser.add_argument('--color_gain_min', type=float, default=0.90)
    parser.add_argument('--color_disable_gain', action='store_true')
    parser.add_argument('--pretrained_seg', action='store_true')
    parser.add_argument('--pretrain_dehaze_epochs', type=int, default=0)
    parser.add_argument('--pretrain_seg_epochs', type=int, default=0)
    parser.add_argument('--finetune_epochs', type=int, default=0)
    parser.add_argument('--dehaze_pretrained_path', type=str, default=None)
    parser.add_argument('--use_attention', action='store_true', help='enable attention gate in segmentation net')
    parser.add_argument('--use_se', action='store_true', help='enable SE blocks in segmentation net')
    parser.add_argument('--disable_ms', action='store_true',
                    help='disable multi-scale side supervision (side2/side3/side4 L1)')
    parser.add_argument('--dehaze_backbone', type=str, default='coloraware', choices=['coloraware', 'dehazenet', 'aodnet', 'grid', 'dehamer', 'ffanet', 'c2pnet', 'msbdn', 'dcp', 'dehazeunet', 'd4', 'psd'], help='which dehaze sub-network to use')
    args = parser.parse_args()
    ms_weights = (0.0, 0.0, 0.0) if args.disable_ms else (0.6, 0.4, 0.2)
    print(f"[Ablation] ms_weights = {ms_weights}")
    finetune_epochs = args.finetune_epochs if args.finetune_epochs > 0 else args.epochs
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    resize_hw = tuple(args.resize) if args.resize else None
    fallback_resize = None
    if args.fallback_resize and len(args.fallback_resize) == 2:
        fh, fw = int(args.fallback_resize[0]), int(args.fallback_resize[1])
        if fh > 0 and fw > 0:
            fallback_resize = (fh, fw)

    ds_meta = HazySegDataset(
        args.data_root,
        num_classes=args.num_classes,
        resize=resize_hw,
        augment=False,
        fallback_resize=fallback_resize,
    )
    n = len(ds_meta)
    if n == 0:
        raise RuntimeError("Dataset empty. Check data_root with hazy/ clear/ mask subfolders.")
    if n < 2:
        raise RuntimeError("Need at least 2 samples to split train/val.")
    split_by_scene = not args.no_split_by_scene
    train_ids, val_ids = split_ids(ds_meta.ids, args.val_ratio, args.seed, split_by_scene=split_by_scene)
    n_train, n_val = len(train_ids), len(val_ids)
    if n_train == 0 or n_val == 0:
        raise RuntimeError(f"Invalid split: train={n_train}, val={n_val}. Adjust --val_ratio.")
    train_ds = HazySegDataset(
        args.data_root,
        num_classes=args.num_classes,
        ids=train_ids,
        resize=resize_hw,
        augment=True,
        fallback_resize=fallback_resize,
    )
    val_ds = HazySegDataset(
        args.data_root,
        num_classes=args.num_classes,
        ids=val_ids,
        resize=resize_hw,
        augment=False,
        fallback_resize=fallback_resize,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=safe_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=safe_collate,)
    print(f"Dataset sizes: total={n} train={n_train} val={n_val}")
    print(f"[Split] mode={'scene' if split_by_scene else 'sample'} val_ratio={args.val_ratio}")

    dehaze_dir = os.path.join(args.save_dir, 'dehaze')
    seg_dir = os.path.join(args.save_dir, 'seg')
    joint_dir = os.path.join(args.save_dir, 'joint')
    os.makedirs(dehaze_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(joint_dir, exist_ok=True)

    # dehazer = ColorAwareUNet(in_ch=3, base_ch=32, residual_scale=0.5, gain_scale=0.65)
    dehazer = build_dehazer_from_args(args)
    seg_net = LiteAttentionUNet(num_classes=args.num_classes, width_mult=1.0, base_ch=32, use_se=args.use_se, attention=args.use_attention, aux=False)
    model = DehazeLightSegWrapper(dehazer, seg_net, imagenet_norm=True).to(device)
    if args.dehaze_pretrained_path:
        try:
            load_dehazer_checkpoint(model.dehazer, args.dehaze_pretrained_path)
            print("Loaded dehazer weights from checkpoint.")
        except Exception as e:
            print("Error loading dehaze_pretrained_path:", e)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None
    vgg_loss = VGGPerceptualLoss(device=device)

    stage1_best = {"psnr": -1e18, "ssim": -1e18, "miou": -1e18, "dice": -1e18}
    stage2_best = {"psnr": -1e18, "ssim": -1e18, "miou": -1e18, "dice": -1e18}
    joint_best = {"psnr": -1e18, "ssim": -1e18, "miou": -1e18, "dice": -1e18}
    
    # Stage 1: pretrain dehaze only
    if args.pretrain_dehaze_epochs > 0:
        print("Stage 1: Pretraining dehaze for", args.pretrain_dehaze_epochs, "epochs")
        opt_d = torch.optim.Adam(model.dehazer.parameters(), lr=args.lr)
        for epoch in range(1, args.pretrain_dehaze_epochs + 1):
            t0 = time.time()
            loss = train_epoch_dehaze_only(model, train_loader, opt_d, device, scaler, vgg_loss=vgg_loss, lam_recon=1.0,
                                           lam_perc=args.lam_perc, lam_color=args.lam_color,
                                           lam_bright=args.lam_bright, lam_contrast=args.lam_contrast,
                                           lam_grad=args.lam_grad, lam_lap=args.lam_lap,
                                           lam_cratio=args.lam_cratio, lam_chroma=args.lam_chroma, lam_sat=args.lam_sat,
                                           ms_weights=ms_weights)
            val_psnr, val_miou, val_ssim, val_dice = validate(model, val_loader, device, args.num_classes)
            print(f"[Dehaze Pretrain] Epoch {epoch} loss:{loss:.4f} VAL PSNR:{val_psnr:.3f} VAL SSIM:{val_ssim:.4f} VAl Dice:{val_dice:.4f} mIoU:{val_miou:.4f} time:{time.time()-t0:.1f}s")

            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': opt_d.state_dict(),
                'psnr': val_psnr,
                'miou': val_miou,
                'ssim': val_ssim,
                'dice': val_dice,
            }
            improved = save_stage_checkpoints(
                dehaze_dir,
                state,
                {"psnr": val_psnr, "ssim": val_ssim, "miou": val_miou, "dice": val_dice},
                stage1_best,
            )
            if improved:
                print(f"[Dehaze Pretrain] New best: {', '.join(improved)}")
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
            val_psnr, val_miou, val_ssim, val_dice = validate(model, val_loader, device, args.num_classes)
            print(f"[Seg Pretrain] Epoch {epoch} loss:{loss:.4f} VAL PSNR:{val_psnr:.3f} VAL SSIM:{val_ssim:.4f} VAL Dice:{val_dice:.4f} mIoU:{val_miou:.4f} time:{time.time()-t0:.1f}s")

            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': opt_s.state_dict(),
                'psnr': val_psnr,
                'miou': val_miou,
                'ssim': val_ssim,
                'dice': val_dice,
            }
            improved = save_stage_checkpoints(
                seg_dir,
                state,
                {"psnr": val_psnr, "ssim": val_ssim, "miou": val_miou, "dice": val_dice},
                stage2_best,
            )
            if improved:
                print(f"[Seg Pretrain] New best: {', '.join(improved)}")

        for p in model.dehazer.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    # Stage 3: joint finetune
    finetune_epochs = args.finetune_epochs if args.finetune_epochs > 0 else args.epochs
    print("Stage 3: Joint finetune for", finetune_epochs, "epochs")
    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch_joint(model, train_loader, optimizer, device, scaler, args.num_classes, args.lam_dehaze, args.lam_seg,
                                      vgg_loss=vgg_loss, lam_perc=args.lam_perc, lam_color=args.lam_color,
                                      lam_bright=args.lam_bright, lam_contrast=args.lam_contrast,
                                      lam_grad=args.lam_grad, lam_lap=args.lam_lap,
                                      lam_cratio=args.lam_cratio, lam_chroma=args.lam_chroma, lam_sat=args.lam_sat,
                                      ms_weights=ms_weights)
        val_psnr, val_miou, val_ssim, val_dice = validate(model, val_loader, device, args.num_classes)
        scheduler.step(1.0 - val_miou)
        elapsed = time.time() - t0
        print(f"[Joint] Epoch {epoch}/{finetune_epochs} time:{elapsed:.1f}s train_loss:{train_loss:.4f} "
      f"VAL PSNR:{val_psnr:.3f} VAL SSIM:{val_ssim:.4f} VAL Dice:{val_dice:.4f} mIoU:{val_miou:.4f}")
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'psnr': val_psnr,
            'miou': val_miou,
            'ssim': val_ssim,
            'dice': val_dice,
        }
        improved = save_stage_checkpoints(
            joint_dir,
            state,
            {"psnr": val_psnr, "ssim": val_ssim, "miou": val_miou, "dice": val_dice},
            joint_best,
        )
        if improved:
            print(f"[Joint] New best: {', '.join(improved)}")
    print("Finished training. Joint best mIoU:", joint_best["miou"], 
              "Best PSNR:", joint_best["psnr"], 
              "Best SSIM:", joint_best["ssim"],
              "Best Dice:", joint_best["dice"])
    print("Checkpoints saved in:", args.save_dir)

if __name__ == '__main__':
    main()

