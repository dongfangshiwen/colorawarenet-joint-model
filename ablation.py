import os
import csv
import time
import random
import argparse
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils.ColorAwareUnet import ColorAwareUNet
from utils.LiteAttentionUnet import LiteAttentionUNet, DehazeLightSegWrapper


# -------------------------
# Repro
# -------------------------
def set_seed(seed=42, deterministic=False, strict_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 更严格（可选，某些算子可能不支持）
    if strict_deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[Warn] use_deterministic_algorithms(True) failed: {e}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------
# Dataset
# -------------------------
class HazySegDataset(Dataset):
    def __init__(self, root, num_classes, ids=None, img_exts=(".png", ".jpg", ".jpeg"), resize=None, augment=True):
        self.root = Path(root)
        self.hazy_dir = self.root / "hazy"
        self.clear_dir = self.root / "clear"
        self.mask_dir = self.root / "mask"
        self.num_classes = int(num_classes)
        self.resize = resize
        self.augment = augment

        if ids is None:
            ids = []
            for p in self.hazy_dir.iterdir():
                if p.suffix.lower() in img_exts:
                    ids.append(p.stem)
            ids.sort()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def _convert_mask_to_labels(self, arr, num_classes):
        if arr.ndim == 3:
            h, w, c = arr.shape
            if c == num_classes:
                return np.argmax(arr, axis=2).astype(np.int64)
            if c == 3:
                arr = arr[..., 0]
            else:
                return np.argmax(arr, axis=2).astype(np.int64)

        arr_f = arr.astype(np.float32)
        vmin, vmax = float(arr_f.min()), float(arr_f.max())

        if np.issubdtype(arr.dtype, np.integer):
            if vmax <= (num_classes - 1):
                return arr.astype(np.int64)
            if vmax <= 255:
                labels = np.round(arr_f * ((num_classes - 1) / 255.0)).astype(np.int64)
                return np.clip(labels, 0, num_classes - 1)

        if vmin >= 0.0 and vmax <= 1.0:
            labels = np.round(arr_f * (num_classes - 1)).astype(np.int64)
            return np.clip(labels, 0, num_classes - 1)

        labels = np.round(arr_f).astype(np.int64)
        return np.clip(labels, 0, num_classes - 1)

    def _sync_transform(self, hazy_pil, clear_pil, mask_pil):
        if self.augment:
            if random.random() > 0.5:
                hazy_pil = TF.hflip(hazy_pil)
                clear_pil = TF.hflip(clear_pil)
                mask_pil = TF.hflip(mask_pil)

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

        if self.resize is not None:
            tw, th = self.resize[1], self.resize[0]
            hazy_pil = hazy_pil.resize((tw, th), Image.BILINEAR)
            clear_pil = clear_pil.resize((tw, th), Image.BILINEAR)
            mask_pil = mask_pil.resize((tw, th), Image.NEAREST)

        return hazy_pil, clear_pil, mask_pil

    def __getitem__(self, idx):
        id0 = self.ids[idx]

        def find_file(dirp, stem):
            for ext in (".png", ".jpg", ".jpeg"):
                p = dirp / f"{stem}{ext}"
                if p.exists():
                    return p
            for p in dirp.glob(f"{stem}.*"):
                return p
            return None

        hazy_path = find_file(self.hazy_dir, id0)
        clear_path = find_file(self.clear_dir, id0)
        mask_path = find_file(self.mask_dir, id0)
        if hazy_path is None or clear_path is None or mask_path is None:
            raise FileNotFoundError(f"Missing file for id {id0}")

        hazy_pil = Image.open(str(hazy_path)).convert("RGB")
        clear_pil = Image.open(str(clear_path)).convert("RGB")

        if mask_path.suffix.lower() == ".npy":
            mask_arr0 = np.load(str(mask_path))
            mask_pil = Image.fromarray(mask_arr0[..., 0].astype(np.uint8)) if mask_arr0.ndim == 3 else Image.fromarray(mask_arr0.astype(np.uint8))
        else:
            mask_pil = Image.open(str(mask_path))

        hazy_pil, clear_pil, mask_pil = self._sync_transform(hazy_pil, clear_pil, mask_pil)

        hazy = TF.to_tensor(hazy_pil)
        clear = TF.to_tensor(clear_pil)

        if mask_path.suffix.lower() == ".npy":
            mask_arr = np.load(str(mask_path))
            if self.resize is not None:
                mask_arr = np.array(Image.fromarray(mask_arr).resize((self.resize[1], self.resize[0]), Image.NEAREST))
        else:
            mask_arr = np.array(mask_pil)

        mask_labels = self._convert_mask_to_labels(mask_arr, self.num_classes)
        mask_tensor = torch.from_numpy(mask_labels).long()
        return hazy, clear, mask_tensor


# -------------------------
# Loss / Metrics
# -------------------------
def l1_loss(a, b): return F.l1_loss(a, b)

def ssim_loss_approx(a, b):
    a_mean = a.mean(dim=[1, 2, 3], keepdim=True)
    b_mean = b.mean(dim=[1, 2, 3], keepdim=True)
    a_var = ((a - a_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    b_var = ((b - b_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    cov = ((a - a_mean) * (b - b_mean)).mean(dim=[1, 2, 3], keepdim=True)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * a_mean * b_mean + c1) * (2 * cov + c2)) / ((a_mean**2 + b_mean**2 + c1) * (a_var + b_var + c2))
    return torch.clamp(1.0 - ssim_map.mean(), 0.0, 1.0)

def ssim_metric(a, b):
    return float((1.0 - ssim_loss_approx(a, b)).item())

def psnr(a, b, max_val=1.0):
    mse = torch.mean((a - b) ** 2)
    return 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))

def dice_loss(pred_logits, target, eps=1e-6):
    probs = F.softmax(pred_logits, dim=1)
    B, C, H, W = probs.shape
    target_one = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
    inter = (probs * target_one).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_one.sum(dim=(2, 3))
    dice = 1.0 - ((2 * inter + eps) / (union + eps))
    return dice.mean()

def compute_mIoU(pred_logits, target, num_classes):
    pred = pred_logits.argmax(dim=1)
    iou_list = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union == 0:
            continue
        iou_list.append(inter / union)
    return float(sum(iou_list) / len(iou_list)) if iou_list else 0.0

def mean_saturation(img):
    p_max, _ = img.max(dim=1)
    p_min, _ = img.min(dim=1)
    sat = (p_max - p_min) / (p_max + 1e-6)
    return float(sat.mean().item())

def color_ratio_loss(pred, target):
    s_pred = pred.sum(dim=1, keepdim=True) + 1e-6
    s_tgt = target.sum(dim=1, keepdim=True) + 1e-6
    pr = pred[:, 0:1, :, :] / s_pred
    pg = pred[:, 1:2, :, :] / s_pred
    pb = pred[:, 2:3, :, :] / s_pred
    tr = target[:, 0:1, :, :] / s_tgt
    tg = target[:, 1:2, :, :] / s_tgt
    tb = target[:, 2:3, :, :] / s_tgt
    return F.l1_loss(pr, tr) + F.l1_loss(pg, tg) + F.l1_loss(pb, tb)


# -------------------------
# Train / Val
# -------------------------
def train_one_epoch(model, loader, optimizer, device, scaler, num_classes,
                    lam_dehaze=1.0, lam_seg=1.0, lam_l1=1.0, ssim_w=0.8, lam_cratio=0.0, print_every=50):
    model.train()
    running = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train")
    for i, (hazy, clear, mask) in pbar:
        hazy = hazy.to(device); clear = clear.to(device); mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            dehazed, seg_out, _ = model(hazy)

            loss_de = lam_l1 * l1_loss(dehazed, clear) + ssim_w * ssim_loss_approx(dehazed, clear)
            if lam_cratio > 0:
                loss_de = loss_de + lam_cratio * color_ratio_loss(dehazed, clear)

            loss_seg = F.cross_entropy(seg_out, mask) + dice_loss(seg_out, mask)
            loss_all = lam_dehaze * loss_de + lam_seg * loss_seg

        if scaler is not None:
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_all.backward()
            optimizer.step()

        running += float(loss_all.item())
        if i % print_every == 0:
            pbar.set_postfix(loss=f"{running/(i+1):.4f}")

    return running / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    tot = dict(psnr=0.0, ssim=0.0, miou=0.0, sat_ref=0.0, sat_pred=0.0, crerr=0.0)
    gain_mean = 0.0
    gain_min = 1e9
    gain_max = -1e9
    n = 0

    for hazy, clear, mask in tqdm(loader, desc="Val"):
        hazy = hazy.to(device); clear = clear.to(device); mask = mask.to(device)

        dehazed, seg_out, _ = model(hazy)

        tot["psnr"] += float(psnr(dehazed, clear).item())
        tot["ssim"] += float(ssim_metric(dehazed, clear))
        tot["miou"] += float(compute_mIoU(seg_out, mask, num_classes))
        tot["sat_ref"] += mean_saturation(clear)
        tot["sat_pred"] += mean_saturation(dehazed)
        tot["crerr"] += float(color_ratio_loss(dehazed, clear).item())

        dz_out = model.dehazer(hazy)
        if isinstance(dz_out, (list, tuple)) and len(dz_out) >= 3:
            cg = dz_out[2]
            gm = float(cg.mean().item())
            gmin = float(cg.min().item())
            gmax = float(cg.max().item())
            gain_mean += gm
            gain_min = min(gain_min, gmin)
            gain_max = max(gain_max, gmax)

        n += 1

    for k in tot:
        tot[k] /= max(1, n)
    dsat = abs(tot["sat_pred"] - tot["sat_ref"])
    gmean = gain_mean / max(1, n)

    return dict(
        ssim=tot["ssim"], psnr=tot["psnr"], miou=tot["miou"],
        sat_ref=tot["sat_ref"], sat_pred=tot["sat_pred"],
        dsat=dsat, crerr=tot["crerr"],
        gain_mean=gmean,
        gain_min=gain_min if gain_min < 1e8 else 0.0,
        gain_max=gain_max if gain_max > -1e8 else 0.0
    )


def append_csv(path, row, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def build_dehazer_safe(args):
    kwargs = dict(
        in_ch=3, base_ch=32,
        residual_scale=args.residual_scale,
        gain_scale=args.gain_scale,
        gain_form=args.gain_form,
        gain_min=args.gain_min,
        disable_gain=args.disable_gain,
    )
    try:
        return ColorAwareUNet(**kwargs)
    except TypeError:
        print("[Warn] ColorAwareUNet signature does not support gain_form/gain_min/disable_gain. Falling back.")
        fallback = dict(in_ch=3, base_ch=32, residual_scale=args.residual_scale, gain_scale=args.gain_scale)
        return ColorAwareUNet(**fallback)


def force_apply_gain_config(dehazer, args):
    """
    关键订正：无论 ColorAwareUNet 支不支持构造参数，都强制把内部属性设置到位
    """
    # NoGain：最稳的是强制 gain_scale=0
    if args.disable_gain:
        args.gain_scale = 0.0

    if hasattr(dehazer, "gain_scale"):
        try:
            dehazer.gain_scale = float(args.gain_scale)
        except Exception:
            pass

    if hasattr(dehazer, "disable_gain"):
        try:
            dehazer.disable_gain = bool(args.disable_gain)
        except Exception:
            pass

    if hasattr(dehazer, "gain_form"):
        try:
            dehazer.gain_form = str(args.gain_form)
        except Exception:
            pass

    if hasattr(dehazer, "gain_min"):
        try:
            dehazer.gain_min = args.gain_min
        except Exception:
            pass


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resize", type=int, nargs=2, default=None)  # H W
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="O-HAZE 很小，建议 0.2 或 0.3")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--strict_deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./runs_gain_ablation")

    # gain knobs
    parser.add_argument("--gain_scale", type=float, default=0.65)
    parser.add_argument("--gain_form", type=str, default="tanh", choices=["tanh", "amp"])
    parser.add_argument("--gain_min", type=float, default=None)
    parser.add_argument("--disable_gain", action="store_true")
    parser.add_argument("--residual_scale", type=float, default=0.5)

    # seg net
    parser.add_argument("--seg_width_mult", type=float, default=1.0)
    parser.add_argument("--seg_attention", action="store_true")
    parser.add_argument("--seg_use_se", action="store_true")

    # losses
    parser.add_argument("--lam_dehaze", type=float, default=1.0)
    parser.add_argument("--lam_seg", type=float, default=1.0)
    parser.add_argument("--lam_l1", type=float, default=1.0)
    parser.add_argument("--ssim_w", type=float, default=0.8)
    parser.add_argument("--lam_cratio", type=float, default=0.0)

    # stability report
    parser.add_argument("--lastk", type=int, default=10)
    parser.add_argument("--best_on", type=str, default="avg_lastk", choices=["val", "avg_lastk"])

    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic, strict_deterministic=args.strict_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- fixed split ids ----
    base_ds = HazySegDataset(
        args.data_root, num_classes=args.num_classes,
        resize=tuple(args.resize) if args.resize else None,
        augment=False
    )
    ids = base_ds.ids
    n = len(ids)
    if n == 0:
        raise RuntimeError("Dataset empty. Check hazy/clear/mask subfolders.")

    val_ratio = float(args.val_ratio)
    val_ratio = min(max(val_ratio, 0.05), 0.5)
    n_val = max(1, int(val_ratio * n))
    n_train = n - n_val

    g_split = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(n, generator=g_split).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_base = HazySegDataset(args.data_root, args.num_classes, ids=ids,
                                resize=tuple(args.resize) if args.resize else None,
                                augment=True)
    val_base = HazySegDataset(args.data_root, args.num_classes, ids=ids,
                              resize=tuple(args.resize) if args.resize else None,
                              augment=False)

    train_ds = torch.utils.data.Subset(train_base, train_idx)
    val_ds = torch.utils.data.Subset(val_base, val_idx)

    g_loader = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g_loader)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            worker_init_fn=seed_worker)

    print(f"Dataset sizes: total={n} train={n_train} val={n_val} (val_ratio={val_ratio})")
    print(f"[Split] split_seed={args.split_seed} fixed | train_seed={args.seed}")
    print(f"[Config] gain_form={args.gain_form} gain_scale={args.gain_scale} disable_gain={args.disable_gain} lastk={args.lastk}")

    # ---- build model ----
    dehazer = build_dehazer_safe(args)
    force_apply_gain_config(dehazer, args)  # ★关键订正：强制 gain 配置生效

    seg = LiteAttentionUNet(
        num_classes=args.num_classes, width_mult=args.seg_width_mult,
        base_ch=32, use_se=args.seg_use_se, attention=args.seg_attention, aux=False
    )
    model = DehazeLightSegWrapper(dehazer, seg, imagenet_norm=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # ---- logging ----
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_csv = os.path.join(args.save_dir, "train_log.csv")

    header = [
        "epoch","train_loss",
        "val_ssim","val_psnr","val_dsat","val_crerr","val_sat_ref","val_sat_pred","val_miou",
        "gain_mean","gain_min","gain_max",
        "avg_lastk_ssim","avg_lastk_dsat","avg_lastk_crerr",
        "lr",
        "seed","split_seed","n_train","n_val","val_ratio",
        "gain_form","gain_scale","gain_min_cfg","disable_gain","residual_scale",
        "ssim_w","lam_l1","lam_cratio",
        "lam_dehaze","lam_seg",
        "seg_width_mult","seg_attention","seg_use_se",
    ]

    best_key = -1e9
    best_dsat = 1e9

    k = max(1, int(args.lastk))
    buf_ssim = deque(maxlen=k)
    buf_dsat = deque(maxlen=k)
    buf_crerr = deque(maxlen=k)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args.num_classes,
            lam_dehaze=args.lam_dehaze, lam_seg=args.lam_seg,
            lam_l1=args.lam_l1, ssim_w=args.ssim_w,
            lam_cratio=args.lam_cratio
        )

        val = validate(model, val_loader, device, args.num_classes)

        buf_ssim.append(val["ssim"])
        buf_dsat.append(val["dsat"])
        buf_crerr.append(val["crerr"])

        avg_lastk_ssim = float(np.mean(buf_ssim))
        avg_lastk_dsat = float(np.mean(buf_dsat))
        avg_lastk_crerr = float(np.mean(buf_crerr))

        # ★关键订正：scheduler 用 avg_lastk
        scheduler.step(1.0 - avg_lastk_ssim)

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:03d}] {dt:.1f}s train:{train_loss:.4f} "
            f"valSSIM:{val['ssim']:.4f} valΔSat:{val['dsat']:.4f} valCRerr:{val['crerr']:.4f} "
            f"avgLast{k} SSIM:{avg_lastk_ssim:.4f} ΔSat:{avg_lastk_dsat:.4f} CRerr:{avg_lastk_crerr:.4f} "
            f"gain(mean/min/max):{val['gain_mean']:.3f}/{val['gain_min']:.3f}/{val['gain_max']:.3f} lr:{lr_now:.2e}"
        )

        row = dict(
            epoch=epoch, train_loss=train_loss,
            val_ssim=val["ssim"], val_psnr=val["psnr"], val_dsat=val["dsat"],
            val_crerr=val["crerr"], val_sat_ref=val["sat_ref"], val_sat_pred=val["sat_pred"],
            val_miou=val["miou"],
            gain_mean=val["gain_mean"], gain_min=val["gain_min"], gain_max=val["gain_max"],
            avg_lastk_ssim=avg_lastk_ssim, avg_lastk_dsat=avg_lastk_dsat, avg_lastk_crerr=avg_lastk_crerr,
            lr=lr_now,
            seed=args.seed, split_seed=args.split_seed, n_train=n_train, n_val=n_val, val_ratio=val_ratio,
            gain_form=args.gain_form, gain_scale=args.gain_scale, gain_min_cfg=args.gain_min,
            disable_gain=bool(args.disable_gain), residual_scale=args.residual_scale,
            ssim_w=args.ssim_w, lam_l1=args.lam_l1, lam_cratio=args.lam_cratio,
            lam_dehaze=args.lam_dehaze, lam_seg=args.lam_seg,
            seg_width_mult=args.seg_width_mult, seg_attention=bool(args.seg_attention), seg_use_se=bool(args.seg_use_se),
        )
        append_csv(log_csv, row, header)

        # ★关键订正：best 用 avg_lastk（或 val）
        if args.best_on == "avg_lastk":
            key = avg_lastk_ssim
            tie = avg_lastk_dsat
        else:
            key = val["ssim"]
            tie = val["dsat"]

        is_best = False
        if key > best_key + 1e-6:
            best_key = key
            best_dsat = tie
            is_best = True
        elif abs(key - best_key) <= 1e-6 and tie < best_dsat:
            best_dsat = tie
            is_best = True

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val": val,
            "args": vars(args),
            "avg_lastk": dict(ssim=avg_lastk_ssim, dsat=avg_lastk_dsat, crerr=avg_lastk_crerr),
        }
        torch.save(state, os.path.join(ckpt_dir, "last.pth"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "best.pth"))

    final_avg_lastk_ssim = float(np.mean(buf_ssim))
    final_avg_lastk_dsat = float(np.mean(buf_dsat))
    final_avg_lastk_crerr = float(np.mean(buf_crerr))

    print(f"Done.")
    print(f"[PaperMetric] avg_last{args.lastk}: SSIM={final_avg_lastk_ssim:.4f} ΔSat={final_avg_lastk_dsat:.4f} CRerr={final_avg_lastk_crerr:.4f}")
    print("Logs:", log_csv)
    print("Checkpoints:", ckpt_dir)


if __name__ == "__main__":
    main()
