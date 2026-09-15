from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def find_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    matches = [p for p in folder.glob(f"{stem}.*") if p.is_file()]
    return matches[0] if matches else None


def pil_to_rgb_tensor(img: Image.Image) -> torch.Tensor:
    return TF.to_tensor(img.convert("RGB"))


def mask_to_label_tensor(mask: Image.Image, num_classes: int) -> torch.Tensor:
    arr = np.array(mask)
    if arr.ndim == 3:
        if arr.shape[2] == num_classes:
            arr = np.argmax(arr, axis=2)
        else:
            arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > max(1, num_classes - 1):
        arr = np.round(arr * ((num_classes - 1) / 255.0))
    arr = np.clip(np.round(arr), 0, num_classes - 1).astype(np.int64)
    return torch.from_numpy(arr)
