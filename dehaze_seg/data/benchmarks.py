from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from .common import IMG_EXTS, find_by_stem, pil_to_rgb_tensor

class HSTSDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        ids: Optional[List[str]] = None,
        resize: Optional[Tuple[int, int]] = (512, 512),
        augment: bool = False,
        train_repeats: int = 1,
    ):
        self.root = Path(root)
        self.hazy_dir = self.root / "synthetic" / "synthetic"
        self.clear_dir = self.root / "synthetic" / "original"
        if not self.hazy_dir.exists():
            raise RuntimeError(f"hazy folder not found: {self.hazy_dir}")
        if not self.clear_dir.exists():
            raise RuntimeError(f"clear folder not found: {self.clear_dir}")

        self.resize = resize
        self.augment = bool(augment)
        self.train_repeats = max(1, int(train_repeats))
        if ids is None:
            ids = []
            for p in self.hazy_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS and find_by_stem(self.clear_dir, p.stem) is not None:
                    ids.append(p.stem)
            ids.sort()
        self.ids = list(ids)
        if not self.ids:
            raise RuntimeError(f"No matched HSTS synthetic/original samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.ids) * (self.train_repeats if self.augment else 1)

    def _sync_transform(self, hazy: Image.Image, clear: Image.Image):
        if hazy.size != clear.size:
            hazy = hazy.resize(clear.size, Image.BICUBIC)

        if self.augment:
            w0, h0 = clear.size
            min_side = min(w0, h0)
            if min_side >= 128:
                area_scale = random.uniform(0.45, 1.0)
                aspect = random.uniform(0.85, 1.15)
                crop_h = int(min_side * (area_scale ** 0.5))
                crop_w = int(crop_h * aspect)
                crop_w = max(96, min(crop_w, w0))
                crop_h = max(96, min(crop_h, h0))
                left = random.randint(0, max(0, w0 - crop_w))
                top = random.randint(0, max(0, h0 - crop_h))
                box = (left, top, left + crop_w, top + crop_h)
                hazy = hazy.crop(box)
                clear = clear.crop(box)

            if random.random() < 0.5:
                hazy = TF.hflip(hazy)
                clear = TF.hflip(clear)
            if random.random() < 0.5:
                hazy = TF.vflip(hazy)
                clear = TF.vflip(clear)
            if random.random() < 0.35:
                angle = random.choice([90, 180, 270])
                hazy = hazy.rotate(angle, expand=True)
                clear = clear.rotate(angle, expand=True)

        if self.resize is not None:
            h, w = int(self.resize[0]), int(self.resize[1])
            hazy = hazy.resize((w, h), Image.BICUBIC)
            clear = clear.resize((w, h), Image.BICUBIC)

        return hazy, clear

    def __getitem__(self, index: int):
        stem = self.ids[index % len(self.ids)]
        hp = find_by_stem(self.hazy_dir, stem)
        cp = find_by_stem(self.clear_dir, stem)
        if hp is None or cp is None:
            raise RuntimeError(f"Missing pair for {stem}: hazy={hp}, clear={cp}")
        hazy = Image.open(hp).convert("RGB")
        clear = Image.open(cp).convert("RGB")
        hazy, clear = self._sync_transform(hazy, clear)
        hazy_t = pil_to_rgb_tensor(hazy)
        clear_t = pil_to_rgb_tensor(clear)
        return hazy_t, clear_t, stem
