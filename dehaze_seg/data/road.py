from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from .common import IMG_EXTS, find_by_stem, pil_to_rgb_tensor, mask_to_label_tensor

class DehazeSegFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        num_classes: int = 2,
        ids: Optional[List[str]] = None,
        resize: Optional[Tuple[int, int]] = (512, 512),
        augment: bool = False,
    ):
        self.root = Path(root)
        self.hazy_dir = self.root / "hazy"
        self.clear_dir = self.root / "clear"
        self.mask_dir = self.root / "masks"
        if not self.hazy_dir.exists():
            raise RuntimeError(f"hazy folder not found: {self.hazy_dir}")
        if not self.clear_dir.exists():
            raise RuntimeError(f"clear folder not found: {self.clear_dir}")
        if not self.mask_dir.exists():
            raise RuntimeError(f"masks folder not found: {self.mask_dir}")

        self.num_classes = int(num_classes)
        self.resize = resize
        self.augment = bool(augment)

        if ids is None:
            stems = []
            for p in self.hazy_dir.iterdir():
                if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                    continue
                stem = p.stem
                if find_by_stem(self.clear_dir, stem) is None or find_by_stem(self.mask_dir, stem) is None:
                    raise ValueError(f"Missing clear image or mask for sample {stem}")
                stems.append(stem)
            self.ids = sorted(stems)
        else:
            self.ids = list(ids)

        if not self.ids:
            raise RuntimeError(f"No matched hazy/clear/masks samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_triplet(self, stem: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
        hp = find_by_stem(self.hazy_dir, stem)
        cp = find_by_stem(self.clear_dir, stem)
        mp = find_by_stem(self.mask_dir, stem)
        if hp is None or cp is None or mp is None:
            raise RuntimeError(f"Missing triplet for sample: {stem}")
        hazy = Image.open(hp).convert("RGB")
        clear = Image.open(cp).convert("RGB")
        mask = Image.open(mp)
        return hazy, clear, mask

    def _sync_transform(self, hazy: Image.Image, clear: Image.Image, mask: Image.Image):
        if self.resize is not None:
            h, w = int(self.resize[0]), int(self.resize[1])
            hazy = hazy.resize((w, h), Image.BILINEAR)
            clear = clear.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
        else:
            ref_size = clear.size
            if hazy.size != ref_size:
                hazy = hazy.resize(ref_size, Image.BILINEAR)
            if mask.size != ref_size:
                mask = mask.resize(ref_size, Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                hazy = TF.hflip(hazy)
                clear = TF.hflip(clear)
                mask = TF.hflip(mask)
            if random.random() < 0.25:
                hazy = TF.adjust_brightness(hazy, random.uniform(0.92, 1.08))
                hazy = TF.adjust_contrast(hazy, random.uniform(0.92, 1.08))
        return hazy, clear, mask

    def __getitem__(self, index: int):
        stem = self.ids[index]
        hazy, clear, mask = self._load_triplet(stem)
        hazy, clear, mask = self._sync_transform(hazy, clear, mask)
        return pil_to_rgb_tensor(hazy), pil_to_rgb_tensor(clear), mask_to_label_tensor(mask, self.num_classes), stem
