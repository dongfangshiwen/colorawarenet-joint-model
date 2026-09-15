"""Dataset adapters expose real supervision using dictionary batches."""
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .benchmarks import HSTSDataset
from .common import IMG_EXTS, find_by_stem, pil_to_rgb_tensor
from .road import DehazeSegFolderDataset

DATASETS = ("paired-road", "sots-indoor", "sots-outdoor", "hsts")


class SOTSDataset(Dataset):
    """Preserve exact-stem then scene-prefix pairing, without synthetic masks."""
    def __init__(self, root, ids=None, resize=(512, 512), augment=False):
        self.root = Path(root)
        self.hazy_dir = self.root / "hazy"
        self.clear_dir = self.root / "clear"
        if not self.clear_dir.is_dir():
            self.clear_dir = self.root / "gt"
        if not self.hazy_dir.is_dir() or not self.clear_dir.is_dir():
            raise ValueError("SOTS requires hazy/ and clear/ (or gt/)")
        self.ids = sorted(p.stem for p in self.hazy_dir.iterdir()
                          if p.suffix.lower() in IMG_EXTS) if ids is None else list(ids)
        self.resize = resize

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stem = self.ids[index]
        hp = find_by_stem(self.hazy_dir, stem)
        cp = find_by_stem(self.clear_dir, stem) or find_by_stem(self.clear_dir, stem.split("_")[0])
        if cp is None:
            raise ValueError(f"Missing SOTS clear image for {stem}")
        with Image.open(hp) as image:
            hazy = image.convert("RGB")
        with Image.open(cp) as image:
            clear = image.convert("RGB")
        # The historical SOTS training loader resizes, with augmentation disabled.
        size = (self.resize[1], self.resize[0]) if self.resize else clear.size
        hazy = hazy.resize(size, Image.Resampling.BILINEAR)
        clear = clear.resize(size, Image.Resampling.BILINEAR)
        return pil_to_rgb_tensor(hazy), pil_to_rgb_tensor(clear), stem


class SupervisedDataset(Dataset):
    def __init__(self, inner, joint):
        self.inner, self.joint = inner, joint
        self.ids = inner.ids

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        item = self.inner[index]
        result = dict(hazy=item[0], clear=item[1], id=item[-1])
        if self.joint:
            result["mask"] = item[2]
        return result


def make_dataset(name, root, ids=None, resize=(512, 512), augment=False, num_classes=2,
                 train_repeats=1):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    kwargs = dict(root=root, ids=ids, resize=resize, augment=augment)
    if name == "paired-road":
        inner = DehazeSegFolderDataset(num_classes=num_classes, **kwargs)
    elif name.startswith("sots-"):
        inner = SOTSDataset(**kwargs)
    elif name == "hsts":
        inner = HSTSDataset(train_repeats=train_repeats, **kwargs)
    if not inner.ids:
        raise ValueError(f"No samples found under {root}")
    if len(inner.ids) != len(set(inner.ids)):
        raise ValueError("Duplicate image stems; use a unique stem for each sample")
    return SupervisedDataset(inner, name == "paired-road")


def split_ids(ids, val_ratio=.15, seed=42):
    if not 0 < val_ratio < 1 or len(ids) < 2:
        raise ValueError("Splitting requires at least two samples and 0 < val_ratio < 1")
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * val_ratio)))
    return ids[n_val:], ids[:n_val]
