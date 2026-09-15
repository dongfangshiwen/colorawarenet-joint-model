"""Small runtime helpers shared by command-line workflows."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name="auto"):
    name = ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(name)


def write_json(path, value):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
