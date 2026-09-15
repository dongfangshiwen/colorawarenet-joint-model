"""Persist sample provenance and keep restoration scenes out of both splits."""
import hashlib
import json
import random
from pathlib import Path

from .common import find_by_stem
from .datasets import split_ids


def canonical_root(root):
    return str(Path(root).resolve())


def paired_directories(dataset, root):
    root = Path(root)
    if dataset == "hsts":
        return root / "synthetic/synthetic", root / "synthetic/original", None
    clear = root / "clear"
    return root / "hazy", clear if clear.is_dir() else root / "gt", root / "masks" if dataset == "paired-road" else None


def clear_reference(directory, stem, dataset):
    path = find_by_stem(directory, stem)
    if path is None and dataset.startswith("sots-"):
        path = find_by_stem(directory, stem.split("_")[0])
    return path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_records(dataset, root, ids):
    """Hash each unique clear reference once, including many-hazy-to-one pairing."""
    _, clear, _ = paired_directories(dataset, root)
    cache, records = {}, {}
    for stem in ids:
        reference = clear_reference(clear, stem, dataset)
        if reference is None:
            raise ValueError(f"Missing clear reference for {stem} under {root}")
        if reference not in cache:
            cache[reference] = file_digest(reference)
        records[stem] = dict(reference=reference.name, clear_sha256=cache[reference])
    return records


def grouped_split(ids, records, val_ratio=.15, seed=42, sample_budget=False):
    if not 0 < val_ratio < 1:
        raise ValueError("Splitting requires 0 < val_ratio < 1")
    groups = {}
    for stem in sorted(ids):
        groups.setdefault(records[stem]["clear_sha256"], []).append(stem)
    if len(groups) < 2:
        raise ValueError("Scene split requires at least two distinct clear references")
    if sample_budget:
        if len(groups) == len(ids):
            return split_ids(ids, val_ratio, seed)
        target = min(len(ids) - 1, max(1, round(len(ids) * val_ratio)))
        order = sorted(groups)
        random.Random(seed).shuffle(order)
        val_groups, count = [], 0
        for group in order:
            if count + len(groups[group]) <= target:
                val_groups.append(group)
                count += len(groups[group])
        if not val_groups:
            val_groups = [min(order, key=lambda group: len(groups[group]))]
        train_groups = [group for group in order if group not in val_groups]
    else:
        train_groups, val_groups = split_ids(list(groups), val_ratio, seed)
    return ([stem for group in train_groups for stem in groups[group]],
            [stem for group in val_groups for stem in groups[group]])


def training_split(dataset, root, ids, val_ratio=.15, seed=42, val_root=None, val_ids=None, split_unit=None):
    records = sample_records(dataset, root, ids)
    if val_root is not None:
        if canonical_root(root) == canonical_root(val_root):
            raise ValueError("--val-root must differ from --data-root; omit it for an internal split")
        train, val = list(ids), list(val_ids or [])
        if not val:
            raise ValueError("Validation root has no samples")
        val_records = sample_records(dataset, val_root, val)
        if {r["clear_sha256"] for r in records.values()} & {r["clear_sha256"] for r in val_records.values()}:
            raise ValueError("Training and validation roots share clear references; use independent scenes")
        grouping = "separate-roots"
    else:
        unit = split_unit or ("scene" if dataset == "paired-road" or dataset.startswith("sots-") else "sample")
        if unit == "scene":
            train, val = grouped_split(ids, records, val_ratio, seed, sample_budget=dataset == "paired-road")
            grouping = "clear-reference-sha256"
        else:
            train, val = split_ids(ids, val_ratio, seed)
            grouping = "sample-id"
        val_root = root
        val_records = {stem: records[stem] for stem in val}
    train_hashes = {records[stem]["clear_sha256"] for stem in train}
    shared = [stem for stem in val if val_records[stem]["clear_sha256"] in train_hashes]
    return dict(version=2, dataset=dataset, seed=seed, val_ratio=val_ratio, grouping=grouping,
                shared_clear_references=shared,
                train_root=canonical_root(root), val_root=canonical_root(val_root), train=train, val=val,
                records={"train": {stem: records[stem] for stem in train}, "val": val_records})


def checkpoint_split(checkpoint_path, checkpoint):
    if checkpoint.get("data_split"):
        return checkpoint["data_split"]
    if checkpoint_path:
        for folder in (Path(checkpoint_path).parent, Path(checkpoint_path).parent.parent):
            manifest = folder / "split.json"
            if manifest.is_file():
                return json.loads(manifest.read_text(encoding="utf-8"))
    return None


def training_overlap(dataset, root, ids, checkpoint_path, checkpoint, records=None):
    """Detect known training samples, including copied roots with identical clear files.

    Unknown historical training data cannot be certified as independent.
    """
    record = checkpoint_split(checkpoint_path, checkpoint)
    saved = checkpoint.get("train_config", checkpoint.get("args", {}))
    if not record or record.get("dataset", saved.get("dataset")) != dataset:
        return []
    train_root = record.get("train_root", saved.get("data_root"))
    overlap = set()
    if train_root and canonical_root(train_root) == canonical_root(root):
        overlap.update(set(ids) & set(record["train"]))
    hashes = {r["clear_sha256"] for r in record.get("records", {}).get("train", {}).values()}
    if hashes:
        records = records if records is not None else sample_records(dataset, root, ids)
        overlap.update(stem for stem in ids if records[stem]["clear_sha256"] in hashes)
    return sorted(overlap)
