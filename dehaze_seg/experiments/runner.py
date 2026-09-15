"""Run registered ablations through the same trainer and fixed data split."""
from pathlib import Path

from .ablations import EXPERIMENTS
from ..engine.train import run_training
from ..utils import write_csv


def run_ablations(args):
    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(f"{name:30} {cfg['description']}")
        return
    if args.dataset != "paired-road" or args.model != "coloraware":
        raise ValueError("Paper ablations require --dataset paired-road --model coloraware")
    names = list(EXPERIMENTS) if args.experiments == "all" else args.experiments.split(",")
    unknown = set(names) - EXPERIMENTS.keys()
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}; use ablate --list")
    if len(names) != len(set(names)):
        raise ValueError("Each experiment may be selected only once")
    rows = []
    for name in names:
        stats = run_training(args, EXPERIMENTS[name], name)
        rows.append(dict(experiment=name, **stats))
        write_csv(Path(args.output) / args.dataset / "ablation_summary.csv", rows)
