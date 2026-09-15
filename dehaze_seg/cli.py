"""Public CLI: all workflows are available through python -m dehaze_seg."""
import argparse
import importlib
import sys

from .data.datasets import DATASETS
from .models.registry import MODELS
from .engine.checkpoint import PROFILES


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def runtime(parser):
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=positive, default=4, help="PyTorch CPU threads")


def training(parser):
    runtime(parser)
    parser.add_argument("--dataset", choices=DATASETS, default="paired-road")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--val-root", help="Independent validation root with the same layout; otherwise split training data")
    parser.add_argument("--model", choices=MODELS, default="coloraware")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--resize", type=positive, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=positive, default=2)
    parser.add_argument("--workers", type=nonnegative, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--split-unit", choices=("sample", "scene"),
                        help="Default: scene for roads/SOTS, sample for HSTS; sample reproduces historical ID splitting")
    parser.add_argument("--pretrain-dehaze-epochs", type=nonnegative, default=60,
                        help="Restoration pretraining epochs; unused for classical DCP")
    parser.add_argument("--pretrain-seg-epochs", type=nonnegative, default=20,
                        help="Segmentation epochs; DCP trains segmentation for this + finetune-epochs")
    parser.add_argument("--finetune-epochs", type=nonnegative, default=20,
                        help="Joint fine-tuning epochs; DCP adds these to its fixed-dehazer segmentation stage")
    parser.add_argument("--direct-joint-epochs", type=nonnegative, default=100)
    parser.add_argument("--num-classes", type=positive, default=2)
    parser.add_argument("--color-base-ch", type=positive, default=32)
    parser.add_argument("--seg-base-ch", type=positive, default=32)
    parser.add_argument("--seg-width-mult", type=float, default=1.)
    parser.add_argument("--use-se", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-imagenet-norm", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--train-repeats", type=positive, default=1, help="HSTS augmentation repeats per epoch")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.)
    for name, value in (("dehaze", 1.), ("seg", 1.), ("ssim", .40), ("perc", .05), ("ce", 1.), ("dice", 1.)):
        parser.add_argument("--lam-" + name, type=float, default=value)
    parser.add_argument("--score-metric", choices=("balanced", "psnr", "ssim", "miou", "f1"), default="balanced")


def inference(parser, evaluate=False):
    runtime(parser)
    parser.add_argument("--checkpoint", nargs="+", help="One or more checkpoints")
    parser.add_argument("--model", choices=MODELS, help="Identifier for legacy weights; --model dcp also runs without weights")
    parser.add_argument("--include-dcp", action="store_true", help="Include parameter-free DCP in a checkpoint comparison")
    parser.add_argument("--segmenter-checkpoint", help="Reuse this joint checkpoint's frozen segmenter for EVERY dehazer")
    parser.add_argument("--segmenter-legacy-profile", choices=PROFILES)
    parser.add_argument("--segmenter-model", choices=MODELS, help="Model identifier for a metadata-free segmenter source")
    parser.add_argument("--segmenter-model-config", help="JSON model config for a metadata-free segmenter source")
    parser.add_argument("--legacy-profile", choices=PROFILES)
    parser.add_argument("--model-config", help="Complete model configuration JSON for metadata-free weights")
    parser.add_argument("--output", default="results/evaluate" if evaluate else "results/predict")
    parser.add_argument("--resize", type=positive, nargs=2, metavar=("H", "W"), help="Optional inference size; outputs return to original resolution")
    parser.add_argument("--limit", type=nonnegative, default=0, help="0 processes all images")
    if evaluate:
        parser.add_argument("--dataset", choices=DATASETS, default="paired-road")
        parser.add_argument("--data-root", default="datasets")
        parser.add_argument("--split", choices=("all", "train", "val"), help="Default: saved validation on a known training root; all on an independent benchmark root")
        parser.add_argument("--split-file", help="Saved split.json")
        parser.add_argument("--allow-training-overlap", action="store_true", help="Explicit diagnostic only: allow known training samples in evaluation")
        parser.add_argument("--metric-align", choices=("crop", "resize", "none"), default="crop")
        parser.add_argument("--save-images", action="store_true")
    else:
        parser.add_argument("--input", required=True, help="Hazy image or directory")


def build_parser(training_prog=None):
    parser = argparse.ArgumentParser(description="Color-gain-guided image dehazing and semantic segmentation")
    commands = parser.add_subparsers(dest="command", required=True)
    training(commands.add_parser("train", help="Train restoration or the staged joint framework",
                                 **({"prog": training_prog} if training_prog else {})))
    inference(commands.add_parser("predict", help="Predict one image or a folder"))
    inference(commands.add_parser("evaluate", help="Evaluate paired references"), evaluate=True)
    ablate = commands.add_parser("ablate", help="Run controlled paper ablations")
    training(ablate)
    ablate.add_argument("--list", action="store_true", help="List experiments without training")
    ablate.add_argument("--experiments", default="arch_baseline_unet,arch_no_color_gain,arch_no_refine,arch_full",
                        help="Comma-separated names or all")
    visualize = commands.add_parser("visualize", help="Visualize gains, components, or training curves")
    visualize.add_argument("kind", choices=("gain", "components", "curves", "introduction"))
    visualize.add_argument("options", nargs=argparse.REMAINDER, help="Use visualize KIND --help")
    return parser


def main(argv=None, training_prog=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(argv) >= 2 and argv[0] == "visualize" and argv[1] in ("gain", "components", "introduction"):
            import torch
            torch.set_num_threads(4)
            module = importlib.import_module(".visualization." + argv[1], package="dehaze_seg")
            return module.main(argv[2:])
        if argv[:2] == ["visualize", "curves"]:
            parser = argparse.ArgumentParser(prog="dehaze_seg visualize curves")
            parser.add_argument("--metrics", required=True)
            parser.add_argument("--output", default="results/curves")
            args = parser.parse_args(argv[2:])
            from pathlib import Path
            from .visualization.curves import save_metric_plots_from_csv
            return save_metric_plots_from_csv(Path(args.metrics), Path(args.output))
        args = build_parser(training_prog=training_prog).parse_args(argv)
        if args.command == "visualize":
            raise ValueError("Use visualize KIND --help to select visualization options")
        import torch
        torch.set_num_threads(args.threads)
        if args.command in ("train", "ablate"):
            if min(args.resize) < 32 or args.num_classes < 2 or args.seg_width_mult <= 0:
                raise ValueError("Training requires resize >= 32, num-classes >= 2 and seg-width-mult > 0")
            if args.lr <= 0 or args.weight_decay < 0 or any(getattr(args, "lam_" + k) < 0 for k in ("dehaze", "seg", "ssim", "perc", "ce", "dice")):
                raise ValueError("Learning rate must be positive; weight decay and loss weights must be nonnegative")
            if args.command == "train":
                from .engine.train import run_training
                return run_training(args)
            from .experiments.runner import run_ablations
            return run_ablations(args)
        from .engine.inference import run_inference
        return run_inference(args, evaluate=args.command == "evaluate")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def entrypoint():
    """Console scripts must return None rather than a result dictionary."""
    main()
