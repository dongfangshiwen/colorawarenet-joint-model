"""Train the paper models from the repository root: python train.py --help."""
import sys

from dehaze_seg.cli import main as run_cli


def main(argv=None):
    """Reuse the shared training command and its argument validation."""
    return run_cli(["train", *(sys.argv[1:] if argv is None else argv)], training_prog="train.py")


if __name__ == "__main__":
    main()
