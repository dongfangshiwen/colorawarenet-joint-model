"""Direct entry point for the complete six-model segmentation comparison."""
import sys

from dehaze_seg.cli import main


if __name__ == "__main__":
    main(["visualize", "segmentation", *sys.argv[1:]])
