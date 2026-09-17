"""Figure annotations, CSV and saved masks must describe the same predictions."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from dehaze_seg.data.splits import training_split
from dehaze_seg.engine.checkpoint import save_checkpoint
from dehaze_seg.models.registry import build_model, model_config
from dehaze_seg.visualization.segmentation import generate, parse_args, PAPER_MODELS


class SegmentationFigureTests(unittest.TestCase):
    def test_shared_segmenter_figure_and_training_sample_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(12)
            for folder in ("hazy", "clear", "masks"):
                (root / "data" / folder).mkdir(parents=True)
            ids = ["000", "001", "002", "003"]
            for sample in ids:
                rgb = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(root / "data/hazy" / f"{sample}.png")
                Image.fromarray(255-rgb).save(root / "data/clear" / f"{sample}.png")
                Image.fromarray((rgb[..., 0] > 128).astype(np.uint8)).save(root / "data/masks" / f"{sample}.png")
            split = training_split("paired-road", root / "data", ids, val_ratio=.25)
            checkpoints = []
            for name in PAPER_MODELS:
                if name == "dcp":
                    continue
                cfg = model_config(name)
                cfg["dehazer"]["base_ch"] = cfg["segmenter"]["base_ch"] = 8
                cfg = json.loads(json.dumps(cfg))
                checkpoint = root / f"{name}.pth"
                torch.manual_seed(3)
                model = build_model(cfg).eval()
                save_checkpoint(checkpoint, model, cfg, {}, "joint", 1, {}, data_split=split)
                checkpoints.append(str(checkpoint))
            args = parse_args(["--checkpoint", *reversed(checkpoints), "--segmenter-checkpoint", str(root/"coloraware.pth"),
                "--data-root", str(root/"data"), "--output", str(root/"figure"), "--include-dcp",
                "--resize", "32", "48", "--device", "cpu", "--threads", "2"])
            with patch("dehaze_seg.visualization.segmentation.save_figure") as render:
                record = generate(args)
            self.assertEqual(record["sample"], split["val"][0])
            self.assertEqual([m["model_config"]["model"] for m in record["methods"]], list(PAPER_MODELS))
            self.assertEqual(len(render.call_args.args[0]), 8)  # Hazy + six methods + GT.
            self.assertTrue(record["shared_segmenter"]["frozen"])
            with Image.open(root/"figure/target_mask.png") as image:
                target = np.asarray(image)
            plotted_records = render.call_args.args[1]
            for i, method in enumerate(record["methods"]):
                with Image.open(root/"figure"/method["prediction_mask"]) as image:
                    pred = np.asarray(image)
                cm = np.array([[( (target == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])
                self.assertEqual(cm.tolist(), method["confusion_matrix"])
                dice = 2*np.diag(cm)/(cm.sum(0)+cm.sum(1)+1e-6)
                self.assertAlmostEqual(dice.mean(), method["metrics"]["mdice"], places=6)
                self.assertEqual(plotted_records[i+1], method["metrics"])
            args.sample = split["train"][0]
            with self.assertRaisesRegex(ValueError, "outside the selected validation"):
                generate(args)
            args.sample = None
            args.checkpoint = [str(root/"coloraware.pth")]
            args.output = str(root/"incomplete")
            with self.assertRaisesRegex(ValueError, "requires all six methods"):
                generate(args)
            self.assertFalse(Path(args.output).exists())


if __name__ == "__main__":
    unittest.main()
