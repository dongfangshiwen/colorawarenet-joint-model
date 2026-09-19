"""Figure annotations, CSV and saved masks must describe the same predictions."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import font_manager
from PIL import Image
import torch

from dehaze_seg.data.splits import training_split
from dehaze_seg.engine.checkpoint import save_checkpoint, load_checkpoint
from dehaze_seg.data.common import pil_to_rgb_tensor
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
                "--segmentation-protocol", "shared-frozen",
                "--data-root", str(root/"data"), "--output", str(root/"figure"), "--include-dcp",
                "--resize", "32", "48", "--device", "cpu", "--threads", "2",
                "--metric-font", font_manager.findfont("DejaVu Serif")])
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
            # Joint protocol must preserve each checkpoint's actual segmenter.
            cfg = model_config("dcp", dcp_mode="learned")
            cfg["segmenter"]["base_ch"] = 8
            save_checkpoint(root/"dcp.pth", build_model(cfg).eval(), cfg, {}, "joint", 1, {}, data_split=split)
            joint_args = parse_args(["--checkpoint", *checkpoints, str(root/"dcp.pth"),
                "--data-root", str(root/"data"), "--output", str(root/"joint"),
                "--resize", "32", "48", "--device", "cpu", "--threads", "2",
                "--metric-font", font_manager.findfont("DejaVu Serif")])
            with patch("dehaze_seg.visualization.segmentation.save_figure"):
                joint = generate(joint_args)
            self.assertEqual(joint["segmentation_protocol"], "joint")
            self.assertIsNone(joint["shared_segmenter"])
            with Image.open(root/"data/hazy"/f"{joint['sample']}.png") as source:
                x = pil_to_rgb_tensor(source)[None]
            for method in joint["methods"]:
                model, _, _ = load_checkpoint(method["checkpoint"])
                with torch.inference_mode():
                    expected = model(x)[1].argmax(1)[0].numpy()
                with Image.open(root/"joint"/method["prediction_mask"]) as source:
                    actual = np.asarray(source)
                np.testing.assert_array_equal(actual, expected)
                cm = np.array([[((target == a) & (actual == b)).sum() for b in (0, 1)] for a in (0, 1)])
                iou = np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm)+1e-6)
                dice = 2*np.diag(cm)/(cm.sum(0)+cm.sum(1)+1e-6)
                self.assertAlmostEqual(iou.mean(), method["metrics"]["miou"], places=6)
                self.assertAlmostEqual(dice.mean(), method["metrics"]["mdice"], places=6)
                self.assertGreaterEqual(method["metrics"]["mdice"], method["metrics"]["miou"])
            args.sample = split["train"][0]
            with self.assertRaisesRegex(ValueError, "outside the selected validation"):
                generate(args)
            args.sample = None
            args.checkpoint = [str(root/"coloraware.pth")]
            args.output = str(root/"incomplete")
            with self.assertRaisesRegex(ValueError, "requires all six methods"):
                generate(args)
            self.assertFalse(Path(args.output).exists())

    def test_protocol_options_cannot_silently_replace_joint_segmenters(self):
        with self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "a.pth", "--segmenter-checkpoint", "b.pth"])
        with self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "a.pth", "--segmentation-protocol", "shared-frozen"])


if __name__ == "__main__":
    unittest.main()
