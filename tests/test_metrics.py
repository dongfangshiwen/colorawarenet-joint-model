"""Checks for hard-mask metrics and reproducible evaluation preprocessing."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dehaze_seg.cli import build_parser
from dehaze_seg.data.datasets import make_dataset
from dehaze_seg.data.splits import training_overlap
from dehaze_seg.engine.checkpoint import save_checkpoint
from dehaze_seg.engine.inference import align_pair, run_inference
from dehaze_seg.engine.train import validate
from dehaze_seg.losses import dice_loss_from_logits
from dehaze_seg.metrics import confusion_matrix, segmentation_metrics
from dehaze_seg.models.registry import build_model, model_config

torch.set_num_threads(2)


class MetricTests(unittest.TestCase):
    def test_cropped_masks_match_independent_pixel_counts(self):
        pred = torch.tensor([[[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]])
        target = torch.tensor([[[[0, 0], [1, 1]]]])
        pred, target = align_pair(pred, target, "crop", label=True)
        self.assertFalse(pred.is_contiguous())
        cm = confusion_matrix(pred, target, 2)
        self.assertEqual(cm.tolist(), [[1, 1], [0, 2]])
        metrics = segmentation_metrics(cm)
        self.assertAlmostEqual(metrics["miou"], (1/2 + 2/3)/2, places=6)
        self.assertAlmostEqual(metrics["mdice"], (2/3 + 4/5)/2, places=6)

    def test_macro_dice_bounds_and_class_identity(self):
        rng = np.random.default_rng(7)
        for counts in rng.integers(0, 100000, size=(100, 2, 2)):
            metrics = segmentation_metrics(torch.from_numpy(counts))
            iou = np.diag(counts) / (counts.sum(0) + counts.sum(1) - np.diag(counts))
            dice = 2*np.diag(counts) / (counts.sum(0) + counts.sum(1))
            np.testing.assert_allclose(metrics["class_iou"], iou, atol=1e-7)
            np.testing.assert_allclose(metrics["class_dice"], dice, atol=1e-7)
            np.testing.assert_allclose(dice, 2*iou/(1+iou))
            self.assertGreaterEqual(metrics["mdice"], metrics["miou"])
        # A foreground-only Dice can be lower than macro IoU; label it explicitly.
        metrics = segmentation_metrics(torch.tensor([[950, 20], [20, 10]]))
        self.assertLess(metrics["class_dice"][1], metrics["miou"])
        self.assertGreater(metrics["mdice"], metrics["miou"])

    def test_empty_class_convention_and_soft_dice_are_explicit(self):
        metrics = segmentation_metrics(torch.tensor([[10, 0], [0, 0]]))
        self.assertEqual(metrics["class_iou"][1], 0.)
        self.assertEqual(metrics["class_dice"][1], 0.)
        target = torch.tensor([[[0, 1]]])
        logits = torch.tensor([[[[.01, 0.]], [[0., .01]]]])
        hard = segmentation_metrics(confusion_matrix(logits.argmax(1), target, 2))
        soft = float(1 - dice_loss_from_logits(logits, target, 2))
        self.assertGreater(hard["mdice"], .99)
        self.assertLess(soft, .51)

    def test_historical_road_checkpoint_detects_shared_training_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("hazy", "clear", "masks"):
                (root / folder).mkdir()
            # Every clear image is identical, so even held-out IDs overlap.
            for i in range(4):
                for folder in ("hazy", "clear", "masks"):
                    Image.new("L", (4, 4), 1).save(root / folder / f"{i}.png")
            checkpoint = {"args": dict(data_root=str(root), seed=42, val_ratio=.25,
                                       pretrain_seg_epochs=20, finetune_epochs=20)}
            self.assertEqual(training_overlap("paired-road", root, ["0", "1", "2", "3"], None, checkpoint),
                             ["0", "1", "2", "3"])
            self.assertEqual(training_overlap("paired-road", root, ["0"], None, {}), [])

    def test_evaluate_inference_grid_matches_training_validation(self):
        torch.manual_seed(42)
        parser = build_parser()
        self.assertEqual(parser.parse_args(["evaluate"]).metric_resolution, "original")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(3)
            for folder in ("hazy", "clear", "masks"):
                (root / "data" / folder).mkdir(parents=True)
            for i in range(2):
                rgb = rng.integers(0, 255, (37, 53, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(root / "data/hazy" / f"{i}.png")
                Image.fromarray(255-rgb).save(root / "data/clear" / f"{i}.png")
                Image.fromarray((rgb[..., 0] > 128).astype(np.uint8)*255).save(root / "data/masks" / f"{i}.png")
            config = model_config()
            config["dehazer"]["base_ch"] = config["segmenter"]["base_ch"] = 8
            model = build_model(config).eval()
            checkpoint = root / "model.pth"
            save_checkpoint(checkpoint, model, config, {}, "joint", 1, {})
            loader = DataLoader(make_dataset("paired-road", root / "data", resize=(32, 48)), batch_size=1)
            training_metrics = validate(model, loader, torch.device("cpu"))
            args = parser.parse_args(["evaluate", "--checkpoint", str(checkpoint), "--data-root", str(root/"data"),
                "--split", "all", "--resize", "32", "48", "--metric-resolution", "inference",
                "--output", str(root/"evaluation"), "--save-images", "--device", "cpu"])
            evaluation = run_inference(args, evaluate=True)[0]
            for key in ("miou", "mdice", "pixel_acc", "psnr", "ssim"):
                self.assertAlmostEqual(evaluation[key], training_metrics[key], places=6)
            records = json.loads((root/"evaluation/segmentation_metrics.json").read_text())
            cms = np.array([row["confusion_matrix"] for row in records["samples"]])
            self.assertEqual(cms.sum((1, 2)).tolist(), [32*48, 32*48])
            np.testing.assert_equal(cms.sum(0), evaluation["confusion_matrix"])
            self.assertEqual(records["checkpoint_sha256"], json.loads((root/"evaluation/protocol.json").read_text())["checkpoint_sha256"])
            with (root/"evaluation/metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(rows[0]["dice_class_1"]), records["samples"][0]["class_dice"][1])
            self.assertAlmostEqual(float(rows[0]["mdice"]), records["samples"][0]["mdice"])
            with Image.open(root/"evaluation/masks/0.png") as image:
                self.assertEqual(image.size, (53, 37))


if __name__ == "__main__":
    unittest.main()
