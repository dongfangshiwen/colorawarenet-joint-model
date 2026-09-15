"""CPU regression checks; no external datasets or pretrained downloads required."""
from argparse import Namespace
import importlib
from pathlib import Path
import pkgutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import dehaze_seg
from dehaze_seg.cli import build_parser
from dehaze_seg.data.common import mask_to_label_tensor
from dehaze_seg.data.datasets import make_dataset, split_ids
from dehaze_seg.engine.checkpoint import load_checkpoint, save_checkpoint
from dehaze_seg.engine.inference import align_pair, infer_image
from dehaze_seg.engine.train import configure_stage, train_epoch, validate
from dehaze_seg.losses import VGGPerceptualLoss, dehaze_losses
from dehaze_seg.models.joint import unpack_dehaze_output
from dehaze_seg.models.registry import MODELS, build_model, model_config

torch.set_num_threads(2)


def small_config():
    cfg = model_config()
    cfg["dehazer"]["base_ch"] = 8
    cfg["segmenter"]["base_ch"] = 8
    return cfg


class CoreTests(unittest.TestCase):
    def test_import_all_modules(self):
        for module in pkgutil.walk_packages(dehaze_seg.__path__, dehaze_seg.__name__ + "."):
            importlib.import_module(module.name)

    def test_model_registry_forward(self):
        x = torch.rand(1, 3, 32, 48)
        for name in MODELS:
            with self.subTest(name=name), torch.no_grad():
                model = build_model(model_config(name, joint=False)).eval()
                out, _ = unpack_dehaze_output(model(x))
                self.assertEqual(out.shape, x.shape)
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_variants_and_gain(self):
        x = torch.rand(2, 3, 35, 49)
        for mode, attention, se in (("global", True, False), ("local", False, True), ("global", True, True)):
            cfg = small_config()
            cfg["dehazer"]["gain_mode"] = mode
            cfg["segmenter"].update(attention=attention, use_se=se)
            model = build_model(cfg).eval()
            with torch.no_grad():
                out, logits, aux = model(x)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(logits.shape, (2, 2, 35, 49))
            expected = (2, 3, 1, 1) if mode == "global" else x.shape
            self.assertEqual(aux["color_gain"].shape, expected)
            self.assertTrue(torch.equal(out, x))  # zero initialization starts at identity
            self.assertEqual(len(aux["sides"]), 3)
            gain = model.dehazer._make_gain(torch.tensor([-100., 100.]))
            torch.testing.assert_close(gain, torch.tensor([.95, 1.3]))
            model.dehazer.gain_form = "amp"
            gain = model.dehazer._make_gain(torch.tensor([-100., 100.]))
            torch.testing.assert_close(gain, torch.tensor([1., 1.3]))

    def test_dictionary_output_does_not_test_tensor_truth(self):
        tensor = torch.rand(1, 3, 8, 8)
        for key in ("out", "dehazed", "dehaze"):
            pred, _ = unpack_dehaze_output({key: tensor})
            torch.testing.assert_close(pred, tensor)

    def test_staged_updates_and_roundtrip(self):
        config = small_config()
        model = build_model(config)
        batches = [{"hazy": torch.rand(3, 32, 32), "clear": torch.rand(3, 32, 32),
                    "mask": torch.randint(0, 2, (32, 32)), "id": str(i)} for i in range(2)]
        loader = DataLoader(batches, batch_size=2)
        args = Namespace(lam_dehaze=1., lam_seg=1., lam_ssim=.4, lam_perc=0., lam_ce=1., lam_dice=1., grad_clip=1.)
        for stage in ("dehaze", "seg", "joint"):
            configure_stage(model, stage)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            stats = train_epoch(model, loader, optimizer, torch.device("cpu"), args, stage)
            self.assertTrue(np.isfinite(stats["total"]))
            changed = {key for key, value in model.state_dict().items() if not torch.equal(value, before[key])}
            self.assertTrue(changed)
            if stage == "dehaze":
                self.assertFalse(any(k.startswith("segmenter.") for k in changed))
            elif stage == "seg":
                self.assertFalse(any(k.startswith("dehazer.") for k in changed))
            else:
                self.assertTrue(any(k.startswith("dehazer.") for k in changed))
                self.assertTrue(any(k.startswith("segmenter.") for k in changed))
        self.assertIn("miou", validate(model, loader, torch.device("cpu")))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            save_checkpoint(path, model, config, vars(args), "joint", 1, {})
            loaded, loaded_config, _ = load_checkpoint(path)
            model.eval()
            x = torch.rand(1, 3, 32, 48)
            with torch.no_grad():
                for a, b in zip(model(x)[:2], loaded(x)[:2]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(config, loaded_config)
            torch.save({"state_dict": model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "lack"):
                load_checkpoint(path)

    def test_loss_and_vgg_failure(self):
        args = Namespace(lam_perc=0., lam_ssim=.4)
        pred, clear = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
        losses = dehaze_losses(pred, clear, args)
        torch.testing.assert_close(losses["dehaze"], losses["l1"] + .4 * losses["ssim_loss"])
        with patch("torchvision.models.vgg16", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "TORCH_HOME"):
                VGGPerceptualLoss(torch.device("cpu"))

    def test_native_size_and_alignment(self):
        model = build_model(small_config()).eval()
        image = Image.fromarray(np.zeros((19, 27, 3), dtype=np.uint8))
        pred, logits, _ = infer_image(model, image, torch.device("cpu"), resize=(32, 48))
        self.assertEqual(pred.shape[-2:], (19, 27))
        self.assertEqual(logits.shape[-2:], (19, 27))
        a, b = align_pair(torch.ones(1, 3, 10, 12), torch.ones(1, 3, 12, 10))
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape[-2:], (10, 10))

    def test_data_pairing_and_split(self):
        train, val = split_ids([f"{i:03d}" for i in range(1, 186)])
        self.assertEqual((len(train), len(val)), (157, 28))
        self.assertFalse(set(train) & set(val))
        self.assertEqual((train, val), split_ids([f"{i:03d}" for i in range(1, 186)]))
        with self.assertRaises(ValueError):
            split_ids(["one"])
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        self.assertEqual(mask_to_label_tensor(Image.fromarray(mask), 2).tolist(), [[0, 1], [1, 0]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((40, 48, 3), dtype=np.uint8)
            image[:, :24] = 255
            for folder in ("hazy", "clear", "masks"):
                (root / folder).mkdir()
            Image.fromarray(image).save(root / "hazy/001.png")
            Image.fromarray(image).save(root / "clear/001.png")
            Image.fromarray((image[..., 0] > 0).astype(np.uint8)).save(root / "masks/001.png")
            ds = make_dataset("paired-road", root, resize=(32, 48), augment=True)
            # Force horizontal flip and disable photometric augmentation.
            with patch("random.random", side_effect=[.1, .9]):
                sample = ds[0]
            torch.testing.assert_close(sample["hazy"], sample["clear"])
            self.assertTrue(torch.equal(sample["hazy"][0] > .5, sample["mask"].bool()))
            (root / "hazy/001.png").rename(root / "hazy/001_1.png")
            ds = make_dataset("sots-indoor", root, resize=(32, 32))
            self.assertNotIn("mask", ds[0])
            ds = make_dataset("sots-outdoor", root, resize=(32, 32))
            torch.testing.assert_close(ds[0]["hazy"], ds[0]["clear"])
            (root / "synthetic/synthetic").mkdir(parents=True)
            (root / "synthetic/original").mkdir(parents=True)
            Image.fromarray(image).save(root / "synthetic/synthetic/test.png")
            Image.fromarray(image).save(root / "synthetic/original/test.png")
            ds = make_dataset("hsts", root, resize=(32, 32), augment=True, train_repeats=3)
            self.assertEqual(len(ds), 3)
            self.assertNotIn("mask", ds[0])

    def test_cli_parser(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--model", "ffanet", "--dataset", "sots-outdoor"])
        self.assertEqual(args.model, "ffanet")
        self.assertEqual(args.pretrain_dehaze_epochs, 60)

    def test_plot_stage_axis_and_negative_gain(self):
        from dehaze_seg.visualization.curves import _epoch_axis
        from dehaze_seg.visualization.components import save_gain_deviation_plot
        import warnings
        self.assertEqual(_epoch_axis([{"stage": "dehaze", "epoch": "1"}, {"stage": "seg", "epoch": "1"}]), [1., 2.])
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            path = Path(directory) / "gain.png"
            save_gain_deviation_plot({"gain": np.array([.98, 1., 1.1])}, path)
            self.assertTrue(path.is_file())

    def test_metadata_prefix_and_mismatch(self):
        config = model_config(joint=False)
        config["dehazer"]["base_ch"] = 8
        model = build_model(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"model_config": config, "state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, path)
            loaded, _, _ = load_checkpoint(path)
            self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))
            broken = model.state_dict()
            broken.pop(next(iter(broken)))
            torch.save({"model_config": config, "model_state": broken}, path)
            with self.assertRaisesRegex(ValueError, "No partial load"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
