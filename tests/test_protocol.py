"""Regression tests for the manuscript's training and comparison protocols."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from dehaze_seg.cli import build_parser
from dehaze_seg.data.datasets import make_dataset
from dehaze_seg.data.splits import training_split, training_overlap
from dehaze_seg.engine.checkpoint import load_checkpoint, save_checkpoint, historical_config
from dehaze_seg.engine.inference import infer_image, run_inference, select_split, use_shared_segmenter
from dehaze_seg.engine.train import configure_stage, run_training, stage_schedule, train_epoch
from dehaze_seg.experiments.ablations import EXPERIMENTS
from dehaze_seg.models.DCP import ClassicalDCP, DCPDehaze
from dehaze_seg.models.registry import build_model, model_config, constructor_defaults

torch.set_num_threads(2)


def small_config(joint=True):
    config = model_config(joint=joint)
    config["dehazer"]["base_ch"] = 8
    config["segmenter"]["base_ch"] = 8
    return config


def road_data(root, count=4):
    rng = np.random.default_rng(42)
    for folder in ("hazy", "clear", "masks"):
        (root / folder).mkdir(parents=True)
    for i in range(count):
        clear = rng.integers(0, 180, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(clear).save(root / "clear" / f"{i:03d}.png")
        Image.fromarray(clear + 40).save(root / "hazy" / f"{i:03d}.png")
        Image.fromarray((clear[..., 0] > 90).astype(np.uint8) * 255).save(root / "masks" / f"{i:03d}.png")
    return [f"{i:03d}" for i in range(count)]


class ProtocolTests(unittest.TestCase):
    def test_amplify_gain_learns_from_initialization(self):
        for mode in ("global", "local"):
            torch.manual_seed(42)
            config = small_config(False)
            config["dehazer"].update(EXPERIMENTS["gain_amp"]["dehazer"], gain_mode=mode)
            model = build_model(config)
            head = model.gain_fc[-1] if mode == "global" else model.local_gain_head[-1]
            before = head.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.rand(2, 3, 32, 32) * .5
            for _ in range(3):
                optimizer.zero_grad()
                gain = model(x)[2]
                loss = (gain - 1.15).square().mean()
                loss.backward()
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
                optimizer.step()
            self.assertFalse(torch.equal(before, head.weight))
            self.assertGreater(model(x)[2].mean().item(), 1.0003)

    def test_local_gain_and_historical_configuration(self):
        config = small_config(False)
        config["dehazer"].update(EXPERIMENTS["gain_local"]["dehazer"])
        model = build_model(config)
        torch.testing.assert_close(model._make_gain(torch.tensor([-100., 100.])), torch.tensor([.7, 1.3]))
        old_args = dict(dehaze_backbone="coloraware", lam_ssim=.4, experiment_name="gain_local")
        legacy = historical_config({"args": old_args}, model.state_dict())
        self.assertEqual(legacy["dehazer"]["gain_min"], .95)
        old_args["color_gain_min"] = .91
        self.assertEqual(historical_config({"args": old_args}, model.state_dict())["dehazer"]["gain_min"], .91)

    def test_classical_dcp_formula_and_native_size(self):
        dcp = ClassicalDCP(patch_size=1, guided=False, top_percent=.5)
        self.assertEqual(sum(p.numel() for p in dcp.parameters()), 0)
        # Pixel 2 has the largest dark channel and supplies atmospheric RGB.
        x = torch.tensor([[[[.2, .8]], [[.3, .9]], [[.4, 1.0]]]])
        atmosphere = x[..., 1:2]
        transmission = 1 - .95 * (x / atmosphere).amin(1, keepdim=True)
        expected = ((x - atmosphere) / transmission.clamp_min(.1) + atmosphere).clamp(0, 1)
        torch.testing.assert_close(dcp(x)[0], expected)
        image = Image.fromarray(np.random.default_rng(3).integers(0, 255, (19, 27, 3), dtype=np.uint8))
        from dehaze_seg.data.common import pil_to_rgb_tensor
        prediction = infer_image(dcp, image, torch.device("cpu"))[0]
        torch.testing.assert_close(prediction, dcp(pil_to_rgb_tensor(image)[None])[0], rtol=0, atol=0)

    def test_dcp_segmentation_training_and_roundtrip(self):
        parser = build_parser()
        defaults = parser.parse_args(["train", "--model", "dcp", "--dcp-mode", "classical"])
        self.assertEqual(stage_schedule(defaults), [("seg", 40)])
        for dataset in ("sots-indoor", "sots-outdoor", "hsts"):
            unlabelled = parser.parse_args(["train", "--model", "dcp", "--dcp-mode", "classical", "--dataset", dataset,
                                           "--data-root", "nonexistent"])
            with self.assertRaisesRegex(ValueError, "requires --dataset paired-road"):
                run_training(unlabelled)
        empty = parser.parse_args(["train", "--model", "dcp", "--dcp-mode", "classical", "--pretrain-seg-epochs", "0",
                                   "--finetune-epochs", "0", "--data-root", "nonexistent"])
        with self.assertRaisesRegex(ValueError, "finetune-epochs > 0"):
            run_training(empty)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            road_data(root / "data")
            args = parser.parse_args(["train", "--model", "dcp", "--dcp-mode", "classical", "--data-root", str(root / "data"),
                "--output", str(root / "runs"), "--resize", "32", "32", "--seg-base-ch", "8",
                "--pretrain-seg-epochs", "1", "--finetune-epochs", "1", "--no-plots", "--device", "cpu"])
            config = model_config("dcp", joint=True)
            config["segmenter"]["base_ch"] = 8
            model = build_model(config)
            before = {k: v.clone() for k, v in model.segmenter.named_parameters()}
            probe = torch.rand(1, 3, 32, 32)
            fixed_prediction = model.dehazer(probe)[0].clone()
            with self.assertRaisesRegex(ValueError, "stays fixed"):
                configure_stage(model, "joint")
            with patch("dehaze_seg.engine.train.build_model", return_value=model), \
                 patch("dehaze_seg.engine.train.VGGPerceptualLoss", side_effect=AssertionError("DCP needs no VGG")):
                stats = run_training(args)
            self.assertIn("miou", stats)
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.segmenter.named_parameters()))
            self.assertEqual(sum(p.numel() for p in model.dehazer.parameters()), 0)
            self.assertFalse(model.dehazer.training)
            torch.testing.assert_close(fixed_prediction, model.dehazer(probe)[0], rtol=0, atol=0)
            output = root / "runs/paired-road/dcp"
            with (output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["stage"] for row in rows], ["seg", "seg"])
            self.assertEqual(rows[0]["val_psnr"], rows[1]["val_psnr"])
            self.assertFalse((output / "dehaze").exists() or (output / "joint").exists())
            loaded, loaded_config, checkpoint = load_checkpoint(output / "last.pth")
            self.assertIsInstance(loaded.dehazer, ClassicalDCP)
            self.assertEqual(loaded_config, config)
            self.assertEqual(checkpoint["train_config"]["training_protocol"], "fixed-dcp-segmentation")
            self.assertEqual(checkpoint["train_config"]["stages"], [dict(stage="seg", epochs=2)])
            self.assertEqual(checkpoint["stage"], "seg")
            self.assertEqual(checkpoint["data_split"]["shared_clear_references"], [])
            with torch.no_grad():
                for expected, actual in zip(model(probe)[:2], loaded(probe)[:2]):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            predict_args = parser.parse_args(["predict", "--checkpoint", str(output / "best.pth"),
                "--input", str(root / "data/hazy/000.png"), "--output", str(root / "predict"), "--device", "cpu"])
            result = run_inference(predict_args)[0]
            self.assertEqual((result["model"], result["implementation"], result["segmentation"]),
                             ("dcp", "classical", "checkpoint"))
            self.assertTrue((root / "predict/masks/000.png").is_file())

    def test_learned_dcp_three_stages_and_segmentation_gradient(self):
        torch.manual_seed(42)
        parser = build_parser()
        defaults = parser.parse_args(["train", "--model", "dcp"])
        self.assertEqual(defaults.dcp_mode, "learned")
        self.assertEqual(stage_schedule(defaults), [("dehaze", 60), ("seg", 20), ("joint", 20)])
        config = model_config("dcp", joint=True, dcp_mode="learned")
        config["segmenter"]["base_ch"] = 8
        model = build_model(config)
        self.assertIsInstance(model.dehazer, DCPDehaze)
        self.assertGreater(sum(p.numel() for p in model.dehazer.parameters()), 0)
        probe = torch.rand(2, 3, 32, 32)
        configure_stage(model, "joint")
        _, logits, _ = model(probe)
        # Isolate segmentation supervision: its gradient must reach restoration.
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 2, (2, 32, 32)))
        loss.backward()
        gradient = model.dehazer.refine_net.conv3.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.)
        model.zero_grad(set_to_none=True)
        stages = []

        def checked_epoch(*args, **kwargs):
            current, stage = args[0], args[5]
            before = {k: v.clone() for k, v in current.state_dict().items()}
            stats = train_epoch(*args, **kwargs)
            changed = {k for k, v in current.state_dict().items() if not torch.equal(v, before[k])}
            self.assertEqual(any(k.startswith("dehazer.") for k in changed), stage != "seg")
            self.assertEqual(any(k.startswith("segmenter.") for k in changed), stage != "dehaze")
            self.assertEqual("dehaze" in stats, stage != "seg")
            self.assertEqual("seg" in stats, stage != "dehaze")
            self.assertTrue(np.isfinite(stats["total"]))
            stages.append(stage)
            return stats

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            road_data(root / "data")
            args = parser.parse_args(["train", "--model", "dcp", "--data-root", str(root / "data"),
                "--output", str(root / "runs"), "--resize", "32", "32", "--seg-base-ch", "8",
                "--pretrain-dehaze-epochs", "1", "--pretrain-seg-epochs", "1", "--finetune-epochs", "1",
                "--lam-perc", "0", "--no-plots", "--device", "cpu"])
            with patch("dehaze_seg.engine.train.build_model", return_value=model), \
                 patch("dehaze_seg.engine.train.train_epoch", side_effect=checked_epoch):
                run_training(args)
            self.assertEqual(stages, ["dehaze", "seg", "joint"])
            output = root / "runs/paired-road/dcp"
            for stage in stages:
                self.assertTrue((output / stage / "best.pth").is_file())
            loaded, loaded_config, checkpoint = load_checkpoint(output / "last.pth")
            self.assertEqual(loaded_config, config)
            self.assertEqual(checkpoint["train_config"]["training_protocol"], "learned-dcp-joint")
            self.assertEqual(checkpoint["stage"], "joint")
            with torch.no_grad():
                for expected, actual in zip(model(probe)[:2], loaded(probe)[:2]):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            infer_args = parser.parse_args(["predict", "--checkpoint", str(output / "best.pth"),
                "--input", str(root / "data/hazy/000.png"), "--output", str(root / "predict"), "--device", "cpu"])
            result = run_inference(infer_args)[0]
            self.assertEqual(result["implementation"], "learned-refinement")
            self.assertTrue((root / "predict/masks/000.png").is_file())

    def test_learned_dcp_restoration_without_masks(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/hazy").mkdir(parents=True)
            (root / "data/clear").mkdir()
            rng = np.random.default_rng(42)
            for i in range(3):
                clear = rng.integers(0, 180, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(clear).save(root / "data/clear" / f"{i}.png")
                Image.fromarray(clear + 40).save(root / "data/hazy" / f"{i}_1.png")
            args = parser.parse_args(["train", "--model", "dcp", "--dataset", "sots-indoor",
                "--data-root", str(root / "data"), "--output", str(root / "runs"), "--resize", "32", "32",
                "--pretrain-dehaze-epochs", "1", "--lam-perc", "0", "--no-plots", "--device", "cpu"])
            self.assertEqual(stage_schedule(args), [("dehaze", 1)])
            stats = run_training(args)
            self.assertNotIn("miou", stats)
            loaded, config, checkpoint = load_checkpoint(root / "runs/sots-indoor/dcp/last.pth")
            self.assertIsInstance(loaded, DCPDehaze)
            self.assertFalse(config["joint"])
            self.assertEqual(checkpoint["train_config"]["training_protocol"], "learned-dcp-restoration")
            self.assertEqual(checkpoint["stage"], "dehaze")

    def test_historical_dcp_roundtrip_keeps_one_public_name(self):
        config = model_config("dcp", joint=False)
        config["dehazer"] = constructor_defaults(DCPDehaze)
        old = DCPDehaze(**config["dehazer"]).eval()
        x = torch.rand(1, 3, 32, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dcp.pth"
            torch.save(dict(format_version=1, model_config=config, model_state=old.state_dict()), path)
            loaded, migrated, _ = load_checkpoint(path, model_name="dcp")
            self.assertEqual(migrated["model"], "dcp")
            self.assertEqual(migrated["dehazer_type"], "legacy-dcp")
            with torch.no_grad():
                torch.testing.assert_close(old(x)[0], loaded(x)[0], rtol=0, atol=0)
            save_checkpoint(path, loaded, migrated, {}, "dehaze", 1, {})
            reloaded, _, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(old(x)[0], reloaded(x)[0], rtol=0, atol=0)

    def test_sots_groups_and_independent_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            (root / "hazy").mkdir(parents=True)
            (root / "gt").mkdir()
            ids = []
            for scene in range(4):
                Image.new("RGB", (32, 32), (20+scene, 30, 40)).save(root / "gt" / f"{scene}.png")
                for haze in range(2):
                    stem = f"{scene}_{haze}"
                    Image.new("RGB", (32, 32), (60+scene, 80, 90)).save(root / "hazy" / f"{stem}.png")
                    ids.append(stem)
            split = training_split("sots-indoor", root, ids, .25, 42)
            self.assertEqual(split, training_split("sots-indoor", root, list(reversed(ids)), .25, 42))
            self.assertEqual((len(split["train"]), len(split["val"])), (6, 2))
            self.assertFalse({s.split("_")[0] for s in split["train"]} & {s.split("_")[0] for s in split["val"]})
            copied = Path(directory) / "copied"
            shutil.copytree(root, copied)
            overlap = training_overlap("sots-indoor", copied, ids, None, {"data_split": split})
            self.assertEqual(set(overlap), set(split["train"]))
            with self.assertRaisesRegex(ValueError, "share clear references"):
                training_split("sots-indoor", root, ids, val_root=copied, val_ids=ids)
            args = build_parser().parse_args(["evaluate", "--model", "dcp", "--dataset", "sots-indoor", "--data-root", str(root)])
            images = sorted((root / "hazy").glob("*.png"))
            self.assertEqual([p.stem for p in select_split(images, args, None, {"data_split": split})], split["val"])

    def test_road_scene_split_preserves_budget_and_legacy_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = road_data(root, 20)
            # The seed-42 sample split separates these two references.
            from dehaze_seg.data.datasets import split_ids
            train, val = split_ids(ids)
            shutil.copyfile(root / "clear" / f"{train[0]}.png", root / "clear" / f"{val[0]}.png")
            legacy = training_split("paired-road", root, ids, split_unit="sample")
            self.assertEqual((legacy["train"], legacy["val"]), (train, val))
            self.assertEqual(legacy["shared_clear_references"], [val[0]])
            grouped = training_split("paired-road", root, ids, split_unit="scene")
            self.assertEqual((len(grouped["train"]), len(grouped["val"])), (17, 3))
            self.assertEqual(grouped["shared_clear_references"], [])
            self.assertEqual(grouped, training_split("paired-road", root, list(reversed(ids)), split_unit="scene"))

    def test_shared_segmenter_and_three_stage_cli_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = road_data(root / "data")
            parser = build_parser()
            args = parser.parse_args(["train", "--data-root", str(root / "data"), "--output", str(root / "runs"),
                "--resize", "32", "32", "--color-base-ch", "8", "--seg-base-ch", "8", "--batch-size", "2",
                "--pretrain-dehaze-epochs", "1", "--pretrain-seg-epochs", "1", "--finetune-epochs", "1",
                "--lam-perc", "0", "--no-plots", "--device", "cpu"])
            run_training(args)
            output = root / "runs/paired-road/coloraware"
            joint_path = output / "last.pth"
            joint, config, checkpoint = load_checkpoint(joint_path)
            split = checkpoint["data_split"]
            self.assertEqual(split, json.loads((output / "split.json").read_text()))
            self.assertEqual(set(split["train"] + split["val"]), set(ids))
            standalone = deepcopy(config)
            standalone["joint"] = False
            dehaze_path = root / "dehazer.pth"
            save_checkpoint(dehaze_path, joint.dehazer, standalone, {}, "dehaze", 1, {}, data_split=split)
            shared = use_shared_segmenter(build_model(model_config("dcp", False)), joint)
            self.assertIs(shared.segmenter, joint.segmenter)
            self.assertFalse(any(p.requires_grad for p in shared.segmenter.parameters()))
            before = {k: v.clone() for k, v in joint.segmenter.state_dict().items()}
            infer_image(shared, Image.new("RGB", (19, 27), (80, 90, 100)), torch.device("cpu"))
            self.assertTrue(all(torch.equal(before[k], v) for k, v in joint.segmenter.state_dict().items()))
            eval_args = parser.parse_args(["evaluate", "--checkpoint", str(joint_path), str(dehaze_path),
                "--include-dcp", "--segmenter-checkpoint", str(joint_path), "--data-root", str(root / "data"),
                "--output", str(root / "eval"), "--device", "cpu"])
            results = run_inference(eval_args, evaluate=True)
            self.assertEqual(len(results), 3)
            self.assertEqual(len({r["segmenter_sha256"] for r in results}), 1)
            self.assertTrue(all(r["segmentation"] == "shared-frozen" and "miou" in r and r["count"] == 1 for r in results))
            self.assertEqual(results[0]["miou"], results[1]["miou"])
            protocols = [json.loads(p.read_text()) for p in (root / "eval").glob("*/protocol.json")]
            self.assertTrue(all(p["samples"] == split["val"] for p in protocols))
            eval_args.split = "all"
            with self.assertRaisesRegex(ValueError, "overlaps known training"):
                run_inference(eval_args, evaluate=True)
            eval_args.allow_training_overlap = True
            eval_args.include_dcp = False
            eval_args.checkpoint = [str(joint_path)]
            diagnostics = run_inference(eval_args, evaluate=True)
            self.assertEqual(diagnostics[0]["known_training_overlap"], 3)
            with self.assertRaisesRegex(ValueError, "must contain a joint"):
                use_shared_segmenter(joint, joint.dehazer)

    def test_explicit_validation_training_and_checkpoint_portability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = road_data(root / "train", 2)
            road_data(root / "val", 2)
            # Different references despite identical local sample names.
            for path in (root / "val/clear").glob("*.png"):
                with Image.open(path) as image:
                    Image.fromarray(255 - np.asarray(image)).save(path)
            split = training_split("paired-road", root / "train", ids, val_root=root / "val", val_ids=ids)
            args = build_parser().parse_args(["evaluate", "--model", "dcp", "--data-root", str(root / "train")])
            with self.assertRaisesRegex(ValueError, "separate validation root"):
                select_split(list((root / "train/hazy").glob("*.png")), args, None, {"data_split": split})
            args.data_root = str(root / "val")
            selected = select_split(list((root / "val/hazy").glob("*.png")), args, None, {"data_split": split})
            self.assertEqual([p.stem for p in selected], ids)
            self.assertEqual(training_overlap("paired-road", root / "val", ids, None, {"data_split": split}), [])
            train_args = build_parser().parse_args(["train", "--data-root", str(root / "train"),
                "--val-root", str(root / "val"), "--output", str(root / "runs"), "--resize", "32", "32",
                "--color-base-ch", "8", "--seg-base-ch", "8", "--pretrain-dehaze-epochs", "1",
                "--pretrain-seg-epochs", "0", "--finetune-epochs", "0", "--lam-perc", "0", "--no-plots"])
            run_training(train_args)
            # Move only the checkpoint: the split no longer depends on split.json.
            relocated = root / "portable.pth"
            shutil.copyfile(root / "runs/paired-road/coloraware/last.pth", relocated)
            _, _, checkpoint = load_checkpoint(relocated)
            self.assertEqual(checkpoint["data_split"], split)


if __name__ == "__main__":
    unittest.main()
