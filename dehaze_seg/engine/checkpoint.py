"""Strict, configuration-aware loading of current and historical weights."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from ..models.registry import build_model, model_config, MODELS

PROFILES = ("paper", "legacy-infer")


def resolve_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dictionary or a dictionary containing weights")
    for key in ("model_state", "model_state_dict", "state_dict", "dehazer_state", "model"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    else:
        state = checkpoint
    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("No supported tensor state dictionary found in checkpoint")
    state = dict(state)
    for prefix in ("module.", "_orig_mod.", "model."):
        while state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    # Historical SOTS wrapper called the segmenter 'seg'.
    state = {("segmenter." + k[4:] if k.startswith("seg.") else k): v for k, v in state.items()}
    return state


def historical_config(checkpoint, state, profile=None, model_name=None):
    saved = checkpoint.get("args", {})
    if not isinstance(saved, dict):
        raise ValueError("Historical args must be a dictionary; provide a JSON --model-config")
    name = saved.get("dehaze_backbone", model_name)
    if not name:
        raise ValueError("Weights lack a model identifier; specify --model and --legacy-profile or --model-config")
    if name not in MODELS:
        raise ValueError(f"Model {name!r} is unavailable; supported: {', '.join(MODELS)}")
    known_args = "dehaze_backbone" in saved and (
        "lam_ssim" in saved or "color_gain_scale" in saved)
    if not known_args and profile is None:
        raise ValueError("Weights lack reconstruction settings; choose --legacy-profile paper/legacy-infer or --model-config")
    joint = any(k.startswith("segmenter.") for k in state)
    config = model_config(name, joint=joint)
    if profile == "legacy-infer":
        legacy = {
            "coloraware": dict(gain_scale=.65, gain_min=None),
            "ffanet": dict(base_ch=32, n_down=2, n_ffab_deep=2),
            "dcp": dict(guided_radius=15, guided_eps=1e-3, use_learned_refine=True),
        }
        config["dehazer"].update(legacy.get(name, {}))
    if name == "coloraware":
        for key in config["dehazer"]:
            if "color_" + key in saved:
                config["dehazer"][key] = saved["color_" + key]
        if "color_disable_gain" in saved:
            config["dehazer"]["disable_gain"] = saved["color_disable_gain"]
        if "color_no_output_clamp" in saved:
            config["dehazer"]["output_clamp"] = not saved["color_no_output_clamp"]
    mapping = dict(num_classes="num_classes", seg_base_ch="base_ch", seg_width_mult="width_mult",
                   seg_use_se="use_se")
    for old, new in mapping.items():
        if old in saved:
            config["segmenter"][new] = saved[old]
    config["segmenter"]["attention"] = not saved.get("seg_no_attention", False)
    config["imagenet_norm"] = not saved.get("no_seg_imagenet_norm", False)
    experiment = saved.get("experiment_name")
    if experiment:
        from ..experiments.ablations import EXPERIMENTS
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown historical ablation {experiment!r}; provide --model-config")
        variant = EXPERIMENTS[experiment]
        config["dehazer"].update(variant.get("dehazer", {}))
        config["dehazer_type"] = variant.get("dehazer_type", "network")
    return config


def load_checkpoint(path, device="cpu", profile=None, model_name=None, config_path=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = resolve_state(checkpoint)
    if "model_config" in checkpoint:
        config = deepcopy(checkpoint["model_config"])
        if config_path or profile:
            raise ValueError("This checkpoint contains a complete model_config; configuration overrides are unnecessary")
    elif config_path:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    else:
        config = historical_config(checkpoint, state, profile, model_name)
    if model_name and config["model"] != model_name:
        raise ValueError(f"Requested {model_name}, but checkpoint is for {config['model']}")
    if not config["joint"] and state and all(k.startswith("dehazer.") for k in state):
        state = {k[len("dehazer."):]: v for k, v in state.items()}
    model = build_model(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint architecture mismatch for {path}. No partial load was performed.\n{exc}") from exc
    return model.to(device).eval(), config, checkpoint


def save_checkpoint(path, model, config, train_config, stage, epoch, stats, optimizer=None,
                    scheduler=None, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(format_version=1, model_state=model.state_dict(), model_config=deepcopy(config),
                 train_config=dict(train_config), stage=stage, epoch=epoch, stats=stats)
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    torch.save(state, path)
