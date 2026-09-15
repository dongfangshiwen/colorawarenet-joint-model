"""Explicit ablation registry; no runtime mutation of training modules."""
from typing import Dict

EXPERIMENTS: Dict[str, Dict[str, object]] = {
    # Architecture and color branch.
    "arch_baseline_unet": {
        "group": "architecture",
        "description": "Residual U-Net only: no color gain and no refine contribution.",
        "dehazer": dict(disable_gain=True, gain_scale=0.0, refine_scale=0.0),
    },
    "arch_no_color_gain": {
        "group": "architecture",
        "description": "Disable color-aware gain, keep residual and refine branches.",
        "dehazer": dict(disable_gain=True, gain_scale=0.0, refine_scale=0.25),
    },
    "arch_no_refine": {
        "group": "architecture",
        "description": "Disable refine contribution, keep global color gain.",
        "dehazer": dict(disable_gain=False, refine_scale=0.0),
    },
    "arch_full": {
        "group": "architecture",
        "description": "Full ColorAwareUNet default: global tanh gain + refine.",
        "dehazer": {},
    },
    "gain_local": {
        "group": "color_gain",
        "description": "Table 4: spatial local tanh gain without a lower clamp.",
        "dehazer": dict(gain_mode="local", gain_min=None),
    },
    "gain_amp": {
        "group": "color_gain",
        "description": "Use amplify-only gain to test one-sided color compensation.",
        "dehazer": dict(gain_form="amp", gain_min=1.0),
    },
    "gain_no_min": {
        "group": "color_gain",
        "description": "Remove gain lower clamp to test whether gain_min prevents gray bias.",
        "dehazer": dict(gain_min=None),
    },
    "gain_weak": {
        "group": "color_gain",
        "description": "Weak gain scale.",
        "dehazer": dict(gain_scale=0.10, gain_min=0.98),
    },
    "gain_strong": {
        "group": "color_gain",
        "description": "Strong gain scale.",
        "dehazer": dict(gain_scale=0.50, gain_min=0.90),
    },
    # Training strategy.
    "strategy_direct_joint": {
        "group": "training_strategy",
        "description": "No staged pretraining; train dehaze and segmentation jointly from scratch.",
        "strategy": "direct_joint",
        "dehazer": {},
    },
    "strategy_staged_no_joint": {
        "group": "training_strategy",
        "description": "Train dehaze, then frozen-dehaze segmentation; no final joint finetune.",
        "strategy": "staged_no_joint",
        "dehazer": {},
    },
    "strategy_full_staged": {
        "group": "training_strategy",
        "description": "Full staged protocol: dehaze pretrain, segmentation pretrain, joint finetune.",
        "strategy": "full_staged",
        "dehazer": {},
    },
    # Loss.
    "loss_l1_only": {
        "group": "loss",
        "description": "Dehaze L1 only and segmentation CE+Dice.",
        "loss": dict(lam_ssim=0.0, lam_perc=0.0),
        "dehazer": {},
    },
    "loss_l1_ssim": {
        "group": "loss",
        "description": "Dehaze L1 + SSIM, no perceptual loss.",
        "loss": dict(lam_ssim=0.40, lam_perc=0.0),
        "dehazer": {},
    },
    "loss_l1_ssim_perc": {
        "group": "loss",
        "description": "Full dehaze loss: L1 + SSIM + perceptual.",
        "loss": dict(lam_ssim=0.40, lam_perc=0.05),
        "dehazer": {},
    },
    "loss_ce_only": {
        "group": "loss",
        "description": "Segmentation CE only, no Dice.",
        "loss": dict(lam_ce=1.0, lam_dice=0.0),
        "dehazer": {},
    },
    # Capacity.
    "capacity_small": {
        "group": "capacity",
        "description": "Small dehazer width, base_ch=16.",
        "dehazer": dict(base_ch=16),
    },
    "capacity_default": {
        "group": "capacity",
        "description": "Default dehazer width, base_ch=32.",
        "dehazer": dict(base_ch=32),
    },
    "capacity_large": {
        "group": "capacity",
        "description": "Large dehazer width, base_ch=48.",
        "dehazer": dict(base_ch=48),
    },
    # Data augmentation.
    "data_no_aug": {
        "group": "data",
        "description": "No training augmentation.",
        "augment": False,
        "dehazer": {},
    },
    "data_default_aug": {
        "group": "data",
        "description": "Default datasets_joint training augmentation.",
        "augment": True,
        "dehazer": {},
    },
    # Downstream segmentation relation.
    "downstream_seg_hazy": {
        "group": "downstream",
        "description": "Train segmentation directly on hazy image with identity dehazer.",
        "strategy": "seg_only",
        "dehazer_type": "identity",
        "seg_source": "hazy",
    },
    "downstream_seg_dehazed": {
        "group": "downstream",
        "description": "Full ColorAwareUNet dehazing before segmentation.",
        "strategy": "full_staged",
        "dehazer": {},
        "seg_source": "dehazed",
    },
}


EXPERIMENTS.update({
    "attention_none": {"group": "attention", "description": "Disable attention gates and SE.", "segmenter": {"attention": False, "use_se": False}},
    "attention_gate": {"group": "attention", "description": "Attention gates without SE (paper default).", "segmenter": {"attention": True, "use_se": False}},
    "attention_se": {"group": "attention", "description": "SE without attention gates.", "segmenter": {"attention": False, "use_se": True}},
    "attention_both": {"group": "attention", "description": "Attention gates and SE.", "segmenter": {"attention": True, "use_se": True}},
    "normalization_off": {"group": "normalization", "description": "Disable ImageNet input normalization.", "imagenet_norm": False},
    "strategy_dehaze_only": {"group": "training_strategy", "description": "Only train restoration.", "strategy": "dehaze_only"},
})
