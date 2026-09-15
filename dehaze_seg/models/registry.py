"""One source of model names, constructor defaults, and saved configuration."""
from copy import deepcopy
import inspect

from .C2PNet import C2PNet
from .ColorAwareUnet import ColorAwareUNet
from .DCP import DCPDehaze
from .FFANet import FFANet
from .GridDehazeNet import GridDehazeNet
from .LiteAttentionUnet import LiteAttentionUNet
from .PSD import PSDDehazeNet
from .joint import IdentityDehazer, JointDehazeSegModel

MODELS = {
    "coloraware": ColorAwareUNet, "c2pnet": C2PNet,
    "dcp": DCPDehaze, "ffanet": FFANet, "grid": GridDehazeNet, "psd": PSDDehazeNet,
}
OVERRIDES = {
    "coloraware": dict(base_ch=32, residual_scale=.5, gain_scale=.30, gain_form="tanh",
                       gain_min=.95, gain_mode="global", norm="inst", refine_scale=.25),
    "c2pnet": dict(base_ch=32, blocks_per_group=6, groups=3, use_pdu=True),
    "dcp": dict(guided_radius=7, guided_eps=1e-4, use_learned_refine=False),
    "ffanet": dict(base_ch=64, n_down=0, n_ffab_deep=19, groups=3),
    "grid": dict(rows=3, cols=6, base_ch=32),
    "psd": dict(base_ch=32, feat_ch=64, residual_scale=.45, refine_scale=.15, t_min=.10),
}


def constructor_defaults(cls):
    return {name: p.default for name, p in inspect.signature(cls).parameters.items()
            if p.default is not inspect.Parameter.empty}


def model_config(name="coloraware", joint=True):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}")
    dehazer = constructor_defaults(MODELS[name])
    dehazer.update(OVERRIDES[name])
    return dict(model=name, joint=joint, dehazer=dehazer,
                segmenter=constructor_defaults(LiteAttentionUNet), imagenet_norm=True,
                dehazer_type="network")


def build_model(config):
    config = deepcopy(config)
    name = config["model"]
    if name not in MODELS:
        raise ValueError(f"Unsupported model {name!r}; supported: {', '.join(MODELS)}")
    dehazer = (IdentityDehazer() if config.get("dehazer_type") == "identity"
               else MODELS[name](**config["dehazer"]))
    if not config["joint"]:
        return dehazer
    return JointDehazeSegModel(dehazer, LiteAttentionUNet(**config["segmenter"]),
                              imagenet_norm=config["imagenet_norm"])
