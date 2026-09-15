"""Output adaptation and the paper's dehaze-to-segmentation connection."""
import torch
from torch import nn


def unpack_dehaze_output(output):
    aux = {"residual": None, "color_gain": None, "sides": []}
    if isinstance(output, (tuple, list)):
        dehazed = output[0]
        for i, key in enumerate(aux, start=1):
            if len(output) > i:
                aux[key] = output[i]
    elif isinstance(output, dict):
        dehazed = next((output[k] for k in ("out", "dehazed", "dehaze")
                        if k in output and output[k] is not None), None)
        aux.update({k: output[k] for k in aux if k in output})
    else:
        dehazed = output
    if not torch.is_tensor(dehazed) or dehazed.ndim != 4:
        raise ValueError("Dehazer must return a BCHW tensor or a supported output container")
    return dehazed.clamp(0, 1), aux


class IdentityDehazer(nn.Module):
    def forward(self, x):
        return x


class JointDehazeSegModel(nn.Module):
    def __init__(self, dehazer, segmenter, imagenet_norm=True):
        super().__init__()
        self.dehazer = dehazer
        self.segmenter = segmenter
        self.imagenet_norm = imagenet_norm
        self.register_buffer("imagenet_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def segment(self, image):
        image = (image - self.imagenet_mean) / self.imagenet_std if self.imagenet_norm else image
        output = self.segmenter(image)
        return output["out"] if isinstance(output, dict) else output

    def forward(self, hazy):
        dehazed, aux = unpack_dehaze_output(self.dehazer(hazy))
        return dehazed, self.segment(dehazed), aux
