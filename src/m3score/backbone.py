"""Loading the encoder and turning images into per-block CLS features."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from . import protocol
from ._errors import M3ScoreError

_REQUIRED = ("patch_embed", "_pos_embed", "norm_pre", "blocks")


class Backbone:
    """A timm vision transformer with the m3-v1 preprocessing in front of it."""

    def __init__(self, model: torch.nn.Module, name: str) -> None:
        missing = [attr for attr in _REQUIRED if not hasattr(model, attr)]
        if missing:
            raise TypeError(
                f"the backbone must be a timm vision transformer; "
                f"{type(model).__name__} has no {', '.join(missing)}"
            )
        self.model = model.eval()
        self.name = name
        size = model.patch_embed.img_size
        self.image_size = (int(size[0]), int(size[1]))
        self.num_layers = len(model.blocks)

    def prepare(self, images: torch.Tensor) -> torch.Tensor:
        """``(N, 1|3, H, W)`` float in [0, 1] to ``(N, 3, S, S)``.

        Grey is repeated to three channels, then one antialiased bilinear resize brings
        the image to the encoder's input size. This is the only resize in the package,
        so every input type is treated identically.
        """
        x = images.float()
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(
                x, size=self.image_size, mode="bilinear", antialias=True, align_corners=False
            )
        return x

    @torch.no_grad()
    def extract(
        self, prepared: torch.Tensor, layers: Iterable[int], device: torch.device | str
    ) -> dict[int, torch.Tensor]:
        """CLS token after each requested block (numbered from 1), as float32 on the CPU."""
        wanted = set(layers)
        last = max(wanted)
        model = self.model
        mean = torch.tensor(protocol.NORM_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(protocol.NORM_STD, device=device).view(1, 3, 1, 1)
        x = (prepared.to(device) - mean) / std
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        if hasattr(model, "patch_drop"):
            x = model.patch_drop(x)
        x = model.norm_pre(x)
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(model.blocks, start=1):
            x = block(x)
            if index in wanted:
                features[index] = x[:, 0, :].float().cpu()
            if index == last:
                break
        return features


def load_backbone(
    spec: str | torch.nn.Module | None = None, device: torch.device | str = "cpu"
) -> Backbone:
    """Build the encoder.

    ``None`` loads the m3-v1 encoder at its pinned Hub revision. A string is a timm
    model name loaded with pretrained weights. A module is used as given.
    """
    if isinstance(spec, torch.nn.Module):
        return Backbone(spec.to(device), type(spec).__name__)
    if spec is not None and not isinstance(spec, str):
        raise TypeError(
            f"backbone must be a timm model name or a torch.nn.Module, got {type(spec).__name__}"
        )

    import timm

    if isinstance(spec, str):
        return Backbone(timm.create_model(spec, pretrained=True, num_classes=0).to(device), spec)

    name = f"{protocol.BACKBONE_ID}@{protocol.BACKBONE_REVISION}"
    try:
        model = timm.create_model(f"hf_hub:{name}", pretrained=True, img_size=protocol.IMAGE_SIZE)
    except Exception as exc:
        raise M3ScoreError(
            f"could not load the M3-Score backbone {protocol.BACKBONE_ID} from the Hugging Face "
            f"Hub ({type(exc).__name__}: {exc}). It is downloaded once, about 90 MB, and cached. "
            f"Check access to huggingface.co, or copy a populated Hugging Face cache to this "
            f"machine and set HF_HUB_OFFLINE=1."
        ) from exc
    return Backbone(model.to(device), name)
