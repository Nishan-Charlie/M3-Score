"""Two fixed synthetic image sets for the golden-value regression test."""

import torch
import torch.nn.functional as F


def synthetic_sets(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth random textures at 224x224.

    A quarter of the second set is a slightly altered copy of real images and the rest
    is a lower-contrast draw, so that fidelity, memorization and coverage all land away
    from 0 and 1. A regression in any of the three feature layers then moves a value.
    """

    def make(seed: int, contrast: float) -> torch.Tensor:
        coarse = torch.rand(n, 1, 14, 14, generator=torch.Generator().manual_seed(seed))
        smooth = F.interpolate(coarse, size=(224, 224), mode="bilinear", align_corners=False)
        return (smooth * contrast).clamp(0.0, 1.0)

    real = make(0, 1.0)
    near_copies = (real[: n // 4] * 0.97 + 0.015).clamp(0.0, 1.0)
    generated = torch.cat([near_copies, make(1, 0.7)[: n - n // 4]])
    return real, generated
