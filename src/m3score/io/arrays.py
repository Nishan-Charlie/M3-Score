"""Images already in memory as a torch tensor or numpy array."""

from __future__ import annotations

import math

import numpy as np
import torch

from .base import ImageSource

_RANGE_TOLERANCE = 1e-6


def check_images(images) -> tuple[torch.Tensor, float]:
    """Validate a batch of images.

    Returns the batch as ``(N, C, H, W)`` in its own dtype, and the divisor that maps it
    to [0, 1]. Conversion to float is left to :func:`to_float` so that a large uint8
    array is never copied to float32 in one piece.
    """
    if isinstance(images, np.ndarray):
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(images))
        except TypeError as exc:
            raise ValueError(
                f"unsupported image dtype {images.dtype}; "
                f"pass uint8 in [0, 255] or floating point in [0, 1]"
            ) from exc
    elif isinstance(images, torch.Tensor):
        tensor = images.detach().cpu()
    else:
        raise TypeError(
            f"expected a torch.Tensor or numpy.ndarray of images, got {type(images).__name__}"
        )
    if tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] not in (1, 3):
        raise ValueError(
            f"expected images shaped (N, H, W), (N, 1, H, W) or (N, 3, H, W); "
            f"got {tuple(images.shape)}"
        )
    if tensor.shape[0] == 0:
        raise ValueError("no images given")
    if tensor.dtype == torch.uint8:
        return tensor, 255.0
    if not tensor.is_floating_point():
        raise ValueError(
            f"unsupported image dtype {tensor.dtype}; "
            f"pass uint8 in [0, 255] or floating point in [0, 1]"
        )
    low, high = float(tensor.min()), float(tensor.max())
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("images contain NaN or infinite values")
    if low < -_RANGE_TOLERANCE or high > 1.0 + _RANGE_TOLERANCE:
        raise ValueError(
            f"image values span [{low:.4g}, {high:.4g}] but must lie in [0, 1]; "
            f"if your generator outputs [-1, 1], pass (x + 1) / 2"
        )
    return tensor, 1.0


def to_float(tensor: torch.Tensor, divisor: float) -> torch.Tensor:
    """Float32 in [0, 1] from a tensor validated by :func:`check_images`."""
    return tensor.float().div(divisor).clamp_(0.0, 1.0)


class ArraySource(ImageSource):
    """A tensor or array of images held in memory. Subjects are unknown."""

    def __init__(self, images) -> None:
        self._tensor, self._divisor = check_images(images)
        self.groups = None
        self.warnings = []

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def load(self, index: int) -> torch.Tensor:
        return to_float(self._tensor[index], self._divisor)
