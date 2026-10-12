"""2D image files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import SkippedItem

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I"})


class ImageFile:
    """One image file, decoded on demand."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label

    def load(self) -> torch.Tensor:
        try:
            with Image.open(self.path) as image:
                if image.mode in _SIXTEEN_BIT_MODES:
                    array = np.asarray(image, dtype=np.float32)
                    # mode "I" is 32-bit; a value outside the 16-bit range has no defined
                    # scale, and clamping it would hide that from the caller
                    if array.min() < 0.0 or array.max() > 65535.0:
                        raise SkippedItem(
                            f"{self.label}: integer image with values outside "
                            f"the 16-bit range [0, 65535]"
                        )
                    return torch.from_numpy(array / 65535.0)[None]
                if image.mode == "F":
                    array = np.asarray(image, dtype=np.float32)
                    if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
                        raise SkippedItem(
                            f"{self.label}: floating-point image with values outside [0, 1]"
                        )
                    return torch.from_numpy(array.copy())[None]
                # 8-bit modes, as the research pipeline reads them
                array = np.array(image.convert("RGB"), dtype=np.uint8)
        except Image.DecompressionBombError as exc:
            # Pillow's own pixel limit; never raised or lowered here
            raise SkippedItem(f"{self.label}: image too large to decode safely ({exc})") from exc
        except (OSError, ValueError) as exc:  # UnidentifiedImageError is an OSError
            raise SkippedItem(f"{self.label}: unreadable image ({exc})") from exc
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
