"""2D image files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import SkippedItem

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I"})
_LOW_RANGE_NOTE = "16-bit image files whose values stay within 12 bits"


class ImageFile:
    """One image file, decoded on demand."""

    def __init__(self, path: Path, label: str, notes: list[str] | None = None) -> None:
        self.path = path
        self.label = label
        self._notes = notes  # the owning source's warnings, shared by all its files

    def _flag_low_range(self, maximum: float) -> None:
        """12-bit data in a 16-bit file is common from DICOM exports and loads almost black."""
        if self._notes is None or not 0.0 < maximum <= 4095.0:
            return
        if any(note.startswith(_LOW_RANGE_NOTE) for note in self._notes):
            return
        self._notes.append(
            f"{_LOW_RANGE_NOTE} are divided by 65535 and will look almost black "
            f"(first seen: {self.label}, maximum {int(maximum)}); rescale them to the full "
            f"16-bit range or convert them to 8-bit"
        )

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
                    self._flag_low_range(float(array.max()))
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
