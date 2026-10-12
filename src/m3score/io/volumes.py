"""What NIfTI and DICOM volumes share: which slices are kept, how intensities are scaled."""

from __future__ import annotations

import math

import numpy as np

from .. import protocol

# About 4 GB as float32, well beyond a clinical volume. A header can claim any size,
# so the claim is checked before any pixel data is decoded.
MAX_VOXELS = 2**30


def check_size(label: str, n_voxels: int) -> None:
    if n_voxels > MAX_VOXELS:
        raise ValueError(
            f"{label}: a volume of {n_voxels:,} voxels is too large to decode safely "
            f"(limit {MAX_VOXELS:,})"
        )


def check_band(band) -> tuple[float, float]:
    low, high = float(band[0]), float(band[1])
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"slice band must satisfy 0 <= LO < HI <= 1, got ({low:g}, {high:g})")
    return low, high


def check_window(window) -> tuple[float, float]:
    low, high = float(window[0]), float(window[1])
    if not high > low:
        raise ValueError(f"intensity window must satisfy LO < HI, got ({low:g}, {high:g})")
    return low, high


def band_indices(depth: int, band: tuple[float, float]) -> range:
    """Slice indices kept from a volume of ``depth`` slices: a fractional band, never empty.

    The default band (55/155, 86/155) keeps slices 55 to 85 of a 155-slice BraTS volume.
    """
    low, high = check_band(band)
    start = min(max(int(math.floor(low * depth + 0.5)), 0), depth - 1)
    stop = min(max(int(math.floor(high * depth + 0.5)), start + 1), depth)
    return range(start, stop)


def scale_intensity(volume: np.ndarray, window: tuple[float, float] | None) -> np.ndarray | None:
    """Map intensities to [0, 1] as float32.

    With a ``window`` the values are clipped to it. Without one, the range is the
    0.5 to 99.5 percentile of the non-zero voxels, so a zero background stays black.
    Returns ``None`` when the volume is empty or constant.
    """
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if window is not None:
        low, high = window
    else:
        foreground = volume[volume != 0]
        if foreground.size == 0:
            return None
        low, high = np.percentile(foreground, protocol.INTENSITY_PERCENTILES)
    if not high > low:
        return None
    return np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32)
