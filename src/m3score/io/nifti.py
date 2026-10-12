"""NIfTI volumes. Needs nibabel: ``pip install "m3-score[nifti]"``."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

from .base import SkippedItem
from .volumes import band_indices, check_size, scale_intensity

NIFTI_SUFFIXES = (".nii", ".nii.gz")
_SUPERIOR_AXIS = 2  # after reorientation to RAS+


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def _require_nibabel():
    try:
        import nibabel
    except ImportError as exc:
        raise ImportError('NIfTI input needs nibabel: pip install "m3-score[nifti]"') from exc
    return nibabel


@functools.lru_cache(maxsize=2)
def _load_volume(path: str, window: tuple[float, float] | None) -> np.ndarray | None:
    """The volume in RAS+ axes scaled to [0, 1], or ``None`` when empty or constant."""
    nibabel = _require_nibabel()
    image = nibabel.as_closest_canonical(nibabel.load(path))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim == 4 and volume.shape[3] == 1:
        volume = volume[..., 0]
    return scale_intensity(volume, window)


class NiftiSlice:
    """One axial slice of a volume, decoded on demand."""

    def __init__(
        self, path: Path, label: str, index: int, window: tuple[float, float] | None
    ) -> None:
        self.path = path
        self.label = label
        self.index = index
        self.window = window

    def load(self) -> torch.Tensor:
        volume = _load_volume(str(self.path), self.window)
        if volume is None:
            raise SkippedItem(f"{self.label}: volume is empty or constant")
        plane = volume[:, :, self.index]
        # A RAS+ plane is indexed [left->right, posterior->anterior]. Radiological
        # display puts anterior at the top and the patient's right on the image left.
        return torch.from_numpy(np.ascontiguousarray(plane[::-1, ::-1].T))[None]


def nifti_items(
    path: Path, label: str, band: tuple[float, float], window: tuple[float, float] | None
) -> list[NiftiSlice]:
    """The slices of one volume that fall inside ``band``. Reads only the header."""
    nibabel = _require_nibabel()
    image = nibabel.load(str(path))
    shape = tuple(image.shape)
    if len(shape) == 4 and shape[3] == 1:
        shape = shape[:3]
    if len(shape) != 3:
        raise ValueError(
            f"{label}: expected a 3D volume but found shape {tuple(image.shape)}; "
            f"split 4D files into 3D volumes first"
        )
    check_size(label, int(np.prod(shape)))
    # which stored axis becomes inferior->superior once the volume is reoriented
    orientation = nibabel.orientations.io_orientation(image.affine)
    stored_axis = np.flatnonzero(orientation[:, 0] == _SUPERIOR_AXIS)
    depth = int(shape[int(stored_axis[0])]) if stored_axis.size else int(shape[2])
    return [NiftiSlice(path, label, index, window) for index in band_indices(depth, band)]
