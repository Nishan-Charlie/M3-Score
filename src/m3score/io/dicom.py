"""DICOM files and series. Needs pydicom: ``pip install "m3-score[dicom]"``."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

from .._errors import M3ScoreError
from .base import SkippedItem
from .volumes import band_indices, check_size, scale_intensity

DICOM_SUFFIXES = frozenset({".dcm", ".dicom", ".ima"})
_DECODER_HINT = (
    "compressed transfer syntaxes need a decoder plugin: "
    "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
)


def is_dicom(path: Path) -> bool:
    """By suffix, otherwise by the ``DICM`` marker (files are often named by UID)."""
    if path.suffix.lower() in DICOM_SUFFIXES:
        return True
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _require_pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError('DICOM input needs pydicom: pip install "m3-score[dicom]"') from exc
    return pydicom


def _number(value, default: float | None) -> float | None:
    """A DICOM numeric element as a float; the first value when it is multi-valued."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value[0])
        except (TypeError, ValueError, IndexError):
            return default


def _stored_window(ds) -> tuple[float, float] | None:
    center = _number(getattr(ds, "WindowCenter", None), None)
    width = _number(getattr(ds, "WindowWidth", None), None)
    if center is None or width is None or width <= 0:
        return None
    return center - width / 2.0, center + width / 2.0


def _raw_pixels(ds, label: str) -> np.ndarray:
    try:
        return ds.pixel_array
    except Exception as exc:
        raise M3ScoreError(
            f"{label}: could not decode DICOM pixel data ({exc}); {_DECODER_HINT}"
        ) from exc


def _pixels(ds, label: str) -> np.ndarray:
    """Stored values after the modality rescale, as float32."""
    slope = _number(getattr(ds, "RescaleSlope", None), 1.0)
    intercept = _number(getattr(ds, "RescaleIntercept", None), 0.0)
    return _raw_pixels(ds, label).astype(np.float32) * slope + intercept


def _to_unit_range(array: np.ndarray, ds, window: tuple[float, float] | None):
    """Explicit window, else the stored window, else the percentile rule; MONOCHROME1 inverted."""
    scaled = scale_intensity(array, window if window is not None else _stored_window(ds))
    if scaled is None:
        return None
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        scaled = 1.0 - scaled
    return scaled


def _subject(ds, label: str) -> str:
    for keyword in ("PatientID", "StudyInstanceUID"):
        value = str(getattr(ds, keyword, "") or "").strip()
        if value:
            return value
    return label.rsplit("/", 1)[0] if "/" in label else "."


def _placement(ds) -> tuple[float, tuple[float, ...]] | None:
    """Position along the slice normal and the rounded orientation, when both are stored."""
    position = getattr(ds, "ImagePositionPatient", None)
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if position is None or orientation is None or len(position) != 3 or len(orientation) != 6:
        return None
    axes = np.asarray(orientation, dtype=np.float64)
    normal = np.cross(axes[:3], axes[3:])
    along = float(np.dot(np.asarray(position, dtype=np.float64), normal))
    return along, tuple(float(v) for v in np.round(axes, 2))


class DicomImage:
    """One single-frame instance scored on its own, such as a chest radiograph."""

    def __init__(self, path: Path, label: str, window: tuple[float, float] | None) -> None:
        self.path = path
        self.label = label
        self.window = window

    def load(self) -> torch.Tensor:
        pydicom = _require_pydicom()
        ds = pydicom.dcmread(str(self.path))
        if int(_number(getattr(ds, "SamplesPerPixel", None), 1.0)) == 3:
            colour = np.ascontiguousarray(_raw_pixels(ds, self.label))
            return torch.from_numpy(colour).permute(2, 0, 1).float() / 255.0
        scaled = _to_unit_range(_pixels(ds, self.label), ds, self.window)
        if scaled is None:
            raise SkippedItem(f"{self.label}: image is empty or constant")
        return torch.from_numpy(scaled)[None]


@functools.lru_cache(maxsize=2)
def _load_stack(
    paths: tuple[str, ...], label: str, window: tuple[float, float] | None
) -> np.ndarray | None:
    """A whole volume as (slices, H, W) in [0, 1], scaled as one unit."""
    pydicom = _require_pydicom()
    datasets = [pydicom.dcmread(path) for path in paths]
    frames = [_pixels(ds, label) for ds in datasets]
    # a multi-frame instance already decodes to (frames, H, W)
    stack = frames[0] if len(frames) == 1 else np.stack(frames)
    return _to_unit_range(stack, datasets[0], window)


class DicomSlice:
    """One slice of a series, or one frame of a multi-frame instance."""

    def __init__(
        self, paths: tuple[str, ...], label: str, index: int, window: tuple[float, float] | None
    ) -> None:
        self.paths = paths
        self.label = label
        self.index = index
        self.window = window

    def load(self) -> torch.Tensor:
        volume = _load_stack(self.paths, self.label, self.window)
        if volume is None:
            raise SkippedItem(f"{self.label}: volume is empty or constant")
        return torch.from_numpy(np.ascontiguousarray(volume[self.index]))[None]


def dicom_items(
    files: list[tuple[Path, str]], band: tuple[float, float], window: tuple[float, float] | None
) -> tuple[list, list[str], list[str], list[str]]:
    """Items for a set of DICOM files, read from their headers only.

    Returns ``(items, labels, groups, notes)``; the first three are aligned. Single
    frames that share a series, an image size and an orientation, and that carry
    positions, form a volume: sorted along the slice normal, with ``band`` applied.
    """
    pydicom = _require_pydicom()
    items: list = []
    labels: list[str] = []
    groups: list[str] = []
    notes: list[str] = []
    series: dict[tuple, list[tuple[float, float, str, str, str]]] = {}

    def add(item, label: str, subject: str) -> None:
        items.append(item)
        labels.append(label)
        groups.append(subject)

    def too_large(label: str, n_voxels: int) -> bool:
        """Headers can claim any size, so the claim is checked before decoding."""
        try:
            check_size(label, n_voxels)
        except ValueError as exc:
            notes.append(f"skipped {exc}")
            return True
        return False

    for path, label in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            notes.append(f"skipped {label}: not a readable DICOM file ({exc})")
            continue
        if "Rows" not in ds or "Columns" not in ds:
            notes.append(f"skipped {label}: DICOM object without an image")
            continue
        subject = _subject(ds, label)
        frames = int(_number(getattr(ds, "NumberOfFrames", None), 1.0))
        if too_large(label, int(ds.Rows) * int(ds.Columns) * max(frames, 1)):
            continue
        if int(_number(getattr(ds, "SamplesPerPixel", None), 1.0)) == 3:
            photometric = str(getattr(ds, "PhotometricInterpretation", ""))
            eight_bit = int(_number(getattr(ds, "BitsAllocated", None), 8.0)) == 8
            if photometric != "RGB" or not eight_bit or frames > 1:
                notes.append(f"skipped {label}: unsupported colour encoding {photometric}")
                continue
            add(DicomImage(path, label, window), label, subject)
        elif frames > 1:
            for index in band_indices(frames, band):
                add(DicomSlice((str(path),), label, index, window), label, subject)
        else:
            placement = _placement(ds)
            uid = getattr(ds, "SeriesInstanceUID", None)
            if placement is None or uid is None:
                add(DicomImage(path, label, window), label, subject)
                continue
            along, orientation = placement
            key = (str(uid), int(ds.Rows), int(ds.Columns), orientation)
            instance = _number(getattr(ds, "InstanceNumber", None), 0.0)
            series.setdefault(key, []).append((along, instance, str(path), label, subject))

    for key, members in series.items():
        if len(members) < 2:
            _, _, path, label, subject = members[0]
            add(DicomImage(Path(path), label, window), label, subject)
            continue
        members.sort(key=lambda member: (member[0], member[1]))
        paths = tuple(member[2] for member in members)
        label, subject = members[0][3], members[0][4]
        if too_large(label, len(members) * key[1] * key[2]):
            continue
        for index in band_indices(len(members), band):
            add(DicomSlice(paths, label, index, window), label, subject)
    return items, labels, groups, notes
