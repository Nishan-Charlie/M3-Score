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


def clear_cache() -> None:
    _load_stack.cache_clear()


def _is_compressed(ds) -> bool:
    syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    try:
        return bool(syntax.is_compressed)
    except Exception:  # no file meta, or a UID that is not a transfer syntax
        return False


def _raw_pixels(ds, label: str) -> np.ndarray:
    """Decoded pixel data.

    A failure on compressed data almost always means a missing decoder and affects
    every file, so it stops the run with the fix. A failure on uncompressed data is a
    damaged file, which is left out.
    """
    try:
        return ds.pixel_array
    except Exception as exc:
        if _is_compressed(ds):
            raise M3ScoreError(
                f"{label}: could not decode DICOM pixel data ({exc}); {_DECODER_HINT}"
            ) from exc
        raise SkippedItem(f"{label}: unreadable pixel data ({type(exc).__name__}: {exc})") from exc


def _functional_rescale(group) -> tuple[float, float] | None:
    sequence = getattr(group, "PixelValueTransformationSequence", None)
    if not sequence:
        return None
    item = sequence[0]
    return (
        _number(getattr(item, "RescaleSlope", None), 1.0),
        _number(getattr(item, "RescaleIntercept", None), 0.0),
    )


def _rescale(ds, n_frames: int):
    """Slope and intercept: scalars, or per-frame arrays shaped ``(frames, 1, 1)``.

    Classic objects carry them at the top level. Enhanced multi-frame objects keep
    them in the shared functional group, or in one functional group per frame.
    """
    slope = _number(getattr(ds, "RescaleSlope", None), None)
    intercept = _number(getattr(ds, "RescaleIntercept", None), None)
    if slope is not None or intercept is not None:
        return (1.0 if slope is None else slope), (0.0 if intercept is None else intercept)
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        found = _functional_rescale(shared[0])
        if found is not None:
            return found
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame and len(per_frame) == n_frames:
        found = [_functional_rescale(group) for group in per_frame]
        if all(item is not None for item in found):
            slopes = np.array([item[0] for item in found], dtype=np.float32)
            intercepts = np.array([item[1] for item in found], dtype=np.float32)
            return slopes.reshape(-1, 1, 1), intercepts.reshape(-1, 1, 1)
    return 1.0, 0.0


def _pixels(ds, label: str) -> np.ndarray:
    """Stored values after the modality rescale, as float32."""
    array = _raw_pixels(ds, label).astype(np.float32)
    slope, intercept = _rescale(ds, array.shape[0] if array.ndim == 3 else 1)
    return array * slope + intercept


def _is_inverted(ds) -> bool:
    return str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1"


def _common_window(datasets) -> tuple[float, float] | None:
    """The stored window of a volume: the median over its slices.

    Some scanners write a window per slice; the median keeps one unusual slice, such
    as the first of the stack, from setting the contrast of the whole volume.
    """
    windows = [w for w in (_stored_window(ds) for ds in datasets) if w is not None]
    if not windows:
        return None
    return (
        float(np.median([low for low, _ in windows])),
        float(np.median([high for _, high in windows])),
    )


def _to_unit_range(array: np.ndarray, window: tuple[float, float] | None, inverted: bool):
    """Clip to ``window`` (or apply the percentile rule) and invert MONOCHROME1."""
    scaled = scale_intensity(array, window)
    if scaled is None:
        return None
    return 1.0 - scaled if inverted else scaled


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
    # Order slices along the positive patient axis whichever way the normal points, so
    # the slice band selects the same anatomy for either stacking direction.
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        along = -along
    return along, tuple(float(v) for v in np.round(axes, 2))


class DicomImage:
    """One single-frame instance scored on its own, such as a chest radiograph."""

    def __init__(self, path: Path, label: str, window: tuple[float, float] | None) -> None:
        self.path = path
        self.label = label
        self.window = window

    def load(self) -> torch.Tensor:
        pydicom = _require_pydicom()
        try:
            ds = pydicom.dcmread(str(self.path))
        except Exception as exc:
            raise SkippedItem(
                f"{self.label}: unreadable DICOM file ({type(exc).__name__}: {exc})"
            ) from exc
        if int(_number(getattr(ds, "SamplesPerPixel", None), 1.0)) == 3:
            colour = np.ascontiguousarray(_raw_pixels(ds, self.label))
            return torch.from_numpy(colour).permute(2, 0, 1).float() / 255.0
        window = self.window if self.window is not None else _stored_window(ds)
        scaled = _to_unit_range(_pixels(ds, self.label), window, _is_inverted(ds))
        if scaled is None:
            raise SkippedItem(f"{self.label}: image is empty or constant")
        return torch.from_numpy(scaled)[None]


@functools.lru_cache(maxsize=2)
def _load_stack(
    paths: tuple[str, ...], label: str, window: tuple[float, float] | None
) -> tuple[np.ndarray | None, str]:
    """A whole volume as (slices, H, W) in [0, 1], scaled as one unit.

    Returns ``None`` and the reason when the volume has to be left out. The reason is
    returned, not raised, so that it is cached with the result.
    """
    pydicom = _require_pydicom()
    try:
        datasets = [pydicom.dcmread(path) for path in paths]
    except Exception as exc:
        return None, f"{label}: unreadable DICOM file ({type(exc).__name__}: {exc})"
    try:
        frames = [_pixels(ds, label) for ds in datasets]
        # a multi-frame instance already decodes to (frames, H, W)
        stack = frames[0] if len(frames) == 1 else np.stack(frames)
    except SkippedItem as skipped:
        return None, str(skipped)
    except ValueError as exc:
        return None, f"{label}: slices cannot be stacked into a volume ({exc})"
    stored = window if window is not None else _common_window(datasets)
    scaled = _to_unit_range(stack, stored, _is_inverted(datasets[0]))
    if scaled is None:
        return None, f"{label}: volume is empty or constant"
    return scaled, ""


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
        volume, reason = _load_stack(self.paths, self.label, self.window)
        if volume is None:
            raise SkippedItem(reason)
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
