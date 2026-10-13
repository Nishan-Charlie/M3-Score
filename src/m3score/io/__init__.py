"""Turning a path, tensor or array into an indexable set of images."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

import numpy as np
import torch

from .. import protocol
from .arrays import ArraySource
from .base import ImageSource, ItemSource, SkippedItem
from .dicom import dicom_items, is_dicom
from .images import IMAGE_SUFFIXES, ImageFile
from .nifti import is_nifti, nifti_items
from .sampling import sample_indices
from .volumes import check_band, check_window

__all__ = [
    "ArraySource",
    "ImageSource",
    "ItemSource",
    "SkippedItem",
    "open_source",
    "sample_indices",
]

_SUPPORTED = ", ".join(sorted(IMAGE_SUFFIXES)) + ", .nii, .nii.gz and DICOM"


def open_source(
    source,
    *,
    include: str | None = None,
    include_optional: bool = False,
    group_pattern: str | None = None,
    group_pattern_optional: bool = False,
    slice_band: tuple[float, float] | None = None,
    window: tuple[float, float] | None = None,
) -> ImageSource:
    """Make an :class:`ImageSource` from a path, a tensor, an array, or a source.

    ``include`` is a glob on each file's path relative to the folder.
    ``group_pattern`` is a regex whose first capture group names the subject of each
    file and must match every file. Both describe one particular folder layout, so
    each has an ``_optional`` switch for a second folder laid out differently: with
    it set, a pattern that matches no file at all is left unapplied, and for
    ``include`` the source's warnings say so. ``slice_band`` and ``window`` apply to
    volumes.
    """
    if isinstance(source, ImageSource):
        return source
    if isinstance(source, torch.Tensor | np.ndarray):
        return ArraySource(source)
    if isinstance(source, str | os.PathLike):
        return _open_path(
            Path(source),
            include,
            include_optional,
            group_pattern,
            group_pattern_optional,
            slice_band,
            window,
        )
    raise TypeError(
        f"cannot read images from {type(source).__name__}; "
        f"pass a path, a torch.Tensor or a numpy.ndarray"
    )


def _list_files(
    root: Path, include: str | None, include_optional: bool = False
) -> tuple[list[tuple[Path, str]], bool]:
    """Files under ``root`` with their labels (posix paths relative to it), sorted by label.

    Also returns whether ``include`` was applied: an optional pattern that matches
    nothing leaves the listing unfiltered.
    """
    if root.is_dir():
        found = [(p, p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file()]
    elif root.is_file():
        found = [(root, root.name)]
    else:
        raise FileNotFoundError(f"no such file or directory: {root}")
    applied = include is not None
    if include is not None:
        kept = [(p, label) for p, label in found if fnmatch.fnmatch(label, include)]
        if kept or not include_optional:
            found = kept
        else:
            applied = False
    # sort on the label string so the order is the same on every platform
    return sorted(found, key=lambda entry: entry[1]), applied


def _open_path(
    root: Path,
    include: str | None,
    include_optional: bool,
    group_pattern: str | None,
    group_pattern_optional: bool,
    slice_band: tuple[float, float] | None,
    window: tuple[float, float] | None,
) -> ImageSource:
    band = check_band(slice_band) if slice_band is not None else protocol.SLICE_BAND
    window = check_window(window) if window is not None else None

    items: list = []
    labels: list[str] = []
    groups: list[str | None] = []
    notes: list[str] = []
    dicom_files: list[tuple[Path, str]] = []
    files, include_applied = _list_files(root, include, include_optional)
    if include is not None and not include_applied:
        notes.append(
            f"include pattern {include!r} matched no file under {root} and was not applied to it"
        )
    for path, label in files:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(ImageFile(path, label))
            labels.append(label)
            groups.append(None)
        elif is_nifti(path):
            for item in nifti_items(path, label, band, window):
                items.append(item)
                labels.append(label)
                groups.append(label)  # one volume file is one subject
        elif is_dicom(path):
            dicom_files.append((path, label))
    if dicom_files:
        # DICOM files are handled together because slices of one series form a volume
        found, found_labels, found_groups, found_notes = dicom_items(dicom_files, band, window)
        items += found
        labels += found_labels
        groups += found_groups
        notes += found_notes
    if not items:
        matching = f" matching {include!r}" if include_applied else ""
        raise FileNotFoundError(
            f"no readable images under {root}{matching}; supported: {_SUPPORTED}"
        )

    if group_pattern is not None:
        matched = _groups_from_pattern(labels, group_pattern, group_pattern_optional)
        if matched is not None:
            groups = list(matched)
    known = None if any(g is None for g in groups) else [str(g) for g in groups]
    return ItemSource(items, known, notes)


def _groups_from_pattern(labels: list[str], pattern: str, optional: bool) -> list[str] | None:
    """Subject of each label from the regex's first capture group."""
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"group pattern {pattern!r} is not a valid regular expression: {exc}"
        ) from exc
    if regex.groups < 1:
        raise ValueError(
            f"group pattern {pattern!r} needs one capture group naming the subject, "
            f"for example '(BraTS2021_\\d+)'"
        )
    matches = [regex.search(label) for label in labels]
    if optional and not any(matches):
        return None
    for label, match in zip(labels, matches, strict=False):
        if match is None:
            raise ValueError(f"group pattern {pattern!r} does not match {label!r}")
    return [match.group(1) for match in matches]
