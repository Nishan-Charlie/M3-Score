"""Turning a path, tensor or array into an indexable set of images."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

import numpy as np
import torch

from .arrays import ArraySource
from .base import ImageSource, ItemSource, SkippedItem
from .images import IMAGE_SUFFIXES, ImageFile
from .sampling import sample_indices

__all__ = [
    "ArraySource",
    "ImageSource",
    "ItemSource",
    "SkippedItem",
    "open_source",
    "sample_indices",
]

_SUPPORTED = ", ".join(sorted(IMAGE_SUFFIXES))


def open_source(
    source,
    *,
    include: str | None = None,
    group_pattern: str | None = None,
    group_pattern_optional: bool = False,
    slice_band: tuple[float, float] | None = None,
    window: tuple[float, float] | None = None,
) -> ImageSource:
    """Make an :class:`ImageSource` from a path, a tensor, an array, or a source.

    ``include`` is a glob on each file's path relative to the folder.
    ``group_pattern`` is a regex whose first capture group names the subject of each
    file. It must match every file, unless ``group_pattern_optional`` is set, in which
    case a pattern that matches no file at all is ignored. ``slice_band`` and
    ``window`` apply to volumes.
    """
    if isinstance(source, ImageSource):
        return source
    if isinstance(source, torch.Tensor | np.ndarray):
        return ArraySource(source)
    if isinstance(source, str | os.PathLike):
        return _open_path(
            Path(source), include, group_pattern, group_pattern_optional, slice_band, window
        )
    raise TypeError(
        f"cannot read images from {type(source).__name__}; "
        f"pass a path, a torch.Tensor or a numpy.ndarray"
    )


def _list_files(root: Path, include: str | None) -> list[tuple[Path, str]]:
    """Files under ``root`` with their labels (posix paths relative to it), sorted by label."""
    if root.is_dir():
        found = [(p, p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file()]
    elif root.is_file():
        found = [(root, root.name)]
    else:
        raise FileNotFoundError(f"no such file or directory: {root}")
    if include is not None:
        found = [(p, label) for p, label in found if fnmatch.fnmatch(label, include)]
    # sort on the label string so the order is the same on every platform
    return sorted(found, key=lambda entry: entry[1])


def _open_path(
    root: Path,
    include: str | None,
    group_pattern: str | None,
    group_pattern_optional: bool,
    slice_band: tuple[float, float] | None,
    window: tuple[float, float] | None,
) -> ImageSource:
    items: list = []
    labels: list[str] = []
    groups: list[str | None] = []
    notes: list[str] = []
    for path, label in _list_files(root, include):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(ImageFile(path, label))
            labels.append(label)
            groups.append(None)
    if not items:
        matching = f" matching {include!r}" if include else ""
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
