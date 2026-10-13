"""What every image source looks like to the metric."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


class SkippedItem(Exception):
    """Raised by an item's ``load`` when it cannot contribute an image.

    The message is the reason, starting with the item's label.
    """


class ImageSource:
    """An indexable set of images.

    ``load(i)`` returns one image as ``(C, H, W)`` float32 in [0, 1] with C in {1, 3}.
    ``groups`` holds one subject identifier per item, or ``None`` when subjects are
    unknown. ``warnings`` lists anything the source left out and why.
    """

    groups: list[str] | None = None
    warnings: list[str]

    def __len__(self) -> int:
        raise NotImplementedError

    def load(self, index: int) -> torch.Tensor:
        raise NotImplementedError


class ItemSource(ImageSource):
    """A source over objects that each know how to ``load()`` themselves."""

    def __init__(
        self, items: Iterable, groups: Sequence[str] | None, warnings: Iterable[str] = ()
    ) -> None:
        self._items = list(items)
        self.groups = list(groups) if groups is not None else None
        # kept as the same list when one is given: readers append to it while loading
        self.warnings = warnings if isinstance(warnings, list) else list(warnings)

    def __len__(self) -> int:
        return len(self._items)

    def load(self, index: int) -> torch.Tensor:
        return self._items[index].load()
