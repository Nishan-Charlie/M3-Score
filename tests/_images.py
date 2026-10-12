"""Helpers that write small image files for tests."""

import numpy as np
from PIL import Image


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def write_pngs(directory, count, *, size=(40, 40), seed=0, name="img_{:04d}.png"):
    """Write ``count`` random 8-bit greyscale PNGs and return their paths."""
    rng = np.random.default_rng(seed)
    return [
        write_png(directory / name.format(i), rng.integers(0, 256, size=size, dtype=np.uint8))
        for i in range(count)
    ]
