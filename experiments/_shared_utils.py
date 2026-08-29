"""
experiments/_shared_utils.py
Shared image-loading, transform, and metric helpers reused across experiment scripts.
"""
from __future__ import annotations

import glob
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

def get_m3_transform():
    """Returns the standard ImageNet-normalised 224×224 transform used by DINOv2 and RadioDino."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_fid_transform(size: int = 299):
    """Returns a uint8-compatible transform for Inception/FID evaluation."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_pils(
    directory: str,
    n: Optional[int] = None,
    exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff'),
) -> List[Image.Image]:
    """Load up to *n* PIL images (RGB) from *directory*, sorted by filename."""
    files = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(exts)
    )
    if n is not None:
        files = files[:n]
    return [Image.open(os.path.join(directory, f)).convert("RGB") for f in files]


def load_pils_recursive(
    directory: str,
    n: Optional[int] = None,
    exts: Tuple[str, ...] = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'),
) -> List[Image.Image]:
    """Recursively load up to *n* PIL images (RGB) from *directory*."""
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(paths)
    if n is not None:
        paths = paths[:n]
    return [Image.open(p).convert("RGB") for p in paths]


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def pil_to_fid_tensor(pil_img: Image.Image, size: int = 299) -> torch.Tensor:
    """Convert a single PIL image to a uint8 tensor [1, 3, size, size] for FID update."""
    return (
        torch.from_numpy(np.array(pil_img.resize((size, size))))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def pils_to_m3_batch(pil_images: List[Image.Image], device: str = "cpu") -> torch.Tensor:
    """Stack PIL images into an ImageNet-normalised [N, 3, 224, 224] tensor."""
    t = get_m3_transform()
    return torch.stack([t(img) for img in pil_images]).to(device)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_psnr(img1, img2) -> float:
    """PSNR between two PIL images or uint8 numpy arrays [0, 255]."""
    if isinstance(img1, Image.Image):
        img1 = np.array(img1).astype(np.float64)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2).astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
