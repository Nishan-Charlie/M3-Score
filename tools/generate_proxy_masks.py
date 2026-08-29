"""
Generate intensity-threshold proxy tumor masks from BraTS FLAIR PNG slices.

BraTS FLAIR images show tumor as hyperintense (bright) regions relative to
normal white matter. This script applies Otsu thresholding + size filtering
to create binary masks that approximate tumor location.

IMPORTANT: These are proxy masks, NOT ground-truth BraTS segmentations.
They are labeled as such in the experiment output.

Output: for each MRI slice that passes the mask-ratio filter, saves:
    <output_dir>/<original_stem>_segmask_slice000.png   (255=tumour, 0=background)

alongside the original image copied as:
    <output_dir>/<original_stem>_slice000.png

Usage:
    python tools/generate_proxy_masks.py \\
        --real_dir  data_mri/brats_axial_multislice \\
        --output_dir data_mri/brats_with_proxymasks \\
        --max_images 500 \\
        --min_mask_ratio 0.01
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def _otsu_threshold(arr: np.ndarray) -> float:
    """Compute Otsu threshold via histogram method (no skimage required)."""
    hist, bins = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)

    sum_bg, w_bg, max_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mu_bg = sum_bg / w_bg
        mu_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mu_bg - mu_fg) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return float(threshold)


def _make_proxy_mask(img_arr: np.ndarray, min_region_px: int = 50) -> np.ndarray:
    """Return a binary mask (0/255) approximating hyperintense tumor regions.

    Steps:
    1. Otsu threshold on the brain region (non-zero pixels).
    2. Keep pixels above threshold AND brighter than the 85th percentile
       of the whole image (tumors are in the top brightness range).
    3. Remove small connected components (< min_region_px pixels).
    """
    brain_mask = img_arr > 5          # exclude near-black background
    if brain_mask.sum() < 100:
        return np.zeros_like(img_arr, dtype=np.uint8)

    otsu  = _otsu_threshold(img_arr[brain_mask])
    p85   = float(np.percentile(img_arr[brain_mask], 85))
    high_thresh = max(otsu, p85)

    binary = (img_arr > high_thresh).astype(np.uint8)

    # Remove small components
    labeled, n = ndimage.label(binary)
    for comp in range(1, n + 1):
        if (labeled == comp).sum() < min_region_px:
            binary[labeled == comp] = 0

    return (binary * 255).astype(np.uint8)


def run(real_dir: str, output_dir: str, max_images: int, min_mask_ratio: float) -> None:
    os.makedirs(output_dir, exist_ok=True)

    all_imgs = sorted(glob.glob(os.path.join(real_dir, "**", "*.png"), recursive=True))
    # Exclude any existing mask files
    all_imgs = [p for p in all_imgs if "_segmask_" not in os.path.basename(p)]
    print(f"Found {len(all_imgs)} images in {real_dir}")

    saved = 0
    skipped_no_mask = 0

    for img_path in tqdm(all_imgs, desc="Generating proxy masks"):
        if saved >= max_images:
            break

        img = Image.open(img_path).convert("L")
        arr = np.array(img)

        mask_arr = _make_proxy_mask(arr)
        ratio = mask_arr.sum() / (255 * mask_arr.size)

        if ratio < min_mask_ratio:
            skipped_no_mask += 1
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        # Save image copy and mask with naming convention for hallucination_detection.py
        img_out  = os.path.join(output_dir, f"{stem}_slice000.png")
        mask_out = os.path.join(output_dir, f"{stem}_segmask_slice000.png")

        img.save(img_out)
        Image.fromarray(mask_arr, mode="L").save(mask_out)
        saved += 1

    print(f"\nSaved {saved} image+mask pairs (skipped {skipped_no_mask} with insufficient mask coverage)")
    print(f"Output: {output_dir}")
    print("NOTE: These are INTENSITY-THRESHOLD PROXY masks, not ground-truth BraTS segmentations.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real_dir",       default="data_mri/brats_axial_multislice")
    p.add_argument("--output_dir",     default="data_mri/brats_with_proxymasks")
    p.add_argument("--max_images",     type=int,   default=500)
    p.add_argument("--min_mask_ratio", type=float, default=0.01,
                   help="Min fraction of pixels in mask (filters near-empty masks)")
    args = p.parse_args()
    run(args.real_dir, args.output_dir, args.max_images, args.min_mask_ratio)
