"""
Download LGG Brain MRI + segmentation masks and prepare them for
experiments/hallucination_detection.py.

Source: gymprathap/Brain-MRI-LGG-Segmentation on HuggingFace
        (re-hosted from TCIA / Kaggle mateuszbuda/lgg-mri-segmentation)

Expected zip structure (kaggle_3m/):
    kaggle_3m/
      TCGA-DU-6407/
        TCGA-DU-6407_59.tif         <- MRI slice (grayscale or RGB)
        TCGA-DU-6407_59_mask.tif    <- binary mask (0/255 tumour)
        ...

Output naming for hallucination_detection.py:
    <output_dir>/lgg_NNNNN_slice000.png
    <output_dir>/lgg_NNNNN_segmask_slice000.png

Only slices with at least --min_mask_ratio tumour pixels are saved.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm


def _process_zip(zip_path: str, output_dir: str, max_samples: int, min_mask_ratio: float) -> int:
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    idx   = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        # Identify MRI slices: .tif files that do NOT end in _mask.tif
        mri_names = sorted(
            n for n in all_names
            if n.lower().endswith(".tif") and not n.lower().endswith("_mask.tif")
        )
        print(f"Found {len(mri_names)} MRI slices in zip")

        for mri_name in tqdm(mri_names, desc="Extracting"):
            if saved >= max_samples:
                break

            # Derive mask path: insert _mask before .tif
            mask_name = mri_name[:-4] + "_mask.tif"
            if mask_name not in zf.namelist():
                continue

            # Load MRI
            with zf.open(mri_name) as f:
                img = Image.open(io.BytesIO(f.read())).convert("L")

            # Load mask
            with zf.open(mask_name) as f:
                mask_np = np.array(Image.open(io.BytesIO(f.read())))

            # Binarise: any non-zero -> 255
            mask_bin = (mask_np > 0).astype(np.uint8) * 255
            ratio = mask_bin.sum() / (255 * mask_bin.size)
            if ratio < min_mask_ratio:
                idx += 1
                continue

            tag      = f"lgg_{idx:05d}"
            img.save(os.path.join(output_dir, f"{tag}_slice000.png"))
            Image.fromarray(mask_bin, mode="L").save(
                os.path.join(output_dir, f"{tag}_segmask_slice000.png")
            )
            saved += 1
            idx   += 1

    return saved


def main(output_dir: str, max_samples: int, min_mask_ratio: float, zip_path: str | None) -> None:
    # ------------------------------------------------------------------
    # 1. Obtain the zip (download if not already present)
    # ------------------------------------------------------------------
    if zip_path and os.path.isfile(zip_path):
        print(f"Using existing zip: {zip_path}")
    else:
        default_zip = "data_mri/lgg_download/Brain-MRI-LGG-Segmentation.zip"
        # Check if already downloaded via hf_hub_download
        if os.path.isfile(default_zip):
            zip_path = default_zip
            print(f"Found zip at {zip_path}")
        else:
            print("Downloading from HuggingFace hub...")
            from huggingface_hub import hf_hub_download
            zip_path = hf_hub_download(
                repo_id="gymprathap/Brain-MRI-LGG-Segmentation",
                filename="Brain-MRI-LGG-Segmentation.zip",
                repo_type="dataset",
                local_dir="data_mri/lgg_download",
            )
            print(f"Downloaded to {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # 2. Extract and save image/mask pairs
    # ------------------------------------------------------------------
    n_saved = _process_zip(zip_path, output_dir, max_samples, min_mask_ratio)
    print(f"\nDone. Saved {n_saved} image+mask pairs to: {output_dir}")
    if n_saved == 0:
        print("WARNING: 0 pairs saved. Check zip structure or lower --min_mask_ratio.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir",      default="data_mri/lgg_with_masks")
    p.add_argument("--max_samples",     type=int,   default=500)
    p.add_argument("--min_mask_ratio",  type=float, default=0.001,
                   help="Min fraction of non-zero mask pixels to keep a slice")
    p.add_argument("--zip_path",        default=None,
                   help="Path to an already-downloaded zip (skips HF download)")
    args = p.parse_args()
    main(args.output_dir, args.max_samples, args.min_mask_ratio, args.zip_path)
