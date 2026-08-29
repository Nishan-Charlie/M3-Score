"""
tools/download_medmnist.py
==========================
Downloads MedMNIST v2 datasets and saves them as individual PNG files
for use with the M3-Score evaluation pipeline.

Datasets downloaded:
  retinamnist    -- retinal fundus (1,480 images)  → real images for retinal eval
  pneumoniamnist -- chest X-ray / pneumonia (5,332) → real images for CXR eval
  chestmnist     -- chest X-ray 14 conditions (100K) → large CXR real set
  organamnist    -- abdominal CT axial slices (52K)  → CT modality real set

Usage:
    python tools/download_medmnist.py \
        --datasets retinamnist pneumoniamnist organamnist \
        --size 64 \
        --output_dir data_mri/medmnist
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

AVAILABLE = ["retinamnist", "pneumoniamnist", "chestmnist", "organamnist",
             "octmnist", "bloodmnist", "dermamnist"]

DATASET_INFO = {
    "retinamnist":    ("RetinaMNIST",    "Retinal fundus (ordinal regression)",  "retinal"),
    "pneumoniamnist": ("PneumoniaMNIST", "Chest X-ray pneumonia (binary)",        "chest_xray"),
    "chestmnist":     ("ChestMNIST",     "Chest X-ray 14 conditions (multi-label)", "chest_xray"),
    "organamnist":    ("OrganAMNIST",    "Abdominal CT axial slices (multi-class)", "ct_abdomen"),
    "octmnist":       ("OCTMNIST",       "Retinal OCT (multi-class)",             "retinal_oct"),
    "bloodmnist":     ("BloodMNIST",     "Blood cell microscopy (multi-class)",   "blood_cell"),
    "dermamnist":     ("DermaMNIST",     "Dermoscopy skin lesions (multi-class)", "dermoscopy"),
}


def save_split(dataset_cls, split: str, output_dir: Path, size: int, dataset_name: str):
    import medmnist
    cls = getattr(medmnist, dataset_cls)
    ds = cls(split=split, size=size, download=True, root=str(output_dir / "raw"))

    split_dir = output_dir / dataset_name / split
    split_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, (img, _) in enumerate(tqdm(ds, desc=f"  {split}", leave=False)):
        # img is a PIL Image from MedMNIST (either L or RGB depending on dataset)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        # Ensure RGB for consistent pipeline input (RadioDino expects 3-ch or will be grayscaled)
        if img.mode not in ("RGB", "L"):
            img = img.convert("L")
        fname = split_dir / f"{dataset_name}_{split}_{idx:05d}.png"
        img.save(fname)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["retinamnist", "pneumoniamnist"],
                        choices=AVAILABLE, help="Which MedMNIST datasets to download")
    parser.add_argument("--size", type=int, default=64,
                        choices=[28, 64, 128, 224], help="Image size to download")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="data_mri/medmnist",
                        help="Root directory for downloaded datasets")
    parser.add_argument("--also_merge", action="store_true",
                        help="Merge all splits into a single 'all/' directory (used by eval pipeline)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Image size: {args.size}×{args.size}")
    print()

    for ds_name in args.datasets:
        cls_name, description, modality = DATASET_INFO[ds_name]
        print(f"[{ds_name}] {description}")

        total = 0
        for split in args.splits:
            try:
                n = save_split(cls_name, split, output_dir, args.size, ds_name)
                print(f"  {split}: {n} images saved to {output_dir / ds_name / split}/")
                total += n
            except Exception as e:
                print(f"  {split}: FAILED — {e}")

        # Merge splits into one flat 'all/' directory for eval pipeline
        if args.also_merge:
            merged_dir = output_dir / ds_name / "all"
            merged_dir.mkdir(parents=True, exist_ok=True)
            count_merged = 0
            for split in args.splits:
                split_dir = output_dir / ds_name / split
                if not split_dir.exists():
                    continue
                for f in split_dir.glob("*.png"):
                    dest = merged_dir / f.name
                    if not dest.exists():
                        import shutil
                        shutil.copy2(f, dest)
                    count_merged += 1
            print(f"  merged:  {count_merged} images → {merged_dir}/")

        print(f"  TOTAL: {total} images\n")

    print("Done.")


if __name__ == "__main__":
    main()
