"""
experiments/supplement_fid_kid.py
====================================
Fills missing FID and KID values in an existing multi_metric_comparison
results.json file. Runs when torchmetrics was not available during the
initial comparison run.

Usage:
    python experiments/supplement_fid_kid.py \
        --results_json results/multi_metric_comparison/results.json

Reads each comparison's (name, n_real, n_gen) from the JSON,
loads the corresponding images from the default directories,
computes FID + KID, and writes an updated JSON in-place.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG_SIZE    = 224
N_SAMPLE    = 500
RANDOM_SEED = 42

# Directory map keyed by comparison name
_DIR_MAP = {
    "BraTS_DDPM":         ("data_mri/brats_axial_multislice",   "output/generated_500_best"),
    "BraTS_WDM3D":        ("data_mri/brats_axial_multislice",   "output/generated_wdm3d/brats"),
    "BraTS_vs_LIDC_OOD":  ("data_mri/brats_axial_multislice",   "output/generated_wdm3d/lidc"),
    "Retinal_DDPM":       ("data_mri/medmnist/retinamnist/all", "output/generated_retinal"),
    "Retinal_GaussNoise": ("data_mri/medmnist/retinamnist/all", None),   # synthetic noise, skip
    "CXR_GaussNoise":     ("data_mri/medmnist/pneumoniamnist/all", None),  # synthetic noise, skip
}


def _collect(d: str | Path, max_n: int = N_SAMPLE) -> List[Path]:
    d = Path(d)
    paths = sorted(p for p in d.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) > max_n:
        rng = random.Random(RANDOM_SEED)
        paths = rng.sample(paths, max_n)
        paths = sorted(paths)
    return paths


def _to_uint8_299(paths: List[Path]) -> torch.Tensor:
    arrs = [np.array(Image.open(p).convert("RGB").resize((299, 299))) for p in paths]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(torch.uint8)


def compute_fid_kid(
    real_paths: List[Path],
    gen_paths:  List[Path],
    device:     str,
) -> dict:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    subset = min(50, len(real_paths), len(gen_paths))
    fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_m = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)

    BATCH = 64
    real_t = _to_uint8_299(real_paths).to(device)
    gen_t  = _to_uint8_299(gen_paths).to(device)

    for i in range(0, len(real_t), BATCH):
        fid_m.update(real_t[i:i+BATCH], real=True)
        kid_m.update(real_t[i:i+BATCH], real=True)
    for i in range(0, len(gen_t), BATCH):
        fid_m.update(gen_t[i:i+BATCH], real=False)
        kid_m.update(gen_t[i:i+BATCH], real=False)

    fid_val = float(fid_m.compute().item())
    kid_mean, kid_std = kid_m.compute()
    return {
        "fid":      fid_val,
        "kid_mean": float(kid_mean.item()) * 100,
        "kid_std":  float(kid_std.item()) * 100,
    }


def main(args):
    device = args.device
    results_path = Path(args.results_json)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    updated = 0
    for entry in tqdm(results, desc="Supplementing FID/KID"):
        name = entry.get("name", "")
        if name not in _DIR_MAP:
            continue

        real_dir, gen_dir = _DIR_MAP[name]
        if gen_dir is None:
            print(f"  {name}: skipped (synthetic noise — no paths)")
            continue

        # Only fill if currently nan
        if not (np.isnan(entry.get("fid", float("nan"))) or
                np.isnan(entry.get("kid_mean", float("nan")))):
            print(f"  {name}: FID/KID already present, skipping")
            continue

        real_paths = _collect(real_dir)
        gen_paths  = _collect(gen_dir)

        if not gen_paths:
            print(f"  {name}: gen_dir empty or missing ({gen_dir})")
            continue

        print(f"  {name}: computing FID/KID...")
        try:
            fk = compute_fid_kid(real_paths, gen_paths, device)
            entry.update(fk)
            print(f"    FID={fk['fid']:.2f}  KID×100={fk['kid_mean']:.4f}")
            updated += 1
        except Exception as e:
            print(f"    FAILED: {e}")

    if updated > 0:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nUpdated {updated} entries -> {results_path}")
    else:
        print("Nothing to update.")

    # Print summary
    print("\n── Updated Results ──────────────────────────────────────────────")
    print(f"{'Name':<25} {'M3':>10} {'FID':>8} {'KID×100':>10} {'CMMD':>12}")
    print("-"*70)
    for r in results:
        if isinstance(r.get("name"), str) and r.get("name") != "bootstrap_auc":
            print(
                f"{r.get('name','?'):<25} "
                f"{r.get('m3',float('nan')):>10.5f} "
                f"{r.get('fid',float('nan')):>8.2f} "
                f"{r.get('kid_mean',float('nan')):>10.4f} "
                f"{r.get('cmmd',float('nan')):>12.6f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", default="results/multi_metric_comparison/results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
