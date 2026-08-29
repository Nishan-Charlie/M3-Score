"""
CMMD Noise/Blur Robustness Sweep
=================================
Computes CLIP Maximum Mean Discrepancy (Jayasumana et al., 2024) across the
same Gaussian noise and blur perturbation levels used in noise_robustness.py.

Reuses the pre-saved perturbed image directories from the noise_robustness run
so no re-perturbation is needed. Outputs CMMD values and Spearman correlations
that can be inserted into the paper's Tab. II (noise Spearman table) and used
to update the Figure 1 robustness plots with a CMMD curve.

Usage:
    python experiments/cmmd_noise_sweep.py \
        --real_dir   data_mri/brats_axial_multislice \
        --pert_dir   results/experiments_output_v5/noise_robustness \
        --output_dir results/experiments_output_v5/cmmd_sweep \
        --device     cpu
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sp_stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.cmmd_metric import CMMDMetric


def _image_paths(directory: str, n: int | None = None) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))
    if n is not None:
        paths = paths[:n]
    return paths


def _spearman_with_ci(levels: list[float], vals: list[float]) -> dict:
    x = np.array(levels)
    y = np.array(vals)
    r, p = sp_stats.spearmanr(x, y)
    n = len(x)
    if abs(r) >= 1.0:
        ci_lo = ci_hi = float(r)
    else:
        z  = np.arctanh(r)
        se = 1.0 / np.sqrt(max(n - 3, 1))
        ci_lo = float(np.tanh(z - 1.96 * se))
        ci_hi = float(np.tanh(z + 1.96 * se))
    return {
        "spearman_r": round(float(r), 4),
        "spearman_p": float(p),
        "ci_95":      [round(ci_lo, 4), round(ci_hi, 4)],
        "n":          n,
    }


def run_cmmd_sweep(
    real_dir:    str,
    pert_dir:    str,
    output_dir:  str,
    device:      str = "cpu",
    num_images:  int = 200,
    noise_levels: list | None = None,
    blur_levels:  list | None = None,
) -> dict:
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    if blur_levels is None:
        blur_levels = [0, 1, 2, 3, 4]

    os.makedirs(output_dir, exist_ok=True)

    metric = CMMDMetric(device=device, batch_size=32)

    # Extract features for the clean real set once
    real_paths = _image_paths(real_dir, num_images)
    print(f"[CMMD] Extracting features for {len(real_paths)} clean real images ...")
    real_feats = metric.extract_features(real_paths, desc="CLIP [real]")

    def _sweep(levels: list, key: str, folder_prefix: str) -> dict:
        vals:   list[float] = []
        rows:   list[dict]  = []
        for lvl in levels:
            folder = os.path.join(pert_dir, f"{folder_prefix}{lvl}")
            pert_paths = _image_paths(folder, num_images)
            if not pert_paths:
                print(f"  [WARN] No images found in {folder} – skipping {key}={lvl}")
                continue
            print(f"[CMMD] {key}={lvl}  n_pert={len(pert_paths)}")
            pert_feats = metric.extract_features(pert_paths, desc=f"CLIP [{key}={lvl}]")
            score = metric.compute_from_features(real_feats, pert_feats)
            print(f"  CMMD = {score:.6f}")
            rows.append({key: float(lvl), "cmmd": round(score, 6)})
            vals.append(score)

        level_vals = [r[key] for r in rows]
        corr = _spearman_with_ci(level_vals, vals)
        return {"metric_values": rows, "spearman": corr}

    print("\n[CMMD] --- Noise sweep ---")
    noise_result = _sweep(noise_levels, "sigma",  "pert_noise_sigma")
    print("\n[CMMD] --- Blur sweep ---")
    blur_result  = _sweep(blur_levels,  "radius", "pert_blur_radius")

    report = {
        "noise": noise_result,
        "blur":  blur_result,
    }
    report_path = os.path.join(output_dir, "cmmd_robustness_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[CMMD] Report saved: {report_path}")

    # Print summary
    print(f"\n[CMMD] Spearman noise rho = {noise_result['spearman']['spearman_r']:.4f}")
    print(f"[CMMD] Spearman blur  rho = {blur_result['spearman']['spearman_r']:.4f}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir",    required=True)
    parser.add_argument("--pert_dir",    required=True,
                        help="Directory containing pert_noise_sigma* and pert_blur_radius* sub-dirs")
    parser.add_argument("--output_dir",  default="results/experiments_output_v5/cmmd_sweep")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--num_images",  type=int, default=200)
    args = parser.parse_args()

    run_cmmd_sweep(
        real_dir    = args.real_dir,
        pert_dir    = args.pert_dir,
        output_dir  = args.output_dir,
        device      = args.device,
        num_images  = args.num_images,
    )
