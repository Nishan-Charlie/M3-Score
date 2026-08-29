"""
experiments/progressive_degradation_sweep.py
===============================================
Progressive image degradation sweep: Gaussian noise + Gaussian blur.

Tests whether M3-Score, FID, KID, SSIM, and PSNR respond monotonically to
controlled degradation of real images. This is a key validation experiment
for Section 3.12 of the M3-Score paper.

NOTE: The `noise_robustness.py` experiment runs a more comprehensive version
of this sweep (larger image set, more sigma levels, additional metrics, full
Spearman analysis). This module provides a lighter, self-contained version
with an orchestrator-compatible run_progressive_degradation() entry point,
and is explicitly listed as a separate experiment to validate the KID baseline.

All metrics computed per degradation level:
  M3-Score, FID, KID, SSIM, PSNR

Usage (CLI):
    python progressive_degradation_sweep.py \\
        --real_dir data_mri/brats_axial_multislice \\
        --output_dir results/prog_degradation \\
        --n 100 --device cuda

Usage (from orchestrator):
    from experiments.progressive_degradation_sweep import run_progressive_degradation
    results = run_progressive_degradation(real_dir=..., output_dir=..., n=100, device="cuda")
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from torchvision import transforms
from evaluation.m3_score_v2 import M3V2Metric
from experiments._shared_utils import load_pils, compute_psnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_gaussian_noise(image_np: np.ndarray, var: float) -> np.ndarray:
    """Add Gaussian noise to a uint8 numpy image (0-255)."""
    sigma = var ** 0.5
    gauss = np.random.normal(0, sigma, image_np.shape)
    return np.clip(image_np + gauss * 255, 0, 255).astype(np.uint8)


def _load_tensors(pil_images: list, size_m3: int = 224, size_fid: int = 299):
    """
    Convert a list of PIL images into:
      m3_batch  : (N, 3, size_m3, size_m3) float32 in [0,1]
      fid_batch : (N, 3, size_fid, size_fid) uint8
    """
    t_m3 = transforms.Compose([
        transforms.Resize((size_m3, size_m3)),
        transforms.ToTensor(),
    ])
    t_fid = transforms.Compose([
        transforms.Resize((size_fid, size_fid)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte()),
    ])
    m3_imgs, fid_imgs = [], []
    for p in pil_images:
        rgb = p.convert("RGB")
        m3_imgs.append(t_m3(rgb))
        fid_imgs.append(t_fid(rgb))
    return torch.stack(m3_imgs), torch.stack(fid_imgs)


def _compute_spearman(levels: list, values: list) -> float:
    """Return Spearman r between levels and values (NaN entries excluded)."""
    from scipy import stats as sp_stats
    lvl = np.array(levels, dtype=float)
    val = np.array(values, dtype=float)
    mask = ~np.isnan(val)
    if mask.sum() < 3:
        return float("nan")
    r, _ = sp_stats.spearmanr(lvl[mask], val[mask])
    return round(float(r), 4)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_progressive_degradation(
    real_dir:     str,
    output_dir:   str          = "./progressive_degradation_output",
    n:            int          = 100,
    noise_levels: Optional[list] = None,
    blur_levels:  Optional[list] = None,
    device:       Optional[str]  = None,
    seed:         int          = 42,
) -> dict:
    """
    Run the progressive degradation sweep.

    Args:
        real_dir:     Directory of real images (searched recursively).
        output_dir:   Destination for CSV, JSON, and plots.
        n:            Number of real images to use.
        noise_levels: Gaussian noise variance levels (0 = clean baseline).
        blur_levels:  Gaussian blur radii in pixels (0 = clean baseline).
        device:       Torch device string (None = auto-detect).
        seed:         Random seed.

    Returns:
        dict with noise and blur sweep results and Spearman correlations,
        compatible with master_report[\"progressive_degradation\"].
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    if blur_levels is None:
        blur_levels = [0, 1, 2, 3, 4]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load real images ──────────────────────────────────────────────────
    print(f"[ProgDeg] Loading {n} real images from {real_dir} ...")
    real_pils = load_pils(real_dir, n=n)
    if not real_pils:
        raise FileNotFoundError(f"No images found in {real_dir}")
    real_pils = real_pils[:n]
    actual_n = len(real_pils)
    print(f"[ProgDeg] Loaded {actual_n} images.")

    real_m3_batch, real_fid_batch = _load_tensors(real_pils)

    # ── Initialise models once ────────────────────────────────────────────
    print("[ProgDeg] Initialising models ...")
    m3_metric = M3V2Metric(device=device)
    m3_metric.prune_layers_via_cka(real_m3_batch[:20])
    print(f"  M3 active layers: {m3_metric.active_layers}")

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        try:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            kid_metric = KernelInceptionDistance(subset_size=min(50, actual_n), normalize=True).to(device)
        except TypeError:
            fid_metric = FrechetInceptionDistance(feature=2048).to(device)
            kid_metric = KernelInceptionDistance(subset_size=min(50, actual_n)).to(device)
        fid_kid_available = True
    except ImportError:
        print("  [WARN] torchmetrics not installed; FID and KID skipped.")
        fid_metric = kid_metric = None
        fid_kid_available = False

    def _eval_one_level(deg_pils: list) -> dict:
        """Compute all metrics between real_pils and deg_pils."""
        deg_m3_batch, deg_fid_batch = _load_tensors(deg_pils)

        # M3-Score
        with torch.no_grad():
            m3_res = m3_metric(real_m3_batch, deg_m3_batch)
        m3_val = float(m3_res.get("m3_v2_final_score", float("nan")))

        # FID + KID
        fid_val = kid_val = float("nan")
        if fid_kid_available:
            try:
                fid_metric.reset()
                fid_metric.update(real_fid_batch.to(device), real=True)
                fid_metric.update(deg_fid_batch.to(device), real=False)
                fid_val = float(fid_metric.compute().item())

                kid_metric.reset()
                kid_metric.update(real_fid_batch.to(device), real=True)
                kid_metric.update(deg_fid_batch.to(device), real=False)
                kid_mean, kid_std = kid_metric.compute()
                kid_val = float(kid_mean.item())
            except Exception as e:
                print(f"  FID/KID failed: {e}")

        # SSIM and PSNR (per-image, on [0,255] uint8)
        ssim_vals, psnr_vals = [], []
        try:
            from skimage.metrics import structural_similarity
            for orig, deg in zip(real_pils, deg_pils):
                orig_arr = np.array(orig.convert("L"))
                deg_arr  = np.array(deg.convert("L"))
                ssim_vals.append(structural_similarity(orig_arr, deg_arr, data_range=255))
                psnr_vals.append(compute_psnr(orig, deg))
        except Exception as e:
            print(f"  SSIM/PSNR failed: {e}")

        psnr_clean = [p for p in psnr_vals if p != float("inf")]

        return {
            "m3_score": round(m3_val, 6),
            "fid":      round(fid_val, 4) if not np.isnan(fid_val) else fid_val,
            "kid":      round(kid_val, 6) if not np.isnan(kid_val) else kid_val,
            "ssim":     round(float(np.mean(ssim_vals)), 6) if ssim_vals else float("nan"),
            "psnr":     round(float(np.mean(psnr_clean)), 4) if psnr_clean else float("nan"),
        }

    # ── Noise sweep ───────────────────────────────────────────────────────
    print("\n[ProgDeg] --- Gaussian Noise Sweep ---")
    noise_results = []
    for var in tqdm(noise_levels, desc="Noise sweep"):
        if var == 0.0:
            deg_pils = real_pils
        else:
            deg_pils = []
            for pil_img in real_pils:
                img_np = np.array(pil_img)
                noisy_np = _add_gaussian_noise(img_np, var)
                deg_pils.append(Image.fromarray(noisy_np))
        row = {"level_type": "noise", "level": var, **_eval_one_level(deg_pils)}
        noise_results.append(row)
        print(f"  var={var:.3f}  M3={row['m3_score']:.5f}  FID={row['fid']}  KID={row['kid']}  SSIM={row['ssim']:.4f}")

    # ── Blur sweep ────────────────────────────────────────────────────────
    print("\n[ProgDeg] --- Gaussian Blur Sweep ---")
    blur_results = []
    for radius in tqdm(blur_levels, desc="Blur sweep"):
        if radius == 0:
            deg_pils = real_pils
        else:
            deg_pils = [
                pil_img.filter(ImageFilter.GaussianBlur(radius))
                for pil_img in real_pils
            ]
        row = {"level_type": "blur", "level": radius, **_eval_one_level(deg_pils)}
        blur_results.append(row)
        print(f"  radius={radius}  M3={row['m3_score']:.5f}  FID={row['fid']}  KID={row['kid']}  SSIM={row['ssim']:.4f}")

    # ── Spearman correlations ─────────────────────────────────────────────
    def _spearman_for_sweep(rows: list) -> dict:
        levels = [r["level"] for r in rows]
        out = {}
        # Higher perturbation → M3/FID/KID should increase; SSIM/PSNR decrease
        for metric, expected_sign in [
            ("m3_score", +1), ("fid", +1), ("kid", +1),
            ("ssim", -1), ("psnr", -1),
        ]:
            vals = [r.get(metric, float("nan")) for r in rows]
            r_val = _compute_spearman(levels, vals)
            out[metric] = {
                "spearman_r": r_val,
                "monotone": (float(r_val) * expected_sign > 0) and (abs(r_val) >= 0.7)
                if not np.isnan(r_val) else False,
                "expected": "increase" if expected_sign > 0 else "decrease",
            }
        return out

    noise_corr = _spearman_for_sweep(noise_results)
    blur_corr  = _spearman_for_sweep(blur_results)

    # ── Save outputs ──────────────────────────────────────────────────────
    all_rows = noise_results + blur_results
    try:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(output_dir, "progressive_degradation.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[ProgDeg] CSV saved: {csv_path}")
    except ImportError:
        csv_path = None

    results = {
        "n_images":        actual_n,
        "active_layers":   m3_metric.active_layers,
        "noise": {
            "levels":          noise_levels,
            "metric_values":   noise_results,
            "spearman":        noise_corr,
        },
        "blur": {
            "levels":          blur_levels,
            "metric_values":   blur_results,
            "spearman":        blur_corr,
        },
    }
    json_path = os.path.join(output_dir, "progressive_degradation_report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"[ProgDeg] JSON saved: {json_path}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Progressive degradation sweep: M3, FID, KID, SSIM, PSNR"
    )
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--output_dir", default="./progressive_degradation_output")
    parser.add_argument("--n",          type=int,   default=100)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    run_progressive_degradation(
        real_dir   = args.real_dir,
        output_dir = args.output_dir,
        n          = args.n,
        device     = args.device,
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()
