"""
FID∞ — Extrapolated Fréchet Inception Distance
================================================
Implements the FID∞ estimator from Chong & Forsyth (2020):

  "Effectively Unbiased FID and IS Estimation"

Computes FID at multiple sample sizes, then fits a linear regression in
1/N space to extrapolate to N→∞, removing finite sample bias.

Protocol:
  1. Fix real set (all images)
  2. For each gen subset size M ∈ {50, 100, 200, 500, 1000}:
     - Compute FID(real, gen[:M])
  3. Fit: FID(M) = FID∞ + b / M
  4. The intercept FID∞ is the bias-corrected estimate
"""

from __future__ import annotations

import os
import sys
import json
import glob
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm


def _load_paths(directory, n=None):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    paths = []
    # Non-recursive to avoid picking up evaluation plots in subfolders
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(paths)
    if n:
        paths = paths[:n]
    return paths


def _load_rgb_uint8(paths, size=(256, 256)):
    imgs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB").resize(size), dtype=np.uint8)
        imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
    return torch.stack(imgs)


def run_fid_infinity(
    real_dir: str,
    gen_dir: str,
    output_dir: str = "./fid_infinity_output",
    sample_sizes: Optional[list[int]] = None,
    num_repeats: int = 3,
    device: str = "cpu",
    use_tqdm: bool = True,
    seed: int = 42,
) -> dict:
    """
    Compute FID∞ by extrapolating FID at multiple sample sizes.

    Returns dict with fid_infinity, regression coefficients, and per-size FID values.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    real_paths = _load_paths(real_dir)
    gen_paths = _load_paths(gen_dir)
    n_gen = len(gen_paths)

    if sample_sizes is None:
        candidates = [50, 100, 200, 500, 1000]
        sample_sizes = [s for s in candidates if s <= n_gen]

    print(f"[FID∞] Real: {len(real_paths)} | Gen: {n_gen}")
    print(f"[FID∞] Sample sizes: {sample_sizes}, repeats: {num_repeats}")

    # Load all real images once
    real_rgb = _load_rgb_uint8(real_paths).to(device)

    fid_values = {}  # {M: [fid_1, fid_2, ...]}

    for M in tqdm(sample_sizes, desc="FID∞ sizes", disable=not use_tqdm):
        fid_values[M] = []
        for rep in range(num_repeats):
            # Random subset of gen images
            idx = np.random.choice(n_gen, size=min(M, n_gen), replace=False)
            gen_subset_paths = [gen_paths[i] for i in idx]
            gen_rgb = _load_rgb_uint8(gen_subset_paths).to(device)

            fid = FrechetInceptionDistance(feature=64).to(device)
            fid.update(real_rgb[:min(len(real_rgb), M)], real=True)
            fid.update(gen_rgb, real=False)
            fid_val = float(fid.compute().item())
            fid_values[M].append(fid_val)

            del fid, gen_rgb
            torch.cuda.empty_cache() if "cuda" in device else None

        mean_fid = np.mean(fid_values[M])
        print(f"  M={M}: FID = {mean_fid:.2f} ± {np.std(fid_values[M]):.2f}")

    # ---- Linear regression: FID(M) = FID∞ + b / M ----
    # x = 1/M, y = FID(M)
    x_vals = []
    y_vals = []
    for M in sample_sizes:
        for v in fid_values[M]:
            x_vals.append(1.0 / M)
            y_vals.append(v)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Fit y = a + b * x where a = FID∞
    if len(x_vals) >= 2:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        b_coeff = coeffs[0]
        fid_inf = coeffs[1]
    else:
        fid_inf = float("nan")
        b_coeff = float("nan")

    print(f"[FID∞] FID∞ = {fid_inf:.4f} (bias slope b = {b_coeff:.4f})")

    # Also try clean-fid if available
    clean_fid_val = None
    try:
        from cleanfid import fid as cleanfid
        clean_fid_val = cleanfid.compute_fid(real_dir, gen_dir, device=torch.device(device))
        print(f"[FID∞] Clean-FID = {clean_fid_val:.4f}")
    except Exception as e:
        print(f"[FID∞] Clean-FID skipped: {e}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    fig.patch.set_facecolor("white")

    # Panel 1: FID vs M
    ax = axes[0]
    ax.set_facecolor("white")
    means = [np.mean(fid_values[M]) for M in sample_sizes]
    stds = [np.std(fid_values[M]) for M in sample_sizes]
    ax.errorbar(sample_sizes, means, yerr=stds, fmt="o-", color="#1565C0",
                linewidth=2, markersize=8, capsize=5, capthick=2)
    ax.axhline(fid_inf, color="#c62828", linestyle="--", linewidth=2,
               label=f"FID∞ = {fid_inf:.2f}")
    if clean_fid_val is not None:
        ax.axhline(clean_fid_val, color="#2e7d32", linestyle=":", linewidth=2,
                   label=f"Clean-FID = {clean_fid_val:.2f}")
    ax.set_xlabel("Sample Size M", fontsize=11)
    ax.set_ylabel("FID", fontsize=11)
    ax.set_title("FID vs Sample Size", fontsize=13)
    ax.legend(fontsize=10, facecolor="white", edgecolor="#cccccc")
    ax.tick_params(colors="#333333")
    ax.grid(True, alpha=0.4)

    # Panel 2: FID vs 1/M with regression line
    ax = axes[1]
    ax.set_facecolor("white")
    ax.scatter(x_vals, y_vals, c="#1565C0", s=30, alpha=0.6, edgecolors="none")
    x_fit = np.linspace(0, max(x_vals) * 1.1, 100)
    y_fit = fid_inf + b_coeff * x_fit
    ax.plot(x_fit, y_fit, "--", color="#c62828", linewidth=2,
            label=f"FID = {fid_inf:.2f} + {b_coeff:.1f}/M")
    ax.scatter([0], [fid_inf], c="#c62828", s=100, zorder=5, marker="*",
               label=f"FID∞ = {fid_inf:.2f}")
    ax.set_xlabel("1/M", fontsize=11)
    ax.set_ylabel("FID", fontsize=11)
    ax.set_title("FID vs 1/M — Extrapolation to ∞", fontsize=13)
    ax.legend(fontsize=10, facecolor="white", edgecolor="#cccccc")
    ax.tick_params(colors="#333333")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fid_infinity.png")
    plt.savefig(plot_path, bbox_inches="tight", facecolor="white")
    plt.close()

    results = {
        "fid_infinity": round(float(fid_inf), 4),
        "bias_slope": round(float(b_coeff), 4),
        "per_size": {
            str(M): {
                "mean": round(float(np.mean(fid_values[M])), 4),
                "std": round(float(np.std(fid_values[M])), 4),
            }
            for M in sample_sizes
        },
        "clean_fid": round(float(clean_fid_val), 4) if clean_fid_val is not None else None,
        "plots": {
            "regression": plot_path
        },
    }

    report_path = os.path.join(output_dir, "fid_infinity_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[FID∞] Report saved → {report_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--output_dir", default="./fid_infinity_output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no_tqdm", action="store_true")
    args = parser.parse_args()

    run_fid_infinity(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir,
        device=args.device,
        use_tqdm=not args.no_tqdm,
    )
