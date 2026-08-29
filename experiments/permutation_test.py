"""
Permutation Test -- Statistical Significance of M3-Score (Section 3.7)
========================================================================
Establishes the null distribution by computing M3 on real-vs-real random
splits of the real dataset, then compares the observed real-vs-generated
score to that null to produce a Z-score and empirical p-value.

This directly validates the claim in Section 3.7 of the paper:
  "Z-Score of 314.86 (p << 0.01)"

Protocol:
  1. Pre-extract RadioDino features for all real and generated images.
  2. For each of N_PERM permutations, randomly split real features into two
     halves of size N//2 and compute the M3 score between them.
     This builds the null distribution.
  3. Compute the observed M3 score on a single random draw of N//2 real
     images vs N//2 generated images (matching the null sample size to
     avoid inflating the Z-score via variance mismatch).
  4. Z = (observed - mu_null) / sigma_null.
  5. Empirical p = fraction of null scores >= observed.

Sample-size note:
  The null and observed scores are both computed at N//2 samples to ensure
  the MMD^2 estimator operates under identical variance conditions. Using
  all N samples for observed while using N//2 for null would inflate the
  Z-score because MMD^2 variance scales as O(1/N).

Weighting note:
  The permutation test uses equal-weight averaging of per-layer MMD^2
  values rather than SNR-based weights, because the SNR weights require
  a bootstrap sub-batch pass that would add O(K * N_PERM) forward calls.
  Both null and observed use the same weighting function, so the Z-score
  is internally consistent. The observed value reported here will differ
  slightly from the full M3-Score reported in other experiments.

Usage:
    python permutation_test.py \\
        --real_dir <path> --gen_dir <path> --output_dir <path> \\
        --n_perm 50 --n_images 500 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.m3_score_v2 import M3V2Metric     # corrected import path


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_images(directory: str, n: int) -> torch.Tensor:
    """
    Load up to n images as (N, 3, 224, 224) float tensors in [0, 1].
    M3V2Metric._preprocess handles normalisation internally.
    """
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))[:n]

    if not paths:
        raise FileNotFoundError(
            f"No images found in {directory}. "
            "Check the directory path and supported extensions."
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imgs = [
        transform(Image.open(p).convert("RGB"))
        for p in tqdm(paths, desc=f"Loading {os.path.basename(directory)}", leave=False)
    ]
    return torch.stack(imgs)


# ---------------------------------------------------------------------------
# Score helper (equal-weight, fast)
# ---------------------------------------------------------------------------

def _equal_weight_score(
    metric:        M3V2Metric,
    feats_a:       list[torch.Tensor],
    feats_b:       list[torch.Tensor],
    active_layers: list[int],
    device:        str,
) -> float:
    """
    Compute the mean MMD^2 across active layers with equal weights.

    Equal weighting is used here (rather than SNR-based weights from
    M3V2Metric.forward) to avoid O(K * N_PERM) additional forward calls.
    Both null permutations and the observed score use this same function,
    so the Z-score is internally consistent.

    Args:
        metric:        Initialised M3V2Metric (used for _compute_kid_distance).
        feats_a:       List of (n, D) feature tensors indexed by layer position.
        feats_b:       List of (n, D) feature tensors indexed by layer position.
        active_layers: 1-indexed list of retained layer indices.
        device:        Torch device string.

    Returns:
        Scalar mean MMD^2 across active layers.
    """
    total = 0.0
    for idx in active_layers:
        fa = feats_a[idx - 1].to(device)
        fb = feats_b[idx - 1].to(device)
        total += metric._compute_mmd2(fa, fb).item()
    return total / max(len(active_layers), 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_permutation_test(
    real_dir:   str,
    gen_dir:    str,
    output_dir: str = "./permutation_test_output",
    n_perm:     int = 50,
    n_images:   int = 500,
    device:     str = None,
    seed:       int = 42,
) -> dict:
    """
    Run the permutation significance test for M3-Score.

    Args:
        real_dir:   Directory of real images.
        gen_dir:    Directory of generated images.
        output_dir: Destination for plot and JSON report.
        n_perm:     Number of real-vs-real null permutations.
        n_images:   Images loaded per set.
        device:     Torch device string (None = auto-detect).
        seed:       Random seed.

    Returns:
        dict with null distribution statistics, Z-score, empirical p-value,
        compatible with master_report["permutation_test"] in run_experiments.py.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Loading images (n={n_images}) ...")
    real_imgs = _load_images(real_dir, n_images)
    gen_imgs  = _load_images(gen_dir,  n_images)

    if len(real_imgs) == 0 or len(gen_imgs) == 0:
        raise ValueError(
            f"Image loading returned empty tensors. "
            f"real: {len(real_imgs)}, gen: {len(gen_imgs)}."
        )

    N = len(real_imgs)
    half = N // 2

    # ── Initialise metric and select layers ───────────────────────────────────
    print("\nInitialising M3-Score and running CKA layer selection ...")
    metric = M3V2Metric(device=device)
    metric.prune_layers_via_cka(real_imgs[:20])
    active_layers = metric.active_layers
    active_set    = set(active_layers)
    print(f"  Active layers: {active_layers}")

    # ── Pre-extract features for all images (done once) ───────────────────────
    print("Extracting real features ...")
    with torch.no_grad():
        real_feats = metric._extract_raw_features(
            real_imgs, use_attention=True, layers_to_keep=active_set
        )

    print("Extracting generated features ...")
    with torch.no_grad():
        gen_feats = metric._extract_raw_features(
            gen_imgs, use_attention=True, layers_to_keep=active_set
        )

    # ── Null distribution: real-vs-real random N//2 splits ───────────────────
    print(f"\nRunning {n_perm} real-vs-real permutations (half size = {half}) ...")
    null_scores: list[float] = []
    for _ in tqdm(range(n_perm), desc="Permutations"):
        perm  = np.random.permutation(N)
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]
        feats_a = [f[idx_a] if f.numel() > 0 else f for f in real_feats]
        feats_b = [f[idx_b] if f.numel() > 0 else f for f in real_feats]
        null_scores.append(
            _equal_weight_score(metric, feats_a, feats_b, active_layers, device)
        )

    null_arr = np.array(null_scores)
    mu_null  = float(null_arr.mean())
    std_null = float(null_arr.std())

    # ── Observed score: real-vs-gen at the same N//2 sample size ─────────────
    # Matching sample sizes ensures the MMD^2 estimator operates under the same
    # variance conditions as the null, preventing Z-score inflation.
    rng_obs = np.random.default_rng(seed + 1)
    obs_real_idx = rng_obs.choice(N, size=half, replace=False)
    obs_gen_idx  = rng_obs.choice(len(gen_imgs), size=half, replace=False)

    obs_feats_real = [f[obs_real_idx] if f.numel() > 0 else f for f in real_feats]
    obs_feats_gen  = [f[obs_gen_idx]  if f.numel() > 0 else f for f in gen_feats]
    observed_score = _equal_weight_score(
        metric, obs_feats_real, obs_feats_gen, active_layers, device
    )

    # ── Z-score and empirical p-value ─────────────────────────────────────────
    z_score = (observed_score - mu_null) / (std_null + 1e-12)
    emp_p   = float((null_arr >= observed_score).mean())

    print(f"\n  Null:     mu={mu_null:.6f}  sigma={std_null:.6f}")
    print(f"  Observed: {observed_score:.6f}  (N//2 = {half} images)")
    print(f"  Z-score:  {z_score:.2f}")
    print(f"  Empirical p-value: {emp_p:.4f}")

    # ── Assemble results ──────────────────────────────────────────────────────
    results: dict = {
        "M3-Score": {
            "null_mean":     round(mu_null, 6),
            "null_std":      round(std_null, 6),
            "null_scores":   [round(float(s), 6) for s in null_scores],
            "observed":      round(float(observed_score), 6),
            "z_score":       round(float(z_score), 4),
            "empirical_p":   round(emp_p, 6),
            "significant":   bool(z_score > 3.0),
            "active_layers": active_layers,
            "sample_size":   half,
        }
    }

    # ── Plot ──────────────────────────────────────────────────────────────────
    r = results["M3-Score"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.hist(
        null_arr,
        bins=max(10, len(null_arr) // 3),
        color="#4fc3f7", alpha=0.7,
        label="Null distribution (real-vs-real)",
        density=True, edgecolor="white",
    )
    ax.axvline(
        r["observed"], color="#ef5350", lw=2.5, linestyle="--",
        label=f"Observed (real-vs-gen) = {r['observed']:.4f}",
    )
    ax.axvline(
        r["null_mean"], color="#66bb6a", lw=1.5, linestyle=":",
        label=f"Null mean = {r['null_mean']:.4f}",
    )
    ax.set_xlabel("M3-Score (equal-weight, N//2 samples)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"M3-Score permutation test (RadioDino)\n"
        f"Z = {r['z_score']:.1f}   empirical p = {r['empirical_p']:.4f}",
        fontsize=12,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.suptitle(
        "Null vs observed distribution (M3-Score)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "permutation_test.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    results["plots"] = {"null_distribution": os.path.abspath(plot_path)}

    report_path = os.path.join(output_dir, "permutation_test_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\nReport saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Permutation significance test for M3-Score (Section 3.7)"
    )
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="./permutation_test_output")
    parser.add_argument("--n_perm",     type=int, default=50)
    parser.add_argument("--n_images",   type=int, default=500)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run_permutation_test(
        real_dir   = args.real_dir,
        gen_dir    = args.gen_dir,
        output_dir = args.output_dir,
        n_perm     = args.n_perm,
        n_images   = args.n_images,
        device     = args.device,
        seed       = args.seed,
    )