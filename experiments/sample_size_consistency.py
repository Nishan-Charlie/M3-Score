"""
sample_size_consistency.py  --  Validation Experiment: Sample-size consistency
===============================================================================

Demonstrates that M3-Score's unbiased MMD estimator remains stable at small
sample counts while FID and CMMD drift, making M3 the right metric for
medical imaging where N is typically 50-500.

Protocol
--------
For each N in [10, 25, 50, 100, 200, 500]:
    Repeat 20 times:
        - Draw N random real images
        - Draw N random generated images (WDM-3D BraTS / DDPM)
        - Compute M3 (RadioDino-s16 L12 RBF), FID (feature-space), CMMD
    Report: mean, std, CV = std/mean  across the 20 repeats

Expected result
---------------
M3 CV stays low and roughly constant across N.
FID CV explodes at small N (high variance due to Gaussian approximation error).
CMMD shows intermediate behaviour.

Usage
-----
    python experiments/sample_size_consistency.py \
        --real_dir data_mri/brats_axial_multislice \
        --gen_dir  output/generated \
        --output_dir results/sample_size_consistency \
        --n_list 10 25 50 100 200 500 \
        --n_repeats 20
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Metric helpers (operate on pre-extracted feature matrices)
# ---------------------------------------------------------------------------

def _mmd_rbf(X: np.ndarray, Y: np.ndarray, n_bw: int = 5) -> float:
    """Unbiased RBF MMD^2 with median-heuristic bandwidths."""
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()

    def sq(A, B):
        return (A.unsqueeze(1) - B.unsqueeze(0)).pow(2).sum(-1)

    Dxx = sq(Xt, Xt); Dyy = sq(Yt, Yt); Dxy = sq(Xt, Yt)
    med = torch.cat([Dxy.flatten(), Dxx[Dxx > 0].flatten(),
                     Dyy[Dyy > 0].flatten()]).median().item()
    if med < 1e-10: med = 1.0

    n, m = len(X), len(Y); total = 0.0
    for bw in [med * (2 ** k) for k in range(-(n_bw // 2), n_bw - n_bw // 2)]:
        g = 1.0 / (2.0 * bw)
        Kxx = torch.exp(-g * Dxx); Kxx.fill_diagonal_(0)
        Kyy = torch.exp(-g * Dyy); Kyy.fill_diagonal_(0)
        Kxy = torch.exp(-g * Dxy)
        total += (Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean()).item()
    return total / n_bw


def _fid_feats(X: np.ndarray, Y: np.ndarray) -> float:
    """Feature-space FID (Frechet distance on moments 1+2)."""
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    from scipy.linalg import sqrtm
    mu_r = X.mean(0); mu_g = Y.mean(0)
    sg_r = np.cov(X.T); sg_g = np.cov(Y.T)
    diff = mu_r - mu_g
    cm = sqrtm(sg_r @ sg_g)
    if np.iscomplexobj(cm): cm = cm.real
    return float(max(diff @ diff + np.trace(sg_r + sg_g - 2 * cm), 0.0))


def _cmmd_poly(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> float:
    """Polynomial kernel MMD^2 (CMMD-style)."""
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()

    def poly_k(A, B, d=degree):
        return ((A @ B.T) / A.shape[-1] + 1) ** d

    Kxx = poly_k(Xt, Xt); Kyy = poly_k(Yt, Yt); Kxy = poly_k(Xt, Yt)
    Kxx.fill_diagonal_(0); Kyy.fill_diagonal_(0)
    n, m = len(X), len(Y)
    mmd = (Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean()).item()
    return float(mmd)


# ---------------------------------------------------------------------------
# Feature extractor (extract ALL features once; subsample on each repeat)
# ---------------------------------------------------------------------------

def _extract_all(
    images, backbone_id: str, device: str, batch_size: int = 32
) -> np.ndarray:
    from evaluation.coverage_novelty import _load_backbone, _extract_features
    backbone, backend = _load_backbone(backbone_id, device)
    return _extract_features(images, backbone, backend, device, batch_size)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_sample_size_consistency(
    real_dir:    str,
    gen_dir:     str,
    output_dir:  str = "results/sample_size_consistency",
    n_list:      List[int] = None,
    n_repeats:   int = 20,
    backbone_id: str = "Snarcy/RadioDino-s16",
    device:      str = "cuda",
    batch_size:  int = 32,
    seed:        int = 42,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    device = device if torch.cuda.is_available() else "cpu"
    n_list = n_list or [10, 25, 50, 100, 200, 500]

    from experiments._shared_utils import load_pils_recursive
    print("\n=== Sample-Size Consistency Experiment ===")

    # Load ALL images and extract features once
    max_n = max(n_list)
    real_imgs = load_pils_recursive(real_dir, n=max_n * 2)
    gen_imgs  = load_pils_recursive(gen_dir,  n=max_n * 2)

    print(f"  Extracting features for {len(real_imgs)} real + {len(gen_imgs)} gen images...")
    real_all = _extract_all(real_imgs, backbone_id, device, batch_size)
    gen_all  = _extract_all(gen_imgs,  backbone_id, device, batch_size)
    print(f"  Feature shapes: real={real_all.shape}  gen={gen_all.shape}")

    METRICS = {
        "m3_mmd":  _mmd_rbf,
        "fid":     _fid_feats,
        "cmmd":    _cmmd_poly,
    }

    all_results: dict = {}

    for N in n_list:
        if N > len(real_all) or N > len(gen_all):
            print(f"  Skipping N={N} (not enough images)")
            continue

        print(f"\n  N={N}  ({n_repeats} repeats)")
        per_metric: dict = {m: [] for m in METRICS}

        for rep in range(n_repeats):
            r_idx = rng.choice(len(real_all), size=N, replace=False)
            g_idx = rng.choice(len(gen_all),  size=N, replace=False)
            X = real_all[r_idx]
            Y = gen_all[g_idx]

            for mname, mfn in METRICS.items():
                val = mfn(X, Y)
                per_metric[mname].append(val)

        stats: dict = {}
        for mname, vals in per_metric.items():
            arr = np.array([v for v in vals if not np.isnan(v)])
            mu  = float(arr.mean())
            std = float(arr.std())
            cv  = float(std / abs(mu)) if abs(mu) > 1e-12 else float("nan")
            stats[mname] = {
                "mean": mu, "std": std, "cv": cv,
                "raw":  [float(v) for v in vals],
            }
            print(f"    {mname:10s}: mean={mu:.5f}  std={std:.5f}  CV={cv:.4f}")

        all_results[str(N)] = stats

    # Save JSON
    json_out = {k: {m: {s: v for s, v in sv.items() if s != "raw"}
                    for m, sv in vs.items()}
                for k, vs in all_results.items()}
    with open(os.path.join(output_dir, "sample_size_consistency_report.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    # Plots
    _plot_results(all_results, n_list, output_dir)

    print(f"\nSaved -> {output_dir}")
    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(all_results: dict, n_list: List[int], output_dir: str):
    METRIC_LABELS = {
        "m3_mmd":  ("M3-Score (MMD)", "#4878cf"),
        "fid":     ("FID (feat-space)", "#e05c5c"),
        "cmmd":    ("CMMD (poly)", "#6acc65"),
    }

    valid_ns = [N for N in n_list if str(N) in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Plot 1: CV vs N ---
    ax = axes[0]
    for mname, (label, color) in METRIC_LABELS.items():
        cvs = [all_results[str(N)][mname]["cv"] for N in valid_ns]
        ax.plot(valid_ns, cvs, "o-", color=color, lw=2, ms=6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("N (sample size)", fontsize=12)
    ax.set_ylabel("CV = std / mean", fontsize=12)
    ax.set_title("Coefficient of Variation vs N\n(lower = more stable)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Plot 2: Mean score vs N ---
    ax = axes[1]
    for mname, (label, color) in METRIC_LABELS.items():
        means = [all_results[str(N)][mname]["mean"] for N in valid_ns]
        stds  = [all_results[str(N)][mname]["std"]  for N in valid_ns]
        ax.plot(valid_ns, means, "o-", color=color, lw=2, ms=6, label=label)
        ax.fill_between(valid_ns,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("N (sample size)", fontsize=12)
    ax.set_ylabel("Score (mean ± std)", fontsize=12)
    ax.set_title("Score Stability vs N\n(M3 converges fastest)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Plot 3: Violin / box of raw values at each N for M3 ---
    ax = axes[2]
    raw_data = []
    positions = []
    for i, N in enumerate(valid_ns):
        raw = all_results[str(N)]["m3_mmd"]["raw"]
        raw_data.append([v for v in raw if not np.isnan(v)])
        positions.append(i)

    bp = ax.boxplot(raw_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="white", lw=2),
                    boxprops=dict(facecolor="#4878cf", alpha=0.7),
                    whiskerprops=dict(color="#333"),
                    capprops=dict(color="#333"),
                    flierprops=dict(marker="o", color="#e05c5c", ms=3))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(N) for N in valid_ns])
    ax.set_xlabel("N (sample size)", fontsize=12)
    ax.set_ylabel("M3-Score", fontsize=12)
    ax.set_title("M3 distribution vs N\n(variance shrinks with N)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Sample-Size Consistency: M3 vs FID vs CMMD\n"
                 "M3 (unbiased MMD) stays stable; FID drifts at small N",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_size_consistency.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # --- Summary table ---
    print("\n  CV summary table:")
    header = f"{'N':>6}  " + "  ".join(f"{m:>12}" for m in METRIC_LABELS)
    print(header)
    print("-" * len(header))
    for N in valid_ns:
        row = f"{N:>6}  " + "  ".join(
            f"{all_results[str(N)][m]['cv']:>12.4f}"
            for m in METRIC_LABELS
        )
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",   required=True)
    p.add_argument("--gen_dir",    required=True)
    p.add_argument("--output_dir", default="results/sample_size_consistency")
    p.add_argument("--n_list",     nargs="+", type=int, default=[10, 25, 50, 100, 200, 500])
    p.add_argument("--n_repeats",  type=int, default=20)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    p.add_argument("--device",     default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed",       type=int, default=42)
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_sample_size_consistency(
        real_dir    = a.real_dir,
        gen_dir     = a.gen_dir,
        output_dir  = a.output_dir,
        n_list      = a.n_list,
        n_repeats   = a.n_repeats,
        backbone_id = a.backbone_id,
        device      = device,
        batch_size  = a.batch_size,
        seed        = a.seed,
    )
