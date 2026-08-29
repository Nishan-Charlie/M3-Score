"""
Feature Orthogonality -- Section 3.10
======================================
Validates that the M3-Score active sub-scales carry non-redundant information
by computing the Spearman correlation matrix between per-image pairwise cosine
distances at each CKA-selected layer.

Protocol:
  1. Run CKA layer selection on a subset of real images to obtain
     active_layers (M layers, data-adaptive).
  2. Extract attention-aware embeddings for all N images at each active layer.
  3. Fix a single set of P random image-pair indices shared across all layers.
  4. For each layer, compute cosine distances over the fixed pairs,
     yielding a vector of length P.
  5. Compute the M x M Spearman correlation matrix across the P-length vectors.
  6. Low off-diagonal correlations indicate that the active layers carry
     representationally independent information.

Critical implementation note:
  The pair indices MUST be fixed before any per-layer distance computation.
  Sampling independently per layer would compare different image pairs across
  layers, rendering the inter-layer correlation meaningless.

Usage:
    python feature_orthogonality.py \\
        --real_dir <path> --gen_dir <path> --output_dir <path> \\
        --n 300 --device cuda
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import transforms
from scipy.stats import spearmanr
from tqdm.auto import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.m3_score_v2 import M3V2Metric   # corrected import path


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_images(directory: str, n: int) -> torch.Tensor:
    """Load up to n images from directory as (N, 3, 224, 224) float tensors
    in [0, 1]. Grayscale images are converted to RGB."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(paths)[:n]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imgs = []
    for p in tqdm(paths, desc=f"Loading {os.path.basename(directory)}", leave=False):
        imgs.append(transform(Image.open(p).convert("RGB")))
    return torch.stack(imgs)  # (N, 3, 224, 224) in [0, 1]


# ---------------------------------------------------------------------------
# Pairwise distance helpers
# ---------------------------------------------------------------------------

def _sample_pair_indices(N: int, max_pairs: int,
                          rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample up to max_pairs index pairs from the upper triangle of an N x N
    matrix.  The same returned (rows, cols) must be passed to every
    subsequent call to _pairwise_cosine_distances_fixed within one experiment
    so that all layers are compared on identical image pairs.
    """
    rows, cols = np.triu_indices(N, k=1)
    if len(rows) > max_pairs:
        choice = rng.choice(len(rows), max_pairs, replace=False)
        rows = rows[choice]
        cols = cols[choice]
    return rows, cols


def _pairwise_cosine_distances_fixed(feats: torch.Tensor,
                                      rows: np.ndarray,
                                      cols: np.ndarray) -> np.ndarray:
    """
    Compute cosine distances for a fixed set of image-pair indices.

    Args:
        feats: (N, D) embedding matrix.
        rows:  First index of each pair.
        cols:  Second index of each pair.

    Returns:
        Array of shape (P,) where P = len(rows).
    """
    f = feats.float()
    norms = f.norm(dim=1, keepdim=True).clamp(min=1e-8)
    f = f / norms
    fi = f[rows]   # (P, D)
    fj = f[cols]   # (P, D)
    cos_sim = (fi * fj).sum(dim=1).cpu().numpy()
    return 1.0 - cos_sim


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def _correlation_matrix(dist_vectors: list[np.ndarray],
                         scale_names: list[str]) -> np.ndarray:
    """
    Compute the M x M Spearman rank correlation matrix from a list of M
    distance vectors, all of the same length and computed on the same
    image pairs.
    """
    n = len(scale_names)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(dist_vectors[i], dist_vectors[j])
            mat[i, j] = mat[j, i] = float(r)
    return mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_corr_heatmap(corr_mat: np.ndarray, scale_names: list[str],
                       title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(corr_mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(range(len(scale_names)))
    ax.set_yticks(range(len(scale_names)))
    ax.set_xticklabels(scale_names, fontsize=10)
    ax.set_yticklabels(scale_names, fontsize=10)
    for i in range(len(scale_names)):
        for j in range(len(scale_names)):
            ax.text(j, i, f"{corr_mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if corr_mat[i, j] > 0.7 else "black")
    plt.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_summary_bar(off_diag: list[dict], output_path: str) -> None:
    """Grouped bar chart showing r_real and r_gen for each layer pair."""
    pairs      = [d["pair"]   for d in off_diag]
    r_real_arr = [d["r_real"] for d in off_diag]
    r_gen_arr  = [d["r_gen"]  for d in off_diag]
    x = np.arange(len(pairs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.4), 4), dpi=120)
    bars_r = ax.bar(x - width / 2, r_real_arr, width,
                    label="Real",      color="#4fc3f7", edgecolor="#333")
    bars_g = ax.bar(x + width / 2, r_gen_arr,  width,
                    label="Generated", color="#ff8a65", edgecolor="#333")

    for bar in bars_r:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_g:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="black", lw=0.8, linestyle="--")
    ax.axhline(0.5, color="gray",  lw=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Spearman r")
    ax.set_ylim(0, 1.15)
    ax.set_title("M3-Score: Inter-Layer Spearman Correlation\n"
                 "(lower values indicate more independent information per layer)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _analyze_backbone(metric: M3V2Metric,
                       imgs_real: torch.Tensor,
                       imgs_gen: torch.Tensor,
                       active_layers: list[int],
                       max_pairs: int,
                       rng: np.random.RandomState) -> dict:
    """
    Extract attention-aware embeddings at all active layers, then compute
    pairwise cosine distances over a FIXED set of image pairs shared across
    all layers.  The fixed pairs are essential: using independently sampled
    pairs per layer would compare different image pairs across layers, making
    the resulting Spearman correlation statistically meaningless.

    Args:
        metric:        Initialised M3V2Metric with pruned active_layers.
        imgs_real:     (N, 3, 224, 224) real image tensor.
        imgs_gen:      (N, 3, 224, 224) generated image tensor.
        active_layers: List of 1-indexed active layer indices.
        max_pairs:     Maximum number of image pairs to sample.
        rng:           Seeded numpy RandomState for reproducible pair sampling.

    Returns:
        Dict with scale_names, corr_real, corr_gen, off_diagonal_pairs.
    """
    active_set = set(active_layers)
    scale_names = [f"L{l}" for l in active_layers]
    N = imgs_real.shape[0]

    # Fix pairs ONCE before any per-layer computation
    rows, cols = _sample_pair_indices(N, max_pairs, rng)
    print(f"  Using {len(rows):,} fixed image pairs for correlation.")

    # Extract features for both sets inside a single no_grad context
    print(f"  Extracting features (real) ...")
    with torch.no_grad():
        feats_real = metric._extract_raw_features(
            imgs_real, use_attention=True, layers_to_keep=active_set)

    print(f"  Extracting features (generated) ...")
    with torch.no_grad():
        feats_gen = metric._extract_raw_features(
            imgs_gen, use_attention=True, layers_to_keep=active_set)

    # Index into the flat list (0-indexed internally, 1-indexed externally)
    active_real = [feats_real[i - 1] for i in active_layers]
    active_gen  = [feats_gen[i - 1]  for i in active_layers]

    # Compute distances using the SAME fixed pairs for every layer
    print(f"  Computing pairwise cosine distances (real, {len(active_layers)} layers) ...")
    dist_real = [_pairwise_cosine_distances_fixed(f.cpu(), rows, cols)
                 for f in active_real]

    print(f"  Computing pairwise cosine distances (generated, {len(active_layers)} layers) ...")
    dist_gen  = [_pairwise_cosine_distances_fixed(f.cpu(), rows, cols)
                 for f in active_gen]

    corr_real = _correlation_matrix(dist_real, scale_names)
    corr_gen  = _correlation_matrix(dist_gen,  scale_names)

    print(f"\n  Spearman correlation matrix (real images):")
    header = "  " + "".join(f"{s:>12}" for s in scale_names)
    print(header)
    for i, sn in enumerate(scale_names):
        row_str = "".join(f"{corr_real[i, j]:+12.4f}" for j in range(len(scale_names)))
        print(f"  {sn:<8} {row_str}")

    off_diag = []
    for i in range(len(scale_names)):
        for j in range(i + 1, len(scale_names)):
            off_diag.append({
                "pair":   f"{scale_names[i]}-{scale_names[j]}",
                "r_real": round(float(corr_real[i, j]), 4),
                "r_gen":  round(float(corr_gen[i, j]),  4),
            })

    return {
        "scale_names":        scale_names,
        "corr_real":          corr_real.tolist(),
        "corr_gen":           corr_gen.tolist(),
        "off_diagonal_pairs": off_diag,
        "num_pairs_used":     int(len(rows)),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_feature_orthogonality(
    real_dir:   str,
    gen_dir:    str,
    output_dir: str  = "./orthogonality_output",
    n:          int  = 300,
    max_pairs:  int  = 50_000,
    device:     str  = None,
    seed:       int  = 42,
) -> dict:
    """
    Run the feature orthogonality experiment (Section 3.10).

    Args:
        real_dir:   Directory of real images.
        gen_dir:    Directory of generated images.
        output_dir: Directory for output plots and JSON report.
        n:          Number of images to load per set.
        max_pairs:  Maximum image pairs for pairwise distance computation.
        device:     Torch device string.
        seed:       Random seed for reproducible pair sampling.

    Returns:
        Dict containing correlation matrices and off-diagonal pair statistics,
        compatible with master_report["orthogonality"] in run_experiments.py.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    print("Loading images ...")
    imgs_real = _load_images(real_dir, n)
    imgs_gen  = _load_images(gen_dir,  n)

    print(f"\nInitialising M3-Score (device={device}) and running CKA layer selection ...")
    metric = M3V2Metric(device=device)
    metric.prune_layers_via_cka(imgs_real[:20])
    active_layers = metric.active_layers
    print(f"  Active layers: {active_layers}")

    res = _analyze_backbone(metric, imgs_real, imgs_gen, active_layers,
                            max_pairs, rng)
    results = {"M3-Score": res}

    # Heatmaps
    scale_names = res["scale_names"]
    corr_real = np.array(res["corr_real"])
    corr_gen  = np.array(res["corr_gen"])
    _plot_corr_heatmap(
        corr_real, scale_names,
        "M3-Score (RadioDino): Active-Layer Correlation (Real)",
        os.path.join(output_dir, "orthogonality_real.png"),
    )
    _plot_corr_heatmap(
        corr_gen, scale_names,
        "M3-Score (RadioDino): Active-Layer Correlation (Generated)",
        os.path.join(output_dir, "orthogonality_gen.png"),
    )

    # Grouped bar chart (real and generated side by side)
    summary_path = os.path.join(output_dir, "orthogonality_summary.png")
    _plot_summary_bar(res["off_diagonal_pairs"], summary_path)

    print(f"\n{'='*60}")
    print("  FEATURE ORTHOGONALITY SUMMARY")
    print(f"{'='*60}")
    for d in res["off_diagonal_pairs"]:
        print(f"  {d['pair']:20}  r_real={d['r_real']:+.4f}  r_gen={d['r_gen']:+.4f}")

    results["plots"] = {
        "real_heatmap": os.path.join(output_dir, "orthogonality_real.png"),
        "gen_heatmap":  os.path.join(output_dir, "orthogonality_gen.png"),
        "summary_bar":  summary_path,
    }
    report_path = os.path.join(output_dir, "orthogonality_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nReport saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature orthogonality experiment")
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="./orthogonality_output")
    parser.add_argument("--n",          type=int, default=300)
    parser.add_argument("--max_pairs",  type=int, default=50_000)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run_feature_orthogonality(
        real_dir   = args.real_dir,
        gen_dir    = args.gen_dir,
        output_dir = args.output_dir,
        n          = args.n,
        max_pairs  = args.max_pairs,
        device     = args.device,
        seed       = args.seed,
    )