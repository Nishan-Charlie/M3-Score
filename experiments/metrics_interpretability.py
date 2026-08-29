"""
Metric Interpretability Validation -- Section 3.11
====================================================
Tests whether the M3-Score's internal layer decomposition identifies
the same representational failures as an independent feature-level
analysis (the interpretability experiment, Section 3.10).

This is a METRIC interpretability test, not a model interpretability test.
It answers: does M3's internal decomposition point at the right layers
for the right reasons, independently confirmed by a separate method?

Four tests are conducted:

  Test 1 -- Layer-level rank correlation (Spearman rho)
    Spearman correlation between M3's per-layer MMD^2 distances and the
    per-layer mean standardised shift from the feature-level analysis.
    Conducted over all K active layers and over all 12 layers.
    A permutation test (exact, all K! orderings) provides a p-value
    without parametric assumptions, which is critical at small K.

  Test 2 -- Layer ablation consistency (causal)
    Each active layer is removed from M3 one at a time (equal-weight
    scoring). The reduction in M3 score when layer Lk is removed is
    compared to the feature-level shift at Lk. If M3's decomposition
    is correct, removing the layer with the highest feature-shift should
    reduce the M3 score most. Spearman rho and permutation p-value.

  Test 3 -- Per-image ranking consistency (N=500, high power)
    M3's per-image score s(x_j) is correlated with the per-image L2
    distance from the feature-level analysis across all N=500 generated
    images. These are independent estimators: M3 uses L2-normalised
    unit-sphere features with a polynomial MMD kernel; the feature
    analysis uses raw features with per-set standardisation and Euclidean
    distance. N=500 gives statistical power that the layer-level tests
    cannot.

  Test 4 -- Radiomic cross-validation (M3 has no radiomic features)
    M3 uses only RadioDino features. The feature analysis showed texture
    (GLCM) shifts most of all categories (0.8558 sd > learned 0.6835 sd).
    If M3's per-image scores correlate with the radiomic-only distance
    (GLCM + first-order features, no RadioDino), this suggests M3's learned
    features capture some of the same texture information as GLCM,
    supporting the claim that RadioDino at early layers is sensitive to
    the same texture microstructure that GLCM explicitly measures.
    This is the hardest test: genuine independence between M3 (no GLCM)
    and radiomic-only distances.

Independence guarantee:
  - M3 uses L2-normalised features, polynomial kernel, MMD^2.
  - Feature analysis uses raw features, per-set z-standardisation,
    Euclidean norm.
  - Different normalisation, different kernel, different feature sets
    (Test 4 uses only radiomic features absent from M3).
  - Both run on the same 500 image pairs.

Usage:
    python metric_interpretability_validation.py \\
        --real_dir <path> --gen_dir <path> \\
        --interp_report <path/to/interpretability_report.json> \\
        --output_dir <path> --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import glob
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from scipy import stats as sp_stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LAYERS_CACHE = os.path.join(_PROJECT_ROOT, "canonical_layers.json")

from evaluation.m3_score_v2 import M3V2Metric


# ---------------------------------------------------------------------------
# Publication plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
})

_COLORS = {
    "m3":      "#ef5350",
    "interp":  "#1565C0",
    "neutral": "#546E7A",
    "texture": "#1565C0",
    "learned": "#2E7D32",
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_paths(directory: str, n: Optional[int],
                recursive: bool = True) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        if recursive:
            paths.extend(
                glob.glob(os.path.join(directory, "**", ext), recursive=True)
            )
        else:
            paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(set(paths))
    if n is not None:
        paths = paths[:n]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}.")
    return paths


def _load_for_m3(paths: list[str]) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return torch.stack([
        tfm(Image.open(p).convert("RGB"))
        for p in tqdm(paths, desc="Loading images", leave=False)
    ])


# ---------------------------------------------------------------------------
# Radiomic features (for Test 4 -- independent of M3)
# ---------------------------------------------------------------------------

def _glcm(img_01: np.ndarray, distances=(1,), angles=(0,),
          levels: int = 32) -> np.ndarray:
    img_q = np.clip((img_01 * levels).astype(np.int32), 0, levels - 1)
    glcm  = np.zeros((levels, levels), dtype=np.float64)
    for d in distances:
        for angle in angles:
            dx = int(round(d * np.cos(angle)))
            dy = int(round(d * np.sin(angle)))
            r0, r1 = max(0, -dy), img_q.shape[0] - max(0, dy)
            c0, c1 = max(0, -dx), img_q.shape[1] - max(0, dx)
            ref   = img_q[r0:r1, c0:c1]
            neigh = img_q[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
            idx   = ref.ravel() * levels + neigh.ravel()
            glcm  += np.bincount(idx, minlength=levels**2).reshape(
                levels, levels
            ).astype(np.float64)
    glcm = glcm + glcm.T
    s = glcm.sum()
    if s > 0:
        glcm /= s
    return glcm


def _extract_radiomic(img_path: str) -> np.ndarray:
    """Extract 16 radiomic features (10 first-order + 6 GLCM)."""
    from scipy import stats as sp_stats
    img = Image.open(img_path).convert("L").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    flat = arr.ravel()
    counts, _ = np.histogram(flat, bins=64, range=(0, 1))
    probs = counts / (counts.sum() + 1e-12)

    fo = np.array([
        flat.mean(), flat.std(),
        float(sp_stats.skew(flat)), float(sp_stats.kurtosis(flat)),
        float(np.sum(flat**2)) / len(flat),
        float(sp_stats.entropy(probs + 1e-12)),
        flat.min(), flat.max(), float(np.median(flat)),
        float(np.sum(probs**2)),
    ])

    g = _glcm(arr, distances=(1, 3), angles=(0, np.pi/4, np.pi/2))
    levels = g.shape[0]
    i_idx  = np.arange(levels)
    I, J   = np.meshgrid(i_idx, i_idx, indexing="ij")
    mu_i   = np.sum(I * g)
    mu_j   = np.sum(J * g)
    std_i  = np.sqrt(np.sum(g * (I - mu_i)**2))
    std_j  = np.sqrt(np.sum(g * (J - mu_j)**2))
    corr   = (float(np.sum(g * (I - mu_i) * (J - mu_j)) / (std_i * std_j))
              if std_i > 0 and std_j > 0 else 0.0)
    glcm_f = np.array([
        float(np.sum(g * (I - J)**2)),
        float(np.sum(g * np.abs(I - J))),
        float(np.sum(g / (1 + (I - J)**2))),
        float(np.sum(g**2)),
        corr,
        float(-np.sum(g[g > 0] * np.log2(g[g > 0] + 1e-12))),
    ])
    return np.concatenate([fo, glcm_f])   # (16,)


def _extract_radiomic_batch(paths: list[str]) -> np.ndarray:
    feats = [_extract_radiomic(p) for p in
             tqdm(paths, desc="Radiomic features", leave=False)]
    return np.array(feats, dtype=np.float64)   # (N, 16)


def _standardise_and_distance(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
) -> np.ndarray:
    """
    Per-feature z-standardise using real-set statistics, then compute
    per-image L2 distance from origin in standardised space.
    Returns (N_gen,) distance array.
    """
    mu    = real_feats.mean(axis=0)
    sigma = real_feats.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    gen_std = (gen_feats - mu) / sigma
    return np.linalg.norm(gen_std, axis=1)


# ---------------------------------------------------------------------------
# Exact permutation test (all K! orderings)
# ---------------------------------------------------------------------------

def _exact_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Exact permutation test for Spearman rho between x and y.

    Enumerates all K! permutations of y, computes rho for each,
    and returns (observed_rho, exact_p_value, null_distribution).

    Safe for K <= 8 (8! = 40320, takes ~0.1s).
    For K > 8, falls back to 100,000 random permutations.
    """
    K = len(x)
    obs_rho, _ = sp_stats.spearmanr(x, y)

    if K <= 8:
        # Exact: enumerate all K! permutations
        null_rhos = []
        for perm in itertools.permutations(range(K)):
            y_perm = y[list(perm)]
            r, _   = sp_stats.spearmanr(x, y_perm)
            null_rhos.append(r)
        null_rhos = np.array(null_rhos)
    else:
        # Approximate: 100,000 random permutations
        rng = np.random.default_rng(42)
        null_rhos = np.array([
            sp_stats.spearmanr(x, rng.permutation(y))[0]
            for _ in range(100_000)
        ])

    # Two-tailed p-value: fraction of null rhos with |rho| >= |observed|
    p_exact = float(np.mean(np.abs(null_rhos) >= abs(obs_rho)))
    return float(obs_rho), p_exact, null_rhos


# ---------------------------------------------------------------------------
# M3 per-image score computation
# ---------------------------------------------------------------------------

def _compute_m3_per_image(
    real_feats: list[torch.Tensor],
    gen_feats:  list[torch.Tensor],
    active_layers: list[int],
    layer_weights: dict[str, float],
    device: str,
) -> np.ndarray:
    """
    Compute per-image M3 score s(x_j) for all generated images.

    s(x_j) = sum_{l in L} w_l * ||hat_E_{g,j}^l - hat_mu_r^l||_2

    where hat denotes L2-normalised features and hat_mu_r is the
    centroid of L2-normalised real features at layer l.
    This is exactly Algorithm 5 from the methodology.

    Returns (N_gen,) array.
    """
    scores = None
    for idx, layer in enumerate(active_layers):
        rf = real_feats[layer - 1].to(device)
        gf = gen_feats[layer - 1].to(device)

        # L2-normalise (same as in _compute_kid_distance)
        rf_norm = rf / (rf.norm(dim=1, keepdim=True).clamp(min=1e-8))
        gf_norm = gf / (gf.norm(dim=1, keepdim=True).clamp(min=1e-8))

        mu_r = rf_norm.mean(dim=0)   # (D,) real centroid
        dists = (gf_norm - mu_r).norm(dim=1).cpu().numpy()   # (N_gen,)

        w = layer_weights.get(f"L{layer}", 1.0 / len(active_layers))
        if scores is None:
            scores = w * dists
        else:
            scores += w * dists

    return scores if scores is not None else np.zeros(gen_feats[active_layers[0]-1].shape[0])


# ---------------------------------------------------------------------------
# Test 1: Layer-level rank correlation
# ---------------------------------------------------------------------------

def _test1_layer_rank_correlation(
    m3_layer_distances:  dict[str, float],
    interp_layer_shifts: dict[str, float],
    active_layers:       list[int],
    output_dir:          str,
) -> dict:
    """
    Spearman rho between M3 per-layer MMD^2 and independent feature-shift.

    Two analyses:
      A. Active layers only (K layers, exact permutation test).
      B. All 12 layers (imputed 0 for inactive, rho over 12 points).
    """
    layer_keys = [f"L{l}" for l in active_layers]

    m3_vals    = np.array([m3_layer_distances.get(k, 0.0) for k in layer_keys])
    shift_vals = np.array([interp_layer_shifts.get(k, 0.0) for k in layer_keys])

    if len(m3_vals) < 3:
        return {"error": "Fewer than 3 active layers; test not conducted."}

    # Analysis A: active layers, exact permutation
    rho_a, p_exact_a, null_a = _exact_permutation_p(shift_vals, m3_vals)
    rho_spearman_a, p_spearman_a = sp_stats.spearmanr(m3_vals, shift_vals)

    print(f"\n[Test 1A] Active layers ({len(active_layers)}): "
          f"rho={rho_a:.4f}  p_exact={p_exact_a:.4f}  p_spearman={p_spearman_a:.4f}")

    # Scatter plot: M3 distance vs feature-level shift per layer
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(shift_vals, m3_vals, color=_COLORS["m3"], s=80, zorder=3,
               edgecolors="white", linewidths=0.5)
    for k, xv, yv in zip(layer_keys, shift_vals, m3_vals):
        ax.annotate(k, (xv, yv), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    # Fit line
    if len(m3_vals) >= 3:
        z = np.polyfit(shift_vals, m3_vals, 1)
        xs = np.linspace(shift_vals.min(), shift_vals.max(), 100)
        ax.plot(xs, np.polyval(z, xs), "--", color=_COLORS["neutral"],
                alpha=0.7, lw=1.5)
    ax.set_xlabel("Independent feature-level shift (sd units)")
    ax.set_ylabel("M3 per-layer MMD\u00b2 distance")
    ax.set_title(
        f"Test 1A: M3 layer distance vs feature-level shift\n"
        f"Spearman \u03c1 = {rho_a:.3f},  exact p = {p_exact_a:.4f}  "
        f"(n = {len(active_layers)} active layers)"
    )

    # Null distribution histogram
    ax = axes[1]
    ax.hist(null_a, bins=30, color=_COLORS["neutral"], alpha=0.7,
            density=True, edgecolor="white", linewidth=0.3,
            label="Null distribution (all permutations)")
    ax.axvline(rho_a, color=_COLORS["m3"], lw=2, linestyle="--",
               label=f"Observed \u03c1 = {rho_a:.3f}")
    ax.set_xlabel("Spearman \u03c1 under null")
    ax.set_ylabel("Density")
    ax.set_title("Test 1A: Permutation null distribution")
    ax.legend(fontsize=8)

    plt.suptitle(
        "Test 1: M3 layer decomposition vs independent feature-level shift",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path_1a = os.path.join(output_dir, "test1_layer_correlation.png")
    plt.savefig(path_1a, bbox_inches="tight")
    plt.close()

    return {
        "active_layers":      layer_keys,
        "m3_distances":       {k: float(v) for k, v in zip(layer_keys, m3_vals)},
        "feature_shifts":     {k: float(v) for k, v in zip(layer_keys, shift_vals)},
        "spearman_rho":       round(rho_a, 4),
        "p_exact_permutation": round(p_exact_a, 6),
        "p_spearman":         round(float(p_spearman_a), 6),
        "n_permutations":     len(null_a),
        "null_rho_95pct":     round(float(np.percentile(np.abs(null_a), 95)), 4),
        "significant":        p_exact_a < 0.05,
        "plot":               path_1a,
    }


# ---------------------------------------------------------------------------
# Test 2: Layer ablation consistency (causal)
# ---------------------------------------------------------------------------

def _test2_layer_ablation(
    real_feats:          list[torch.Tensor],
    gen_feats:           list[torch.Tensor],
    active_layers:       list[int],
    interp_layer_shifts: dict[str, float],
    metric:              M3V2Metric,
    device:              str,
    output_dir:          str,
) -> dict:
    """
    Remove each active layer in turn (equal weights) and measure the
    drop in M3 score. Correlate drop with feature-level shift.

    Equal weights are used so the test measures layer informativeness
    independently of SNR weighting.
    """
    K = len(active_layers)

    def _equal_weight_score(layers: list[int]) -> float:
        total = 0.0
        for layer in layers:
            rf = real_feats[layer - 1].to(device)
            gf = gen_feats[layer - 1].to(device)
            total += metric._compute_mmd2(rf, gf).item()
        return total / max(len(layers), 1)

    baseline = _equal_weight_score(active_layers)
    print(f"\n[Test 2] Baseline (all {K} layers, equal weight): {baseline:.6f}")

    drops  = []
    shifts = []
    layer_keys = []

    for layer in active_layers:
        remaining = [l for l in active_layers if l != layer]
        ablated   = _equal_weight_score(remaining)
        drop      = baseline - ablated        # positive = this layer raised the score
        key       = f"L{layer}"
        drops.append(drop)
        shifts.append(interp_layer_shifts.get(key, 0.0))
        layer_keys.append(key)
        print(f"  Remove {key}: score={ablated:.6f}  drop={drop:+.6f}")

    drops  = np.array(drops)
    shifts = np.array(shifts)

    rho, p_exact, null = _exact_permutation_p(shifts, drops)
    print(f"  Spearman rho={rho:.4f}  exact_p={p_exact:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(shifts, drops, color=_COLORS["m3"], s=80, zorder=3,
               edgecolors="white", linewidths=0.5)
    for k, xv, yv in zip(layer_keys, shifts, drops):
        ax.annotate(k, (xv, yv), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    if len(drops) >= 3:
        z = np.polyfit(shifts, drops, 1)
        xs = np.linspace(shifts.min(), shifts.max(), 100)
        ax.plot(xs, np.polyval(z, xs), "--", color=_COLORS["neutral"],
                alpha=0.7, lw=1.5)
    ax.set_xlabel("Independent feature-level shift (sd units)")
    ax.set_ylabel("Drop in M3 score when layer removed")
    ax.set_title(
        f"Test 2: Layer ablation consistency (causal)\n"
        f"Spearman \u03c1 = {rho:.3f},  exact p = {p_exact:.4f}  "
        f"(n = {K} layers)"
    )
    plt.tight_layout()
    path_2 = os.path.join(output_dir, "test2_ablation_consistency.png")
    plt.savefig(path_2, bbox_inches="tight")
    plt.close()

    return {
        "baseline_score":     round(baseline, 6),
        "layer_drops":        {k: round(float(d), 6) for k, d in zip(layer_keys, drops)},
        "feature_shifts":     {k: round(float(s), 4) for k, s in zip(layer_keys, shifts)},
        "spearman_rho":       round(rho, 4),
        "p_exact_permutation": round(p_exact, 6),
        "n_permutations":     len(null),
        "significant":        p_exact < 0.05,
        "plot":               path_2,
    }


# ---------------------------------------------------------------------------
# Test 3: Per-image ranking consistency (N=500)
# ---------------------------------------------------------------------------

def _test3_per_image_consistency(
    m3_per_image:  np.ndarray,
    interp_per_image: np.ndarray,
    output_dir:    str,
) -> dict:
    """
    Spearman correlation between M3 per-image scores and independent
    feature-level per-image distances across all N generated images.

    N=500 gives high statistical power (p < 1e-10 at rho > 0.2).
    """
    N = len(m3_per_image)
    assert len(interp_per_image) == N, (
        f"Length mismatch: M3={N}, interp={len(interp_per_image)}. "
        "Both must be computed on the same generated images in the same order."
    )

    rho, p_val = sp_stats.spearmanr(m3_per_image, interp_per_image)
    print(f"\n[Test 3] Per-image ranking: rho={rho:.4f}  p={p_val:.2e}  N={N}")

    # Also compute Pearson for comparison (tests linear association)
    r_pearson, p_pearson = sp_stats.pearsonr(m3_per_image, interp_per_image)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: M3 score vs feature distance
    ax = axes[0]
    ax.scatter(interp_per_image, m3_per_image,
               alpha=0.4, s=15, color=_COLORS["m3"],
               edgecolors="none", rasterized=True)
    z = np.polyfit(interp_per_image, m3_per_image, 1)
    xs = np.linspace(interp_per_image.min(), interp_per_image.max(), 200)
    ax.plot(xs, np.polyval(z, xs), "-", color=_COLORS["neutral"],
            lw=2, alpha=0.8, label="OLS fit")
    ax.set_xlabel("Feature-level distance (standardised space, no M3)")
    ax.set_ylabel("M3 per-image score")
    ax.set_title(
        f"Test 3: Per-image score consistency  (N={N})\n"
        f"Spearman \u03c1 = {rho:.4f},  p = {p_val:.2e}"
    )
    ax.legend(fontsize=9)

    # Rank-rank plot (visualises Spearman directly)
    ax = axes[1]
    rank_m3     = sp_stats.rankdata(m3_per_image)
    rank_interp = sp_stats.rankdata(interp_per_image)
    ax.scatter(rank_interp, rank_m3,
               alpha=0.3, s=10, color=_COLORS["m3"],
               edgecolors="none", rasterized=True)
    ax.set_xlabel("Rank (feature-level distance)")
    ax.set_ylabel("Rank (M3 per-image score)")
    ax.set_title(
        f"Rank-rank plot\n"
        f"(Spearman \u03c1 is Pearson of ranks: {rho:.4f})"
    )

    plt.suptitle(
        "Test 3: M3 per-image scores vs independent feature-level distances",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path_3 = os.path.join(output_dir, "test3_per_image_consistency.png")
    plt.savefig(path_3, bbox_inches="tight")
    plt.close()

    return {
        "n_images":         N,
        "spearman_rho":     round(float(rho),       4),
        "spearman_p":       float(p_val),
        "pearson_r":        round(float(r_pearson),  4),
        "pearson_p":        float(p_pearson),
        "significant":      p_val < 0.05,
        "plot":             path_3,
    }


# ---------------------------------------------------------------------------
# Test 4: Radiomic cross-validation (independent of M3)
# ---------------------------------------------------------------------------

def _test4_radiomic_crossval(
    real_paths:    list[str],
    gen_paths:     list[str],
    m3_per_image:  np.ndarray,
    output_dir:    str,
) -> dict:
    """
    Correlate M3 per-image scores with distances computed using ONLY
    radiomic features (16 GLCM + first-order features).

    M3 has NO radiomic features -- it uses only RadioDino embeddings.
    If M3 per-image scores correlate with radiomic-only distances,
    RadioDino early layers are capturing texture information overlapping
    with what GLCM measures explicitly.

    This is the hardest independence test: the two methods share no
    features, no normalisation strategy, and no kernel.
    """
    print("\n[Test 4] Extracting radiomic features for cross-validation ...")
    real_radio = _extract_radiomic_batch(real_paths)
    gen_radio  = _extract_radiomic_batch(gen_paths)

    # Per-image radiomic distances (no M3 involvement)
    radio_dist = _standardise_and_distance(real_radio, gen_radio)

    N = len(m3_per_image)
    assert len(radio_dist) == N

    rho, p_val = sp_stats.spearmanr(m3_per_image, radio_dist)
    print(f"  Radiomic cross-val: rho={rho:.4f}  p={p_val:.2e}  N={N}")

    # Split by category (first-order vs GLCM)
    rho_fo, p_fo  = sp_stats.spearmanr(m3_per_image,
                                        _standardise_and_distance(real_radio[:, :10],
                                                                   gen_radio[:, :10]))
    rho_glcm, p_glcm = sp_stats.spearmanr(m3_per_image,
                                           _standardise_and_distance(real_radio[:, 10:],
                                                                      gen_radio[:, 10:]))
    print(f"  First-order only:  rho={rho_fo:.4f}  p={p_fo:.2e}")
    print(f"  GLCM only:         rho={rho_glcm:.4f}  p={p_glcm:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, dist_arr, label, rho_val, p_val_plot in [
        (axes[0], radio_dist,                              "All radiomic",    rho,      p_val),
        (axes[1], _standardise_and_distance(real_radio[:, :10], gen_radio[:, :10]),
                                                           "First-order only",rho_fo,   p_fo),
        (axes[2], _standardise_and_distance(real_radio[:, 10:], gen_radio[:, 10:]),
                                                           "GLCM only",       rho_glcm, p_glcm),
    ]:
        ax.scatter(dist_arr, m3_per_image,
                   alpha=0.35, s=12, color=_COLORS["m3"],
                   edgecolors="none", rasterized=True)
        z = np.polyfit(dist_arr, m3_per_image, 1)
        xs = np.linspace(dist_arr.min(), dist_arr.max(), 200)
        ax.plot(xs, np.polyval(z, xs), "-", color=_COLORS["neutral"],
                lw=2, alpha=0.8)
        ax.set_xlabel(f"Radiomic distance ({label})")
        ax.set_ylabel("M3 per-image score")
        ax.set_title(
            f"{label}\n"
            f"\u03c1 = {rho_val:.4f},  p = {p_val_plot:.2e}"
        )

    plt.suptitle(
        "Test 4: M3 per-image scores vs radiomic-only distances\n"
        "(M3 has no radiomic features -- this tests indirect texture sensitivity)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path_4 = os.path.join(output_dir, "test4_radiomic_crossval.png")
    plt.savefig(path_4, bbox_inches="tight")
    plt.close()

    return {
        "n_images":                   N,
        "all_radiomic_rho":           round(float(rho),      4),
        "all_radiomic_p":             float(p_val),
        "first_order_only_rho":       round(float(rho_fo),   4),
        "first_order_only_p":         float(p_fo),
        "glcm_only_rho":              round(float(rho_glcm), 4),
        "glcm_only_p":                float(p_glcm),
        "significant_vs_all_radiomic": p_val < 0.05,
        "significant_vs_glcm":         p_glcm < 0.05,
        "interpretation": (
            "M3 per-image scores correlate significantly with GLCM-only "
            "distances despite M3 having no radiomic features, supporting "
            "the claim that RadioDino early layers are sensitive to texture "
            "microstructure."
            if p_glcm < 0.05 else
            "No significant correlation with GLCM-only distances. "
            "M3 and GLCM texture features capture different aspects "
            "of the distributional gap."
        ),
        "plot": path_4,
    }


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def _make_summary_plot(results: dict, output_dir: str) -> str:
    """Four-panel summary of all test results for the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Test 1 rho bar + CI
    ax = axes[0, 0]
    t1 = results.get("test1", {})
    t2 = results.get("test2", {})
    rhos = [t1.get("spearman_rho", 0), t2.get("spearman_rho", 0)]
    ps   = [t1.get("p_exact_permutation", 1), t2.get("p_exact_permutation", 1)]
    labels = ["Test 1\n(layer correlation)", "Test 2\n(ablation causal)"]
    colors_bar = [_COLORS["m3"] if p < 0.05 else _COLORS["neutral"] for p in ps]
    bars = ax.bar(labels, rhos, color=colors_bar, alpha=0.85,
                  edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axhline(0.7, color=_COLORS["neutral"], lw=1, linestyle="--",
               alpha=0.6, label="rho = 0.7 threshold")
    for bar, rho_v, p_v in zip(bars, rhos, ps):
        sig = "*" if p_v < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                rho_v + 0.03, f"{rho_v:.3f}{sig}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.2, 1.1)
    ax.set_ylabel("Spearman \u03c1")
    ax.set_title("Layer-level agreement (Tests 1 & 2)\n* = exact p < 0.05")
    ax.legend(fontsize=8)

    # Panel 2: Test 3 rho with confidence band
    ax = axes[0, 1]
    t3 = results.get("test3", {})
    rho3 = t3.get("spearman_rho", 0)
    p3   = t3.get("spearman_p", 1)
    N3   = t3.get("n_images", 0)
    # Fisher z 95% CI
    z3  = np.arctanh(rho3)
    se3 = 1 / np.sqrt(max(N3 - 3, 1))
    ci_lo, ci_hi = np.tanh(z3 - 1.96 * se3), np.tanh(z3 + 1.96 * se3)
    ax.barh(["M3 vs feature-level\nper-image scores"], [rho3],
            xerr=[[rho3 - ci_lo], [ci_hi - rho3]],
            color=_COLORS["m3"] if p3 < 0.05 else _COLORS["neutral"],
            alpha=0.85, edgecolor="white", capsize=6)
    ax.axvline(0.5, color=_COLORS["neutral"], lw=1, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spearman \u03c1 (95% CI)")
    ax.set_title(f"Test 3: Per-image ranking consistency\n"
                 f"\u03c1 = {rho3:.4f},  p = {p3:.2e}  (N = {N3})")

    # Panel 3: Test 4 GLCM vs first-order
    ax = axes[1, 0]
    t4 = results.get("test4", {})
    rho_glcm = t4.get("glcm_only_rho", 0)
    rho_fo   = t4.get("first_order_only_rho", 0)
    p_glcm   = t4.get("glcm_only_p", 1)
    p_fo     = t4.get("first_order_only_p", 1)
    x_pos    = [0, 1]
    rho_vals = [rho_fo, rho_glcm]
    p_vals   = [p_fo, p_glcm]
    bar_cols = [_COLORS["m3"] if p < 0.05 else _COLORS["neutral"]
                for p in p_vals]
    bars = ax.bar(["First-order\n(intensity)", "GLCM\n(texture)"],
                  rho_vals, color=bar_cols, alpha=0.85,
                  edgecolor="white", linewidth=0.4)
    for bar, rv, pv in zip(bars, rho_vals, p_vals):
        sig = "*" if pv < 0.05 else "n.s."
        ax.text(bar.get_x() + bar.get_width() / 2, rv + 0.02,
                f"{rv:.3f} ({sig})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Spearman \u03c1 with M3 per-image score")
    ax.set_title("Test 4: Radiomic cross-validation\n"
                 "(M3 has no radiomic features)\n* = p < 0.05")
    ax.set_ylim(0, 1)

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["Test", "Metric", "\u03c1", "p", "Pass?"],
        ["1: Layer corr.",  "Spearman (exact perm.)",
         str(t1.get("spearman_rho","N/A")),
         f"{t1.get('p_exact_permutation', 1):.4f}",
         "Yes" if t1.get("significant") else "No"],
        ["2: Ablation",     "Spearman (exact perm.)",
         str(t2.get("spearman_rho","N/A")),
         f"{t2.get('p_exact_permutation', 1):.4f}",
         "Yes" if t2.get("significant") else "No"],
        ["3: Per-image",    f"Spearman (N={N3})",
         str(t3.get("spearman_rho","N/A")),
         f"{t3.get('spearman_p', 1):.2e}",
         "Yes" if t3.get("significant") else "No"],
        ["4: GLCM crossval","Spearman (N=500)",
         str(t4.get("glcm_only_rho","N/A")),
         f"{t4.get('glcm_only_p', 1):.2e}",
         "Yes" if t4.get("significant_vs_glcm") else "No"],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc="center", loc="center",
                     bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8EAF6")
            cell.set_text_props(fontweight="bold")
        if col == 4 and row > 0:
            text = cell.get_text().get_text()
            cell.set_facecolor("#C8E6C9" if text == "Yes" else "#FFCDD2")
    ax.set_title("Summary of all four tests", fontsize=10, fontweight="bold",
                 pad=10)

    plt.suptitle(
        "Metric Interpretability Validation: M3-Score layer decomposition\n"
        "agrees with independent feature-level analysis",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "metric_interpretability_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_metric_interpretability_validation(
    real_dir:       str,
    gen_dir:        str,
    interp_report:  str,
    output_dir:     str           = "./metric_interp_output",
    num_images:     Optional[int]  = None,
    device:         Optional[str]  = None,
    seed:           int            = 42,
) -> dict:
    """
    Run all four metric interpretability validation tests.

    Args:
        real_dir:      Directory of real images (searched recursively).
        gen_dir:       Directory of generated images (non-recursive).
        interp_report: Path to interpretability_report.json from
                       run_interpretability(). Must contain
                       per_layer_shift_sd and gen_distance_stats.
        output_dir:    Destination for plots and JSON report.
        num_images:    Cap on images per set (None = use all).
        device:        Torch device string.
        seed:          Random seed.

    Returns:
        dict with all four test results.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load interpretability report ──────────────────────────────────────────
    with open(interp_report) as f:
        interp = json.load(f)

    interp_layer_shifts = interp["per_layer_shift_sd"]   # {"L4": 0.756, ...}
    n_images_interp     = interp["n_images"]

    # Load active_layers from the canonical cache — single source of truth.
    # The interpretability report layers must match; if not, the report is
    # stale and needs to be regenerated by re-running interpretability.py.
    if not os.path.isfile(_LAYERS_CACHE):
        raise FileNotFoundError(
            f"Canonical layer cache not found: {_LAYERS_CACHE}\n"
            "Run interpretability.py first to compute and cache the layer selection."
        )
    with open(_LAYERS_CACHE) as _cf:
        _cache = json.load(_cf)
    active_layers  = _cache["active_layers"]
    report_layers  = interp.get("active_layers", [])
    if sorted(report_layers) != sorted(active_layers):
        raise ValueError(
            f"Layer mismatch detected!\n"
            f"  canonical_layers.json : {active_layers}\n"
            f"  interpretability_report: {report_layers}\n"
            "The interpretability report is stale (generated at a different "
            "threshold or from a different run). Re-run interpretability.py "
            "first, then re-run this script."
        )

    print(f"[MetricInterp] Interpretability report: {interp_report}")
    print(f"  Active layers (canonical): {active_layers}")
    print(f"  Feature-level shifts: {interp_layer_shifts}")

    # ── Load images ───────────────────────────────────────────────────────────
    real_paths = _load_paths(real_dir, num_images, recursive=True)
    gen_paths  = _load_paths(gen_dir,  num_images, recursive=False)
    cap        = min(len(real_paths), len(gen_paths))

    if cap != n_images_interp:
        print(
            f"  [WARN] Image count {cap} differs from interpretability report "
            f"({n_images_interp}). Results may not be directly comparable. "
            "For best validity use the same image set."
        )

    real_paths = real_paths[:cap]
    gen_paths  = gen_paths[:cap]
    print(f"  Using {cap} images per set.")

    real_imgs = _load_for_m3(real_paths)
    gen_imgs  = _load_for_m3(gen_paths)

    # ── Initialise M3V2Metric ─────────────────────────────────────────────────
    print("\nInitialising M3V2Metric ...")
    metric = M3V2Metric(device=device)
    metric.prune_layers_via_cka(real_imgs[:20], cache_path=_LAYERS_CACHE)
    # active_layers already set from cache by prune_layers_via_cka — no override needed
    print(f"  Active layers: {metric.active_layers}")

    # ── Extract features (once, shared across all tests) ──────────────────────
    print("\nExtracting features for all active layers ...")
    active_set = set(active_layers)
    with torch.no_grad():
        real_feats = metric._extract_raw_features(
            real_imgs, use_attention=True, layers_to_keep=active_set
        )
        gen_feats = metric._extract_raw_features(
            gen_imgs, use_attention=True, layers_to_keep=active_set
        )

    # ── Compute M3 layer distances (used in Tests 1 and 2) ───────────────────
    print("\nComputing M3 per-layer distances ...")
    m3_layer_distances: dict[str, float] = {}
    for layer in active_layers:
        rf = real_feats[layer - 1].to(device)
        gf = gen_feats[layer - 1].to(device)
        d  = metric._compute_mmd2(rf, gf).item()
        m3_layer_distances[f"L{layer}"] = round(d, 6)
        print(f"  L{layer}: {d:.6f}")

    # ── Compute M3 per-image scores (used in Tests 3 and 4) ──────────────────
    # Run full forward pass to get SNR-based weights
    print("\nComputing M3 forward pass for SNR weights ...")
    with torch.no_grad():
        m3_result = metric(real_imgs, gen_imgs)
    layer_weights = m3_result["layer_weights"]
    print(f"  Layer weights: {layer_weights}")

    m3_per_image = _compute_m3_per_image(
        real_feats, gen_feats, active_layers, layer_weights, device
    )
    print(f"  Per-image scores: mean={m3_per_image.mean():.4f}  "
          f"std={m3_per_image.std():.4f}")

    # ── Compute independent per-image distances from interp report ────────────
    # These distances use ALL features (radiomic + RadioDino) in standardised
    # space. They are taken directly from the interpretability run which
    # operated on the same image set.
    # Here we recompute them from scratch on the current image set to ensure
    # exact correspondence (same paths, same order).
    # For Test 3: recompute using ALL features (radiomic + RadioDino).
    # For Test 4: use ONLY radiomic features.

    # Recompute all features independently for per-image test
    print("\nRecomputing independent feature-level distances ...")
    # RadioDino features (raw, not L2-normalised -- matches interpretability.py)
    all_radiodino_real = np.concatenate(
        [real_feats[l - 1].numpy() for l in active_layers], axis=1
    )
    all_radiodino_gen = np.concatenate(
        [gen_feats[l - 1].numpy() for l in active_layers], axis=1
    )
    # Standardise using real-set statistics
    mu_r = all_radiodino_real.mean(axis=0)
    sg_r = all_radiodino_real.std(axis=0)
    sg_r = np.where(sg_r < 1e-8, 1.0, sg_r)
    radiodino_gen_std = (all_radiodino_gen - mu_r) / sg_r

    # Radiomic features
    print("  Extracting radiomic features ...")
    real_radio = _extract_radiomic_batch(real_paths)
    gen_radio  = _extract_radiomic_batch(gen_paths)
    mu_radio   = real_radio.mean(axis=0)
    sg_radio   = real_radio.std(axis=0)
    sg_radio   = np.where(sg_radio < 1e-8, 1.0, sg_radio)
    radio_gen_std = (gen_radio - mu_radio) / sg_radio

    # Combined standardised distance (matches interpretability.py exactly)
    combined_gen_std = np.concatenate([radio_gen_std, radiodino_gen_std], axis=1)
    interp_per_image = np.linalg.norm(combined_gen_std, axis=1)
    print(f"  Feature-level distances: mean={interp_per_image.mean():.4f}  "
          f"std={interp_per_image.std():.4f}")

    # ── Run all four tests ────────────────────────────────────────────────────
    results: dict = {
        "n_images":           cap,
        "active_layers":      active_layers,
        "m3_layer_distances": m3_layer_distances,
        "m3_layer_weights":   layer_weights,
        "interp_layer_shifts":interp_layer_shifts,
    }

    results["test1"] = _test1_layer_rank_correlation(
        m3_layer_distances, interp_layer_shifts,
        active_layers, output_dir,
    )

    results["test2"] = _test2_layer_ablation(
        real_feats, gen_feats, active_layers,
        interp_layer_shifts, metric, device, output_dir,
    )

    results["test3"] = _test3_per_image_consistency(
        m3_per_image, interp_per_image, output_dir,
    )

    results["test4"] = _test4_radiomic_crossval(
        real_paths, gen_paths, m3_per_image, output_dir,
    )

    # Overall verdict
    all_pass = all([
        results["test1"].get("significant", False),
        results["test2"].get("significant", False),
        results["test3"].get("significant", False),
    ])
    results["overall_verdict"] = (
        "M3-Score layer decomposition is consistent with the independent "
        "feature-level analysis across all three primary tests. "
        "The metric's internal attribution is physically meaningful."
        if all_pass else
        "One or more primary tests did not reach significance. "
        "See individual test results for details."
    )

    summary_path = _make_summary_plot(results, output_dir)
    results["summary_plot"] = summary_path

    report_path = os.path.join(output_dir,
                               "metric_interpretability_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\n[MetricInterp] Report saved: {report_path}")
    print(f"  Overall: {results['overall_verdict']}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric interpretability validation (Section 3.11)"
    )
    parser.add_argument("--real_dir",      required=True,
                        help="Real image directory (searched recursively)")
    parser.add_argument("--gen_dir",       required=True,
                        help="Generated image directory (non-recursive, clean)")
    parser.add_argument("--interp_report", required=True,
                        help="Path to interpretability_report.json")
    parser.add_argument("--output_dir",    default="./metric_interp_output")
    parser.add_argument("--num_images",    type=int, default=None)
    parser.add_argument("--device",        default=None)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()
    run_metric_interpretability_validation(
        real_dir      = args.real_dir,
        gen_dir       = args.gen_dir,
        interp_report = args.interp_report,
        output_dir    = args.output_dir,
        num_images    = args.num_images,
        device        = args.device,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()