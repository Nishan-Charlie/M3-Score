"""
Interpretability Analysis -- Feature-Level Decomposition (Section 3.10)
=======================================================================
Audits which radiomic and learned features differ most between real and
generated image distributions.

Feature categories:
  - Intensity:  First-order statistical moments (mean, std, skewness, etc.)
  - Texture:    Grey-level co-occurrence matrix (GLCM) features
  - Learned:    CLS-token embeddings from CKA-selected RadioDino layers

CRITICAL NOTE on feature standardisation:
  Raw first-order features operate on pixel values in [0, 1] (after
  normalisation applied in this file). However, fo_energy = sum(x^2)/N
  is still on a different scale to glcm_entropy ~ [0, 5] and RadioDino
  embeddings ~ [-3, 3]. Concatenating and computing z-scores without
  per-feature standardisation produces physically meaningless results
  (z-scores in the millions, as observed in the original output).

  The correct procedure used here:
    1. Extract each feature on [0,1]-normalised images.
    2. Standardise EACH feature to zero mean and unit variance using
       the real-set statistics (per-feature z-scoring within the real set).
    3. Compute shift = |mean of standardised generated values| per feature.
       This gives shifts in units of real-set standard deviations, making
       all features directly comparable regardless of original scale.

NOTE on directory structure:
  real_dir is searched recursively (BraTS data may be in subdirectories).
  gen_dir is searched non-recursively. Pass a clean gen_dir containing
  only generated MRI files. Do not point gen_dir at a directory that
  contains evaluation output PNGs or other non-MRI files; those will be
  loaded and scored as massively out-of-distribution, contaminating all
  results.

Usage:
    python interpretability.py \\
        --real_dir <path> --gen_dir <path> --output_dir <path> \\
        --num_images 500 --device cuda
"""

from __future__ import annotations

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
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from scipy import stats as sp_stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LAYERS_CACHE = os.path.join(_PROJECT_ROOT, "canonical_layers.json")

from evaluation.m3_score_v2 import M3V2Metric     # corrected import path


# ---------------------------------------------------------------------------
# Publication plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

_COLORS = {
    "intensity": "#C62828",
    "texture":   "#1565C0",
    "learned":   "#2E7D32",
    "neutral":   "#546E7A",
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_paths(
    directory:  str,
    num_images: Optional[int],
    recursive:  bool = True,
) -> list[str]:
    """
    Load image paths from a directory.

    Args:
        directory:  Directory to search.
        num_images: Maximum number of paths to return.
        recursive:  If True, search subdirectories as well.
                    Set to True for real_dir (BraTS data is often nested).
                    Set to False for gen_dir (pass a clean flat directory
                    containing only generated MRI files).

    Raises:
        FileNotFoundError if no images are found.
    """
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        if recursive:
            paths.extend(
                glob.glob(os.path.join(directory, "**", ext), recursive=True)
            )
        else:
            paths.extend(glob.glob(os.path.join(directory, ext)))

    paths = sorted(set(paths))
    if num_images is not None:
        paths = paths[:num_images]

    if not paths:
        raise FileNotFoundError(
            f"No images found in {directory} "
            f"(recursive={recursive}). "
            "Check the directory path and that it contains supported image files."
        )
    return paths


# ---------------------------------------------------------------------------
# Radiomic feature extraction (on [0, 1] normalised images)
# ---------------------------------------------------------------------------

def _glcm(img_01: np.ndarray,
          distances: tuple = (1,),
          angles: tuple = (0,),
          levels: int = 32) -> np.ndarray:
    img_q = (img_01 * levels).astype(np.int32)
    img_q = np.clip(img_q, 0, levels - 1)
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
            counts = np.bincount(idx, minlength=levels * levels)
            glcm  += counts.reshape((levels, levels)).astype(np.float64)
    glcm = glcm + glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def _glcm_features(glcm: np.ndarray) -> dict:
    levels = glcm.shape[0]
    i_idx  = np.arange(levels)
    I, J   = np.meshgrid(i_idx, i_idx, indexing="ij")
    contrast      = float(np.sum(glcm * (I - J) ** 2))
    dissimilarity = float(np.sum(glcm * np.abs(I - J)))
    homogeneity   = float(np.sum(glcm / (1.0 + (I - J) ** 2)))
    energy        = float(np.sum(glcm ** 2))
    mu_i  = np.sum(I * glcm)
    mu_j  = np.sum(J * glcm)
    std_i = np.sqrt(np.sum(glcm * (I - mu_i) ** 2))
    std_j = np.sqrt(np.sum(glcm * (J - mu_j) ** 2))
    corr  = float(
        np.sum(glcm * (I - mu_i) * (J - mu_j)) / (std_i * std_j)
    ) if std_i > 0 and std_j > 0 else 0.0
    entropy = float(
        -np.sum(glcm[glcm > 0] * np.log2(glcm[glcm > 0] + 1e-12))
    )
    return {
        "glcm_contrast":      contrast,
        "glcm_dissimilarity": dissimilarity,
        "glcm_homogeneity":   homogeneity,
        "glcm_energy":        energy,
        "glcm_correlation":   corr,
        "glcm_entropy":       entropy,
    }


def _first_order_features(img_01: np.ndarray) -> dict:
    arr = img_01.ravel()
    if len(arr) == 0:
        return {}
    counts, _ = np.histogram(arr, bins=64, range=(0, 1))
    probs = counts / (counts.sum() + 1e-12)
    return {
        "fo_mean":       float(np.mean(arr)),
        "fo_std":        float(np.std(arr)),
        "fo_skewness":   float(sp_stats.skew(arr)),
        "fo_kurtosis":   float(sp_stats.kurtosis(arr)),
        "fo_energy":     float(np.sum(arr ** 2)) / len(arr),
        "fo_entropy":    float(sp_stats.entropy(probs + 1e-12)),
        "fo_min":        float(np.min(arr)),
        "fo_max":        float(np.max(arr)),
        "fo_median":     float(np.median(arr)),
        "fo_uniformity": float(np.sum(probs ** 2)),
    }


def extract_radiomic_features(img_path: str) -> dict:
    img  = Image.open(img_path).convert("L").resize((224, 224))
    arr  = np.array(img, dtype=np.float32) / 255.0
    feats = _first_order_features(arr)
    glcm  = _glcm(arr, distances=(1, 3), angles=(0, np.pi / 4, np.pi / 2))
    feats.update(_glcm_features(glcm))
    return feats


def extract_radiomic_batch(
    paths:    list[str],
    use_tqdm: bool = True,
    label:    str  = "",
) -> tuple[np.ndarray, list[str]]:
    all_feats:     list[list[float]] = []
    feature_names: Optional[list[str]] = None
    for p in tqdm(paths, desc=f"Radiomic [{label}]", disable=not use_tqdm):
        feats = extract_radiomic_features(p)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        all_feats.append([feats[k] for k in feature_names])
    return np.array(all_feats, dtype=np.float64), feature_names  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# RadioDino learned features
# ---------------------------------------------------------------------------

def extract_radiodino_features(
    paths:         list[str],
    metric:        M3V2Metric,
    active_layers: list[int],
    use_tqdm:      bool = True,
    label:         str  = "",
) -> tuple[np.ndarray, list[str]]:
    tfm  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    imgs = torch.stack([
        tfm(Image.open(p).convert("RGB"))
        for p in tqdm(paths, desc=f"Load [{label}]", disable=not use_tqdm)
    ])
    active_set = set(active_layers)
    with torch.no_grad():
        feats_list = metric._extract_raw_features(
            imgs, use_attention=True, layers_to_keep=active_set
        )
    layer_arrays = [feats_list[l - 1].numpy() for l in active_layers]
    features     = np.concatenate(layer_arrays, axis=1)
    D = layer_arrays[0].shape[1]
    feat_names = [
        f"radiodino_L{l}_{i:03d}" for l in active_layers for i in range(D)
    ]
    return features, feat_names


# ---------------------------------------------------------------------------
# Per-feature standardisation
# ---------------------------------------------------------------------------

def _standardise(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu    = real_feats.mean(axis=0)
    sigma = real_feats.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (real_feats - mu) / sigma, (gen_feats - mu) / sigma


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_feature_ranking(
    feature_names: list[str],
    shift:         np.ndarray,
    output_dir:    str,
    top_n:         int = 20,
) -> str:
    idx    = np.argsort(shift)[::-1][:top_n]
    names  = [feature_names[i] for i in idx]
    vals   = [float(shift[i])  for i in idx]

    def _cat(n: str) -> str:
        if n.startswith("fo_"):  return "intensity"
        if n.startswith("glcm"): return "texture"
        return "learned"

    colors = [_COLORS[_cat(n)] for n in names]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * top_n + 1.2)))
    ax.barh(range(len(names)), vals, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature shift (real-set standard deviations)")
    ax.set_title(f"Top {top_n} most shifted features: real vs generated")
    legend_items = [
        mpatches.Patch(color=_COLORS["intensity"], label="Intensity (first-order)"),
        mpatches.Patch(color=_COLORS["texture"],   label="Texture (GLCM)"),
        mpatches.Patch(color=_COLORS["learned"],   label="Learned (RadioDino)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_ranking.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _plot_category_summary(
    radiomic_names: list[str],
    radiomic_shift: np.ndarray,
    dino_shift:     np.ndarray,
    active_layers:  list[int],
    output_dir:     str,
) -> str:
    fo_mask   = np.array([n.startswith("fo_")  for n in radiomic_names])
    glcm_mask = np.array([n.startswith("glcm") for n in radiomic_names])

    cat_labels = ["Intensity\n(fo)", "Texture\n(GLCM)", "Learned\n(RadioDino)"]
    cat_vals   = [
        float(radiomic_shift[fo_mask].mean())   if fo_mask.any()   else 0.0,
        float(radiomic_shift[glcm_mask].mean()) if glcm_mask.any() else 0.0,
        float(dino_shift.mean()),
    ]
    cat_colors = [_COLORS["intensity"], _COLORS["texture"], _COLORS["learned"]]

    D = dino_shift.shape[0] // max(len(active_layers), 1)
    layer_means = [
        float(dino_shift[li * D: (li + 1) * D].mean())
        for li in range(len(active_layers))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    bars = ax.bar(cat_labels, cat_vals, color=cat_colors,
                  edgecolor="white", linewidth=0.4, width=0.5, alpha=0.9)
    offset_a = max(max(cat_vals) * 0.015, 0.002)
    for bar, v in zip(bars, cat_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset_a,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_ylabel("Mean shift (real-set std units)")
    ax.set_title("Mean distributional shift by feature category")

    ax = axes[1]
    layer_labels = [f"L{l}" for l in active_layers]
    x = np.arange(len(active_layers))
    ax.bar(x, layer_means, color=_COLORS["learned"],
           edgecolor="white", linewidth=0.4, alpha=0.9)
    offset_b = max(max(layer_means) * 0.015, 0.002)
    for xi, v in zip(x, layer_means):
        ax.text(xi, v + offset_b, f"{v:.3f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Mean shift (real-set std units)")
    ax.set_title("Mean RadioDino shift by CKA-selected layer")

    plt.suptitle("Distributional shift: real vs generated",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "category_shift.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _plot_per_image_dist(gen_dist: np.ndarray, real_dist: np.ndarray,
                          output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_dist, bins=40, alpha=0.55, color=_COLORS["neutral"],
            label=f"Real-to-real  (mean={real_dist.mean():.2f})",
            density=True, edgecolor="white", linewidth=0.3)
    ax.hist(gen_dist, bins=40, alpha=0.65, color=_COLORS["intensity"],
            label=f"Generated-to-real  (mean={gen_dist.mean():.2f})",
            density=True, edgecolor="white", linewidth=0.3)
    ax.axvline(real_dist.mean(), color=_COLORS["neutral"],
               lw=1.5, linestyle="--", alpha=0.8)
    ax.axvline(gen_dist.mean(), color=_COLORS["intensity"],
               lw=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("L2 distance from real centroid (standardised space)")
    ax.set_ylabel("Density")
    ax.set_title("Per-image distributional shift: real vs generated")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(output_dir, "per_image_distance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _plot_top_feature_distributions(
    real_feats_raw: np.ndarray,
    gen_feats_raw:  np.ndarray,
    all_names:      list[str],
    shift:          np.ndarray,
    output_dir:     str,
    top_n:          int = 6,
) -> str:
    idx   = np.argsort(shift)[::-1][:top_n]
    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()

    def _cat(n: str) -> str:
        if n.startswith("fo_"):  return "intensity"
        if n.startswith("glcm"): return "texture"
        return "learned"

    for plot_i, feat_i in enumerate(idx):
        ax  = axes[plot_i]
        r_v = real_feats_raw[:, feat_i]
        g_v = gen_feats_raw[:, feat_i]
        col = _COLORS[_cat(all_names[feat_i])]

        ax.hist(r_v, bins=30, alpha=0.55, color=_COLORS["neutral"],
                label="Real", density=True, edgecolor="white", linewidth=0.3)
        ax.hist(g_v, bins=30, alpha=0.65, color=col,
                label="Generated", density=True, edgecolor="white", linewidth=0.3)
        ax.axvline(r_v.mean(), color=_COLORS["neutral"],
                   lw=1.2, linestyle="--", alpha=0.85)
        ax.axvline(g_v.mean(), color=col, lw=1.2, linestyle="--", alpha=0.85)

        name_short = all_names[feat_i].replace("radiodino_", "dino_")
        ax.set_title(f"{name_short}\nshift = {shift[feat_i]:.2f} sd", fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if plot_i == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    for j in range(len(idx), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Value distributions of top shifted features",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "top_feature_distributions.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _save_image_grid(
    paths:            list[str],
    scores:           list[float],
    output_dir:       str,
    filename:         str,
    title:            str,
    top_n:            int  = 5,
    save_individuals: bool = True,
) -> str:
    n = min(top_n, len(paths))
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = Image.open(paths[i]).convert("L")
        ax.imshow(np.array(img), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"d = {scores[i]:.2f}", fontsize=9)
        ax.axis("off")
        if save_individuals:
            stem = filename.rsplit(".", 1)[0]
            img.save(os.path.join(output_dir, f"{stem}_rank_{i + 1}.png"))
    plt.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_interpretability(
    real_dir:   str,
    gen_dir:    str,
    output_dir: str           = "./interpretability_output",
    num_images: Optional[int]  = None,
    device:     str            = "cuda",
    use_tqdm:   bool           = True,
    seed:       int            = 42,
    cka_threshold: float       = 0.8,
) -> dict:
    """
    Run the interpretability feature decomposition experiment.

    real_dir is searched recursively (BraTS data may be nested).
    gen_dir is searched non-recursively; pass a clean flat directory
    containing only generated MRI files to avoid contamination by
    evaluation PNGs or other non-MRI files.

    All features are per-feature standardised before shift computation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # real_dir: recursive (BraTS may be nested)
    # gen_dir:  non-recursive (must be a clean flat directory)
    all_real = _load_paths(real_dir, num_images, recursive=True)
    all_gen  = _load_paths(gen_dir,  num_images, recursive=False)
    cap        = min(len(all_real), len(all_gen))
    real_paths = all_real[:cap]
    gen_paths  = all_gen[:cap]
    print(f"  Using {cap} images per set.")

    # ── Radiomic features ─────────────────────────────────────────────────────
    print("\n  Extracting radiomic features ...")
    r_radio_raw, r_names = extract_radiomic_batch(
        real_paths, use_tqdm=use_tqdm, label="real"
    )
    g_radio_raw, _ = extract_radiomic_batch(
        gen_paths, use_tqdm=use_tqdm, label="gen"
    )

    # ── RadioDino learned features ──────────────────────────────────────────────
    print("\n  Running CKA layer selection ...")
    metric = M3V2Metric(device=device, cka_threshold=cka_threshold)
    tfm    = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ref20  = torch.stack([
        tfm(Image.open(p).convert("RGB")) for p in real_paths[:20]
    ])
    metric.prune_layers_via_cka(ref20, cache_path=_LAYERS_CACHE)
    active_layers = metric.active_layers
    print(f"  Active layers: {active_layers}")

    print("\n  Extracting RadioDino features ...")
    r_dino_raw, d_names = extract_radiodino_features(
        real_paths, metric, active_layers, use_tqdm=use_tqdm, label="real"
    )
    g_dino_raw, _ = extract_radiodino_features(
        gen_paths, metric, active_layers, use_tqdm=use_tqdm, label="gen"
    )

    # ── Per-feature standardisation ───────────────────────────────────────────
    r_radio_std, g_radio_std = _standardise(r_radio_raw, g_radio_raw)
    r_dino_std,  g_dino_std  = _standardise(r_dino_raw,  g_dino_raw)

    radio_shift = np.abs(g_radio_std.mean(axis=0))
    dino_shift  = np.abs(g_dino_std.mean(axis=0))

    all_names    = list(r_names) + list(d_names)
    all_shift    = np.concatenate([radio_shift, dino_shift])
    real_std_all = np.concatenate([r_radio_std, r_dino_std], axis=1)
    gen_std_all  = np.concatenate([g_radio_std, g_dino_std], axis=1)

    top_idx = int(np.argmax(all_shift))
    print(f"\n  Most shifted feature: {all_names[top_idx]}")
    print(f"  Shift = {all_shift[top_idx]:.4f} real-set standard deviations")
    if "fo_median" in r_names:
        fo_med_idx = list(r_names).index("fo_median")
        print(f"  fo_median shift = {radio_shift[fo_med_idx]:.4f} sd "
              f"(expected < 10 if standardisation is correct)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    rank_path    = _plot_feature_ranking(all_names, all_shift, output_dir)
    cat_path     = _plot_category_summary(
        r_names, radio_shift, dino_shift, active_layers, output_dir
    )
    gen_dist     = np.linalg.norm(gen_std_all,  axis=1)
    real_dist    = np.linalg.norm(real_std_all, axis=1)
    dist_path    = _plot_per_image_dist(gen_dist, real_dist, output_dir)
    raw_all      = np.concatenate([r_radio_raw, r_dino_raw], axis=1)
    raw_gen      = np.concatenate([g_radio_raw, g_dino_raw], axis=1)
    distpan_path = _plot_top_feature_distributions(
        raw_all, raw_gen, all_names, all_shift, output_dir
    )
    most_idx  = np.argsort(gen_dist)[::-1][:5]
    least_idx = np.argsort(gen_dist)[:5]
    most_grid = _save_image_grid(
        [gen_paths[i] for i in most_idx],
        [float(gen_dist[i]) for i in most_idx],
        output_dir, "most_shifted.png",
        "Top-5 most shifted generated images",
    )
    least_grid = _save_image_grid(
        [gen_paths[i] for i in least_idx],
        [float(gen_dist[i]) for i in least_idx],
        output_dir, "least_shifted.png",
        "Top-5 least shifted generated images",
        save_individuals=False,
    )

    # ── Results dict (every plotted value captured) ───────────────────────────
    top20_idx = np.argsort(all_shift)[::-1][:20]
    ranked_features_top20 = [
        {
            "rank":     int(r + 1),
            "feature":  all_names[i],
            "category": (
                "intensity" if all_names[i].startswith("fo_")
                else "texture" if all_names[i].startswith("glcm")
                else "learned"
            ),
            "shift_sd": round(float(all_shift[i]), 4),
        }
        for r, i in enumerate(top20_idx)
    ]

    D = dino_shift.shape[0] // max(len(active_layers), 1)
    per_layer_shift_sd = {
        f"L{layer}": round(float(dino_shift[li * D: (li + 1) * D].mean()), 4)
        for li, layer in enumerate(active_layers)
    }

    gen_dist_stats = {
        "mean":   round(float(gen_dist.mean()),       4),
        "std":    round(float(gen_dist.std()),         4),
        "median": round(float(np.median(gen_dist)),    4),
        "min":    round(float(gen_dist.min()),         4),
        "max":    round(float(gen_dist.max()),         4),
    }
    real_dist_stats = {
        "mean":   round(float(real_dist.mean()),      4),
        "std":    round(float(real_dist.std()),        4),
        "median": round(float(np.median(real_dist)),   4),
        "min":    round(float(real_dist.min()),        4),
        "max":    round(float(real_dist.max()),        4),
    }

    top6_idx = np.argsort(all_shift)[::-1][:6]
    top6_feature_distributions = [
        {
            "feature":   all_names[fi],
            "shift_sd":  round(float(all_shift[fi]), 4),
            "real_mean": round(float(raw_all[:, fi].mean()), 4),
            "real_std":  round(float(raw_all[:, fi].std()),  4),
            "gen_mean":  round(float(raw_gen[:, fi].mean()), 4),
            "gen_std":   round(float(raw_gen[:, fi].std()),  4),
        }
        for fi in top6_idx
    ]

    most_shifted_images = [
        {"rank": int(r + 1), "path": gen_paths[i],
         "distance": round(float(gen_dist[i]), 4)}
        for r, i in enumerate(most_idx)
    ]
    least_shifted_images = [
        {"rank": int(r + 1), "path": gen_paths[i],
         "distance": round(float(gen_dist[i]), 4)}
        for r, i in enumerate(least_idx)
    ]

    results = {
        "n_images":      cap,
        "active_layers": active_layers,
        "top_feature":   all_names[top_idx],
        "top_shift_sd":  round(float(all_shift[top_idx]), 4),

        # Consistent key names with previous version
        "ranked_features_top20": ranked_features_top20,

        "category_shifts": {
            "intensity_mean_sd": round(float(
                radio_shift[np.array([n.startswith("fo_") for n in r_names])].mean()
            ), 4),
            "texture_mean_sd": round(float(
                radio_shift[np.array([n.startswith("glcm") for n in r_names])].mean()
            ), 4),
            "learned_mean_sd": round(float(dino_shift.mean()), 4),
        },

        "per_layer_shift_sd":          per_layer_shift_sd,
        "gen_distance_stats":          gen_dist_stats,
        "real_distance_stats":         real_dist_stats,
        "top6_feature_distributions":  top6_feature_distributions,
        "most_shifted_images":         most_shifted_images,
        "least_shifted_images":        least_shifted_images,

        "plots": {
            "ranking":            rank_path,
            "category_summary":   cat_path,
            "per_image_distance": dist_path,
            "top_distributions":  distpan_path,
            "most_shifted":       most_grid,
            "least_shifted":      least_grid,
        },
    }
    report_path = os.path.join(output_dir, "interpretability_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\n  Report saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Interpretability feature decomposition (Section 3.10)"
    )
    parser.add_argument("--real_dir",   required=True,
                        help="Real image directory (searched recursively)")
    parser.add_argument("--gen_dir",    required=True,
                        help="Generated image directory (non-recursive; "
                             "must contain only generated MRI files)")
    parser.add_argument("--output_dir", default="./interpretability_output")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--no_tqdm",    action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--cka_threshold", type=float, default=0.95)
    args = parser.parse_args()
    run_interpretability(
        real_dir   = args.real_dir,
        gen_dir    = args.gen_dir,
        output_dir = args.output_dir,
        num_images = args.num_images,
        device     = args.device,
        use_tqdm   = not args.no_tqdm,
        seed       = args.seed,
        cka_threshold = args.cka_threshold,
    )