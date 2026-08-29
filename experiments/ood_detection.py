"""
OOD / Anomaly Detection on Generated Images -- Section 3.x
============================================================
Identifies out-of-distribution generated images and correlates anomaly scores
with per-image M3-Score distances to understand which images are hardest to
generate faithfully.

Protocol:
  1. Extract ResNet-18 embeddings from real and generated images.
  2. Fit an Isolation Forest on real embeddings.
  3. Score each generated image to produce an anomaly score.
  4. Compute per-image distances in three feature spaces:
       - M3-Score V2 (RadioDino, CKA-selected layers, L2-normalised) -- consistent
         with Eq. perim from the paper methodology.
       - InceptionV3 (2048-d pool features, same backbone as FID).
       - Pixel MSE distance to real centroid.
  5. Compute Spearman correlation + 95% CI between anomaly scores and each distance.
  6. Failure analysis: identify cases where M3 and FID proxy disagree,
     using rank-based selection among OOD-flagged images only.
  7. ROC-AUC / PR-AUC: binary classification (real=0, generated=1) using
     centroid-L2 distance as the classifier score for RadioDino and InceptionV3.
     Reproduces Table 4 from the paper.
  8. t-SNE manifold topology: 2D projection of ResNet-18 embeddings for real
     and generated images. Tight generated clusters indicate mode collapse.
     Reproduces Fig 5 (Section 3.5) from the paper.

Notes:
  - real_dir is searched recursively (BraTS data may be nested).
  - gen_dir is searched non-recursively; pass a clean flat directory.
  - M3 per-image distances use L2-normalised features (unit sphere) to
    match the hat_E notation in the per-image score equation.
  - Layer weights are obtained from a full M3 forward pass after CKA pruning.

Usage:
    python ood_detection.py \\
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
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from sklearn.ensemble import IsolationForest
from tqdm.auto import tqdm
from scipy import stats as sp_stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LAYERS_CACHE = os.path.join(_PROJECT_ROOT, "canonical_layers.json")

from evaluation.m3_score_v2 import M3V2Metric        # corrected import path


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _resnet_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _inception_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _m3_transform() -> transforms.Compose:
    """Raw [0, 1] float tensors at 224x224. M3V2Metric._preprocess normalises internally."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Path loading
# ---------------------------------------------------------------------------

def _load_paths(
    directory:  str,
    n:          Optional[int] = None,
    recursive:  bool = True,
) -> list[str]:
    """
    Load image paths from directory.

    Args:
        recursive: True for real_dir (BraTS may be nested); False for gen_dir
                   to avoid loading evaluation output PNGs from subdirectories.
    """
    paths: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
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
        raise FileNotFoundError(
            f"No images found in {directory} (recursive={recursive})."
        )
    return paths


# ---------------------------------------------------------------------------
# ResNet-18 embedding network
# ---------------------------------------------------------------------------

class _EmbedNet(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.features.eval()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.to(self.device)).squeeze(-1).squeeze(-1)


def _extract_embeddings(
    paths:      list[str],
    net:        _EmbedNet,
    transform:  transforms.Compose,
    batch_size: int  = 64,
    use_tqdm:   bool = True,
    label:      str  = "",
) -> np.ndarray:
    all_feats: list[np.ndarray] = []
    itr = range(0, len(paths), batch_size)
    if use_tqdm:
        itr = tqdm(itr, desc=label)
    for start in itr:
        batch = [
            transform(Image.open(p).convert("RGB"))
            for p in paths[start:start + batch_size]
        ]
        feats = net(torch.stack(batch))
        all_feats.append(feats.cpu().numpy())
    if not all_feats:
        raise ValueError(f"No embeddings extracted [{label}].")
    return np.concatenate(all_feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-image distance helpers
# ---------------------------------------------------------------------------

def _l2_normalise_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalise each row to unit sphere. Matches hat_E in methodology."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-8, 1.0, norms)


def _compute_m3_distances(
    real_paths:    list[str],
    gen_paths:     list[str],
    metric:        M3V2Metric,
    layer_weights: dict[str, float],
    use_tqdm:      bool = True,
) -> np.ndarray:
    """
    Compute per-generated-image weighted L2 distance to the real embedding
    centroid using CKA-selected RadioDino layers (Eq. perim, Section 3.5).

    Features are L2-normalised to the unit sphere before distance computation,
    matching the hat_E notation in the methodology. Without this normalisation,
    raw RadioDino hidden states have norms 100-300, producing distances in the
    thousands that are incomparable to other feature-space distances.

    Layer weights are taken from the M3 forward pass (SNR-based).
    """
    transform = _m3_transform()

    imgs_real = torch.stack([
        transform(Image.open(p).convert("RGB"))
        for p in tqdm(real_paths, desc="M3 real", disable=not use_tqdm)
    ])
    imgs_gen = torch.stack([
        transform(Image.open(p).convert("RGB"))
        for p in tqdm(gen_paths, desc="M3 gen", disable=not use_tqdm)
    ])

    active_set = set(metric.active_layers)
    with torch.no_grad():
        real_feats_list = metric._extract_raw_features(
            imgs_real, use_attention=True, layers_to_keep=active_set
        )
        gen_feats_list = metric._extract_raw_features(
            imgs_gen, use_attention=True, layers_to_keep=active_set
        )

    distances = np.zeros(len(gen_paths))
    for layer in metric.active_layers:
        w = layer_weights.get(f"L{layer}", 1.0 / len(metric.active_layers))

        # L2-normalise to unit sphere (Fix: matches hat_E in Eq. perim)
        real_f = _l2_normalise_rows(real_feats_list[layer - 1].numpy())
        gen_f  = _l2_normalise_rows(gen_feats_list[layer - 1].numpy())

        centroid  = real_f.mean(axis=0)       # (D,) centroid of real unit vectors
        distances += w * np.linalg.norm(gen_f - centroid, axis=1)

    return distances


def _compute_inception_distances(
    real_paths:  list[str],
    gen_paths:   list[str],
    device:      str,
    batch_size:  int,
    use_tqdm:    bool,
) -> np.ndarray:
    """Per-image L2 distance to real centroid in InceptionV3 2048-d space."""
    net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    net.fc = nn.Identity()
    net.eval().to(device)
    transform = _inception_transform()

    def _extract(paths: list[str], label: str) -> np.ndarray:
        all_feats: list[np.ndarray] = []
        itr = range(0, len(paths), batch_size)
        if use_tqdm:
            itr = tqdm(itr, desc=label)
        with torch.no_grad():
            for s in itr:
                batch = torch.stack([
                    transform(Image.open(p).convert("RGB"))
                    for p in paths[s:s + batch_size]
                ]).to(device)
                out = net(batch)
                # Guard for InceptionOutputs namedtuple (train mode safety)
                if hasattr(out, "logits"):
                    out = out.logits
                all_feats.append(out.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    real_f   = _extract(real_paths, "Inception real")
    gen_f    = _extract(gen_paths,  "Inception gen")
    centroid = real_f.mean(axis=0)
    return np.linalg.norm(gen_f - centroid, axis=1)


def _compute_pixel_distances(
    real_paths: list[str],
    gen_paths:  list[str],
    batch_size: int,
    use_tqdm:   bool,
    size:       int = 128,
) -> np.ndarray:
    """Per-image MSE to real centroid in flattened grayscale pixel space."""
    def _extract(paths: list[str], label: str) -> np.ndarray:
        all_feats: list[np.ndarray] = []
        itr = range(0, len(paths), batch_size)
        if use_tqdm:
            itr = tqdm(itr, desc=label)
        for s in itr:
            batch = [
                np.array(
                    Image.open(p).convert("L").resize((size, size))
                ).flatten() / 255.0
                for p in paths[s:s + batch_size]
            ]
            all_feats.append(np.stack(batch))
        return np.concatenate(all_feats, axis=0)

    real_pix = _extract(real_paths, "Pixel real")
    gen_pix  = _extract(gen_paths,  "Pixel gen")
    centroid = real_pix.mean(axis=0)
    return np.mean((gen_pix - centroid) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _spearman_with_ci(a: np.ndarray, b: np.ndarray) -> dict:
    """Spearman rho with Fisher-z 95% CI."""
    valid = ~(np.isnan(a) | np.isnan(b))
    n = int(valid.sum())
    if n < 3:
        return {"spearman_r": float("nan"), "p": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"), "n": n}
    r, p = sp_stats.spearmanr(a[valid], b[valid])
    if abs(r) >= 1.0:
        return {"spearman_r": float(r), "p": float(p),
                "ci_lower": float(r), "ci_upper": float(r), "n": n}
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    return {
        "spearman_r": round(float(r), 4),
        "p":          float(p),
        "ci_lower":   round(float(np.tanh(z - 1.96 * se)), 4),
        "ci_upper":   round(float(np.tanh(z + 1.96 * se)), 4),
        "n":          n,
    }


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

# Colorblind-safe palette (Wong 2011 / matplotlib tab10 subset)
_C_REAL    = "#1f77b4"   # blue   – real images / RadioDino
_C_GEN     = "#d62728"   # red    – generated images / OOD
_C_INC     = "#ff7f0e"   # orange – InceptionV3
_C_INLIER  = "#2ca02c"   # green  – in-distribution generated
_C_GRID    = "#e0e0e0"   # light gray grid lines
_C_SPINE   = "#cccccc"   # axis spine
_C_TEXT    = "#222222"   # near-black text


def _pub_ax(ax, grid: bool = True) -> None:
    """Apply publication style: white background, dark text, light grid."""
    ax.set_facecolor("white")
    ax.tick_params(colors=_C_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_C_SPINE)
    if grid:
        ax.grid(True, color=_C_GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_grid(
    indices:       np.ndarray,
    paths:         list[str],
    anomaly_scores: np.ndarray,
    m3_norm:       np.ndarray,
    fid_norm:      np.ndarray,
    save_path:     str,
    title:         str,
) -> None:
    """Save a grid of images with their anomaly and metric scores."""
    n = len(indices)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]
    for i, idx in enumerate(indices):
        img = Image.open(paths[idx]).convert("RGB")
        from PIL import ImageOps
        img = ImageOps.invert(img)
        axes[i].imshow(img)
        axes[i].set_title(
            f"Anomaly: {anomaly_scores[idx]:.2f}\n"
            f"M3: {m3_norm[idx]:.2f} | FID: {fid_norm[idx]:.2f}",
            fontsize=8, color=_C_TEXT,
        )
        axes[i].axis("off")
    plt.suptitle(title, fontsize=11, color=_C_TEXT)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()



# ---------------------------------------------------------------------------
# ROC-AUC / PR-AUC helpers (Table 4 from paper)
# ---------------------------------------------------------------------------

def _compute_roc_pr_auc(
    real_distances: np.ndarray,
    gen_distances:  np.ndarray,
) -> dict:
    """
    Binary classification AUC using centroid-L2 distance as the classifier score.

    Protocol (Table 4 from paper):
      - Labels: real images = 0 (in-distribution), generated = 1 (OOD).
      - Score:  distance from real centroid in feature space (higher = more OOD).
      - Metrics: ROC-AUC and PR-AUC (average precision).

    A higher AUC means the feature space separates real from generated more
    reliably. This tests whether RadioDino (M3) or InceptionV3 (FID) provides
    better distributional separation.

    Args:
        real_distances: (N_real,) distances for real images.
        gen_distances:  (N_gen,)  distances for generated images.

    Returns:
        dict with roc_auc, pr_auc, n_real, n_gen.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    valid_r = ~np.isnan(real_distances)
    valid_g = ~np.isnan(gen_distances)
    r = real_distances[valid_r]
    g = gen_distances[valid_g]

    if len(r) < 2 or len(g) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "n_real": int(valid_r.sum()), "n_gen": int(valid_g.sum())}

    scores = np.concatenate([r, g])
    labels = np.concatenate([np.zeros(len(r)), np.ones(len(g))])

    try:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc  = float(average_precision_score(labels, scores))
    except Exception as exc:
        print(f"  [WARN] AUC computation failed: {exc}")
        roc_auc = float("nan")
        pr_auc  = float("nan")

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc":  round(pr_auc,  4),
        "n_real":  int(valid_r.sum()),
        "n_gen":   int(valid_g.sum()),
    }


def _plot_roc_curves(
    real_m3:   np.ndarray,
    gen_m3:    np.ndarray,
    real_inc:  np.ndarray,
    gen_inc:   np.ndarray,
    output_dir: str,
) -> str:
    """
    Plot ROC curves and distance distributions for RadioDino vs InceptionV3.
    Reproduces Fig 20 from the paper.
    """
    from sklearn.metrics import roc_curve, auc as sk_auc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor("white")

    def _roc(real_d, gen_d, label, color, ax):
        r = real_d[~np.isnan(real_d)]
        g = gen_d[~np.isnan(gen_d)]
        scores = np.concatenate([r, g])
        labels = np.concatenate([np.zeros(len(r)), np.ones(len(g))])
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_val = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC = {auc_val:.3f})")
        return auc_val

    ax = axes[0]
    _pub_ax(ax)
    ax.plot([0, 1], [0, 1], "--", color=_C_SPINE, lw=1)
    _roc(real_inc, gen_inc, "Inception-v3 (FID)", _C_INC,  ax)
    _roc(real_m3,  gen_m3,  "RadioDino (M3)",       _C_REAL, ax)
    ax.set_xlabel("False Positive Rate", color=_C_TEXT, fontsize=11)
    ax.set_ylabel("True Positive Rate",  color=_C_TEXT, fontsize=11)
    ax.set_title("ROC Curve: Real vs Generated Separation",
                 color=_C_TEXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="white", edgecolor=_C_SPINE,
              labelcolor=_C_TEXT, framealpha=1)

    ax = axes[1]
    _pub_ax(ax)
    # RadioDino: solid fill = real, step = generated
    for real_d, gen_d, label, c_fill, c_step in [
        (real_inc, gen_inc, "InceptionV3", _C_INC,  _C_INC),
        (real_m3,  gen_m3,  "RadioDino",     _C_REAL, _C_REAL),
    ]:
        r = real_d[~np.isnan(real_d)]
        g = gen_d[~np.isnan(gen_d)]
        ax.hist(r, bins=30, alpha=0.40, color=c_fill, density=True,
                label=f"Real ({label})")
        ax.hist(g, bins=30, alpha=0.85, color=c_step, density=True,
                histtype="step", linewidth=1.8, linestyle="--",
                label=f"Generated ({label})")
    ax.set_xlabel("Distance from real centroid", color=_C_TEXT, fontsize=11)
    ax.set_ylabel("Density",                     color=_C_TEXT, fontsize=11)
    ax.set_title("Feature Distance Distributions",
                 color=_C_TEXT, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, facecolor="white", edgecolor=_C_SPINE,
              labelcolor=_C_TEXT, framealpha=1)

    plt.suptitle("OOD Separation: RadioDino vs InceptionV3 (Table 4)",
                 color=_C_TEXT, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "ood_roc_auc.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    return path



# ---------------------------------------------------------------------------
# t-SNE manifold topology (Section 3.5 from paper)
# ---------------------------------------------------------------------------

def _plot_tsne(
    real_embed: np.ndarray,
    gen_embed:  np.ndarray,
    output_dir: str,
    perplexity: float = 30.0,
    seed:       int   = 42,
) -> str:
    """
    t-SNE projection of ResNet-18 embeddings for real and generated images.

    Projects both sets onto 2D and colour-codes them (real=blue, gen=red).
    Tight clustering of generated images indicates mode collapse.
    Reproduces Fig 5 from the paper (Section 3.5).

    Args:
        real_embed: (N_real, D) ResNet-18 embeddings for real images.
        gen_embed:  (N_gen,  D) ResNet-18 embeddings for generated images.
        perplexity: t-SNE perplexity (default 30, paper used 30).
        seed:       random state for reproducibility.

    Returns:
        Path to saved PNG.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [WARN] sklearn.manifold.TSNE not available; t-SNE skipped.")
        return ""

    N_real = len(real_embed)
    N_gen  = len(gen_embed)
    combined = np.concatenate([real_embed, gen_embed], axis=0)

    print(f"[OOD] Running t-SNE on {N_real + N_gen} images "
          f"(perplexity={perplexity}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    proj = tsne.fit_transform(combined)   # (N_real+N_gen, 2)

    real_proj = proj[:N_real]
    gen_proj  = proj[N_real:]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor("white")
    _pub_ax(ax, grid=True)

    # KDE density contours to visualise manifold overlap
    try:
        from scipy.stats import gaussian_kde
        for pts, color in [(real_proj, _C_REAL), (gen_proj, _C_GEN)]:
            k = gaussian_kde(pts.T, bw_method=0.3)
            xmin, xmax = pts[:, 0].min() - 2, pts[:, 0].max() + 2
            ymin, ymax = pts[:, 1].min() - 2, pts[:, 1].max() + 2
            xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
            zz = k(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=4, colors=color, alpha=0.45,
                       linewidths=0.9)
    except Exception:
        pass

    ax.scatter(real_proj[:, 0], real_proj[:, 1],
               c=_C_REAL, s=18, alpha=0.65, marker="o",
               label=f"Real (n={N_real})", edgecolors="none")
    ax.scatter(gen_proj[:, 0], gen_proj[:, 1],
               c=_C_GEN, s=22, alpha=0.75, marker="^",
               label=f"Generated (n={N_gen})", edgecolors="none")

    ax.set_xlabel("t-SNE Dimension 1", color=_C_TEXT, fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", color=_C_TEXT, fontsize=11)
    ax.set_title(
        f"t-SNE: Real vs Generated MRI Feature Space\n"
        f"n_real={N_real}  n_gen={N_gen}  perplexity={perplexity}",
        color=_C_TEXT, fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, facecolor="white", edgecolor=_C_SPINE,
              labelcolor=_C_TEXT, markerscale=1.5, framealpha=1)

    plt.tight_layout()
    path = os.path.join(output_dir, "tsne_manifold.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OOD] t-SNE saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ood_detection(
    real_dir:      str,
    gen_dir:       str,
    output_dir:    str            = "./ood_output",
    num_images:    Optional[int]  = None,
    device:        str            = "cpu",
    batch_size:    int            = 64,
    contamination: float          = 0.05,
    seed:          int            = 42,
    use_tqdm:      bool           = True,
    load_cache:    bool           = False,
    save_cache:    bool           = True,
) -> dict:
    """
    Run OOD detection on generated images.

    Returns dict with anomaly statistics, per-image scores, and Spearman
    correlations with M3-Score, InceptionV3, and pixel distances.
    """
    # Seeds (Fix: were missing despite seed parameter)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # real_dir: recursive (BraTS may be nested)
    # gen_dir:  non-recursive (must be a clean flat directory)
    real_paths = _load_paths(real_dir, num_images, recursive=True)
    gen_paths  = _load_paths(gen_dir,  num_images, recursive=False)
    cap        = min(len(real_paths), len(gen_paths))
    real_paths = real_paths[:cap]
    gen_paths  = gen_paths[:cap]
    print(f"[OOD] Real: {len(real_paths)} | Gen: {len(gen_paths)}")

    cache_path = os.path.join(output_dir, "ood_cache.npz")
    meta_path = os.path.join(output_dir, "ood_cache_meta.json")

    if load_cache and os.path.exists(cache_path) and os.path.exists(meta_path):
        print(f"[OOD] Loading cached embeddings and distances from {output_dir} ...")
        cache = np.load(cache_path)
        real_embed = cache["real_embed"]
        gen_embed = cache["gen_embed"]
        anomaly_scores = cache["anomaly_scores"]
        is_anomaly = cache["is_anomaly"]
        real_anomaly = cache["real_anomaly"]
        m3_distances = cache["m3_distances"]
        m3_real_distances = cache["m3_real_distances"]
        inc_distances = cache["inc_distances"]
        inc_real_distances = cache["inc_real_distances"]
        pix_distances = cache["pix_distances"]

        with open(meta_path, "r") as f:
            meta = json.load(f)
        active_layers = meta["active_layers"]
        layer_weights = meta["layer_weights"]

        # Verify length consistency
        if len(gen_paths) != len(m3_distances):
            raise ValueError(f"Cache size mismatch: paths have {len(gen_paths)} images, but cache has {len(m3_distances)}.")

        # Re-initialize M3V2Metric just to have it for the report/active_layers
        m3_metric = M3V2Metric(device=device)
        m3_metric.active_layers = active_layers

        pct_ood = float(is_anomaly.mean()) * 100
        print(f"[OOD] Cache loaded successfully. {pct_ood:.1f}% of generated images flagged as OOD.")
    else:
        if load_cache:
            print(f"[OOD] [WARN] Cache files not found in {output_dir}. Running full computation ...")

        # ── ResNet-18 embeddings for Isolation Forest ─────────────────────────────
        print("[OOD] Extracting ResNet-18 embeddings ...")
        embed_net  = _EmbedNet(device)
        resnet_tfm = _resnet_transform()
        real_embed = _extract_embeddings(
            real_paths, embed_net, resnet_tfm, batch_size, use_tqdm, "Real embed"
        )
        gen_embed  = _extract_embeddings(
            gen_paths,  embed_net, resnet_tfm, batch_size, use_tqdm, "Gen embed"
        )

        # ── Isolation Forest ──────────────────────────────────────────────────────
        print("[OOD] Fitting Isolation Forest ...")
        iforest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        iforest.fit(real_embed)
        # Higher = more anomalous (negate so that larger = more OOD)
        anomaly_scores = -iforest.decision_function(gen_embed)
        is_anomaly     =  iforest.predict(gen_embed) == -1
        real_anomaly   = -iforest.decision_function(real_embed)
        pct_ood        = float(is_anomaly.mean()) * 100
        print(f"[OOD] {pct_ood:.1f}% of generated images flagged as OOD")

        # ── M3-Score V2 per-image distances ───────────────────────────────────────
        print("[OOD] Computing M3-Score V2 per-image distances (RadioDino, L2-normalised) ...")
        m3_metric = M3V2Metric(device=device)
        ref20 = torch.stack([
            _m3_transform()(Image.open(p).convert("RGB"))
            for p in real_paths[:20]
        ])
        m3_metric.prune_layers_via_cka(ref20, cache_path=_LAYERS_CACHE)
        print(f"  Active layers: {m3_metric.active_layers}")

        # Obtain SNR-based layer weights via a full forward pass
        # (Fix: hasattr(metric, 'layer_weights') always returns False --
        #  layer_weights is in the forward() result dict, not stored on the object)
        print("  Running M3 forward pass to obtain SNR layer weights ...")
        m3_transform = _m3_transform()
        imgs_real_small = torch.stack([
            m3_transform(Image.open(p).convert("RGB")) for p in real_paths[:50]
        ])
        imgs_gen_small = torch.stack([
            m3_transform(Image.open(p).convert("RGB")) for p in gen_paths[:50]
        ])
        with torch.no_grad():
            m3_result = m3_metric(imgs_real_small, imgs_gen_small)
        layer_weights = m3_result["layer_weights"]   # {"L4": 0.3, "L12": 0.7, ...}
        print(f"  Layer weights: {layer_weights}")

        m3_distances = _compute_m3_distances(
            real_paths, gen_paths, m3_metric, layer_weights, use_tqdm
        )
        # Real images scored against their own centroid (needed for ROC-AUC)
        m3_real_distances = _compute_m3_distances(
            real_paths, real_paths, m3_metric, layer_weights, use_tqdm
        )

        # ── InceptionV3 per-image distances ───────────────────────────────────────
        print("[OOD] Computing InceptionV3 per-image distances ...")
        try:
            inc_distances = _compute_inception_distances(
                real_paths, gen_paths, device, batch_size, use_tqdm
            )
            # Also compute real-to-centroid distances for ROC-AUC (Table 4)
            # Re-use the same InceptionV3 but score real images against their own centroid
            inc_real_distances = _compute_inception_distances(
                real_paths, real_paths, device, batch_size, use_tqdm
            )
        except Exception as e:
            print(f"  [WARN] InceptionV3 distances failed: {e}")
            inc_distances      = np.full(len(gen_paths),  float("nan"))
            inc_real_distances = np.full(len(real_paths), float("nan"))

        # ── Pixel MSE per-image distances ─────────────────────────────────────────
        print("[OOD] Computing pixel MSE per-image distances ...")
        try:
            pix_distances = _compute_pixel_distances(
                real_paths, gen_paths, batch_size, use_tqdm
            )
        except Exception as e:
            print(f"  [WARN] Pixel distances failed: {e}")
            pix_distances = np.full(len(gen_paths), float("nan"))

        if save_cache:
            print(f"[OOD] Saving cache files to {output_dir} ...")
            np.savez(
                cache_path,
                real_embed=real_embed,
                gen_embed=gen_embed,
                anomaly_scores=anomaly_scores,
                is_anomaly=is_anomaly,
                real_anomaly=real_anomaly,
                m3_distances=m3_distances,
                m3_real_distances=m3_real_distances,
                inc_distances=inc_distances,
                inc_real_distances=inc_real_distances,
                pix_distances=pix_distances,
            )
            meta = {
                "active_layers": m3_metric.active_layers,
                "layer_weights": layer_weights,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

    # ── Spearman correlations ─────────────────────────────────────────────────
    res_m3  = _spearman_with_ci(anomaly_scores, m3_distances)
    res_inc = _spearman_with_ci(anomaly_scores, inc_distances)
    res_pix = _spearman_with_ci(anomaly_scores, pix_distances)

    for label, res in [("M3", res_m3), ("FID proxy", res_inc), ("Pixel MSE", res_pix)]:
        print(
            f"[OOD] Spearman {label:12s}: "
            f"r={res['spearman_r']:+.4f}  "
            f"CI=[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]  "
            f"p={res['p']:.2e}  n={res['n']}"
        )

    # ── Scatter plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.patch.set_facecolor("white")

    series = [
        ("M3-Score V2 Distance\n(L2-normalised RadioDino)",   m3_distances,  res_m3),
        ("InceptionV3 Distance\n(FID proxy feature space)", inc_distances, res_inc),
        ("Pixel MSE Distance",                              pix_distances, res_pix),
    ]
    for ax, (title, dists, stats) in zip(axes, series):
        _pub_ax(ax)
        ax.scatter(anomaly_scores[~is_anomaly], dists[~is_anomaly],
                   c=_C_INLIER, s=20, alpha=0.55, label="In-distribution",
                   edgecolors="none")
        ax.scatter(anomaly_scores[is_anomaly], dists[is_anomaly],
                   c=_C_GEN, s=45, alpha=0.85, label="OOD",
                   edgecolors="#7f0e0e", linewidths=0.4)
        r_str  = f"{stats['spearman_r']:+.3f}" if not np.isnan(stats['spearman_r']) else "N/A"
        ci_str = (f"[{stats['ci_lower']:+.2f}, {stats['ci_upper']:+.2f}]"
                  if not np.isnan(stats.get('ci_lower', float('nan'))) else "")
        p_str  = f"{stats['p']:.2e}" if not np.isnan(stats['p']) else "N/A"
        ax.set_xlabel("Anomaly Score (Isolation Forest)", color=_C_TEXT, fontsize=10)
        ax.set_ylabel(title.split("\n")[0], color=_C_TEXT, fontsize=10)
        ax.set_title(
            f"{title}\n\u03c1 = {r_str} {ci_str}  (p = {p_str},  n = {stats['n']})",
            color=_C_TEXT, fontsize=10, fontweight="bold",
        )
        if ax is axes[0]:
            ax.legend(fontsize=9, facecolor="white", edgecolor=_C_SPINE,
                      labelcolor=_C_TEXT, framealpha=1)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "ood_comparison_scatter.png")
    plt.savefig(scatter_path, bbox_inches="tight", facecolor="white")
    plt.close()

    # ── Anomaly score distribution ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("white")
    _pub_ax(ax)
    ax.hist(real_anomaly,   bins=40, alpha=0.55, color=_C_REAL,
            label="Real",      density=True)
    ax.hist(anomaly_scores, bins=40, alpha=0.55, color=_C_GEN,
            label="Generated", density=True)
    ax.set_xlabel("Anomaly Score", color=_C_TEXT, fontsize=11)
    ax.set_ylabel("Density",       color=_C_TEXT, fontsize=11)
    ax.set_title(f"Anomaly Score Distribution  (OOD rate = {pct_ood:.1f}%)",
                 color=_C_TEXT, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor="white", edgecolor=_C_SPINE,
              labelcolor=_C_TEXT, framealpha=1)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "ood_distributions.png")
    plt.savefig(dist_path, bbox_inches="tight", facecolor="white")
    plt.close()

    # ── ROC-AUC / PR-AUC (Table 4 from paper) ────────────────────────────────
    print("[OOD] Computing ROC-AUC and PR-AUC (RadioDino vs InceptionV3) ...")
    auc_m3  = _compute_roc_pr_auc(m3_real_distances,  m3_distances)
    auc_inc = _compute_roc_pr_auc(inc_real_distances,  inc_distances)
    print(f"  RadioDino (M3):      ROC-AUC={auc_m3['roc_auc']:.4f}  "
          f"PR-AUC={auc_m3['pr_auc']:.4f}")
    print(f"  InceptionV3 (FID): ROC-AUC={auc_inc['roc_auc']:.4f}  "
          f"PR-AUC={auc_inc['pr_auc']:.4f}")

    roc_plot_path = ""
    try:
        roc_plot_path = _plot_roc_curves(
            m3_real_distances, m3_distances,
            inc_real_distances, inc_distances,
            output_dir,
        )
        print(f"  ROC plot saved: {roc_plot_path}")
    except Exception as e:
        print(f"  [WARN] ROC plot failed: {e}")

    # ── t-SNE manifold topology (Section 3.5) ─────────────────────────────────
    tsne_path = _plot_tsne(real_embed, gen_embed, output_dir, seed=seed)

    # ── Failure analysis: rank-based, OOD images only ─────────────────────────
    # Fix: the original used diff * anomaly_scores which is unreliable when
    # anomaly_scores are negative (inliers have negative scores from -decision_fn).
    # Correct approach: restrict to OOD-flagged images, then rank by
    # (m3_rank - fid_rank) to find cases where the two metrics disagree.
    print("[OOD] Performing failure analysis (rank-based, OOD images only) ...")

    def _normalise(x: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(x)
        out   = np.full_like(x, float("nan"))
        if valid.any():
            v = x[valid]
            out[valid] = (v - v.min()) / (v.max() - v.min() + 1e-8)
        return out

    m3_norm  = _normalise(m3_distances)
    fid_norm = _normalise(inc_distances)

    ood_idx = np.where(is_anomaly)[0]   # indices of OOD-flagged images

    if len(ood_idx) >= 4:
        # Within OOD images, rank each metric (lower rank = lower normalised distance)
        m3_rank_ood  = np.argsort(np.argsort(m3_norm[ood_idx]))
        fid_rank_ood = np.argsort(np.argsort(fid_norm[ood_idx]))
        rank_diff    = m3_rank_ood.astype(int) - fid_rank_ood.astype(int)

        # M3-advantage: M3 ranks high (large distance), FID ranks low
        m3_adv_ood  = ood_idx[np.argsort(-rank_diff)[:4]]
        # FID-advantage: FID ranks high (large distance), M3 ranks low
        fid_adv_ood = ood_idx[np.argsort(rank_diff)[:4]]
    else:
        # Fallback: use all images ranked by |m3_norm - fid_norm|
        rank_diff   = m3_norm - fid_norm
        m3_adv_ood  = np.argsort(-rank_diff)[:4]
        fid_adv_ood = np.argsort(rank_diff)[:4]

    m3_adv_path  = os.path.join(output_dir, "failure_analysis_m3_advantage.png")
    fid_adv_path = os.path.join(output_dir, "failure_analysis_fid_advantage.png")

    _save_grid(m3_adv_ood,  gen_paths, anomaly_scores, m3_norm, fid_norm,
               m3_adv_path,
               "M3 detects anomaly stronger than FID proxy\n(among OOD images)")
    _save_grid(fid_adv_ood, gen_paths, anomaly_scores, m3_norm, fid_norm,
               fid_adv_path,
               "FID proxy detects anomaly stronger than M3\n(among OOD images)")

    # ── Save report ───────────────────────────────────────────────────────────
    results = {
        "pct_ood":                 round(pct_ood, 2),
        "mean_anomaly_score_gen":  round(float(anomaly_scores.mean()), 6),
        "mean_anomaly_score_real": round(float(real_anomaly.mean()),   6),
        "active_m3_layers":        m3_metric.active_layers,
        "layer_weights":           layer_weights,
        "correlations": {
            "m3":        res_m3,
            "fid_proxy": res_inc,
            "pixel_mse": res_pix,
        },
        "auc": {
            "radiodino_m3":  auc_m3,
            "inceptionv3": auc_inc,
        },
        "plots": {
            "scatter":       scatter_path,
            "distributions": dist_path,
            "m3_advantage":  m3_adv_path,
            "fid_advantage": fid_adv_path,
            "roc_auc":       roc_plot_path,
            "tsne":          tsne_path,
        },
        "failure_analysis": {
            "m3_advantage_samples":  [os.path.basename(gen_paths[i])
                                      for i in m3_adv_ood],
            "fid_advantage_samples": [os.path.basename(gen_paths[i])
                                      for i in fid_adv_ood],
        },
    }
    report_path = os.path.join(output_dir, "ood_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)    # Fix: added default=str
    print(f"[OOD] Report saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OOD detection experiment")
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="./ood_output")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_tqdm",    action="store_true")
    parser.add_argument("--load_cache",  action="store_true", help="Load cached embeddings/distances and run plotting only")
    parser.add_argument("--no_save_cache", action="store_true", help="Do not save cache files")
    args = parser.parse_args()
    run_ood_detection(
        real_dir   = args.real_dir,
        gen_dir    = args.gen_dir,
        output_dir = args.output_dir,
        num_images = args.num_images,
        device     = args.device,
        seed       = args.seed,
        use_tqdm   = not args.no_tqdm,
        load_cache = args.load_cache,
        save_cache = not args.no_save_cache,
    )