"""
Per-Image Spatial Attribution -- 4 Methods + Comprehensive Metrics
===================================================================
For each generated image, computes pixel-level and patch-level heatmaps
showing which spatial regions drove the M3 distance from the real
distribution.

Four complementary attribution methods:

  Method 1 -- Gradient × Input  (pixel-level, 224×224)
    Gradient of the weighted patch-distance loss w.r.t. the processor-
    normalised input tensor, multiplied element-wise by the input.
    A(p) = |∂L/∂x_p| × |x_p|.  Correctly flows through the backbone
    by computing gradients on the normalised pixel_values tensor (not
    the raw image), so numpy conversion never breaks the graph.

  Method 2 -- Patch L2 distance map  (14×14 → 224×224)
    Per-patch-position weighted L2 distance of the generated image's
    patch tokens from the real-set patch centroid at the same position.
    Upsampled to 224×224 with bilinear interpolation.

  Method 3 -- CLS Attention map  (14×14 → 224×224)
    Mean CLS→patch attention weights across all active layers, weighted
    by M3 layer weights.  Shows where RadioDino attends when evaluating
    the image; combined with patch distance, reveals "the model is
    looking at an anomalous region."

  Method 4 -- Cosine anomaly map  (14×14 → 224×224)
    (1 − cosine_similarity) between each patch token and the real-set
    centroid at the same position.  Complementary to L2: measures
    directional alignment in feature space, not magnitude.

Metrics reported per image:
  m3_per_image_score    -- weighted L2 distance in L2-normalised space
  per_layer_distances   -- per active layer contribution
  patch_l2_mean/max     -- statistics of the spatial L2 map
  attn_entropy          -- Shannon entropy of the CLS attention map
                          (high = diffuse; low = concentrated)
  grad_mean/max         -- statistics of the gradient attribution map
  cosine_anomaly_mean   -- mean cosine anomaly across all patches

Summary outputs:
  attribution_{i:03d}.png   -- 5-panel figure per generated image
  summary_distribution.png  -- M3 score distribution histogram
  summary_layer_contrib.png -- per-layer mean distance + weights bar chart
  summary_worst_grid.png    -- top-5 worst images (highest M3 distance)
  summary_best_grid.png     -- top-5 best images  (lowest M3 distance)
  spatial_attribution_report.json -- full metrics

Usage:
    python experiments/spatial_attribution.py \\
        --real_dir  data_mri/brats_axial_multislice \\
        --gen_dir   output/generated \\
        --n_real    200 --n_gen 20 \\
        --output_dir results/spatial_attribution
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

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

_C_TEXT  = "#222222"
_C_REAL  = "#1f77b4"
_C_GEN   = "#d62728"
_C_NEUT  = "#546E7A"
_C_GRID  = "#e0e0e0"

_IMG_SIZE = 224        # display size for all upsampled maps

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_paths(directory: str, n: int, recursive: bool = False) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        if recursive:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        else:
            paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(set(paths))[:n]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}.")
    return paths


def _m3_transform() -> transforms.Compose:
    """Raw [0,1] tensors at 224×224; M3V2Metric._preprocess normalises internally."""
    return transforms.Compose([
        transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
        transforms.ToTensor(),
    ])


def _to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float [0,1] → (H, W, 3) uint8."""
    return (
        img_tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255
    ).astype(np.uint8)


def _normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def _upsample_map(grid: np.ndarray, out_size: int = _IMG_SIZE) -> np.ndarray:
    """Bilinear upsample a 2-D attribution grid to out_size×out_size."""
    t = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(out_size, out_size), mode="bilinear",
                       align_corners=False)
    return up.squeeze().numpy()


# ---------------------------------------------------------------------------
# Single-pass feature extraction  (patch tokens + CLS attn + pooled feat)
# ---------------------------------------------------------------------------

def _extract_one(
    backbone,
    processor,
    img_t: torch.Tensor,
    target_layers_0idx: list[int],
    device: str,
    need_attn: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Single backbone forward for one image.

    Returns
    -------
    patch_tokens : list[Tensor (N_patches, D)]  -- raw patch tokens per layer
    attn_pooled  : list[Tensor (D,)]            -- attention-weighted pooled feat
    attn_maps    : list[Tensor (N_patches,)]    -- normalised CLS→patch attn
    """
    imgs_np = (
        img_t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255
    ).astype("uint8")
    inputs = processor(images=[imgs_np], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = backbone(
            **inputs,
            output_hidden_states=True,
            output_attentions=need_attn,
        )

    hidden     = out.hidden_states[1:]              # skip embedding layer
    attentions = out.attentions if need_attn else None

    patch_tokens: list[torch.Tensor] = []
    attn_pooled:  list[torch.Tensor] = []
    attn_maps:    list[torch.Tensor] = []

    for li in target_layers_0idx:
        h = hidden[li][0]                           # (N_seq, D)
        patch = h[1:, :]                            # (N_patches, D)
        patch_tokens.append(patch)

        if attentions is not None:
            avg_attn = attentions[li][0].mean(0)    # (N_seq, N_seq)
            cls_attn = avg_attn[0, 1:]              # (N_patches,)
            cls_attn = cls_attn / (cls_attn.sum() + 1e-8)
            pooled   = (patch * cls_attn.unsqueeze(-1)).sum(0)  # (D,)
            attn_maps.append(cls_attn.detach())
            attn_pooled.append(pooled.detach())
        else:
            attn_pooled.append(patch.mean(0))
            attn_maps.append(torch.ones(patch.shape[0]) / patch.shape[0])

    return patch_tokens, attn_pooled, attn_maps


# ---------------------------------------------------------------------------
# Build real-set statistics  (run once over all real images)
# ---------------------------------------------------------------------------

def _build_real_stats(
    real_paths:         list[str],
    backbone,
    processor,
    target_layers_0idx: list[int],
    active_layers:      list[int],
    device:             str,
) -> tuple[
    list[torch.Tensor],   # patch_centroids   (N_patches, D) per layer
    dict[int, torch.Tensor],  # m3_centroids_norm (D,) per layer index
    list[torch.Tensor],   # attn_centroids    (N_patches,) per layer
    int,                  # patch_grid (e.g. 14)
]:
    """
    Compute position-wise patch centroids, L2-normalised M3 centroids,
    and mean attention centroids from all real images.
    """
    transform = _m3_transform()
    n_l = len(target_layers_0idx)
    patch_acc: list[list[torch.Tensor]] = [[] for _ in range(n_l)]
    pooled_acc: list[list[torch.Tensor]] = [[] for _ in range(n_l)]
    attn_acc:  list[list[torch.Tensor]] = [[] for _ in range(n_l)]
    patch_grid = None

    for p in tqdm(real_paths, desc="Real centroids", leave=False):
        img_t = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        patch_toks, pooled, attn = _extract_one(
            backbone, processor, img_t, target_layers_0idx, device, need_attn=True
        )
        for si in range(n_l):
            patch_acc[si].append(patch_toks[si].cpu())
            pooled_acc[si].append(pooled[si].cpu())
            attn_acc[si].append(attn[si].cpu())
        if patch_grid is None:
            patch_grid = int(patch_toks[0].shape[0] ** 0.5)

    patch_centroids = [torch.stack(a).mean(0) for a in patch_acc]
    attn_centroids  = [torch.stack(a).mean(0) for a in attn_acc]

    # L2-normalised M3 centroids (keyed by 1-indexed layer)
    m3_centroids_norm: dict[int, torch.Tensor] = {}
    for si, layer in enumerate(active_layers):
        pooled_stack = torch.stack(pooled_acc[si])           # (N_real, D)
        pooled_norm  = pooled_stack / pooled_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
        m3_centroids_norm[layer] = pooled_norm.mean(0)       # (D,)

    return patch_centroids, m3_centroids_norm, attn_centroids, patch_grid


# ---------------------------------------------------------------------------
# Method 1 -- Gradient × Input  (corrected: grad flows on pixel_values)
# ---------------------------------------------------------------------------

def _method1_grad_x_input(
    img_t:              torch.Tensor,
    backbone,
    processor,
    patch_centroids:    list[torch.Tensor],
    target_layers_0idx: list[int],
    weights:            list[float],
    device:             str,
) -> np.ndarray:
    """
    Returns (H, W) attribution map in [0, ∞).

    Key fix: gradients are computed on processor-normalised pixel_values
    (the tensor that actually enters the backbone), so the numpy
    conversion for the HuggingFace processor never breaks the graph.
    """
    imgs_np = (
        img_t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255
    ).astype("uint8")
    inputs = processor(images=[imgs_np], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    pixel_values = pixel_values.detach().requires_grad_(True)

    out    = backbone(pixel_values=pixel_values, output_hidden_states=True)
    hidden = out.hidden_states[1:]

    loss = sum(
        weights[si] * (
            (hidden[li][0, 1:, :].mean(0) - patch_centroids[si].to(device)) ** 2
        ).sum()
        for si, li in enumerate(target_layers_0idx)
    )
    loss.backward()

    grad = pixel_values.grad                         # (1, 3, H', W')
    attr = (grad.abs() * pixel_values.detach().abs()).squeeze(0).mean(0)  # (H', W')

    # Resize to IMG_SIZE if the processor changed the resolution
    if attr.shape[-1] != _IMG_SIZE:
        attr = F.interpolate(
            attr.unsqueeze(0).unsqueeze(0),
            size=(_IMG_SIZE, _IMG_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze()

    return attr.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Method 2 -- Patch L2 distance map
# ---------------------------------------------------------------------------

def _method2_patch_l2(
    patch_tokens:       list[torch.Tensor],
    patch_centroids:    list[torch.Tensor],
    weights:            list[float],
    patch_grid:         int,
) -> np.ndarray:
    """Weighted per-position L2 distance → (patch_grid, patch_grid) → 224×224."""
    n_p = patch_grid * patch_grid
    combined = torch.zeros(n_p)
    for si, patch in enumerate(patch_tokens):
        centroid = patch_centroids[si].to(patch.device)
        dists = ((patch[:n_p] - centroid[:n_p]) ** 2).sum(-1)
        combined += weights[si] * dists.cpu()
    grid = combined.reshape(patch_grid, patch_grid).numpy()
    return _upsample_map(grid)


# ---------------------------------------------------------------------------
# Method 3 -- CLS Attention map
# ---------------------------------------------------------------------------

def _method3_attention(
    attn_maps:  list[torch.Tensor],
    weights:    list[float],
    patch_grid: int,
) -> np.ndarray:
    """Weighted mean CLS→patch attention → (patch_grid, patch_grid) → 224×224."""
    n_p = patch_grid * patch_grid
    combined = torch.zeros(n_p)
    for si, attn in enumerate(attn_maps):
        combined += weights[si] * attn[:n_p].cpu()
    combined = combined / (combined.sum() + 1e-8)
    grid = combined.reshape(patch_grid, patch_grid).numpy()
    return _upsample_map(grid)


# ---------------------------------------------------------------------------
# Method 4 -- Cosine anomaly map  (1 − cosine similarity)
# ---------------------------------------------------------------------------

def _method4_cosine_anomaly(
    patch_tokens:    list[torch.Tensor],
    patch_centroids: list[torch.Tensor],
    weights:         list[float],
    patch_grid:      int,
) -> np.ndarray:
    """
    (1 − cosine_sim) per patch position, weighted across layers.
    High values → patch direction differs from the real distribution.
    """
    n_p = patch_grid * patch_grid
    combined = torch.zeros(n_p)
    for si, patch in enumerate(patch_tokens):
        centroid = patch_centroids[si].to(patch.device)
        p = patch[:n_p]
        c = centroid[:n_p]
        sim     = F.cosine_similarity(p, c, dim=-1)          # (N_p,)
        anomaly = (1.0 - sim).clamp(0.0, 2.0) / 2.0         # → [0, 1]
        combined += weights[si] * anomaly.cpu()
    grid = combined.reshape(patch_grid, patch_grid).numpy()
    return _upsample_map(grid)


# ---------------------------------------------------------------------------
# Per-image M3 score  (consistent with M3V2Metric feature space)
# ---------------------------------------------------------------------------

def _compute_m3_per_image(
    attn_pooled:          list[torch.Tensor],     # (D,) per layer
    m3_centroids_norm:    dict[int, torch.Tensor],
    active_layers:        list[int],
    layer_weights:        dict[str, float],
    device:               str,
) -> tuple[float, dict[str, float]]:
    """
    Weighted L2 distance from real centroid in L2-normalised feature space.
    Uses the same attention-weighted pooled features as M3V2Metric forward().
    """
    total = 0.0
    per_layer: dict[str, float] = {}
    for si, layer in enumerate(active_layers):
        feat = attn_pooled[si].to(device)                    # (D,)
        feat_norm = feat / feat.norm().clamp(min=1e-8)
        mu_r = m3_centroids_norm[layer].to(device)           # (D,)
        dist = (feat_norm - mu_r).norm().item()
        w    = layer_weights.get(f"L{layer}", 1.0 / len(active_layers))
        total += w * dist
        per_layer[f"L{layer}"] = round(dist, 6)
    return round(total, 6), per_layer


# ---------------------------------------------------------------------------
# Per-layer patch distance breakdown (for top-K images)
# ---------------------------------------------------------------------------

def _per_layer_patch_l2(
    patch_tokens:       list[torch.Tensor],
    patch_centroids:    list[torch.Tensor],
    active_layers:      list[int],
    patch_grid:         int,
) -> dict[str, np.ndarray]:
    """Return per-layer (patch_grid×patch_grid) L2 map for detailed breakdown."""
    n_p = patch_grid * patch_grid
    out: dict[str, np.ndarray] = {}
    for si, layer in enumerate(active_layers):
        patch    = patch_tokens[si]
        centroid = patch_centroids[si].to(patch.device)
        dists    = ((patch[:n_p] - centroid[:n_p]) ** 2).sum(-1)
        grid     = dists.cpu().reshape(patch_grid, patch_grid).numpy()
        out[f"L{layer}"] = _upsample_map(grid)
    return out


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def _attention_entropy(attn_map: np.ndarray) -> float:
    """Shannon entropy of an attention distribution (in nats)."""
    p = attn_map.ravel()
    p = p / (p.sum() + 1e-8)
    return float(-np.sum(p * np.log(p + 1e-12)))


# ---------------------------------------------------------------------------
# Visualisation -- per-image 5-panel figure
# ---------------------------------------------------------------------------

def _save_figure(
    img_rgb:      np.ndarray,
    grad_map:     np.ndarray,
    patch_map:    np.ndarray,
    attn_map:     np.ndarray,
    cosine_map:   np.ndarray,
    m3_score:     float,
    per_layer:    dict[str, float],
    output_path:  str,
    idx:          int,
) -> None:
    grad_n   = _normalise(grad_map)
    patch_n  = _normalise(patch_map)
    attn_n   = _normalise(attn_map)
    cosine_n = _normalise(cosine_map)

    fig = plt.figure(figsize=(20, 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 5, figure=fig, wspace=0.08)

    panels = [
        ("Original",                 None,      None),
        ("Gradient × Input\n(pixel)", grad_n,   "hot"),
        ("Patch L2 distance\n(OOD)",  patch_n,  "hot"),
        ("CLS Attention map",         attn_n,   "viridis"),
        ("Cosine anomaly\n(1−sim)",   cosine_n, "RdYlBu_r"),
    ]

    for col, (title, overlay, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[col])
        ax.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
        if overlay is not None:
            ax.imshow(overlay, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
        ax.set_title(title, color=_C_TEXT, fontsize=9, pad=4)
        ax.axis("off")
        ax.set_facecolor("white")

    layer_str = "  ".join(f"{k}:{v:.3f}" for k, v in per_layer.items())
    fig.suptitle(
        f"Image {idx:03d}  |  M3 score = {m3_score:.4f}  |  layers: {layer_str}",
        color=_C_TEXT, fontsize=9, y=1.01,
    )
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Visualisation -- per-layer breakdown for top-K images
# ---------------------------------------------------------------------------

def _save_layer_figure(
    img_rgb:       np.ndarray,
    layer_maps:    dict[str, np.ndarray],
    per_layer:     dict[str, float],
    layer_weights: dict[str, float],
    output_path:   str,
    idx:           int,
) -> None:
    layers = sorted(layer_maps.keys())
    n = len(layers)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 3.2, 3.8), dpi=150)
    fig.patch.set_facecolor("white")

    axes[0].imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
    axes[0].set_title(f"Original #{idx:03d}", fontsize=9, color=_C_TEXT)
    axes[0].axis("off")

    for ai, lk in enumerate(layers):
        ax = axes[ai + 1]
        ax.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
        norm_m = _normalise(layer_maps[lk])
        ax.imshow(norm_m, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        d = per_layer.get(lk, 0)
        w = layer_weights.get(lk, 0)
        ax.set_title(f"{lk}\ndist={d:.3f}  w={w:.3f}", fontsize=8, color=_C_TEXT)
        ax.axis("off")

    plt.suptitle(f"Per-layer patch L2 attribution  |  Image {idx:03d}",
                 fontsize=10, fontweight="bold", color=_C_TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def _make_summary_plots(
    scores:          list[dict],
    active_layers:   list[int],
    layer_weights:   dict[str, float],
    gen_paths:       list[str],
    output_dir:      str,
) -> dict[str, str]:
    paths: dict[str, str] = {}

    m3_vals = np.array([s["m3_per_image_score"] for s in scores])

    # 1. M3 score distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("white")
    ax.hist(m3_vals, bins=max(10, len(scores) // 5),
            color=_C_GEN, alpha=0.8, edgecolor="white", linewidth=0.4, density=True)
    ax.axvline(m3_vals.mean(), color=_C_REAL, lw=1.8, linestyle="--",
               label=f"Mean = {m3_vals.mean():.4f}")
    ax.axvline(np.median(m3_vals), color=_C_NEUT, lw=1.5, linestyle=":",
               label=f"Median = {np.median(m3_vals):.4f}")
    ax.set_xlabel("Per-image M3 score")
    ax.set_ylabel("Density")
    ax.set_title("M3 per-image score distribution (generated images)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(output_dir, "summary_distribution.png")
    plt.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["distribution"] = p

    # 2. Per-layer mean distance + weights
    layer_keys   = [f"L{l}" for l in active_layers]
    mean_dists   = []
    for lk in layer_keys:
        vals = [s["per_layer_distances"].get(lk, 0.0) for s in scores]
        mean_dists.append(float(np.mean(vals)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.set_facecolor("white")
    x = np.arange(len(layer_keys))
    bars = ax.bar(x, mean_dists, color=_C_GEN, alpha=0.85,
                  edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, mean_dists):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(mean_dists) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_keys, fontsize=9)
    ax.set_ylabel("Mean L2 distance (normalised space)")
    ax.set_title("Mean per-layer M3 distance across all generated images")

    ax = axes[1]
    ax.set_facecolor("white")
    w_vals = [layer_weights.get(lk, 0.0) for lk in layer_keys]
    bars2 = ax.bar(x, w_vals, color=_C_REAL, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars2, w_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(w_vals) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_keys, fontsize=9)
    ax.set_ylabel("M3 layer weight")
    ax.set_title("M3 layer weights (SNR-based)")

    plt.suptitle("Layer-level analysis", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, "summary_layer_contrib.png")
    plt.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["layer_contrib"] = p

    # 3. Worst / best grids
    for label, sorted_scores in [
        ("worst", sorted(scores, key=lambda s: s["m3_per_image_score"], reverse=True)),
        ("best",  sorted(scores, key=lambda s: s["m3_per_image_score"], reverse=False)),
    ]:
        top = sorted_scores[:5]
        n   = len(top)
        fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 3.5), dpi=150)
        fig.patch.set_facecolor("white")
        if n == 1:
            axes = [axes]
        for ax, rec in zip(axes, top):
            img = np.array(Image.open(rec["path"]).convert("RGB").resize(
                (_IMG_SIZE, _IMG_SIZE)))
            ax.imshow(img)
            ax.set_title(f"#{rec['idx']:03d}\nM3={rec['m3_per_image_score']:.4f}",
                         fontsize=9, color=_C_TEXT)
            ax.axis("off")
        title = f"Top-{n} {'highest' if label=='worst' else 'lowest'} M3 score images"
        plt.suptitle(title, fontsize=11, fontweight="bold", y=1.02, color=_C_TEXT)
        plt.tight_layout()
        p = os.path.join(output_dir, f"summary_{label}_grid.png")
        plt.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close()
        paths[label] = p

    # 4. Scatter: grad_max vs m3_score (interpretability check)
    if all("grad_max" in s for s in scores):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor("white")
        gmax = np.array([s["grad_max"] for s in scores])
        ax.scatter(_normalise(gmax), _normalise(m3_vals),
                   color=_C_GEN, alpha=0.6, s=25, edgecolors="none")
        z = np.polyfit(_normalise(gmax), _normalise(m3_vals), 1)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, np.polyval(z, xs), "--", color=_C_NEUT, lw=1.5, alpha=0.8)
        ax.set_xlabel("Gradient max (normalised)")
        ax.set_ylabel("M3 per-image score (normalised)")
        ax.set_title("Gradient magnitude vs M3 score")
        plt.tight_layout()
        p = os.path.join(output_dir, "summary_grad_vs_m3.png")
        plt.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close()
        paths["grad_vs_m3"] = p

    return paths


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_spatial_attribution(
    real_dir:        str,
    gen_dir:         str,
    output_dir:      str            = "./results/spatial_attribution",
    n_real:          int            = 200,
    n_gen:           int            = 20,
    device:          Optional[str]  = None,
    seed:            int            = 42,
    layer_breakdown_top_k: int      = 5,
) -> dict:
    """
    Generate per-image spatial attribution maps for generated images.

    Args:
        real_dir:              Directory of real images (searched recursively).
        gen_dir:               Directory of generated images (non-recursive).
        output_dir:            Destination for PNGs and JSON report.
        n_real:                Number of real images for centroid estimation.
        n_gen:                 Number of generated images to attribute.
        device:                Torch device string.
        seed:                  Random seed.
        layer_breakdown_top_k: Save per-layer figures for the top-K worst images.

    Returns:
        Full results dict (also saved as JSON).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # real_dir searched recursively (BraTS may be nested); gen_dir non-recursive
    real_paths = _load_paths(real_dir, n_real, recursive=True)
    gen_paths  = _load_paths(gen_dir,  n_gen,  recursive=False)
    print(f"Real: {len(real_paths)} | Gen: {len(gen_paths)}")

    # ── CKA layer selection (shared canonical cache) ──────────────────────────
    print("Loading M3V2Metric and running CKA layer selection ...")
    metric    = M3V2Metric(device=device)
    transform = _m3_transform()
    prune_imgs = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in real_paths[:20]
    ])
    metric.prune_layers_via_cka(prune_imgs, cache_path=_LAYERS_CACHE)
    active_layers_1idx = metric.active_layers          # 1-indexed
    target_layers_0idx = [l - 1 for l in active_layers_1idx]
    print(f"  Active layers (1-indexed): {active_layers_1idx}")

    # ── Get SNR-based layer weights from M3 forward pass ─────────────────────
    print("Computing M3 layer weights via forward pass ...")
    sample_real = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in real_paths[:min(30, len(real_paths))]
    ])
    sample_gen = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in gen_paths[:min(30, len(gen_paths))]
    ])
    with torch.no_grad():
        m3_result    = metric(sample_real, sample_gen)
    layer_weights = m3_result["layer_weights"]          # {"L4": 0.23, ...}
    print(f"  Layer weights: {layer_weights}")
    weights_list = [layer_weights.get(f"L{l}", 1.0 / len(active_layers_1idx))
                    for l in active_layers_1idx]

    # Reuse the backbone and processor from the metric (no duplicate model load)
    backbone  = metric.backbone
    processor = metric.processor

    # ── Determine patch grid from backbone output ─────────────────────────────
    with torch.no_grad():
        _test_img = transform(Image.open(real_paths[0]).convert("RGB")).unsqueeze(0)
        _pt, _, _ = _extract_one(backbone, processor, _test_img.to(device),
                                  target_layers_0idx, device, need_attn=False)
    patch_grid = int(_pt[0].shape[0] ** 0.5)
    print(f"  Patch grid: {patch_grid}×{patch_grid}  ({_pt[0].shape[0]} patches)")
    del _test_img, _pt

    # ── Build real-set statistics ─────────────────────────────────────────────
    print(f"Building real centroids from {len(real_paths)} images ...")
    patch_centroids, m3_centroids_norm, attn_centroids, _ = _build_real_stats(
        real_paths, backbone, processor,
        target_layers_0idx, active_layers_1idx, device,
    )
    # Move centroids to device
    patch_centroids = [c.to(device) for c in patch_centroids]
    attn_centroids  = [c.to(device) for c in attn_centroids]
    m3_centroids_norm = {k: v.to(device) for k, v in m3_centroids_norm.items()}

    # ── Per-image attribution ─────────────────────────────────────────────────
    print(f"Computing attributions for {len(gen_paths)} generated images ...")
    records: list[dict] = []

    for i, p in enumerate(tqdm(gen_paths, desc="Attribution")):
        img_t   = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        img_rgb = _to_rgb(img_t)

        # Single backbone pass → patch tokens + attention-weighted pooling + attention
        patch_toks, attn_pooled, attn_maps = _extract_one(
            backbone, processor, img_t, target_layers_0idx, device, need_attn=True
        )

        # Method 1: Gradient × Input  (separate pass with gradient enabled)
        try:
            grad_map = _method1_grad_x_input(
                img_t, backbone, processor,
                patch_centroids, target_layers_0idx, weights_list, device,
            )
        except Exception as e:
            print(f"  [WARN] Grad×Input failed for image {i}: {e}")
            grad_map = np.zeros((_IMG_SIZE, _IMG_SIZE))

        # Method 2: Patch L2 distance
        patch_map = _method2_patch_l2(
            patch_toks, patch_centroids, weights_list, patch_grid,
        )

        # Method 3: CLS Attention map
        attn_map = _method3_attention(attn_maps, weights_list, patch_grid)

        # Method 4: Cosine anomaly map
        cosine_map = _method4_cosine_anomaly(
            patch_toks, patch_centroids, weights_list, patch_grid,
        )

        # Per-image M3 score
        m3_score, per_layer = _compute_m3_per_image(
            attn_pooled, m3_centroids_norm,
            active_layers_1idx, layer_weights, device,
        )

        # Scalar metrics
        attn_ent   = _attention_entropy(attn_map)
        grad_mean  = float(grad_map.mean())
        grad_max   = float(grad_map.max())
        patch_mean = float(patch_map.mean())
        patch_max  = float(patch_map.max())
        cos_mean   = float(cosine_map.mean())

        # 5-panel figure
        fig_path = os.path.join(output_dir, f"attribution_{i:03d}.png")
        _save_figure(img_rgb, grad_map, patch_map, attn_map, cosine_map,
                     m3_score, per_layer, fig_path, i)

        records.append({
            "idx":                i,
            "path":               p,
            "m3_per_image_score": m3_score,
            "per_layer_distances": per_layer,
            "grad_mean":          round(grad_mean,  6),
            "grad_max":           round(grad_max,   6),
            "patch_l2_mean":      round(patch_mean, 6),
            "patch_l2_max":       round(patch_max,  6),
            "attn_entropy":       round(attn_ent,   6),
            "cosine_anomaly_mean": round(cos_mean,  6),
            "attribution_plot":   fig_path,
        })

    # ── Per-layer breakdown for top-K worst images ───────────────────────────
    records_ranked = sorted(records, key=lambda r: r["m3_per_image_score"], reverse=True)
    layer_fig_paths: list[str] = []
    print(f"Generating per-layer breakdown for top-{layer_breakdown_top_k} worst images ...")
    for rec in records_ranked[:layer_breakdown_top_k]:
        i   = rec["idx"]
        p   = rec["path"]
        img_t   = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        img_rgb = _to_rgb(img_t)
        patch_toks, _, _ = _extract_one(
            backbone, processor, img_t, target_layers_0idx, device, need_attn=False
        )
        layer_maps = _per_layer_patch_l2(
            patch_toks, patch_centroids, active_layers_1idx, patch_grid,
        )
        lfig = os.path.join(output_dir, f"layer_breakdown_{i:03d}.png")
        _save_layer_figure(img_rgb, layer_maps, rec["per_layer_distances"],
                           layer_weights, lfig, i)
        layer_fig_paths.append(lfig)
        rec["layer_breakdown_plot"] = lfig

    # ── Summary plots ─────────────────────────────────────────────────────────
    print("Generating summary plots ...")
    summary_paths = _make_summary_plots(
        records, active_layers_1idx, layer_weights, gen_paths, output_dir,
    )

    # ── Summary statistics ────────────────────────────────────────────────────
    m3_vals = np.array([r["m3_per_image_score"] for r in records])
    layer_mean_dists = {
        f"L{l}": round(float(np.mean([r["per_layer_distances"].get(f"L{l}", 0.0)
                                       for r in records])), 6)
        for l in active_layers_1idx
    }

    results = {
        "active_layers_1indexed":  active_layers_1idx,
        "layer_weights":           layer_weights,
        "patch_grid":              patch_grid,
        "n_real":                  len(real_paths),
        "n_gen":                   len(gen_paths),
        "m3_score_stats": {
            "mean":   round(float(m3_vals.mean()),          4),
            "std":    round(float(m3_vals.std()),           4),
            "median": round(float(np.median(m3_vals)),      4),
            "min":    round(float(m3_vals.min()),           4),
            "max":    round(float(m3_vals.max()),           4),
        },
        "layer_mean_distances":  layer_mean_dists,
        "ranked_by_m3_score":    records_ranked,
        "summary_plots":         summary_paths,
        "layer_breakdown_plots": layer_fig_paths,
    }

    report_path = os.path.join(output_dir, "spatial_attribution_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\nReport saved: {report_path}")
    print(f"  M3 score  mean={m3_vals.mean():.4f}  std={m3_vals.std():.4f}  "
          f"min={m3_vals.min():.4f}  max={m3_vals.max():.4f}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial attribution experiment (4 methods + comprehensive metrics)"
    )
    parser.add_argument("--real_dir",   required=True,
                        help="Real image directory (searched recursively)")
    parser.add_argument("--gen_dir",    required=True,
                        help="Generated image directory (non-recursive)")
    parser.add_argument("--output_dir", default="./results/spatial_attribution")
    parser.add_argument("--n_real",     type=int, default=200)
    parser.add_argument("--n_gen",      type=int, default=20)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--top_k",      type=int, default=5,
                        help="Number of worst images for per-layer breakdown")
    args = parser.parse_args()
    run_spatial_attribution(
        real_dir               = args.real_dir,
        gen_dir                = args.gen_dir,
        output_dir             = args.output_dir,
        n_real                 = args.n_real,
        n_gen                  = args.n_gen,
        device                 = args.device,
        seed                   = args.seed,
        layer_breakdown_top_k  = args.top_k,
    )


if __name__ == "__main__":
    main()
