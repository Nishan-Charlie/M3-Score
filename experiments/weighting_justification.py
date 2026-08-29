"""
Weighting Justification Experiment
====================================
Compares three M3 weighting strategies on per-image OOD discrimination AUC:

  1. Adaptive  — entropy × stability × uniqueness (current scheme)
  2. Uniform   — equal weight across all active layers
  3. Best-single — the single layer with highest individual OOD AUC

If adaptive does not beat both baselines, the weighting adds no value and
the simpler alternatives should be reported in the paper instead.

Output:
  weighting_justification_report.json   — per-strategy AUC table
  weighting_justification.png           — grouped bar chart
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
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.m3_score_v2 import M3V2Metric


def _load_paths(directory: str, n: Optional[int] = None, recursive: bool = True) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        if recursive:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        else:
            paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(set(paths))
    return paths[:n] if n else paths


def _l2norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-8, 1.0, norms)


def run_weighting_justification(
    real_dir:    str,
    gen_dir:     str,
    output_dir:  str           = "./weighting_justification",
    n_images:    int           = 500,
    device:      Optional[str] = None,
    seed:        int           = 42,
    backbone_id: str           = "Snarcy/RadioDino-s16",
    cka_tau:     float         = 0.80,
    layers_cache: Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load images ──────────────────────────────────────────────────────────
    real_paths = _load_paths(real_dir, n_images)
    gen_paths  = _load_paths(gen_dir,  n_images, recursive=False)
    cap = min(len(real_paths), len(gen_paths))
    real_paths, gen_paths = real_paths[:cap], gen_paths[:cap]

    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    real_imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in real_paths])
    gen_imgs  = torch.stack([tfm(Image.open(p).convert("RGB")) for p in gen_paths])

    # ── Initialise metric ─────────────────────────────────────────────────────
    metric = M3V2Metric(device=device, backbone_id=backbone_id,
                        cka_threshold=cka_tau, seed=seed)
    cka_ref_n    = min(200, len(real_imgs))
    gen_seed     = torch.Generator().manual_seed(seed)
    cka_ref_imgs = real_imgs[torch.randperm(len(real_imgs), generator=gen_seed)[:cka_ref_n]]
    metric.prune_layers_via_cka(cka_ref_imgs, cache_path=layers_cache, seed=seed)
    active = metric.active_layers
    print(f"Active layers: {active}")

    # ── Extract per-layer features ────────────────────────────────────────────
    print("Extracting features ...")
    active_set = set(active)
    with torch.no_grad():
        real_feats = metric._extract_raw_features(real_imgs, layers_to_keep=active_set)
        gen_feats  = metric._extract_raw_features(gen_imgs,  layers_to_keep=active_set)

    # per-layer features indexed by layer number
    rf_map = {l: _l2norm(real_feats[l - 1].numpy()) for l in active}
    gf_map = {l: _l2norm(gen_feats[l - 1].numpy())  for l in active}

    # ── Per-layer OOD AUC (centroid-L2 distance, held-out centroid split) ────
    rng_split = np.random.default_rng(seed)
    all_idx   = rng_split.permutation(cap)
    fit_idx   = all_idx[:cap // 2]
    eval_idx  = all_idx[cap // 2:]
    labels    = np.concatenate([np.zeros(len(eval_idx)), np.ones(cap)])

    layer_auc: dict[int, float] = {}
    layer_real_dist: dict[int, np.ndarray] = {}
    layer_gen_dist:  dict[int, np.ndarray] = {}

    for l in active:
        rf, gf    = rf_map[l], gf_map[l]
        centroid  = rf[fit_idx].mean(axis=0)
        r_dist    = np.linalg.norm(rf[eval_idx] - centroid, axis=1)
        g_dist    = np.linalg.norm(gf           - centroid, axis=1)
        all_dist  = np.concatenate([r_dist, g_dist])
        layer_auc[l] = float(roc_auc_score(labels, all_dist))
        layer_real_dist[l] = r_dist
        layer_gen_dist[l]  = g_dist
        print(f"  L{l:2d}  AUC = {layer_auc[l]:.4f}")

    # ── Run full forward() to get adaptive weights ────────────────────────────
    print("Running adaptive forward() ...")
    with torch.no_grad():
        result = metric(real_imgs, gen_imgs)
    adaptive_weights = {l: result["layer_weights"][f"L{l}"] for l in active}

    # ── Strategy 1: Adaptive ─────────────────────────────────────────────────
    def _weighted_auc(weights: dict[int, float]) -> float:
        r_comb = sum(weights[l] * layer_real_dist[l] for l in active)
        g_comb = sum(weights[l] * layer_gen_dist[l]  for l in active)
        return float(roc_auc_score(labels, np.concatenate([r_comb, g_comb])))

    auc_adaptive = _weighted_auc(adaptive_weights)

    # ── Strategy 2: Uniform ───────────────────────────────────────────────────
    unif = {l: 1.0 / len(active) for l in active}
    auc_uniform = _weighted_auc(unif)

    # ── Strategy 3: Best single layer ────────────────────────────────────────
    best_layer = max(layer_auc, key=layer_auc.__getitem__)
    auc_best_single = layer_auc[best_layer]

    print(f"\nWeighting strategy comparison:")
    print(f"  Adaptive      AUC = {auc_adaptive:.4f}  weights = { {f'L{l}': round(w,3) for l,w in adaptive_weights.items()} }")
    print(f"  Uniform       AUC = {auc_uniform:.4f}")
    print(f"  Best single (L{best_layer}) AUC = {auc_best_single:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    strategies  = ["Adaptive\n(entropy×SNR×CKA)", f"Uniform\n(1/{len(active)} each)", f"Best single\n(L{best_layer})"]
    auc_vals    = [auc_adaptive, auc_uniform, auc_best_single]
    bar_colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=130)

    ax = axes[0]
    bars = ax.bar(strategies, auc_vals, color=bar_colors, edgecolor="black", width=0.5)
    for bar, v in zip(bars, auc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(min(auc_vals) * 0.97, 1.02)
    ax.set_ylabel("OOD ROC-AUC", fontsize=12)
    ax.set_title("Weighting strategy comparison\n(higher = better)", fontsize=12)
    ax.axhline(auc_adaptive, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    layer_labels = [f"L{l}" for l in active]
    layer_aucs   = [layer_auc[l] for l in active]
    layer_wts    = [adaptive_weights[l] for l in active]
    x = np.arange(len(active))
    ax2 = ax.twinx()
    bars2 = ax.bar(x - 0.2, layer_aucs, width=0.35, color="#4fc3f7", edgecolor="black", label="Layer AUC")
    bars3 = ax2.bar(x + 0.2, layer_wts,  width=0.35, color="#ef9a9a", edgecolor="black", label="Adaptive weight")
    ax.set_xticks(x); ax.set_xticklabels(layer_labels, fontsize=11)
    ax.set_ylabel("Per-layer OOD AUC", fontsize=11)
    ax2.set_ylabel("Adaptive weight", fontsize=11)
    ax.set_title("Per-layer AUC vs adaptive weight\n(ideal: high weight on high-AUC layers)", fontsize=11)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "weighting_justification.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    report = {
        "backbone":          backbone_id,
        "active_layers":     active,
        "layer_auc":         {f"L{l}": round(v, 4) for l, v in layer_auc.items()},
        "adaptive_weights":  {f"L{l}": round(v, 4) for l, v in adaptive_weights.items()},
        "auc_adaptive":      round(auc_adaptive,    4),
        "auc_uniform":       round(auc_uniform,     4),
        "auc_best_single":   round(auc_best_single, 4),
        "best_layer":        best_layer,
        "adaptive_beats_uniform":     bool(auc_adaptive > auc_uniform),
        "adaptive_beats_best_single": bool(auc_adaptive > auc_best_single),
        "plots": {"weighting_justification": plot_path},
    }
    report_path = os.path.join(output_dir, "weighting_justification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",    required=True)
    p.add_argument("--gen_dir",     required=True)
    p.add_argument("--output_dir",  default="results/weighting_justification")
    p.add_argument("--n_images",    type=int,   default=500)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    a = p.parse_args()
    run_weighting_justification(**vars(a))
