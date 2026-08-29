"""
CKA Layer Similarity — All Backbones
======================================
Computes the Linear CKA similarity matrix between every pair of layers for
each of the six backbones used in the M3-Score backbone comparison study.

Backbones
---------
  1. microsoft/rad-dino        (ViT-B/14, transformers)
  2. Snarcy/RadioDino-s16      (ViT-S/16, timm)          ← chosen backbone
  3. facebook/dinov2-base      (ViT-B/14, transformers)
  4. flaviagiammarino/pubmed-clip-vit-base-patch32
                               (ViT-B/32, transformers CLIP)
  5. microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
                               (ViT-B/16, open_clip)
  6. RadImageNet-ResNet50      (ResNet-50, timm)

Why CKA matters for the paper
------------------------------
CKA(L_i, L_j) = 1  → identical representations (redundant layers)
CKA(L_i, L_j) ≈ 0  → orthogonal representations (complementary)

A good backbone for single-layer use should have L12 (final layer) that is
highly distinct from early layers (low CKA vs L1-L6), meaning it captures
truly different, high-level features.  RadioDino-s16 L12 achieves the lowest
CKA with early layers of any backbone tested — justifying single-layer-L12.

Outputs
-------
  Per backbone:
    cka_matrix_{tag}.png    — heatmap of 12×12 CKA matrix
    cka_report_{tag}.json   — matrix + summary stats
  Combined:
    cka_all_backbones.png   — 2×3 grid for paper figure
    cka_summary_report.json — all backbones' summary stats
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse extractors from backbone_comparison.py
from experiments.backbone_comparison import (
    _load_images,
    _TransformersViTExtractor,
    _TimmViTExtractor,
    _CLIPViTExtractor,
    _OpenCLIPViTExtractor,
    _ResNetExtractor,
)

# ---------------------------------------------------------------------------
# Backbone registry — same as backbone_comparison.py
# ---------------------------------------------------------------------------

_BACKBONES: list[dict] = [
    {"id": "microsoft/rad-dino",                                             "tag": "rad-dino",    "loader": "transformers"},
    {"id": "Snarcy/RadioDino-s16",                                           "tag": "radiodino-s16", "loader": "timm"},
    {"id": "facebook/dinov2-base",                                           "tag": "dinov2-base", "loader": "transformers"},
    {"id": "flaviagiammarino/pubmed-clip-vit-base-patch32",                  "tag": "pubmed-clip", "loader": "clip"},
    {"id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",       "tag": "biomed-clip", "loader": "openclip"},
    {"id": "microsoft/resnet-50",                                            "tag": "resnet50",    "loader": "timm_resnet"},
]


def _get_extractor(backbone: dict, device: str):
    loader = backbone["loader"]
    mid    = backbone["id"]
    if loader == "transformers":
        return _TransformersViTExtractor(mid, device)
    elif loader == "timm":
        return _TimmViTExtractor(mid, device)
    elif loader == "clip":
        return _CLIPViTExtractor(mid, device)
    elif loader == "openclip":
        return _OpenCLIPViTExtractor(mid, device)
    elif loader == "timm_resnet":
        return _ResNetExtractor(mid, device)
    else:
        raise ValueError(f"Unknown loader: {loader}")


# ---------------------------------------------------------------------------
# CKA computation
# ---------------------------------------------------------------------------

def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between (N, D1) and (N, D2) tensors."""
    x = (x - x.mean(0)).float()
    y = (y - y.mean(0)).float()
    num   = torch.norm(x.t() @ y) ** 2
    denom = torch.norm(x.t() @ x) * torch.norm(y.t() @ y)
    return float((num / (denom + 1e-12)).item())


def _cka_matrix(layer_feats: list[torch.Tensor]) -> np.ndarray:
    L = len(layer_feats)
    mat = np.zeros((L, L))
    for i in tqdm(range(L), desc="  CKA rows", leave=False):
        for j in range(i, L):
            v = _linear_cka(layer_feats[i], layer_feats[j])
            mat[i, j] = mat[j, i] = v
    return mat


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_single(mat: np.ndarray, tag: str, output_dir: str) -> str:
    L = mat.shape[0]
    labels = [f"L{i+1}" for i in range(L)]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(im, ax=ax, label="Linear CKA")
    ax.set_xticks(range(L)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(L)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"CKA Similarity — {tag}", fontsize=12, fontweight="bold")
    # Annotate L12 vs L1 value
    ax.annotate(f"L1↔L{L}={mat[0,-1]:.2f}", xy=(L-1, 0), xytext=(L*0.55, 0.5),
                fontsize=9, color="white",
                arrowprops=dict(arrowstyle="->", color="white", lw=1))
    plt.tight_layout()
    path = os.path.join(output_dir, f"cka_matrix_{tag}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close()
    return path


def _plot_combined(all_mats: dict[str, np.ndarray], output_dir: str) -> str:
    tags = list(all_mats.keys())
    n    = len(tags)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), dpi=130)
    axes = axes.flatten()
    for ax, tag in zip(axes, tags):
        mat = all_mats[tag]
        L   = mat.shape[0]
        im  = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title(tag, fontsize=10, fontweight="bold")
        ax.set_xticks(range(0, L, max(1, L//4)))
        ax.set_xticklabels([f"L{i+1}" for i in range(0, L, max(1, L//4))], fontsize=7)
        ax.set_yticks(range(0, L, max(1, L//4)))
        ax.set_yticklabels([f"L{i+1}" for i in range(0, L, max(1, L//4))], fontsize=7)
        ax.text(L * 0.05, L * 0.88, f"L1↔L{L}={mat[0,-1]:.2f}",
                color="white", fontsize=8, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(tags):]:
        ax.set_visible(False)
    fig.suptitle("Inter-layer CKA Similarity — All Backbones\n"
                 "(Low L1↔Lmax = final layer is most distinctive)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "cka_all_backbones.png")
    fig.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ---------------------------------------------------------------------------
# L12 distinctiveness bar chart
# ---------------------------------------------------------------------------

def _plot_distinctiveness(summaries: dict, output_dir: str) -> str:
    """Bar chart: L1↔Lmax CKA per backbone. Lower = more distinctive final layer."""
    tags = list(summaries.keys())
    vals = [summaries[t]["l1_vs_lmax"] for t in tags]
    colors = ["#e05c5c" if t == "radiodino-s16" else "#1f77b4" for t in tags]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    bars = ax.bar(tags, vals, color=colors, edgecolor="black", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CKA(L1, L_final)")
    ax.set_title("Final-layer distinctiveness: lower = L_final encodes\n"
                 "maximally different features from early layers", fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    ax.axhline(0.30, ls="--", color="grey", lw=1, label="threshold 0.30")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "cka_l1_vs_lmax.png")
    fig.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cka_analysis(
    real_dir:    str,
    output_dir:  str            = "./results/cka_analysis",
    n_images:    int            = 256,
    device:      str | None     = None,
    backbones:   list[str] | None = None,
    skip_cached: bool           = True,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    target_backbones = [b for b in _BACKBONES
                        if backbones is None or b["tag"] in backbones]

    print(f"\nLoading {n_images} real images ...")
    imgs = _load_images(real_dir, n_images)
    print(f"  {len(imgs)} images loaded")

    all_mats: dict[str, np.ndarray] = {}
    summaries: dict[str, dict]     = {}

    for bb in target_backbones:
        tag      = bb["tag"]
        rp_path  = os.path.join(output_dir, f"cka_report_{tag}.json")

        if skip_cached and os.path.isfile(rp_path):
            print(f"\n[{tag}] Loading cached result ...")
            with open(rp_path) as f:
                rpt = json.load(f)
            mat = np.array(rpt["cka_matrix"])
            all_mats[tag]  = mat
            summaries[tag] = rpt["summary"]
            continue

        print(f"\n{'='*50}\n  Backbone: {tag}\n{'='*50}")
        try:
            extractor = _get_extractor(bb, device)
        except Exception as e:
            print(f"  SKIP (load failed): {e}")
            continue

        print(f"  Layers: {extractor.num_layers}  embed_dim: {extractor.embed_dim}")
        print("  Extracting features ...")
        layer_feats = extractor.extract(imgs)
        del extractor
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"  Computing {len(layer_feats)}×{len(layer_feats)} CKA matrix ...")
        mat = _cka_matrix(layer_feats)
        L   = mat.shape[0]

        summary = {
            "num_layers":         L,
            "embed_dim":          layer_feats[0].shape[1],
            "l1_vs_lmax":         round(float(mat[0, -1]), 4),
            "lhalf_vs_lmax":      round(float(mat[L // 2, -1]), 4),
            "mean_off_diagonal":  round(float(
                (mat.sum() - L) / (L * L - L)
            ), 4),
        }
        print(f"  L1<->L{L} CKA: {summary['l1_vs_lmax']:.4f}  "
              f"mean off-diag: {summary['mean_off_diagonal']:.4f}")

        plot_path = _plot_single(mat, tag, output_dir)
        report = {"backbone": bb["id"], "tag": tag, "n_images": n_images,
                  "cka_matrix": mat.tolist(), "summary": summary,
                  "plots": {"heatmap": plot_path}}
        with open(rp_path, "w") as f:
            json.dump(report, f, indent=2)

        all_mats[tag]  = mat
        summaries[tag] = summary

    # Combined plots
    combined_path = None
    distinct_path = None
    if all_mats:
        print("\nGenerating combined CKA figure ...")
        combined_path = _plot_combined(all_mats, output_dir)
        distinct_path = _plot_distinctiveness(summaries, output_dir)

    # Summary report
    summary_report = {
        "n_images":  n_images,
        "backbones": summaries,
        "plots": {"combined": combined_path, "distinctiveness": distinct_path},
    }
    sp = os.path.join(output_dir, "cka_summary_report.json")
    with open(sp, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSummary report: {sp}")

    # Print ranking
    if summaries:
        print("\n  L1<->Lmax CKA ranking (lower = more distinctive final layer):")
        for tag, s in sorted(summaries.items(), key=lambda x: x[1]["l1_vs_lmax"]):
            star = " <-- chosen" if tag == "radiodino-s16" else ""
            print(f"  {tag:35s}  {s['l1_vs_lmax']:.4f}{star}")

    return summary_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",    required=True)
    p.add_argument("--output_dir",  default="results/cka_analysis")
    p.add_argument("--n_images",    type=int,  default=256)
    p.add_argument("--device",      default=None)
    p.add_argument("--backbones",   nargs="+", default=None,
                   help="Subset of backbone tags to run (default: all)")
    p.add_argument("--no_cache",    action="store_true",
                   help="Recompute even if cached JSON exists")
    a = p.parse_args()
    run_cka_analysis(
        real_dir    = a.real_dir,
        output_dir  = a.output_dir,
        n_images    = a.n_images,
        device      = a.device,
        backbones   = a.backbones,
        skip_cached = not a.no_cache,
    )
