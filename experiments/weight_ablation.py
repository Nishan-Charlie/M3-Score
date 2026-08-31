"""
CKA Threshold Ablation Study (Section 3.8)
============================================
The M3-Score uses CKA to prune redundant transformer layers before
computing weighted MMD² distances. The CKA threshold tau controls how
aggressively layers are pruned: higher tau retains fewer layers.

This study sweeps tau over a range and at each value reports:
  - How many layers are retained.
  - The M3 score on real-vs-generated (RG) and real-vs-real (RR).
  - Discriminability = (RG - RR) / RR, measuring how well the threshold
    separates generated from real images.

The threshold with the highest discriminability is identified as the
recommended setting.

Note on naming: this function is named run_weight_ablation to match the
import in run_experiments.py (which refers to Section 3.8 weight/threshold
ablation). It ablates the CKA threshold, not the layer weights.

Implementation note:
  RadioDino features are extracted ONCE for all images across ALL thresholds.
  Only the CKA selection step (which operates on pre-extracted features via
  gram matrices) is repeated per threshold. This avoids loading RadioDino
  from disk once per threshold (previously 8 loads).

Usage:
    python weight_ablation.py \\
        --real_dir <path> --gen_dir <path> --output_dir <path> \\
        --n 500 --device cuda
"""

from __future__ import annotations

import argparse
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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.m3_score_v2 import M3V2Metric


# CKA thresholds to evaluate
CKA_THRESHOLDS = [0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_images(directory: str, n: int) -> torch.Tensor:
    """
    Load up to n images as a CPU float tensor of shape (N, 3, 224, 224)
    in [0, 1]. Device transfer is handled inside M3V2Metric._preprocess.
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
    return torch.stack(imgs)   # CPU tensor


# ---------------------------------------------------------------------------
# CKA layer selection on pre-extracted gram matrices
# ---------------------------------------------------------------------------

def _cka_select_layers(
    cls_feats:  list[torch.Tensor],
    tau:        float,
    num_layers: int,
    metric:     "M3V2Metric",
) -> list[int]:
    """
    Apply CKA greedy backward selection at threshold tau using CLS token
    features, matching M3V2Metric.prune_layers_via_cka exactly.

    Args:
        cls_feats:  List of (N, D) CLS feature tensors, one per layer (0-indexed).
        tau:        Redundancy threshold.
        num_layers: Total number of transformer layers.
        metric:     M3V2Metric instance used for _linear_cka.

    Returns:
        Sorted list of 1-indexed selected layer indices.
    """
    selected = [num_layers]
    for i in range(num_layers - 1, 0, -1):
        redundant = False
        for s in selected:
            f_i = cls_feats[i - 1]
            f_s = cls_feats[s - 1]
            if f_i.numel() == 0 or f_s.numel() == 0:
                continue
            cka = metric._linear_cka(f_i, f_s)
            if cka > tau:
                redundant = True
                break
        if not redundant:
            selected.append(i)
    return sorted(selected)


# ---------------------------------------------------------------------------
# Equal-weight score from pre-extracted features
# ---------------------------------------------------------------------------

def _equal_weight_score(
    metric:        M3V2Metric,
    feats_a:       list[torch.Tensor],
    feats_b:       list[torch.Tensor],
    active_layers: list[int],
    device:        str,
) -> float:
    """
    Compute mean MMD² across active layers with equal weights.
    Operates on pre-extracted feature lists to avoid redundant backbone calls.
    """
    total = 0.0
    for layer in active_layers:
        fa = feats_a[layer - 1].to(device)
        fb = feats_b[layer - 1].to(device)
        total += metric._compute_mmd2(fa, fb).item()
    return total / max(len(active_layers), 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_weight_ablation(
    real_dir:    str,
    gen_dir:     str,
    output_dir:  str          = "./weight_ablation_output",
    n:           int          = 500,
    device:      Optional[str] = None,
    seed:        int          = 42,
    backbone_id: str          = "Snarcy/RadioDino-s16",
) -> dict:
    """
    Run the CKA threshold ablation study (Section 3.8).

    RadioDino features are extracted once for all images. CKA layer selection
    is recomputed at each threshold using the pre-extracted gram matrices,
    avoiding repeated backbone loading.

    Args:
        real_dir:   Directory of real images.
        gen_dir:    Directory of generated images.
        output_dir: Destination for plots and JSON report.
        n:          Images loaded per set.
        device:     Torch device string.
        seed:       Random seed.

    Returns:
        dict with per-threshold results and best threshold, compatible with
        master_report["weight_ablation"] in run_experiments.py.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading images ...")
    real_imgs = _load_images(real_dir, n)
    gen_imgs  = _load_images(gen_dir,  n)

    # Deterministic real-vs-real split
    half   = len(real_imgs) // 2
    real_a = real_imgs[:half]
    real_b = real_imgs[half:2 * half]

    # Guard: ensure gen set has at least half images for a fair comparison
    gen_half = gen_imgs[:half]
    if len(gen_imgs) < half:
        print(
            f"  [WARN] gen_dir has only {len(gen_imgs)} images; "
            f"using all for the generated half (expected >= {half})."
        )
        gen_half = gen_imgs

    # ── Instantiate metric once (backbone loaded once) ────────────────────────
    print("\nInitialising M3-Score backbone (loaded once for all thresholds) ...")
    metric = M3V2Metric(device=device, backbone_id=backbone_id)

    # ── Extract features for all layers once ─────────────────────────────────
    all_layers = set(range(1, metric.num_layers + 1))
    print("Extracting all-layer features (real_a, real_b, gen) ...")

    with torch.no_grad():
        feats_real_a = metric._extract_raw_features(
            real_a, use_attention=True, layers_to_keep=all_layers
        )
        feats_real_b = metric._extract_raw_features(
            real_b, use_attention=True, layers_to_keep=all_layers
        )
        feats_gen = metric._extract_raw_features(
            gen_half, use_attention=True, layers_to_keep=all_layers
        )

    # ── Extract CLS features for CKA selection (matches M3V2Metric.prune_layers_via_cka) ──
    prune_n = min(20, len(real_a))
    print("Extracting CLS features for CKA layer selection ...")
    with torch.no_grad():
        cls_feats = metric._extract_cls_features(real_a[:prune_n])

    # ── Threshold sweep (no backbone calls inside the loop) ───────────────────
    rows: list[dict] = []
    print(f"\nEvaluating {len(CKA_THRESHOLDS)} CKA thresholds ...")

    for tau in tqdm(CKA_THRESHOLDS, desc="CKA thresholds"):
        active_layers = _cka_select_layers(cls_feats, tau, metric.num_layers, metric)
        n_active = len(active_layers)

        score_rg = _equal_weight_score(
            metric, feats_real_a, feats_gen,    active_layers, device
        )
        score_rr = _equal_weight_score(
            metric, feats_real_a, feats_real_b, active_layers, device
        )

        # Discriminability: how much larger is RG than RR?
        # Positive values indicate the metric correctly ranks generated images
        # as further from real than real images are from each other.
        # Negative values indicate the threshold is too aggressive (all layers
        # pruned, score collapses, or generated accidentally scores below RR).
        disc = (score_rg - score_rr) / (score_rr + 1e-10)

        rows.append({
            "threshold":        tau,
            "n_active_layers":  n_active,
            "active_layers":    active_layers,
            "score_rg":         round(score_rg, 6),
            "score_rr":         round(score_rr, 6),
            "discriminability": round(disc, 4),
        })
        print(
            f"  tau={tau:.2f}  layers={n_active}  "
            f"RG={score_rg:.4f}  RR={score_rr:.4f}  Disc={disc:.3f}"
        )

    best_row = max(rows, key=lambda r: r["discriminability"])
    best_tau = best_row["threshold"]
    print(
        f"\nBest CKA threshold: {best_tau}  "
        f"(discriminability={best_row['discriminability']:.3f})"
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    sorted_rows = sorted(rows, key=lambda r: r["threshold"])
    thresholds  = [r["threshold"]        for r in sorted_rows]
    discs       = [r["discriminability"] for r in sorted_rows]
    n_layers    = [r["n_active_layers"]  for r in sorted_rows]
    scores_rg   = [r["score_rg"]        for r in sorted_rows]
    scores_rr   = [r["score_rr"]        for r in sorted_rows]

    color_disc   = "#1565C0"   # publication-quality deep blue
    color_layers = "#c62828"   # deep red for bars
    color_rg     = "#e65100"   # deep orange — real vs generated
    color_rr     = "#2e7d32"   # deep green  — real vs real
    color_best   = "#6a1a9a"   # purple dashed — best tau marker

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor("white")

    # Panel 1: Discriminability + active layer count
    ax1 = axes[0]
    ax1.set_facecolor("white")
    ax2 = ax1.twinx()
    ax1.plot(thresholds, discs, "o-", color=color_disc, lw=2.5, markersize=8,
             label="Discriminability", zorder=3)
    ax1.axvline(best_tau, color=color_best, lw=1.8, linestyle="--",
                label=f"Best $\\tau$ = {best_tau}", zorder=2)
    ax1.set_xlabel("CKA threshold $\\tau$", fontsize=11, color="#222222")
    ax1.set_ylabel("Discriminability  $(S_\\mathrm{RG} - S_\\mathrm{RR})\\ /\\ S_\\mathrm{RR}$",
                   color=color_disc, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color_disc)
    ax1.tick_params(axis="x", colors="#333333")
    ax1.set_title(
        "Discriminability vs. CKA threshold $\\tau$\n"
        "($\\tau=0.80$: recommended default, underlined)",
        fontsize=11, color="#222222"
    )
    ax1.grid(alpha=0.35, color="#dddddd")
    ax2.bar(thresholds, n_layers, width=0.015, color=color_layers,
            alpha=0.30, label="Active layers $M$", zorder=1)
    ax2.set_ylabel("Active layers retained ($M$)", color=color_layers, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color_layers)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # legend placed below the axes so it never sits on top of the curves
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
               facecolor="white", edgecolor="#cccccc",
               loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=3, borderaxespad=0.0)

    # Panel 2: Raw RG and RR scores by threshold
    ax = axes[1]
    ax.set_facecolor("white")
    ax.plot(thresholds, scores_rg, "o-", color=color_rg, lw=2.5,
            markersize=8, label="$S_{\\mathrm{RG}}$ (real vs generated)")
    ax.plot(thresholds, scores_rr, "s--", color=color_rr, lw=2.5,
            markersize=8, label="$S_{\\mathrm{RR}}$ (real vs real)")
    ax.axvline(best_tau, color=color_best, lw=1.8, linestyle="--",
               label=f"Best $\\tau$ = {best_tau}")
    ax.set_xlabel("CKA threshold $\\tau$", fontsize=11, color="#222222")
    ax.set_ylabel("M3 score (equal-weight)", fontsize=11, color="#222222")
    ax.set_title("Raw $S_\\mathrm{RG}$ and $S_\\mathrm{RR}$ vs. CKA threshold $\\tau$",
                 fontsize=11, color="#222222")
    ax.tick_params(colors="#333333")
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, borderaxespad=0.0)
    ax.grid(alpha=0.35, color="#dddddd")

    plt.suptitle(
        "M3-Score: CKA Threshold Ablation",
        fontsize=13, fontweight="bold", color="#222222",
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "weight_ablation_discriminability.png")
    plt.savefig(plot_path, bbox_inches="tight", facecolor="white")
    plt.close()

    # ── Save report ───────────────────────────────────────────────────────────
    results = {
        "cka_thresholds_tested": CKA_THRESHOLDS,
        "best_threshold":        best_tau,
        "best_discriminability": best_row["discriminability"],
        "rows":                  rows,
        "plots":                 {"discriminability": plot_path},
    }
    report_path = os.path.join(output_dir, "weight_ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Report saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CKA threshold ablation study (Section 3.8)"
    )
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="./weight_ablation_output")
    parser.add_argument("--n",          type=int, default=500)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run_weight_ablation(
        real_dir   = args.real_dir,
        gen_dir    = args.gen_dir,
        output_dir = args.output_dir,
        n          = args.n,
        device     = args.device,
        seed       = args.seed,
    )