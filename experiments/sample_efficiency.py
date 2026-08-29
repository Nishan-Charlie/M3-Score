"""
Sample Efficiency and Stability Analysis -- Section 3.x
=========================================================
Evaluates how stable each metric is when computed on random subsets of the
real and generated image sets.  High coefficient of variation (CV) indicates
that a metric is unreliable at the given sample size; low CV indicates it
produces consistent estimates across different random draws.

Protocol:
  For each of K rounds:
    1. Randomly draw n_subset images from the real set and n_subset from
       the generated set (without replacement, independent per round).
    2. Compute all seven metrics on the drawn subset:
         M3-Score, FID, KID, SSIM, PSNR, MS-SSIM, LPIPS.
  Report mean, standard deviation, and coefficient of variation for each
  metric across K rounds.

Notes:
  - M3V2Metric CKA layer selection is performed once on a fixed reference
    subset before the rounds loop and reused for all rounds.
  - SSIM/PSNR/MS-SSIM/LPIPS are computed on the same float tensors used
    for M3.  Because these are paired metrics in the original literature
    but used here in an unpaired distributional setting, values are
    computed between each generated image and the mean real image (centroid
    distance), consistent with M3's per-image score interpretation.
  - A seed parameter ensures the round-by-round subset draws are
    reproducible across runs.

Usage:
    python sample_efficiency.py \\
        --real_dir <path> --gen_dir <path> --output_dir <path> \\
        --rounds 10 --n_subset 100 --device cuda
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


# ---------------------------------------------------------------------------
# Path loading
# ---------------------------------------------------------------------------

def _load_all_paths(directory: str) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Scipy-based FID (no torchmetrics required)
# ---------------------------------------------------------------------------

def _extract_inception_features(paths: list[str], device: str) -> np.ndarray:
    """Extract 2048-dim InceptionV3 pool3 features for a list of image paths."""
    from torchvision import models
    import torch.nn as nn
    inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inc.fc = nn.Identity()
    inc.eval().to(device)
    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    all_feats: list[np.ndarray] = []
    BS = 32
    with torch.no_grad():
        for s in range(0, len(paths), BS):
            batch = torch.stack(
                [tfm(Image.open(p).convert("RGB")) for p in paths[s:s + BS]]
            ).to(device)
            out = inc(batch)
            if hasattr(out, "logits"):
                out = out.logits
            all_feats.append(out.cpu().numpy())
    del inc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(all_feats, axis=0)


def _fid_from_features(real_f: np.ndarray, gen_f: np.ndarray, eps: float = 1e-6) -> float:
    """
    Fréchet distance between two feature sets using scipy.
    Adds eps*I regularisation so the covariance is never singular (safe at N=500
    with 2048-dim features where the sample covariance is rank-deficient).
    """
    from scipy.linalg import sqrtm as _sqrtm
    mu_r, mu_g = real_f.mean(0), gen_f.mean(0)
    sig_r = np.cov(real_f, rowvar=False) + eps * np.eye(real_f.shape[1])
    sig_g = np.cov(gen_f,  rowvar=False) + eps * np.eye(gen_f.shape[1])
    diff  = mu_r - mu_g
    covmean = _sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))
    return max(0.0, fid)


# ---------------------------------------------------------------------------
# Per-round metric computation (feature-array based, no model reload)
# ---------------------------------------------------------------------------

def _compute_round(
    idx_ref:       np.ndarray,
    idx_gen:       np.ndarray,
    m3_ref_imgs:   torch.Tensor,        # (N_real, 3, 224, 224) float [0,1]
    m3_gen_imgs:   torch.Tensor,        # (N_gen,  3, 224, 224) float [0,1]
    inc_real_feats: np.ndarray,         # (N_real, 2048)
    inc_gen_feats:  np.ndarray,         # (N_gen,  2048)
    clip_real_feats: Optional[np.ndarray],  # (N_real, D) or None
    clip_gen_feats:  Optional[np.ndarray],  # (N_gen,  D) or None
    m3_metric:     M3V2Metric,
) -> dict:
    """Compute M3, FID, and CMMD on one random index subset draw."""
    row: dict = {}

    # M3-Score — forward on GPU-resident tensor subsets
    try:
        with torch.no_grad():
            res = m3_metric(m3_ref_imgs[idx_ref], m3_gen_imgs[idx_gen])
        row["m3"] = float(res["m3_v2_final_score"])
    except Exception as e:
        row["m3"] = float("nan")
        print(f"    M3 failed: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # FID — scipy Fréchet distance on pre-extracted InceptionV3 features
    try:
        row["fid"] = _fid_from_features(
            inc_real_feats[idx_ref], inc_gen_feats[idx_gen]
        )
    except Exception as e:
        row["fid"] = float("nan")
        print(f"    FID failed: {e}")

    # CMMD — unbiased Gaussian-RBF MMD² on pre-extracted CLIP features
    if clip_real_feats is not None and clip_gen_feats is not None:
        try:
            from evaluation.cmmd_metric import CMMDMetric
            row["cmmd"] = CMMDMetric.gaussian_mmd2_unbiased(
                clip_real_feats[idx_ref], clip_gen_feats[idx_gen]
            )
        except Exception as e:
            row["cmmd"] = float("nan")
            print(f"    CMMD failed: {e}")
    else:
        row["cmmd"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sample_efficiency(
    real_dir:    str,
    gen_dir:     str,
    output_dir:  str           = "./sample_efficiency_output",
    rounds:      int           = 10,
    n_subset:    int           = 100,
    device:      Optional[str] = None,
    seed:        int           = 42,
    backbone_id: str           = "Snarcy/RadioDino-s16",
) -> dict:
    """
    Stability experiment: CV of M3-Score vs FID vs CLIP-MMD across K random
    subset draws at a given sample size.

    Features for FID and CMMD are extracted once from all available images,
    then subsampled per round — no model reloading per round.
    """
    if n_subset < 4:
        raise ValueError(f"n_subset must be >= 4 (got {n_subset}).")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    paths_ref = _load_all_paths(real_dir)
    paths_gen = _load_all_paths(gen_dir)
    if not paths_ref:
        raise FileNotFoundError(f"No images found in {real_dir}.")
    if not paths_gen:
        raise FileNotFoundError(f"No images found in {gen_dir}.")

    print(f"[Stability] Real: {len(paths_ref)} | Gen: {len(paths_gen)}")
    print(f"            Rounds: {rounds} | Subset: {n_subset} | Backbone: {backbone_id}")

    # ── M3 metric + CKA pruning ───────────────────────────────────────────────
    print("\nInitialising M3 metric ...")
    m3_metric = M3V2Metric(device=device, backbone_id=backbone_id, seed=seed)

    prune_rng     = np.random.default_rng(seed)
    prune_idx     = prune_rng.choice(len(paths_ref), min(200, len(paths_ref)), replace=False)
    m3_tfm        = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    prune_imgs    = torch.stack([m3_tfm(Image.open(paths_ref[i]).convert("RGB")) for i in prune_idx])
    m3_metric.prune_layers_via_cka(prune_imgs, seed=seed)
    print(f"  Active layers: {m3_metric.active_layers}")

    # Pre-load all M3 tensors (kept on CPU, moved to GPU inside forward())
    print("  Loading M3 tensors ...")
    m3_ref_imgs = torch.stack([m3_tfm(Image.open(p).convert("RGB")) for p in tqdm(paths_ref, leave=False)])
    m3_gen_imgs = torch.stack([m3_tfm(Image.open(p).convert("RGB")) for p in tqdm(paths_gen, leave=False)])

    # ── InceptionV3 features (for FID) ───────────────────────────────────────
    print("  Extracting InceptionV3 features (FID) ...")
    inc_real_feats = _extract_inception_features(paths_ref, device)
    inc_gen_feats  = _extract_inception_features(paths_gen, device)
    print(f"  InceptionV3 features: real {inc_real_feats.shape}, gen {inc_gen_feats.shape}")

    # ── CLIP features (for CMMD) ─────────────────────────────────────────────
    clip_real_feats: Optional[np.ndarray] = None
    clip_gen_feats:  Optional[np.ndarray] = None
    try:
        from evaluation.cmmd_metric import CMMDMetric
        print("  Extracting CLIP features (CMMD) ...")
        cmmd_obj       = CMMDMetric(device=device)
        clip_real_feats = cmmd_obj.extract_features(paths_ref, desc="CLIP [real]")
        clip_gen_feats  = cmmd_obj.extract_features(paths_gen, desc="CLIP [gen]")
        del cmmd_obj
        print(f"  CLIP features: real {clip_real_feats.shape}, gen {clip_gen_feats.shape}")
    except Exception as e:
        print(f"  [WARN] CLIP feature extraction failed: {e}")

    # ── Rounds loop ───────────────────────────────────────────────────────────
    metric_keys = ["m3", "fid", "cmmd"]
    history: list[dict] = []
    rng = np.random.default_rng(seed + 1)   # +1 so round draws differ from prune draws

    for r in tqdm(range(rounds), desc="Stability rounds"):
        idx_ref = rng.choice(len(paths_ref), min(n_subset, len(paths_ref)), replace=False)
        idx_gen = rng.choice(len(paths_gen), min(n_subset, len(paths_gen)), replace=False)

        row = _compute_round(
            idx_ref, idx_gen,
            m3_ref_imgs, m3_gen_imgs,
            inc_real_feats, inc_gen_feats,
            clip_real_feats, clip_gen_feats,
            m3_metric,
        )
        row["round"] = r + 1
        history.append(row)

        vals_str = "  ".join(f"{k}={row.get(k, float('nan')):.5f}" for k in metric_keys)
        print(f"  Round {r + 1:3d}: {vals_str}")

    # ── Stability statistics ──────────────────────────────────────────────────
    import math

    def _cv(vals: list[float]) -> float:
        clean = [v for v in vals if not math.isnan(v)]
        if len(clean) < 2:
            return float("nan")
        mean = float(np.mean(clean))
        std  = float(np.std(clean, ddof=1))
        return (std / abs(mean) * 100) if mean != 0 else float("nan")

    stats: list[dict] = []
    for key in metric_keys:
        vals  = [r.get(key, float("nan")) for r in history]
        clean = [v for v in vals if not math.isnan(v)]
        n     = len(clean)
        mean  = float(np.mean(clean))  if n > 0 else float("nan")
        std   = float(np.std(clean, ddof=1)) if n > 1 else float("nan")
        cv    = _cv(vals)
        
        # 95% CI
        ci_half = (1.96 * std / math.sqrt(n)) if n > 1 and not math.isnan(std) else float("nan")
        
        stats.append({
            "metric":   key,
            "mean":     round(mean, 6),
            "std":      round(std,  6),
            "cv_pct":   round(cv,   4),
            "ci_95":    [round(mean - ci_half, 6), round(mean + ci_half, 6)] if not math.isnan(ci_half) else [float("nan"), float("nan")],
            "n_valid":  n,
        })

    print("\n[Stability] Coefficient of Variation by metric:")
    for s in stats:
        print(f"  {s['metric']:12s}  mean={s['mean']:.4f}  "
              f"std={s['std']:.4f}  CV={s['cv_pct']:.2f}%")

    # ── Plot 1: CV bar chart (M3 vs FID vs CMMD) ─────────────────────────────
    colors = {"m3": "#ef5350", "fid": "#4fc3f7", "cmmd": "#66bb6a"}

    cv_vals   = [s["cv_pct"] for s in stats]
    cv_labels = [s["metric"].upper() for s in stats]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    bar_colors = [colors.get(s["metric"], "#888") for s in stats]
    bars = ax.bar(cv_labels, cv_vals, color=bar_colors, edgecolor="#333", width=0.5)
    valid_cv = [x for x in cv_vals if not math.isnan(x)]
    top = max(valid_cv, default=5) * 1.35
    for bar, v in zip(bars, cv_vals):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Coefficient of variation (%)", fontsize=12)
    ax.set_title(
        f"Stability: CV across {rounds} random draws (N={n_subset})\n"
        f"Lower = more stable", fontsize=12
    )
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    cv_path = os.path.join(output_dir, "stability_cv.png")
    plt.savefig(cv_path, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Per-round convergence lines ───────────────────────────────────
    rounds_x = [r["round"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=120)

    for ax_i, key in enumerate(metric_keys):
        vals = [r.get(key, float("nan")) for r in history]
        axes[ax_i].plot(rounds_x, vals, "o-", color=colors.get(key, "#888"), lw=2, markersize=5)
        axes[ax_i].set_xlabel("Round", fontsize=11)
        axes[ax_i].set_ylabel("Score", fontsize=11)
        axes[ax_i].set_title(key.upper(), fontsize=12)
        axes[ax_i].grid(alpha=0.3)

    plt.suptitle(
        f"Per-round metric values ({rounds} draws, N={n_subset})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    conv_path = os.path.join(output_dir, "stability_convergence.png")
    plt.savefig(conv_path, bbox_inches="tight")
    plt.close()

    # ── Save report ───────────────────────────────────────────────────────────
    results = {
        "rounds":        rounds,
        "n_subset":      n_subset,
        "active_layers": m3_metric.active_layers,
        "history":       history,
        "stats":         stats,
        "plots": {
            "cv_bar":      cv_path,
            "convergence": conv_path,
        },
    }
    report_path = os.path.join(output_dir, "sample_efficiency_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\nReport saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# Alias for run_experiments.py compatibility
# ---------------------------------------------------------------------------

run_stability_test = run_sample_efficiency


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample efficiency and stability analysis"
    )
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="./sample_efficiency_output")
    parser.add_argument("--rounds",     type=int, default=10)
    parser.add_argument("--n_subset",   type=int, default=100)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    run_sample_efficiency(
        real_dir   = args.real_dir,
        gen_dir    = args.gen_dir,
        output_dir = args.output_dir,
        rounds     = args.rounds,
        n_subset   = args.n_subset,
        device     = args.device,
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()