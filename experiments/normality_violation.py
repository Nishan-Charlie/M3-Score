"""
normality_violation.py  --  Validation Experiment: FID blind-spot construction
===============================================================================

Constructs synthetic feature distributions that progressively depart from
the real (reference) distribution while KEEPING MOMENTS 1 AND 2 FIXED.
FID uses only mean and covariance => FID reports ~zero.
M3-Score (kernel MMD) is sensitive to all moments => M3 tracks the departure.

Three departure types
---------------------
1. Bimodal mixture
   Split real features into two halves; shift them apart symmetrically so
   the combined mean and variance are preserved.  Increasing shift = more
   bimodal departure.

2. Skewness injection
   Apply a Box-Cox-like power transform that skews each dimension while
   preserving the linear mean and variance (via analytic re-centering/rescaling).

3. Mode collapse simulation
   Replace features with repeated copies of a centroid plus small Gaussian
   noise, scaled so mean/variance match.  Increasing radius = more collapsed.

For each departure type, sweep over 8 levels (0=identity => reference).
Measure FID (Inception) and M3-Score (RadioDino-s16 L12 RBF).
Plot: departure level vs metric value, showing FID stays near zero while
      M3 increases monotonically.

Usage
-----
    python experiments/normality_violation.py \
        --real_dir data_mri/brats_axial_multislice \
        --gen_dir  output/generated \
        --output_dir results/normality_violation \
        --n_images 500 \
        --device cuda
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Departure constructors
# ---------------------------------------------------------------------------

def _departure_bimodal(feats: np.ndarray, shift: float) -> np.ndarray:
    """
    Split into two equal halves, shift apart symmetrically.
    Mean preserved; covariance increased proportionally to shift.
    We re-scale to preserve both mean AND variance.

    feats: (N, D)
    shift: 0 = identity, larger = more bimodal
    """
    N, D = feats.shape
    mu   = feats.mean(0, keepdims=True)
    out  = feats.copy()

    mid = N // 2
    # Direction: first PCA component is most natural; use mean diff as proxy
    direction = np.ones(D) / np.sqrt(D)   # unit vector

    out[:mid]  = feats[:mid]  + shift * direction
    out[mid:]  = feats[mid:]  - shift * direction

    # Re-centre to exactly preserve mean
    out -= (out.mean(0, keepdims=True) - mu)

    # Re-scale each dim to preserve variance (component-wise)
    orig_std = feats.std(0) + 1e-8
    new_std  = out.std(0)   + 1e-8
    out = mu + (out - out.mean(0, keepdims=True)) * (orig_std / new_std)

    return out.astype(np.float32)


def _departure_skewness(feats: np.ndarray, skew_power: float) -> np.ndarray:
    """
    Apply power transform: x -> sign(x) * |x|^p  (p > 1 increases skewness).
    Re-centre and re-scale to preserve mean and variance exactly.

    skew_power: 1.0 = identity, larger = more skewed
    """
    mu   = feats.mean(0, keepdims=True)
    std  = feats.std(0, keepdims=True)  + 1e-8
    z    = (feats - mu) / std            # standardised

    # Power transform in standardised space
    z_p  = np.sign(z) * (np.abs(z) ** skew_power)

    # Re-standardise to mean=0, std=1 then project back
    z_p  = (z_p - z_p.mean(0, keepdims=True)) / (z_p.std(0, keepdims=True) + 1e-8)
    out  = mu + z_p * std
    return out.astype(np.float32)


def _departure_mode_collapse(feats: np.ndarray, collapse: float) -> np.ndarray:
    """
    Linearly interpolate each feature toward the dataset centroid.
    collapse=0 => identity; collapse=1 => all samples = centroid.
    Re-scale to preserve variance.

    collapse: 0.0 = identity, larger (up to 0.99) = more collapsed
    """
    mu   = feats.mean(0, keepdims=True)
    out  = feats * (1.0 - collapse) + mu * collapse

    # Restore variance
    orig_std = feats.std(0)  + 1e-8
    new_std  = out.std(0)    + 1e-8
    out = mu + (out - out.mean(0, keepdims=True)) * (orig_std / new_std)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# FID computation (moments 1+2 only, in feature space)
# ---------------------------------------------------------------------------

def _fid_from_feats(real_f: np.ndarray, gen_f: np.ndarray) -> float:
    """
    Feature-space FID = Frechet distance using mean and covariance.
    (Same as standard FID but operating on RadioDino features directly.)
    """
    from scipy.linalg import sqrtm

    mu_r = real_f.mean(0)
    mu_g = gen_f.mean(0)
    sigma_r = np.cov(real_f.T)
    sigma_g = np.cov(gen_f.T)

    diff  = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return max(fid, 0.0)


# ---------------------------------------------------------------------------
# M3-Score (MMD)  in feature space
# ---------------------------------------------------------------------------

def _m3_from_feats(real_f: np.ndarray, gen_f: np.ndarray) -> float:
    """Unbiased RBF MMD^2 (multi-bandwidth median-heuristic)."""
    Xt = torch.from_numpy(real_f).float()
    Yt = torch.from_numpy(gen_f).float()

    def sq(A, B):
        return (A.unsqueeze(1) - B.unsqueeze(0)).pow(2).sum(-1)

    Dxx = sq(Xt, Xt)
    Dyy = sq(Yt, Yt)
    Dxy = sq(Xt, Yt)

    med = torch.cat([Dxy.flatten(),
                     Dxx[Dxx > 0].flatten(),
                     Dyy[Dyy > 0].flatten()]).median().item()
    if med < 1e-10:
        med = 1.0

    n, m  = len(real_f), len(gen_f)
    total = 0.0
    for bw in [med * (2 ** k) for k in range(-2, 3)]:
        g   = 1.0 / (2.0 * bw)
        Kxx = torch.exp(-g * Dxx); Kxx.fill_diagonal_(0)
        Kyy = torch.exp(-g * Dyy); Kyy.fill_diagonal_(0)
        Kxy = torch.exp(-g * Dxy)
        total += (Kxx.sum() / (n*(n-1)) + Kyy.sum() / (m*(m-1))
                  - 2 * Kxy.mean()).item()
    return total / 5.0


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

def _extract_real_feats(
    real_dir: str,
    n_images: int,
    backbone_id: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    from evaluation.coverage_novelty import _load_backbone, _extract_features
    from experiments._shared_utils import load_pils_recursive

    imgs = load_pils_recursive(real_dir, n=n_images)
    backbone, backend = _load_backbone(backbone_id, device)
    feats = _extract_features(imgs, backbone, backend, device, batch_size)
    return feats


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_normality_violation(
    real_dir:     str,
    gen_dir:      str,
    output_dir:   str = "results/normality_violation",
    n_images:     int = 500,
    backbone_id:  str = "Snarcy/RadioDino-s16",
    device:       str = "cuda",
    batch_size:   int = 32,
    seed:         int = 42,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    device = device if torch.cuda.is_available() else "cpu"

    print("\n=== Normality Violation Experiment ===")
    print(f"Extracting real features from: {real_dir}")
    real_feats = _extract_real_feats(real_dir, n_images, backbone_id, device, batch_size)

    # Use generated images as the BASE (real-like) distribution
    from evaluation.coverage_novelty import _load_backbone, _extract_features
    from experiments._shared_utils import load_pils_recursive
    gen_imgs_base = load_pils_recursive(gen_dir, n=n_images)
    backbone, backend = _load_backbone(backbone_id, device)
    gen_feats_base = _extract_features(gen_imgs_base, backbone, backend, device, batch_size)

    # Departure levels and constructors
    DEPARTURES = {
        "bimodal_shift": {
            "fn":     _departure_bimodal,
            "levels": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
            "xlabel": "Bimodal shift magnitude",
        },
        "skewness_power": {
            "fn":     _departure_skewness,
            "levels": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "xlabel": "Skewness power (1=identity)",
        },
        "mode_collapse": {
            "fn":     _departure_mode_collapse,
            "levels": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
            "xlabel": "Collapse fraction (0=identity)",
        },
    }

    all_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax_idx, (dep_name, dep_cfg) in enumerate(DEPARTURES.items()):
        levels   = dep_cfg["levels"]
        fn       = dep_cfg["fn"]
        xlabel   = dep_cfg["xlabel"]

        fid_vals = []
        m3_vals  = []

        print(f"\n  Departure: {dep_name}")
        for level in levels:
            dist_feats = fn(gen_feats_base, level)
            fid = _fid_from_feats(real_feats, dist_feats)
            m3  = _m3_from_feats(real_feats, dist_feats)
            fid_vals.append(fid)
            m3_vals.append(m3)
            print(f"    level={level:.2f}  FID={fid:.4f}  M3={m3:.6f}")

        all_results[dep_name] = {
            "levels":   levels,
            "fid_vals": fid_vals,
            "m3_vals":  m3_vals,
        }

        # Normalise both to [0,1] for comparison on same axis
        fid_norm = np.array(fid_vals)
        m3_norm  = np.array(m3_vals)
        fid_range = fid_norm.max() - fid_norm.min() + 1e-10
        m3_range  = m3_norm.max()  - m3_norm.min()  + 1e-10
        fid_norm = (fid_norm - fid_norm.min()) / fid_range
        m3_norm  = (m3_norm  - m3_norm.min())  / m3_range

        ax = axes[ax_idx]
        ax.plot(levels, fid_norm, "o--", color="#e05c5c", lw=2, ms=6, label="FID (normalised)")
        ax.plot(levels, m3_norm,  "s-",  color="#4878cf", lw=2, ms=6, label="M3 (normalised)")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Normalised score", fontsize=12)
        ax.set_title(dep_name.replace("_", " ").title(), fontsize=13)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Normality Violation: FID blind-spot vs M3-Score sensitivity\n"
                 "(distributions preserve mean & covariance — FID can't detect departure)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "normality_violation.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    # Save raw numbers
    with open(os.path.join(output_dir, "normality_violation_report.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {output_dir}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",    required=True)
    p.add_argument("--gen_dir",     required=True)
    p.add_argument("--output_dir",  default="results/normality_violation")
    p.add_argument("--n_images",    type=int, default=500)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    p.add_argument("--device",      default=None)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_normality_violation(
        real_dir    = a.real_dir,
        gen_dir     = a.gen_dir,
        output_dir  = a.output_dir,
        n_images    = a.n_images,
        backbone_id = a.backbone_id,
        device      = device,
        batch_size  = a.batch_size,
        seed        = a.seed,
    )
