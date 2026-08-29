"""
Per-Axis Validation of the Three-Axis M3 Design
================================================
Validates that each M3 axis is sensitive to the failure mode it is designed
to detect, and insensitive to failure modes that belong to the other axes.

Two sub-experiments
-------------------
1. Memorization validation (L9 axis)
   Inject known near-duplicates (copies of real images with tiny jitter) into
   the generated set at a known rate (e.g. 10 %, 20 %, 50 %).
   Expected: memorization/rate increases monotonically with injection rate.
   L12 fidelity/mmd2 should stay roughly constant (near-dups are in-distribution).

2. Coverage validation (L4 axis)
   Apply mode-drop: restrict the generated set to images from only a subset
   of the real distribution (e.g. first half of sorted real images, proxying
   a collapsed generator that ignores inferior slices).
   Expected: coverage/recall decreases monotonically as mode-drop fraction rises.
   L12 fidelity/mmd2 may stay high because the retained modes match real statistics.

Protocol
--------
- Uses M3EntropyMetric.compute_axes() with a-priori layers (L4, L9, L12).
- Injection / mode-drop rates: [0.0, 0.1, 0.2, 0.3, 0.5]
- Reports whether each axis correctly detects the targeted perturbation.

Usage
-----
    python experiments/per_axis_validation.py \\
        --real_dir  data_mri/brats_axial_multislice \\
        --gen_dir   output/generated_500_best \\
        --output_dir results/per_axis_validation \\
        --num_images 200 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from evaluation.m3_score_v2 import M3EntropyMetric

_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _glob_images(directory: str, n: Optional[int] = None) -> List[str]:
    import glob
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))
    return paths[:n] if n else paths


def _load_tensor(paths: List[str]) -> torch.Tensor:
    return torch.stack([_TFM(Image.open(p).convert("RGB")) for p in paths])


# ---------------------------------------------------------------------------
# Jitter: tiny random perturbation to create near-duplicates
# ---------------------------------------------------------------------------

def _jitter(t: torch.Tensor, sigma: float = 1e-3, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.randn(t.shape, generator=g) * sigma
    return (t + noise).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Sub-experiment 1: memorization injection
# ---------------------------------------------------------------------------

def _memorization_experiment(
    real_t: torch.Tensor,
    gen_t: torch.Tensor,
    metric: M3EntropyMetric,
    rates: List[float],
    seed: int,
) -> List[dict]:
    """
    Inject near-duplicates of real images into gen_t at increasing rates.

    At rate=r, floor(r * len(gen_t)) generated images are replaced by
    jittered copies of real images sampled at random.
    """
    rng  = np.random.default_rng(seed)
    N_g  = len(gen_t)
    N_r  = len(real_t)
    rows = []

    for rate in rates:
        n_inject = int(rate * N_g)
        gen_mod  = gen_t.clone()

        if n_inject > 0:
            real_idx = rng.choice(N_r, size=n_inject, replace=True)
            gen_idx  = rng.choice(N_g, size=n_inject, replace=False)
            near_dup = _jitter(real_t[real_idx], sigma=5e-4, seed=int(seed + rate * 1000))
            gen_mod[gen_idx] = near_dup

        print(f"  [memorization] rate={rate:.1f}  n_inject={n_inject} ...")
        axes = metric.compute_axes(real_t, gen_mod, n_permutations=50, n_bootstrap=50)
        rows.append({
            "injection_rate":         rate,
            "n_injected":             n_inject,
            "fidelity/mmd2":          axes["fidelity/mmd2"],
            "memorization/rate":      axes["memorization/rate"],
            "memorization/mean_nn":   axes["memorization/mean_nn_dist"],
            "coverage/precision":     axes["coverage/precision"],
            "coverage/recall":        axes["coverage/recall"],
        })
        print(f"    fidelity/mmd2={axes['fidelity/mmd2']:.4f}  "
              f"mem/rate={axes['memorization/rate']:.4f}  "
              f"recall={axes['coverage/recall']:.4f}")

    return rows


# ---------------------------------------------------------------------------
# Sub-experiment 2: mode-drop (coverage)
# ---------------------------------------------------------------------------

def _coverage_experiment(
    real_t: torch.Tensor,
    gen_t: torch.Tensor,
    metric: M3EntropyMetric,
    drop_fractions: List[float],
) -> List[dict]:
    """
    Simulate mode-drop by restricting gen_t to the first (1-frac) fraction of
    images (sorted by filename order, which roughly corresponds to inferior
    slices).  This leaves a gap in the distribution that coverage/recall
    should detect.
    """
    N_g  = len(gen_t)
    rows = []

    for frac in drop_fractions:
        n_keep = max(2, int((1.0 - frac) * N_g))
        gen_sub = gen_t[:n_keep]

        print(f"  [mode-drop] frac={frac:.1f}  n_keep={n_keep}/{N_g} ...")
        axes = metric.compute_axes(real_t, gen_sub, n_permutations=50, n_bootstrap=50)
        rows.append({
            "drop_fraction":        frac,
            "n_gen":                n_keep,
            "fidelity/mmd2":        axes["fidelity/mmd2"],
            "memorization/rate":    axes["memorization/rate"],
            "coverage/precision":   axes["coverage/precision"],
            "coverage/recall":      axes["coverage/recall"],
        })
        print(f"    fidelity/mmd2={axes['fidelity/mmd2']:.4f}  "
              f"recall={axes['coverage/recall']:.4f}  "
              f"precision={axes['coverage/precision']:.4f}")

    return rows


# ---------------------------------------------------------------------------
# Monotonicity helpers
# ---------------------------------------------------------------------------

def _mono_increasing(vals: List[float]) -> bool:
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def _mono_decreasing(vals: List[float]) -> bool:
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    real_dir: str,
    gen_dir: str,
    output_dir: str,
    num_images: int = 200,
    device: str = "cpu",
    seed: int = 42,
    injection_rates: Optional[List[float]] = None,
    drop_fractions:  Optional[List[float]] = None,
) -> dict:
    if injection_rates is None:
        injection_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    if drop_fractions is None:
        drop_fractions  = [0.0, 0.1, 0.2, 0.3, 0.5]

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load images
    real_paths = _glob_images(real_dir, num_images)
    gen_paths  = _glob_images(gen_dir,  num_images)
    if not real_paths:
        raise FileNotFoundError(f"No real images in {real_dir}")
    if not gen_paths:
        raise FileNotFoundError(f"No gen images in {gen_dir}")

    real_t = _load_tensor(real_paths)
    gen_t  = _load_tensor(gen_paths)
    N      = min(len(real_t), len(gen_t))
    real_t = real_t[:N]
    gen_t  = gen_t[:N]
    print(f"Loaded {N} real and {N} gen images")

    # Metric
    metric = M3EntropyMetric(device=device, single_layer=12, seed=seed)

    # Sub-experiment 1: memorization
    print("\n=== Sub-experiment 1: Memorization injection ===")
    mem_rows = _memorization_experiment(real_t, gen_t, metric, injection_rates, seed)

    mem_rates = [r["memorization/rate"] for r in mem_rows]
    fid_at_mem = [r["fidelity/mmd2"]    for r in mem_rows]
    mem_mono   = _mono_increasing(mem_rates)
    fid_stable = abs(fid_at_mem[-1] - fid_at_mem[0]) / (abs(fid_at_mem[0]) + 1e-8) < 0.5

    print(f"\n  memorization/rate increases monotonically: {mem_mono}")
    print(f"  fidelity/mmd2 stays relatively stable:    {fid_stable}  "
          f"(change={100*(fid_at_mem[-1]-fid_at_mem[0])/(abs(fid_at_mem[0])+1e-8):.1f}%)")

    # Sub-experiment 2: coverage (mode-drop)
    print("\n=== Sub-experiment 2: Coverage (mode-drop) ===")
    cov_rows = _coverage_experiment(real_t, gen_t, metric, drop_fractions)

    recall_vals  = [r["coverage/recall"]  for r in cov_rows]
    fid_at_cov   = [r["fidelity/mmd2"]   for r in cov_rows]
    recall_mono  = _mono_decreasing(recall_vals)

    print(f"\n  coverage/recall decreases monotonically:   {recall_mono}")
    print(f"  fidelity/mmd2 range: {min(fid_at_cov):.4f} - {max(fid_at_cov):.4f}")

    report = {
        "config": {
            "real_dir":        real_dir,
            "gen_dir":         gen_dir,
            "num_images":      N,
            "device":          device,
            "injection_rates": injection_rates,
            "drop_fractions":  drop_fractions,
        },
        "memorization_experiment": {
            "rows":         mem_rows,
            "pass_rate_increases_monotonically": mem_mono,
            "pass_fidelity_stable":              fid_stable,
            "passed": mem_mono,
        },
        "coverage_experiment": {
            "rows":          cov_rows,
            "pass_recall_decreases_monotonically": recall_mono,
            "passed": recall_mono,
        },
        "summary": {
            "memorization_axis_valid": mem_mono,
            "coverage_axis_valid":     recall_mono,
            "both_axes_valid":         mem_mono and recall_mono,
        },
    }

    out_path = os.path.join(output_dir, "per_axis_validation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")

    _plot(mem_rows, cov_rows, output_dir)
    return report


def _plot(mem_rows: List[dict], cov_rows: List[dict], output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    # Memorization panel
    ax = axes[0]
    rates  = [r["injection_rate"]    for r in mem_rows]
    mrates = [r["memorization/rate"] for r in mem_rows]
    fids   = [r["fidelity/mmd2"]     for r in mem_rows]
    ax2    = ax.twinx()
    ax.plot(rates, mrates, "o-", color="#1f77b4", lw=2, label="mem/rate (L9)")
    ax2.plot(rates, fids,  "s--", color="#d62728", lw=1.5, label="fidelity MMD2 (L12)")
    ax.set_xlabel("Near-duplicate injection rate")
    ax.set_ylabel("memorization/rate (L9)", color="#1f77b4")
    ax2.set_ylabel("fidelity/mmd2 (L12)", color="#d62728")
    ax.set_title("Memorization axis\n(L9 should rise; L12 should stay flat)")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    # Coverage panel
    ax = axes[1]
    fracs   = [r["drop_fraction"]    for r in cov_rows]
    recalls = [r["coverage/recall"]  for r in cov_rows]
    fids_c  = [r["fidelity/mmd2"]   for r in cov_rows]
    ax3     = ax.twinx()
    ax.plot(fracs, recalls, "o-", color="#2ca02c", lw=2, label="coverage/recall (L4)")
    ax3.plot(fracs, fids_c, "s--", color="#d62728", lw=1.5, label="fidelity MMD2 (L12)")
    ax.set_xlabel("Mode-drop fraction")
    ax.set_ylabel("coverage/recall (L4)", color="#2ca02c")
    ax3.set_ylabel("fidelity/mmd2 (L12)", color="#d62728")
    ax.set_title("Coverage axis\n(L4 recall should fall; L12 may stay flat)")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    fig.suptitle("Per-Axis Validation: each axis detects its targeted failure mode", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "per_axis_validation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="results/per_axis_validation")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--injection_rates", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--drop_fractions",  type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5])
    args = parser.parse_args()

    run(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        device=args.device,
        seed=args.seed,
        injection_rates=args.injection_rates,
        drop_fractions=args.drop_fractions,
    )
