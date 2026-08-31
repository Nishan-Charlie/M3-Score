"""
Layer-depth ablation for the memorization and coverage axes (EXP 33).
=====================================================================

The canonical M3-Score fixes three a-priori layers: L12 (fidelity),
L9 (memorization), L4 (coverage).  L12 is independently corroborated by the
CKA analysis, but L9 and L4 were chosen by representational position alone.
This script supplies the per-task depth ablation that Section III-B of the
paper previously deferred to future work, so the a-priori choice can be
checked against a task-specific optimum.

Two sweeps, each over all 12 RadioDINO-s16 blocks:

  1. MEMORIZATION.  Replace a known fraction of the generated set with
     jittered near-duplicates of real images (sigma = 5e-4, matching
     experiments/per_axis_validation.py).  At each depth, measure how well
     the 1-NN memorization rate recovers the injected fraction.
     Figure of merit: mean absolute recovery error (lower is better).

  2. COVERAGE.  Truncate the generated set (mode-drop) and measure k-NN
     manifold recall at each depth.  A well-behaved coverage layer should
     show recall FALLING as diversity is removed.
     Figure of merit: Spearman rho between drop fraction and recall
     (strongly negative is better; positive means the known k-NN radius
     inflation dominates).

Feature extraction is done once per image set per condition and reused
across all 12 depths, so the sweep costs about as much as a handful of
ordinary evaluations.

Usage
-----
    python experiments/layer_depth_ablation.py \\
        --real_dir data_mri/brats_axial_multislice \\
        --gen_dir  output/generated_500_standard \\
        --num_images 150 \\
        --output_dir results/layer_depth_ablation \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.per_axis_validation import _glob_images, _load_tensor, _jitter

N_LAYERS = 12
ALL_LAYERS = set(range(1, N_LAYERS + 1))

# a-priori choices under test
MEM_LAYER_APRIORI = 9
COV_LAYER_APRIORI = 4


def _extract_all_layers(metric: M3EntropyMetric, imgs: torch.Tensor) -> List[torch.Tensor]:
    """CLS features at every block, as a list indexed by (layer - 1)."""
    cls, _ = metric._extract_features_and_entropy(imgs, layers_to_keep=ALL_LAYERS)
    return [cls[i].to(metric.device).float() for i in range(N_LAYERS)]


# ---------------------------------------------------------------------------
# Sweep 1: memorization depth
# ---------------------------------------------------------------------------

def memorization_depth_sweep(
    metric: M3EntropyMetric,
    real_t: torch.Tensor,
    gen_t: torch.Tensor,
    rates: List[float],
    seed: int,
    sigma: float = 5e-4,
    real_feats: List[torch.Tensor] | None = None,
) -> Dict:
    rng = np.random.default_rng(seed)
    N_g, N_r = len(gen_t), len(real_t)

    if real_feats is None:
        print("  extracting real features at all 12 blocks ...")
        real_feats = _extract_all_layers(metric, real_t)

    # per_layer[layer] = list of recovered rates, aligned with `rates`
    per_layer: Dict[int, List[float]] = {L: [] for L in range(1, N_LAYERS + 1)}

    for rate in rates:
        n_inject = int(rate * N_g)
        gen_mod = gen_t.clone()
        if n_inject > 0:
            real_idx = rng.choice(N_r, size=n_inject, replace=True)
            gen_idx = rng.choice(N_g, size=n_inject, replace=False)
            gen_mod[gen_idx] = _jitter(
                real_t[real_idx], sigma=sigma, seed=int(seed + rate * 1000)
            )

        print(f"  [mem] sigma={sigma:g}  injection rate={rate:.2f} (n={n_inject}) ...")
        gen_feats = _extract_all_layers(metric, gen_mod)

        for L in range(1, N_LAYERS + 1):
            _, mem_rate = metric._compute_memorization(
                real_feats[L - 1], gen_feats[L - 1], k=1
            )
            per_layer[L].append(float(mem_rate))

    # figure of merit: mean |recovered - injected|
    summary = {}
    for L in range(1, N_LAYERS + 1):
        rec = np.array(per_layer[L])
        inj = np.array(rates)
        mae = float(np.mean(np.abs(rec - inj)))
        rho = float(spearmanr(inj, rec).correlation) if len(set(rec)) > 1 else 0.0
        summary[L] = {
            "recovered_rates": per_layer[L],
            "mean_abs_error": mae,
            "spearman_rho": rho,
        }
        print(f"    L{L:<2d}  MAE={mae:.4f}  rho={rho:+.3f}  {per_layer[L]}")

    best = min(summary, key=lambda L: summary[L]["mean_abs_error"])
    maes = [summary[L]["mean_abs_error"] for L in range(1, N_LAYERS + 1)]
    return {
        "injection_rates": rates,
        "jitter_sigma": sigma,
        "per_layer": summary,
        "best_layer": best,
        "a_priori_layer": MEM_LAYER_APRIORI,
        "a_priori_mae": summary[MEM_LAYER_APRIORI]["mean_abs_error"],
        "best_mae": summary[best]["mean_abs_error"],
        # spread across depths: 0 means the test cannot separate layers
        "mae_spread": float(max(maes) - min(maes)),
        "saturated": bool(max(maes) - min(maes) < 1e-6),
    }


# ---------------------------------------------------------------------------
# Sweep 2: coverage depth
# ---------------------------------------------------------------------------

def coverage_depth_sweep(
    metric: M3EntropyMetric,
    real_t: torch.Tensor,
    gen_t: torch.Tensor,
    drops: List[float],
    knn_k: int,
) -> Dict:
    print("  extracting real features at all 12 blocks ...")
    real_feats = _extract_all_layers(metric, real_t)
    print("  extracting generated features at all 12 blocks ...")
    gen_feats = _extract_all_layers(metric, gen_t)

    N_g = gen_feats[0].shape[0]
    per_layer: Dict[int, Dict[str, List[float]]] = {
        L: {"precision": [], "recall": []} for L in range(1, N_LAYERS + 1)
    }

    for drop in drops:
        keep = int(round((1.0 - drop) * N_g))
        print(f"  [cov] mode-drop={drop:.0%}  keeping {keep}/{N_g} ...")
        for L in range(1, N_LAYERS + 1):
            prec, rec = metric._compute_precision_recall(
                real_feats[L - 1], gen_feats[L - 1][:keep], k=knn_k
            )
            per_layer[L]["precision"].append(float(prec))
            per_layer[L]["recall"].append(float(rec))

    summary = {}
    for L in range(1, N_LAYERS + 1):
        rec = np.array(per_layer[L]["recall"])
        rho = float(spearmanr(drops, rec).correlation) if len(set(rec)) > 1 else 0.0
        summary[L] = {
            "precision": per_layer[L]["precision"],
            "recall": per_layer[L]["recall"],
            "spearman_rho_drop_vs_recall": rho,
            "monotone_decreasing": bool(rho < 0),
        }
        print(f"    L{L:<2d}  rho(drop, recall)={rho:+.3f}  recall={per_layer[L]['recall']}")

    # best = most negative rho (recall falls as diversity is removed)
    best = min(summary, key=lambda L: summary[L]["spearman_rho_drop_vs_recall"])
    any_monotone = [L for L in summary if summary[L]["monotone_decreasing"]]
    return {
        "drop_fractions": drops,
        "knn_k": knn_k,
        "per_layer": summary,
        "best_layer": best,
        "best_rho": summary[best]["spearman_rho_drop_vs_recall"],
        "a_priori_layer": COV_LAYER_APRIORI,
        "a_priori_rho": summary[COV_LAYER_APRIORI]["spearman_rho_drop_vs_recall"],
        "layers_with_correct_direction": sorted(any_monotone),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(mem: Dict, cov: Dict, out_path: str,
              mem_by_sigma: Dict | None = None) -> None:
    layers = list(range(1, N_LAYERS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.patch.set_facecolor("white")

    # -- memorization: recovery error vs depth, one line per difficulty --
    ax = axes[0]
    series = mem_by_sigma if mem_by_sigma else {f"{mem['jitter_sigma']:g}": mem}
    cmap = plt.get_cmap("viridis")
    keys = list(series.keys())
    for i, (sig, m) in enumerate(series.items()):
        mae = [m["per_layer"][L]["mean_abs_error"] for L in layers]
        ax.plot(layers, mae, marker="o", ms=4, lw=1.6,
                color=cmap(i / max(len(keys) - 1, 1)),
                label=rf"$\sigma={float(sig):g}$")
    ax.axvline(MEM_LAYER_APRIORI, color="#763494", ls="--", lw=1.6, zorder=1)
    ax.text(MEM_LAYER_APRIORI + 0.15, ax.get_ylim()[1] * 0.94,
            f"a-priori L{MEM_LAYER_APRIORI}", color="#763494", fontsize=8.5)
    ax.set_xlabel("RadioDINO-s16 block")
    ax.set_ylabel("mean |recovered $-$ injected|")
    ax.set_title("Memorization: recovery error by depth\n"
                 "(lower is better; flat line = depth does not matter)",
                 fontsize=10)
    ax.set_xticks(layers)
    ax.legend(fontsize=7.5, ncol=2, loc="center left", framealpha=0.9)
    ax.grid(alpha=0.3, zorder=0)

    # -- coverage --
    ax = axes[1]
    rho = [cov["per_layer"][L]["spearman_rho_drop_vs_recall"] for L in layers]
    colors = ["#b8303e" if r > 0 else "#167042" for r in rho]
    bars = ax.bar(layers, rho, color=colors, edgecolor="#555", zorder=3)
    bars[COV_LAYER_APRIORI - 1].set_edgecolor("#000")
    bars[COV_LAYER_APRIORI - 1].set_linewidth(2.0)
    ax.axhline(0, color="black", lw=1.0, zorder=4)
    ax.set_xlabel("RadioDINO-s16 block")
    ax.set_ylabel(r"Spearman $\rho$ (mode-drop, recall)")
    ax.set_title("Coverage axis: does recall fall as diversity is removed?\n"
                 f"(a-priori L{COV_LAYER_APRIORI} outlined; negative is correct)",
                 fontsize=10)
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    print(f"\n[saved] {out_path}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--num_images", type=int, default=150)
    ap.add_argument("--output_dir", default="results/layer_depth_ablation")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--injection_rates", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--drop_fractions", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--jitter_sigmas", type=float, nargs="+",
                    default=[5e-4, 5e-3, 2e-2, 5e-2, 1e-1, 2e-1],
                    help="near-duplicate difficulty; the first value is the "
                         "canonical protocol used in per_axis_validation.py")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"
    print(f"[device] {device}")

    real_paths = _glob_images(args.real_dir, args.num_images)
    gen_paths = _glob_images(args.gen_dir, args.num_images)
    print(f"[data] real={len(real_paths)}  gen={len(gen_paths)}")
    if len(real_paths) < 10 or len(gen_paths) < 10:
        raise SystemExit("Not enough images found; check --real_dir / --gen_dir.")

    real_t = _load_tensor(real_paths)
    gen_t = _load_tensor(gen_paths)

    metric = M3EntropyMetric(
        device=device, backbone_id=args.backbone_id,
        single_layer=None, seed=args.seed,
    )

    print("\n=== Sweep 1: memorization depth, across near-duplicate difficulty ===")
    print("  extracting real features at all 12 blocks (reused for every sigma) ...")
    real_feats_all = _extract_all_layers(metric, real_t)

    mem_by_sigma = {}
    for sig in args.jitter_sigmas:
        print(f"\n  --- jitter sigma = {sig:g} ---")
        mem_by_sigma[f"{sig:g}"] = memorization_depth_sweep(
            metric, real_t, gen_t, args.injection_rates, args.seed,
            sigma=sig, real_feats=real_feats_all,
        )
    # the canonical protocol (matches per_axis_validation.py) is the first sigma
    mem = mem_by_sigma[f"{args.jitter_sigmas[0]:g}"]

    print("\n=== Sweep 2: coverage depth ===")
    cov = coverage_depth_sweep(
        metric, real_t, gen_t, args.drop_fractions, args.knn_k
    )

    report = {
        "experiment": "layer_depth_ablation",
        "backbone": args.backbone_id,
        "num_images": len(real_paths),
        "seed": args.seed,
        "memorization": mem,
        "memorization_by_sigma": mem_by_sigma,
        "coverage": cov,
    }
    out_json = os.path.join(args.output_dir, "layer_depth_ablation_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] {out_json}")

    make_plot(mem, cov, os.path.join(args.output_dir, "layer_depth_ablation.png"),
              mem_by_sigma=mem_by_sigma)

    print("\n================ SUMMARY ================")
    print("Memorization by near-duplicate difficulty:")
    for sig, m in mem_by_sigma.items():
        tag = "SATURATED (no depth separation)" if m["saturated"] else \
              f"best L{m['best_layer']} (MAE={m['best_mae']:.4f})"
        print(f"  sigma={sig:>7}  a-priori L{m['a_priori_layer']} MAE={m['a_priori_mae']:.4f}"
              f"  spread={m['mae_spread']:.4f}  -> {tag}")
    print(f"Coverage:     a-priori L{cov['a_priori_layer']} rho={cov['a_priori_rho']:+.3f}; "
          f"best L{cov['best_layer']} rho={cov['best_rho']:+.3f}")
    print(f"Coverage layers with correct direction: {cov['layers_with_correct_direction']}")


if __name__ == "__main__":
    main()
