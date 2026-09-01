"""
Depth x task grid for RadioDINO-s16 (EXP 34).
=============================================

The anchor experiment for the depth-resolved reframing.  Earlier work measured
encoder depth only for the two diagnostics that failed (memorization, coverage)
and reported the one that worked (fidelity) at a single a-priori depth, L12.
That asymmetry is what made the a-priori depth choices look arbitrary.  This
script closes it: every task is measured at every one of the 12 blocks, from a
single cached forward pass per image set.

Five sweeps, all over blocks L1..L12:

  T1  FIDELITY / SEVERITY ORDERING.  Unbiased multi-bandwidth RBF MMD^2 from a
      real reference to each of: a held-out real split (null), a DDPM, a weaker
      same-modality generator (WDM-3D BraTS), a cross-modality shift (WDM-3D
      LIDC chest CT), and a far-domain shift (retinal fundus).  Figures of
      merit: permutation Z against the real-vs-real null, and whether the depth
      recovers the a-priori severity ordering.

  T2  PER-IMAGE OOD SCORING.  Depth x scoring-rule grid.  Four rules score a
      single image against the real reference set -- L2 to the real centroid,
      1-NN distance, 5-NN distance, and shrinkage Mahalanobis -- and each is
      scored by ROC-AUC separating held-out real from cross-modality CT.  The
      centroid rule is the one the canonical per-image score uses; this sweep
      tests whether that choice is defensible in a 384-d SSL embedding.

  T3  MEMORIZATION.  Replace a known fraction of the generated set with
      jittered near-duplicates of real images and measure how well the 1-NN
      memorization rate recovers the injected fraction, across a jitter ladder
      that spans the easy (saturated) and discriminative regimes.
      Figure of merit: mean absolute recovery error.

  T4  COVERAGE.  Truncate the generated set (mode drop) and measure k-NN
      manifold recall.  A usable coverage depth shows recall FALLING as
      diversity is removed.  Figure of merit: Spearman rho(drop, recall);
      strongly negative is good, positive means k-NN radius inflation wins.

  T5  DEGRADATION MONOTONICITY.  Gaussian noise ladder applied to the generated
      set; MMD^2 per depth against the fixed real reference.
      Figure of merit: Spearman rho(sigma, MMD^2).

Feature extraction is batched and cached, so the whole grid costs a few minutes
on a laptop GPU rather than the hours the per-task scripts would take
separately.

Usage
-----
    python experiments/depth_task_grid.py \\
        --real_dir data_mri/brats_axial_multislice \\
        --gen_dir  output/generated_500_standard \\
        --num_images 500 \\
        --output_dir results/depth_task_grid \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from sklearn.metrics import roc_auc_score

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.per_axis_validation import _glob_images, _load_tensor, _jitter

N_LAYERS = 12
ALL_LAYERS = set(range(1, N_LAYERS + 1))
LAYERS = list(range(1, N_LAYERS + 1))

# A-priori depths under test (the canonical choices this grid audits).
APRIORI = {"fidelity": 12, "memorization": 9, "coverage": 4}


# ---------------------------------------------------------------------------
# Feature extraction with caching
# ---------------------------------------------------------------------------

def extract_all_layers(
    metric: M3EntropyMetric,
    imgs: torch.Tensor,
) -> List[torch.Tensor]:
    """CLS features at every block, as a list indexed by (layer - 1)."""
    cls, _ = metric._extract_features_and_entropy(imgs, layers_to_keep=ALL_LAYERS)
    return [cls[i].float() for i in range(N_LAYERS)]


# ---------------------------------------------------------------------------
# Real-reference construction
# ---------------------------------------------------------------------------

def _patient_id(path: str) -> str:
    """Subject identifier from a BraTS slice filename.

    Filenames look like ``BraTS2021_00042_slice073.png``; everything before the
    ``_slice`` marker identifies the subject.
    """
    return os.path.basename(path).split("_slice")[0]


def split_real_by_patient(
    paths: List[str],
    n_per_split: int,
    seed: int,
) -> tuple:
    """Two patient-DISJOINT real splits of n_per_split images each.

    Slices from one subject are near-duplicates of each other, so a reference
    set drawn as the first N sorted filenames covers only a handful of
    subjects and is far less dispersed than the real distribution it stands
    for.  Every distance measured against it is inflated, and a real-vs-real
    null built by splitting *within* subjects is optimistically small.  Both
    splits here draw from disjoint subject pools, sampling slices spread across
    subjects rather than consecutively.
    """
    by_patient: Dict[str, List[str]] = {}
    for p in paths:
        by_patient.setdefault(_patient_id(p), []).append(p)

    pids = sorted(by_patient)
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    half = len(pids) // 2
    pools = [pids[:half], pids[half:]]

    splits = []
    for i, pool in enumerate(pools):
        rng_i = np.random.default_rng(seed + i)
        # Round-robin one slice per subject before taking a second from any,
        # so the split spans as many subjects as it has room for.
        buckets = [list(rng_i.permutation(by_patient[q])) for q in pool]
        picked: List[str] = []
        depth = 0
        while len(picked) < n_per_split:
            progressed = False
            for b in buckets:
                if depth < len(b):
                    picked.append(b[depth])
                    progressed = True
                    if len(picked) == n_per_split:
                        break
            if not progressed:
                break
            depth += 1
        if len(picked) < n_per_split:
            raise SystemExit(
                f"real split {i}: only {len(picked)} images available, need {n_per_split}"
            )
        splits.append(sorted(picked))
    return splits[0], splits[1]


def mmd_at(metric: M3EntropyMetric, a: torch.Tensor, b: torch.Tensor) -> float:
    """Unbiased multi-bandwidth RBF MMD^2, on device, returned as a float."""
    return float(
        metric._compute_mmd2_rbf(a.to(metric.device), b.to(metric.device)).item()
    )


# ---------------------------------------------------------------------------
# T1: fidelity and severity ordering
# ---------------------------------------------------------------------------

def sweep_fidelity(
    metric: M3EntropyMetric,
    feats: Dict[str, List[torch.Tensor]],
    n_perm: int,
    seed: int,
) -> Dict:
    """MMD^2 from the real reference to every comparison set, at every depth.

    Severity ranks are assigned a priori: a held-out real split is 0 (no
    shift), the DDPM 1, the weaker same-modality generator 2, and the
    cross-modality CT shift 3.  Retinal fundus is measured but held out of the
    ordering score, because RadioDINO's radiology-ontology ordering places it
    differently from a natural-image backbone -- that difference is a result in
    its own right, not an ordering error.
    """
    ref = "real_A"
    targets = [t for t in feats if t != ref]
    severity = {"real_B": 0, "ddpm": 1, "wdm3d": 2, "lidc": 3}

    per_layer: Dict[int, Dict] = {}
    for L in LAYERS:
        rf = feats[ref][L - 1]
        row: Dict[str, float] = {}
        for t in targets:
            row[t] = mmd_at(metric, rf, feats[t][L - 1])

        # Permutation test for the primary real-vs-DDPM comparison.
        p_val, z_score, null_mean = metric._permutation_test_fidelity(
            rf, feats["ddpm"][L - 1], n_permutations=n_perm, seed=seed
        )

        ranked = [t for t in severity if t in row]
        exp_rank = [severity[t] for t in ranked]
        obs_val = [row[t] for t in ranked]
        rho = float(spearmanr(exp_rank, obs_val).correlation)
        ordered = all(
            row[a] < row[b]
            for a, b in zip(ranked, ranked[1:])
        )

        per_layer[L] = {
            "mmd": row,
            "null_mmd": row.get("real_B", float("nan")),
            "perm_p": float(p_val),
            "perm_z": float(z_score),
            "perm_null_mean": float(null_mean),
            "severity_rho": rho,
            "severity_ordered": bool(ordered),
        }
        print(
            f"    L{L:<2d}  MMD(ddpm)={row['ddpm']:.4f}  null={row.get('real_B', float('nan')):.4f}  "
            f"Z={z_score:8.1f}  rho_sev={rho:+.3f}  ordered={ordered}"
        )

    best = max(LAYERS, key=lambda L: per_layer[L]["perm_z"])
    return {
        "reference": ref,
        "severity_ranks": severity,
        "n_permutations": n_perm,
        "per_layer": per_layer,
        "best_layer_by_z": best,
        "a_priori_layer": APRIORI["fidelity"],
        "a_priori_z": per_layer[APRIORI["fidelity"]]["perm_z"],
        "best_z": per_layer[best]["perm_z"],
        "layers_with_correct_ordering": [
            L for L in LAYERS if per_layer[L]["severity_ordered"]
        ],
    }


# ---------------------------------------------------------------------------
# T2: per-image OOD scoring rules
# ---------------------------------------------------------------------------

def _shrinkage_precision(x: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Inverse covariance with linear shrinkage toward a scaled identity.

    Plain covariance inversion is unstable at N=500 in 384 dimensions, so the
    estimate is shrunk toward its own average eigenvalue before inversion.
    """
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.t() @ xc) / max(x.shape[0] - 1, 1)
    d = cov.shape[0]
    target = torch.eye(d, device=cov.device) * (torch.diagonal(cov).mean())
    cov_s = (1.0 - eps) * cov + eps * target
    return torch.linalg.pinv(cov_s)


def _score_images(
    metric: M3EntropyMetric,
    query: torch.Tensor,
    ref: torch.Tensor,
    rule: str,
) -> np.ndarray:
    """Per-image novelty score against a real reference set (higher = more OOD)."""
    q = query.to(metric.device).float()
    r = ref.to(metric.device).float()

    if rule == "centroid":
        c = r.mean(0, keepdim=True)
        return (q - c).norm(dim=1).cpu().numpy()
    if rule == "knn1":
        return metric._knn_distances(q, r, k=1).cpu().numpy()
    if rule == "knn5":
        return metric._knn_distances(q, r, k=5).cpu().numpy()
    if rule == "mahalanobis":
        c = r.mean(0, keepdim=True)
        prec = _shrinkage_precision(r)
        d = q - c
        m = ((d @ prec) * d).sum(1).clamp(min=0.0).sqrt()
        return m.cpu().numpy()
    raise ValueError(f"unknown scoring rule: {rule}")


def sweep_ood_scoring(
    metric: M3EntropyMetric,
    feats: Dict[str, List[torch.Tensor]],
    rules: List[str],
) -> Dict:
    """Depth x scoring-rule ROC-AUC on two per-image tasks of unequal difficulty.

    The reference set is the real split used everywhere else and the negative
    class is always the disjoint real split.  Two positive classes are scored:

      cross_modality  chest CT.  A coarse shift, expected to saturate.
      same_modality   the DDPM's own samples.  The task a practitioner actually
                      faces -- flagging individual synthetic brain slices -- and
                      the one that separates depths and scoring rules.

    Reporting both is what distinguishes a rule that works from a task that is
    too easy to tell rules apart.
    """
    tasks = {"cross_modality": "lidc", "same_modality": "ddpm"}
    out: Dict[str, Dict] = {}

    for task, pos_set in tasks.items():
        if pos_set not in feats:
            continue
        print(f"    -- {task} (real_B vs {pos_set})")
        per_layer: Dict[int, Dict[str, float]] = {}
        for L in LAYERS:
            ref = feats["real_A"][L - 1]
            neg = feats["real_B"][L - 1]
            pos = feats[pos_set][L - 1]
            row: Dict[str, float] = {}
            for rule in rules:
                s_neg = _score_images(metric, neg, ref, rule)
                s_pos = _score_images(metric, pos, ref, rule)
                y = np.concatenate([np.zeros(len(s_neg)), np.ones(len(s_pos))])
                s = np.concatenate([s_neg, s_pos])
                row[rule] = float(roc_auc_score(y, s))
            per_layer[L] = row
            print(
                "       L{:<2d}  ".format(L)
                + "  ".join(f"{k}={v:.3f}" for k, v in row.items())
            )

        summary = {}
        for rule in rules:
            best = max(LAYERS, key=lambda L: per_layer[L][rule])
            summary[rule] = {
                "best_layer": best,
                "best_auc": per_layer[best][rule],
                "auc_at_L12": per_layer[12][rule],
                "mean_auc": float(np.mean([per_layer[L][rule] for L in LAYERS])),
            }
        aucs = [per_layer[L][r] for L in LAYERS for r in rules]
        out[task] = {
            "positive_set": pos_set,
            "per_layer": per_layer,
            "per_rule": summary,
            "auc_spread": float(max(aucs) - min(aucs)),
            "saturated": bool(min(aucs) > 0.99),
        }

    return {"rules": rules, "tasks": out}


# ---------------------------------------------------------------------------
# T3: memorization
# ---------------------------------------------------------------------------

def sweep_memorization(
    metric: M3EntropyMetric,
    real_t: torch.Tensor,
    gen_t: torch.Tensor,
    real_feats: List[torch.Tensor],
    rates: List[float],
    sigmas: List[float],
    seed: int,
) -> Dict:
    """Injection-recovery error per depth, across a jitter ladder.

    At the smallest jitter the task saturates and every depth recovers the
    injected rate; the ladder is what separates the depths.
    """
    rng_base = seed
    by_sigma: Dict[str, Dict] = {}

    for sigma in sigmas:
        per_layer: Dict[int, List[float]] = {L: [] for L in LAYERS}
        for rate in rates:
            n_inject = int(rate * len(gen_t))
            gen_mod = gen_t.clone()
            if n_inject > 0:
                rng = np.random.default_rng(rng_base)
                real_idx = rng.choice(len(real_t), size=n_inject, replace=True)
                gen_idx = rng.choice(len(gen_t), size=n_inject, replace=False)
                gen_mod[gen_idx] = _jitter(
                    real_t[real_idx], sigma=sigma, seed=int(seed + rate * 1000)
                )
            gen_feats = extract_all_layers(metric, gen_mod)
            for L in LAYERS:
                _, mem_rate = metric._compute_memorization(
                    real_feats[L - 1], gen_feats[L - 1], k=1
                )
                per_layer[L].append(float(mem_rate))

        summary = {}
        inj = np.array(rates)
        for L in LAYERS:
            rec = np.array(per_layer[L])
            mae = float(np.mean(np.abs(rec - inj)))
            rho = (
                float(spearmanr(inj, rec).correlation)
                if len(set(rec.tolist())) > 1
                else 0.0
            )
            summary[L] = {
                "recovered_rates": per_layer[L],
                "mean_abs_error": mae,
                "spearman_rho": rho,
            }
        best = min(LAYERS, key=lambda L: summary[L]["mean_abs_error"])
        maes = [summary[L]["mean_abs_error"] for L in LAYERS]
        by_sigma[f"{sigma:g}"] = {
            "jitter_sigma": sigma,
            "injection_rates": rates,
            "per_layer": summary,
            "best_layer": best,
            "best_mae": summary[best]["mean_abs_error"],
            "a_priori_layer": APRIORI["memorization"],
            "a_priori_mae": summary[APRIORI["memorization"]]["mean_abs_error"],
            "mae_spread": float(max(maes) - min(maes)),
            "saturated": bool(max(maes) - min(maes) < 0.01),
        }
        print(
            f"    sigma={sigma:<8g} best=L{best:<2d} (MAE={summary[best]['mean_abs_error']:.4f})  "
            f"L9 MAE={summary[APRIORI['memorization']]['mean_abs_error']:.4f}  "
            f"spread={by_sigma[f'{sigma:g}']['mae_spread']:.4f}  "
            f"saturated={by_sigma[f'{sigma:g}']['saturated']}"
        )

    discriminative = [
        s for s, d in by_sigma.items() if not d["saturated"]
    ]
    return {
        "injection_rates": rates,
        "sigmas": sigmas,
        "by_sigma": by_sigma,
        "discriminative_sigmas": discriminative,
        "a_priori_layer": APRIORI["memorization"],
        "best_layer_in_discriminative_regime": [
            by_sigma[s]["best_layer"] for s in discriminative
        ],
    }


# ---------------------------------------------------------------------------
# T4: coverage
# ---------------------------------------------------------------------------

def sweep_coverage(
    metric: M3EntropyMetric,
    real_feats: List[torch.Tensor],
    gen_feats: List[torch.Tensor],
    drops: List[float],
    knn_k: int,
    seed: int,
) -> Dict:
    """k-NN manifold recall per depth as diversity is progressively removed."""
    rng = np.random.default_rng(seed)
    n_gen = gen_feats[0].shape[0]
    keep_sets = []
    for d in drops:
        n_keep = max(int((1.0 - d) * n_gen), knn_k + 1)
        keep_sets.append(
            torch.from_numpy(np.sort(rng.choice(n_gen, size=n_keep, replace=False)))
        )

    per_layer: Dict[int, Dict[str, List[float]]] = {
        L: {"precision": [], "recall": []} for L in LAYERS
    }
    for keep in keep_sets:
        for L in LAYERS:
            prec, rec = metric._compute_precision_recall(
                real_feats[L - 1], gen_feats[L - 1][keep], k=knn_k
            )
            per_layer[L]["precision"].append(float(prec))
            per_layer[L]["recall"].append(float(rec))

    summary = {}
    for L in LAYERS:
        rec = np.array(per_layer[L]["recall"])
        rho = (
            float(spearmanr(drops, rec).correlation)
            if len(set(rec.tolist())) > 1
            else 0.0
        )
        summary[L] = {
            "precision": per_layer[L]["precision"],
            "recall": per_layer[L]["recall"],
            "spearman_rho_drop_vs_recall": rho,
        }
        print(
            f"    L{L:<2d}  rho(drop, recall)={rho:+.3f}  recall={[round(v, 3) for v in per_layer[L]['recall']]}"
        )

    correct = [L for L in LAYERS if summary[L]["spearman_rho_drop_vs_recall"] < 0]
    best = min(LAYERS, key=lambda L: summary[L]["spearman_rho_drop_vs_recall"])
    return {
        "drop_fractions": drops,
        "knn_k": knn_k,
        "per_layer": summary,
        "best_layer": best,
        "best_rho": summary[best]["spearman_rho_drop_vs_recall"],
        "a_priori_layer": APRIORI["coverage"],
        "a_priori_rho": summary[APRIORI["coverage"]]["spearman_rho_drop_vs_recall"],
        "layers_with_correct_direction": correct,
        "any_layer_valid": len(correct) > 0,
    }


# ---------------------------------------------------------------------------
# T5: degradation monotonicity
# ---------------------------------------------------------------------------

def sweep_monotonicity(
    metric: M3EntropyMetric,
    real_feats: List[torch.Tensor],
    gen_t: torch.Tensor,
    sigmas: List[float],
    seed: int,
) -> Dict:
    """MMD^2 per depth along a Gaussian-noise degradation ladder."""
    per_layer: Dict[int, List[float]] = {L: [] for L in LAYERS}
    for sigma in sigmas:
        if sigma == 0.0:
            noisy = gen_t
        else:
            g = torch.Generator().manual_seed(int(seed + sigma * 1000))
            noisy = (gen_t + torch.randn(gen_t.shape, generator=g) * sigma).clamp(0.0, 1.0)
        gf = extract_all_layers(metric, noisy)
        for L in LAYERS:
            per_layer[L].append(mmd_at(metric, real_feats[L - 1], gf[L - 1]))
        print(f"    sigma={sigma:.2f}  L12={per_layer[12][-1]:.4f}  L1={per_layer[1][-1]:.4f}")

    summary = {}
    for L in LAYERS:
        vals = per_layer[L]
        rho = float(spearmanr(sigmas, vals).correlation)
        summary[L] = {"mmd": vals, "spearman_rho": rho}
    monotone = [L for L in LAYERS if summary[L]["spearman_rho"] > 0.999]
    return {
        "sigmas": sigmas,
        "per_layer": summary,
        "monotone_layers": monotone,
        "all_layers_monotone": len(monotone) == N_LAYERS,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(res: Dict, out_path: str) -> None:
    """Four-panel depth profile: one panel per task, a-priori depth marked.

    Delegates to the manuscript plotter so a re-run of this experiment
    produces the same styled figure that the paper uses. The local fallback
    below only runs if that module is unavailable.
    """
    try:
        from tools.regen_paper_figures import fig_depth_grid
        from experiments._plot_style import apply_paper_style
        apply_paper_style()
        fig_depth_grid(res, out_path)
        return
    except Exception as exc:      # pragma: no cover - fallback path
        print(f"  [WARN] styled plotter unavailable ({exc}); using basic figure")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ap_col = "#c0392b"

    # (a) fidelity: permutation Z by depth
    ax = axes[0, 0]
    if "fidelity" in res:
        z = [res["fidelity"]["per_layer"][str(L)]["perm_z"] for L in LAYERS]
        ax.plot(LAYERS, z, "o-", color="#2c3e50")
        ax.axvline(APRIORI["fidelity"], color=ap_col, ls="--", lw=1.2,
                   label=f"a priori L{APRIORI['fidelity']}")
        ax.set_ylabel("permutation $Z$ (real vs DDPM)")
        ax.set_title("(a) Fidelity: deeper is better")
        ax.legend(fontsize=8)
    ax.set_xlabel("encoder block")
    ax.grid(alpha=0.3)

    # (b) memorization: MAE by depth, one line per jitter level
    ax = axes[0, 1]
    if "memorization" in res:
        bys = res["memorization"]["by_sigma"]
        cmap = plt.get_cmap("viridis")
        keys = list(bys)
        for i, s in enumerate(keys):
            mae = [bys[s]["per_layer"][str(L)]["mean_abs_error"] for L in LAYERS]
            ax.plot(LAYERS, mae, "o-", ms=3, color=cmap(i / max(len(keys) - 1, 1)),
                    label=f"$\\sigma$={bys[s]['jitter_sigma']:g}")
        ax.axvline(APRIORI["memorization"], color=ap_col, ls="--", lw=1.2,
                   label=f"a priori L{APRIORI['memorization']}")
        ax.set_ylabel("mean abs. recovery error")
        ax.set_title("(b) Memorization: shallower is better")
        ax.legend(fontsize=6, ncol=2)
    ax.set_xlabel("encoder block")
    ax.grid(alpha=0.3)

    # (c) coverage: rho(drop, recall) by depth
    ax = axes[1, 0]
    if "coverage" in res:
        rho = [
            res["coverage"]["per_layer"][str(L)]["spearman_rho_drop_vs_recall"]
            for L in LAYERS
        ]
        ax.bar(LAYERS, rho, color=["#27ae60" if v < 0 else "#95a5a6" for v in rho])
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(APRIORI["coverage"], color=ap_col, ls="--", lw=1.2,
                   label=f"a priori L{APRIORI['coverage']}")
        ax.set_ylabel(r"$\rho$(mode drop, recall)")
        ax.set_title("(c) Coverage: no depth is valid")
        ax.legend(fontsize=8)
    ax.set_xlabel("encoder block")
    ax.grid(alpha=0.3, axis="y")

    # (d) per-image AUC by depth on the discriminative task, one line per rule
    ax = axes[1, 1]
    if "ood_scoring" in res:
        tasks = res["ood_scoring"]["tasks"]
        # Prefer the task that actually separates depths.
        task = "same_modality" if "same_modality" in tasks else list(tasks)[0]
        for rule in res["ood_scoring"]["rules"]:
            auc = [tasks[task]["per_layer"][str(L)][rule] for L in LAYERS]
            ax.plot(LAYERS, auc, "o-", ms=3, label=rule)
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_ylabel("ROC-AUC (real vs DDPM, per image)")
        ax.set_title("(d) Per-image scoring rule x depth")
        ax.legend(fontsize=8)
    ax.set_xlabel("encoder block")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  figure -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", default="data_mri/brats_axial_multislice")
    ap.add_argument("--gen_dir", default="output/generated_500_standard")
    ap.add_argument("--wdm3d_dir", default="output/generated_wdm3d/brats")
    ap.add_argument("--lidc_dir", default="output/generated_wdm3d/lidc")
    ap.add_argument("--retinal_dir", default="output/generated_retinal")
    ap.add_argument("--num_images", type=int, default=500)
    ap.add_argument("--output_dir", default="results/depth_task_grid")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone", default="Snarcy/RadioDino-s16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_permutations", type=int, default=500)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument(
        "--real_sampling", choices=["patient_disjoint", "sorted"],
        default="patient_disjoint",
        help="how the real reference and null splits are drawn; 'sorted' "
             "reproduces the legacy first-N-filenames protocol",
    )
    ap.add_argument(
        "--tasks", nargs="+",
        default=["fidelity", "ood_scoring", "memorization", "coverage", "monotonicity"],
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[depth_task_grid] backbone={args.backbone} device={args.device} N={args.num_images}")
    metric = M3EntropyMetric(
        device=args.device,
        backbone_id=args.backbone,
        single_layer=None,
        seed=args.seed,
    )

    # ---- Load image sets -------------------------------------------------
    # real_A is the reference, real_B the patient-disjoint null probe.
    all_real = _glob_images(args.real_dir)
    if args.real_sampling == "patient_disjoint":
        real_A, real_B = split_real_by_patient(all_real, args.num_images, args.seed)
    else:
        # Legacy protocol, retained so the published numbers can be reproduced
        # and the reference-set effect quantified.
        real_A = all_real[: args.num_images]
        real_B = all_real[args.num_images: 2 * args.num_images]
    n_pat_A = len({_patient_id(p) for p in real_A})
    n_pat_B = len({_patient_id(p) for p in real_B})
    print(
        f"  real sampling = {args.real_sampling}: "
        f"real_A spans {n_pat_A} subjects, real_B spans {n_pat_B} "
        f"(overlap {len({_patient_id(p) for p in real_A} & {_patient_id(p) for p in real_B})})"
    )
    sets = {
        "real_A": real_A,
        "real_B": real_B,
        "ddpm": _glob_images(args.gen_dir, args.num_images),
        "wdm3d": _glob_images(args.wdm3d_dir, args.num_images),
        "lidc": _glob_images(args.lidc_dir, args.num_images),
        "retinal": _glob_images(args.retinal_dir, args.num_images),
    }

    tensors: Dict[str, torch.Tensor] = {}
    feats: Dict[str, List[torch.Tensor]] = {}
    print("  loading and extracting features (12 blocks, one pass per set) ...")
    for name, paths in sets.items():
        if not paths:
            print(f"    [skip] {name}: no images found")
            continue
        t0 = time.time()
        tensors[name] = _load_tensor(paths)
        feats[name] = extract_all_layers(metric, tensors[name])
        print(f"    {name:<8s} n={len(paths):<5d} {time.time() - t0:5.1f}s")

    results: Dict = {
        "experiment": "depth_task_grid",
        "backbone": args.backbone,
        "num_images": args.num_images,
        "seed": args.seed,
        "n_layers": N_LAYERS,
        "a_priori_layers": APRIORI,
        "sets": {k: len(v) for k, v in sets.items() if v},
        "real_sampling": args.real_sampling,
        "real_subjects": {"real_A": n_pat_A, "real_B": n_pat_B},
    }

    if "fidelity" in args.tasks:
        print("\n  [T1] fidelity / severity ordering")
        results["fidelity"] = sweep_fidelity(
            metric, feats, n_perm=args.n_permutations, seed=args.seed
        )

    if "ood_scoring" in args.tasks:
        print("\n  [T2] per-image OOD scoring rule x depth")
        results["ood_scoring"] = sweep_ood_scoring(
            metric, feats, rules=["centroid", "knn1", "knn5", "mahalanobis"]
        )

    if "memorization" in args.tasks:
        print("\n  [T3] memorization depth (jitter ladder)")
        results["memorization"] = sweep_memorization(
            metric,
            tensors["real_A"],
            tensors["ddpm"],
            feats["real_A"],
            rates=[0.0, 0.1, 0.2, 0.3, 0.4],
            sigmas=[5e-4, 5e-3, 0.02, 0.05, 0.1, 0.2],
            seed=args.seed,
        )

    if "coverage" in args.tasks:
        print("\n  [T4] coverage depth (mode drop)")
        results["coverage"] = sweep_coverage(
            metric,
            feats["real_A"],
            feats["ddpm"],
            drops=[0.0, 0.2, 0.4, 0.6, 0.8],
            knn_k=args.knn_k,
            seed=args.seed,
        )

    if "monotonicity" in args.tasks:
        print("\n  [T5] degradation monotonicity")
        results["monotonicity"] = sweep_monotonicity(
            metric,
            feats["real_A"],
            tensors["ddpm"],
            sigmas=[0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
            seed=args.seed,
        )

    out_json = os.path.join(args.output_dir, "depth_task_grid.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  results -> {out_json}")

    # Reload through JSON so the figure sees string keys uniformly.
    with open(out_json) as f:
        make_figure(json.load(f), os.path.join(args.output_dir, "depth_task_grid.png"))


if __name__ == "__main__":
    main()
