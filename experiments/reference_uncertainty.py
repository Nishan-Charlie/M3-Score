"""
Reference-cohort uncertainty, with a fixed-bandwidth kernel control (EXP 38).
============================================================================

This replaces the three-draw sweep in `reference_set_construction.py` with a
proper characterisation, and adds the control that sweep was missing.

Two things are measured at each cohort size m, over many independent draws:

  D(R, G)      distance from a real reference of m subjects to a fixed
               generated set
  D(R_i, R_j)  distance between two independently drawn real references of the
               same size -- the inter-cohort reference distribution

**The kernel control.** The multi-bandwidth RBF kernel used throughout this
project sets its bandwidth by the median heuristic on the pooled sample
R union G. That means the kernel is re-estimated for every comparison, so when
the reference changes, both the sample *and the kernel* change. A claim that
"reference composition changes the metric" is therefore confounded unless the
kernel is held fixed. Every quantity here is computed twice:

  adaptive   median heuristic on each pooled sample (the project default)
  fixed      one bandwidth calibrated once, on a held-out calibration set,
             and then frozen for every comparison

If the reference-composition effect is real it must survive the fixed kernel.
If it appears only under the adaptive kernel, the effect is an artefact of
bandwidth re-estimation and the paper's central claim does not hold.

Features for the whole subject pool are extracted once and reused, so drawing
50 references costs one forward pass, not 50. This runs on CPU by default so it
does not contend with a training job for the GPU.

Usage
-----
    python experiments/reference_uncertainty.py \\
        --n_draws 50 --num_images 500 --device cpu \\
        --output_dir results/reference_uncertainty
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

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.per_axis_validation import _glob_images, _load_tensor
from experiments.depth_task_grid import _patient_id

DEPTHS = [8, 12]
BAND_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


# ---------------------------------------------------------------------------
# MMD with an optionally fixed bandwidth
# ---------------------------------------------------------------------------

def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a * a).sum(1, keepdim=True) + (b * b).sum(1)
            - 2.0 * (a @ b.t())).clamp(min=0.0)


def _l2n(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)


def median_sigma2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Median heuristic on the pooled sample, matching the project default."""
    joint = torch.cat([_l2n(x), _l2n(y)], dim=0)
    d = _sq_dist(joint, joint)
    n = d.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=d.device), diagonal=1)
    return float(d[mask].median().clamp(min=1e-10))


def mmd2(x: torch.Tensor, y: torch.Tensor, sigma2: float | None = None) -> float:
    """Unbiased multi-bandwidth RBF MMD^2.

    sigma2=None reproduces the project default (median heuristic per pair);
    passing a value freezes the kernel across every comparison.
    """
    x, y = _l2n(x), _l2n(y)
    if sigma2 is None:
        sigma2 = median_sigma2(x, y)

    Dxx, Dyy, Dxy = _sq_dist(x, x), _sq_dist(y, y), _sq_dist(x, y)
    m, n = x.shape[0], y.shape[0]
    total = 0.0
    for s in BAND_SCALES:
        g = 1.0 / (2.0 * sigma2 * s)
        Kxx, Kyy, Kxy = torch.exp(-g * Dxx), torch.exp(-g * Dyy), torch.exp(-g * Dxy)
        t_rr = (Kxx.sum() - Kxx.trace()) / (m * (m - 1))
        t_gg = (Kyy.sum() - Kyy.trace()) / (n * (n - 1))
        total += float(t_rr + t_gg - 2.0 * Kxy.mean())
    return max(total / len(BAND_SCALES), 0.0)


# ---------------------------------------------------------------------------

def balanced_draw_idx(
    by_patient: Dict[str, List[int]],
    subjects: List[str],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row indices for n slices spread across the given subjects."""
    buckets = [list(rng.permutation(by_patient[q])) for q in subjects]
    picked: List[int] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for b in buckets:
            if depth < len(b):
                picked.append(b[depth])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        depth += 1
    return np.array(picked)


def describe(vals: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "median": float(np.median(vals)),
        "iqr_lo": float(np.percentile(vals, 25)),
        "iqr_hi": float(np.percentile(vals, 75)),
        "p2_5": float(np.percentile(vals, 2.5)),
        "p97_5": float(np.percentile(vals, 97.5)),
        "cv": float(vals.std(ddof=1) / vals.mean()) if vals.mean() > 0 and vals.size > 1 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", default="data_mri/brats_axial_multislice")
    ap.add_argument("--gen_dir", default="output/generated_500_standard")
    ap.add_argument("--num_images", type=int, default=500)
    ap.add_argument("--subject_counts", type=int, nargs="+",
                    default=[17, 25, 50, 100, 200, 400])
    ap.add_argument("--n_draws", type=int, default=50)
    ap.add_argument("--pool_subjects", type=int, default=500,
                    help="subjects whose slices are cached for drawing")
    ap.add_argument("--calib_subjects", type=int, default=60,
                    help="held-out subjects used once to fix the kernel bandwidth")
    ap.add_argument("--output_dir", default="results/reference_uncertainty")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(args.threads)
    metric = M3EntropyMetric(device=args.device, single_layer=None, seed=args.seed)
    keep = set(DEPTHS)

    # ---- subject pool ----------------------------------------------------
    all_real = _glob_images(args.real_dir)
    by_subj: Dict[str, List[str]] = {}
    for p in all_real:
        by_subj.setdefault(_patient_id(p), []).append(p)
    pids = sorted(by_subj)
    rng0 = np.random.default_rng(args.seed)
    shuffled = list(rng0.permutation(pids))

    calib_ids = shuffled[:args.calib_subjects]
    pool_ids = shuffled[args.calib_subjects:args.calib_subjects + args.pool_subjects]
    print(f"[reference_uncertainty] {len(pids)} subjects; "
          f"{len(calib_ids)} calibration, {len(pool_ids)} pool")

    pool_paths: List[str] = []
    pool_owner: List[str] = []
    for q in pool_ids:
        for p in by_subj[q]:
            pool_paths.append(p)
            pool_owner.append(q)
    print(f"  caching {len(pool_paths)} pool slices ...")

    def extract(paths: List[str]) -> Dict[int, torch.Tensor]:
        feats: Dict[int, List[torch.Tensor]] = {L: [] for L in DEPTHS}
        B = 250
        for i in range(0, len(paths), B):
            f = metric._extract_features_and_entropy(
                _load_tensor(paths[i:i + B]), layers_to_keep=keep)[0]
            for L in DEPTHS:
                feats[L].append(f[L - 1].float())
            print(f"    {min(i + B, len(paths))}/{len(paths)}", end="\r")
        print()
        return {L: torch.cat(feats[L], 0) for L in DEPTHS}

    t0 = time.time()
    pool_f = extract(pool_paths)
    print(f"  pool cached in {time.time() - t0:.0f}s")

    gen_paths = _glob_images(args.gen_dir, args.num_images)
    gen_f = extract(gen_paths)

    calib_paths = [by_subj[q][0] for q in calib_ids for _ in (0,)][:args.num_images]
    calib_paths = []
    for q in calib_ids:
        calib_paths.extend(by_subj[q])
    calib_paths = calib_paths[:args.num_images]
    calib_f = extract(calib_paths)

    # ---- fix the kernel once, on held-out data ---------------------------
    # Calibrated on a calibration cohort against the generated set, so the
    # bandwidth is representative of the comparison being made but does not
    # move when the evaluation reference changes.
    fixed_sigma2 = {L: median_sigma2(calib_f[L], gen_f[L]) for L in DEPTHS}
    print("  fixed bandwidths: " + ", ".join(
        f"L{L}: sigma^2={fixed_sigma2[L]:.5f}" for L in DEPTHS))

    idx_by_subj: Dict[str, List[int]] = {}
    for i, q in enumerate(pool_owner):
        idx_by_subj.setdefault(q, []).append(i)

    results: Dict = {
        "experiment": "reference_uncertainty",
        "num_images": args.num_images,
        "n_draws": args.n_draws,
        "depths": DEPTHS,
        "seed": args.seed,
        "fixed_sigma2": fixed_sigma2,
        "n_calibration_subjects": len(calib_ids),
        "by_cohort_size": {},
    }

    for m in args.subject_counts:
        if m > len(pool_ids):
            print(f"  [skip] m={m} exceeds pool of {len(pool_ids)}")
            continue
        rng = np.random.default_rng(args.seed + m)
        draws: List[np.ndarray] = []
        for d in range(args.n_draws):
            subs = list(rng.choice(pool_ids, size=m, replace=False))
            idx = balanced_draw_idx(idx_by_subj, subs, args.num_images, rng)
            if idx.size == args.num_images:
                draws.append(idx)
        if len(draws) < 4:
            print(f"  [skip] m={m}: only {len(draws)} usable draws")
            continue

        entry: Dict = {"n_subjects": m, "n_draws": len(draws)}
        for L in DEPTHS:
            P = pool_f[L]
            G = gen_f[L]
            rg_ad, rg_fx = [], []
            for idx in draws:
                R = P[torch.from_numpy(idx)]
                rg_ad.append(mmd2(R, G))
                rg_fx.append(mmd2(R, G, fixed_sigma2[L]))

            # Inter-cohort pairs from disjoint draws (consecutive pairs share
            # no draw, though subjects may recur across draws).
            rr_ad, rr_fx = [], []
            for a in range(0, len(draws) - 1, 2):
                Ra = P[torch.from_numpy(draws[a])]
                Rb = P[torch.from_numpy(draws[a + 1])]
                rr_ad.append(mmd2(Ra, Rb))
                rr_fx.append(mmd2(Ra, Rb, fixed_sigma2[L]))

            entry[f"L{L}"] = {
                "real_vs_gen_adaptive": describe(np.array(rg_ad)),
                "real_vs_gen_fixed": describe(np.array(rg_fx)),
                "real_vs_real_adaptive": describe(np.array(rr_ad)),
                "real_vs_real_fixed": describe(np.array(rr_fx)),
            }
            e = entry[f"L{L}"]
            print(f"  m={m:<4d} L{L}: "
                  f"RG adapt={e['real_vs_gen_adaptive']['median']:.4f} "
                  f"fixed={e['real_vs_gen_fixed']['median']:.4f} | "
                  f"RR adapt={e['real_vs_real_adaptive']['median']:.4f} "
                  f"fixed={e['real_vs_real_fixed']['median']:.4f} "
                  f"(CV_RG_fixed={e['real_vs_gen_fixed']['cv']:.3f})")

        results["by_cohort_size"][str(m)] = entry

    path = os.path.join(args.output_dir, "reference_uncertainty.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  results -> {path}")

    # ---- does the composition effect survive a fixed kernel? -------------
    sizes = sorted(int(k) for k in results["by_cohort_size"])
    if len(sizes) >= 2:
        lo, hi = str(sizes[0]), str(sizes[-1])
        print(f"\n  Kernel control ({sizes[0]} vs {sizes[-1]} subjects):")
        control: Dict = {}
        for L in DEPTHS:
            a = results["by_cohort_size"][lo][f"L{L}"]
            b = results["by_cohort_size"][hi][f"L{L}"]
            for kind in ("real_vs_gen", "real_vs_real"):
                ma, mb = a[f"{kind}_adaptive"]["median"], b[f"{kind}_adaptive"]["median"]
                fa, fb = a[f"{kind}_fixed"]["median"], b[f"{kind}_fixed"]["median"]
                # A ratio is only meaningful when the denominator is well away
                # from zero; the inter-cohort distance at large m collapses to
                # the numerical floor, where a ratio is not interpretable.
                floor = 1e-4
                r_ad = ma / mb if mb > floor else None
                r_fx = fa / fb if fb > floor else None
                control[f"L{L}_{kind}"] = {
                    "median_low_m_adaptive": ma, "median_high_m_adaptive": mb,
                    "median_low_m_fixed": fa, "median_high_m_fixed": fb,
                    "ratio_adaptive": r_ad, "ratio_fixed": r_fx,
                }
                shown = (f"ratio adaptive={r_ad:6.2f}x fixed={r_fx:6.2f}x"
                         if r_ad is not None and r_fx is not None
                         else f"{ma:.4f}->{mb:.4f} (adaptive), "
                              f"{fa:.4f}->{fb:.4f} (fixed); "
                              f"ratio undefined, denominator at floor")
                print(f"    L{L} {kind:<13s} {shown}")
        results["kernel_control"] = control
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
