"""
Cohort-heterogeneity null for generative medical image evaluation (EXP 36).
===========================================================================

A permutation test asks whether the generated set and the real set were drawn
from the same distribution.  For any competent medical generator the answer is
always no, at overwhelming significance, so the test is not informative: it
rejects a null nobody believed.  What a practitioner needs to know is different
and comparative --

    is this generator further from the real reference than two real patient
    cohorts are from each other?

That question has an empirical null.  Draw many disjoint blocks of real
subjects, measure the distance between every pair, and the resulting spread is
the irreducible heterogeneity of the dataset at that cohort size.  A generator
whose distance falls inside that spread is closer to the reference than routine
inter-cohort variation; one that falls outside is distinguishable from real data
by more than cohort effects alone.

This script builds that null and calibrates each generator against it:

  1.  Partition the subject list into K disjoint blocks and draw N slices from
      each, balanced across the subjects in the block.
  2.  Compute MMD^2 for all K(K-1)/2 block pairs -> the cohort-heterogeneity
      null at that cohort size.
  3.  Compute MMD^2 from a held-out reference block to each generated set.
  4.  Report, per generator, where it falls in the null: a cohort-calibrated
      z-score and an empirical p-value, alongside the conventional permutation
      p-value for contrast.

Sweeping the block size shows how the null tightens as cohorts grow, which is
what makes a reported score comparable across studies only when the cohort size
is reported with it.

Usage
-----
    python experiments/cohort_heterogeneity_null.py \\
        --real_dir data_mri/brats_axial_multislice \\
        --num_images 500 --block_subjects 17 50 100 \\
        --output_dir results/cohort_heterogeneity_null \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import itertools
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

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.per_axis_validation import _glob_images, _load_tensor
from experiments.depth_task_grid import _patient_id

DEPTHS = [8, 12]


def balanced_draw(
    by_patient: Dict[str, List[str]],
    subjects: List[str],
    n: int,
    seed: int = 0,
) -> List[str]:
    """N slices from the given subjects, one per subject before any second.

    Slice order within a subject is shuffled first.  Taking slices in filename
    order would draw only the lowest indices whenever there are many subjects
    and few slices each, and those are the volume-edge positions -- a second
    composition confound on top of the subject one this function exists to
    control.
    """
    rng = np.random.default_rng(seed)
    buckets = [list(rng.permutation(by_patient[q])) for q in subjects]
    picked: List[str] = []
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
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", default="data_mri/brats_axial_multislice")
    ap.add_argument("--num_images", type=int, default=500)
    ap.add_argument("--block_subjects", type=int, nargs="+", default=[17, 50, 100])
    ap.add_argument("--max_blocks", type=int, default=24,
                    help="cap on disjoint blocks per cohort size (K); the null "
                         "uses all K(K-1)/2 pairs")
    ap.add_argument("--gen_dirs", nargs="+", default=[
        "output/generated_500_standard",
        "output/generated_wdm3d/brats",
        "output/generated_wdm3d/lidc",
        "output/generated_retinal",
    ])
    ap.add_argument("--gen_names", nargs="+", default=["ddpm", "wdm3d", "lidc_ct", "retinal"])
    ap.add_argument("--output_dir", default="results/cohort_heterogeneity_null")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone", default="Snarcy/RadioDino-s16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_permutations", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metric = M3EntropyMetric(device=args.device, backbone_id=args.backbone,
                             single_layer=None, seed=args.seed)
    keep = set(DEPTHS)

    all_real = _glob_images(args.real_dir)
    by_patient: Dict[str, List[str]] = {}
    for p in all_real:
        by_patient.setdefault(_patient_id(p), []).append(p)
    pids = sorted(by_patient)
    print(f"[cohort_heterogeneity_null] {len(all_real)} slices, {len(pids)} subjects")

    def feats(paths: List[str]) -> Dict[int, torch.Tensor]:
        f = metric._extract_features_and_entropy(_load_tensor(paths), layers_to_keep=keep)[0]
        return {L: f[L - 1].float().to(metric.device) for L in DEPTHS}

    def mmd(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(metric._compute_mmd2_rbf(a, b).item())

    # Generated sets are fixed across all cohort sizes.
    gen_feats: Dict[str, Dict[int, torch.Tensor]] = {}
    for name, d in zip(args.gen_names, args.gen_dirs):
        paths = _glob_images(d, args.num_images)
        if not paths:
            print(f"  [skip] {name}: no images in {d}")
            continue
        gen_feats[name] = feats(paths)
        print(f"  generated set {name:<10s} n={len(paths)}")

    results: Dict = {
        "experiment": "cohort_heterogeneity_null",
        "backbone": args.backbone,
        "num_images": args.num_images,
        "depths": DEPTHS,
        "seed": args.seed,
        "n_subjects_total": len(pids),
        "by_cohort_size": {},
    }

    for n_sub in args.block_subjects:
        rng = np.random.default_rng(args.seed)
        shuffled = list(rng.permutation(pids))
        n_blocks = min(args.max_blocks, len(shuffled) // n_sub)
        if n_blocks < 3:
            print(f"  [skip] cohort size {n_sub}: only {n_blocks} disjoint blocks")
            continue

        print(f"\n  cohort size {n_sub} subjects -> {n_blocks} disjoint blocks")
        blocks: List[Dict[int, torch.Tensor]] = []
        usable = 0
        for b in range(n_blocks):
            subs = shuffled[b * n_sub:(b + 1) * n_sub]
            paths = balanced_draw(by_patient, subs, args.num_images, seed=args.seed + b)
            if len(paths) < args.num_images:
                continue
            blocks.append(feats(paths))
            usable += 1
        print(f"    {usable} blocks with a full {args.num_images}-slice draw")
        if usable < 3:
            print("    [skip] too few usable blocks")
            continue

        # ---- null: disjoint real block pairs, excluding the reference ----
        # Block 0 is used as the reference for the generator comparisons
        # below. If it also contributed to the null, then MMD(R_0, G) and the
        # null values MMD(R_0, R_j) would share the same real sample, and the
        # z-score would not have the intended interpretation. The null is
        # therefore built only from blocks 1..K-1, which are disjoint from the
        # reference in subjects and in slices.
        ref_idx = 0
        null_idx = [i for i in range(usable) if i != ref_idx]
        null: Dict[int, List[float]] = {L: [] for L in DEPTHS}
        for i, j in itertools.combinations(null_idx, 2):
            for L in DEPTHS:
                null[L].append(mmd(blocks[i][L], blocks[j][L]))

        entry: Dict = {
            "n_subjects_per_block": n_sub,
            "n_blocks": usable,
            "n_blocks_in_null": len(null_idx),
            "n_null_pairs": len(null[DEPTHS[0]]),
            "reference_excluded_from_null": True,
            "null": {},
            "generators": {},
        }
        for L in DEPTHS:
            arr = np.array(null[L])
            entry["null"][f"L{L}"] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "q95": float(np.percentile(arr, 95)),
                "values": [float(v) for v in arr],
            }
            print(f"    null L{L}: mean={arr.mean():.4f}  sd={arr.std():.4f}  "
                  f"range=[{arr.min():.4f}, {arr.max():.4f}]  q95={np.percentile(arr, 95):.4f}")

        # ---- generators calibrated against that null ---------------------
        # Block 0 is the reference; it is one of the blocks that formed the
        # null, so the comparison is like-for-like in cohort size.
        ref = blocks[0]
        for name, gf in gen_feats.items():
            g_entry: Dict = {}
            for L in DEPTHS:
                obs = mmd(ref[L], gf[L])
                arr = np.array(null[L])
                z = (obs - arr.mean()) / max(arr.std(), 1e-12)
                # +1 smoothing keeps the p-value away from an exact zero.
                p = float((arr >= obs).sum() + 1) / (len(arr) + 1)
                perm_p, perm_z, _ = metric._permutation_test_fidelity(
                    ref[L], gf[L], n_permutations=args.n_permutations, seed=args.seed
                )
                # z is reported alongside its two parts. The excess is the
                # substantive quantity -- how much further the generator sits
                # than two real cohorts typically do -- and it is roughly
                # invariant to cohort size. The null spread is what cohort size
                # actually changes. Reporting only z invites reading a rising
                # z as a growing effect when it is a shrinking denominator.
                g_entry[f"L{L}"] = {
                    "mmd": obs,
                    "excess_over_null_mean": float(obs - arr.mean()),
                    "null_mean": float(arr.mean()),
                    "null_sd": float(arr.std()),
                    "cohort_z": float(z),
                    "cohort_p": p,
                    "ratio_to_null_mean": float(obs / max(arr.mean(), 1e-12)),
                    "exceeds_null_max": bool(obs > arr.max()),
                    "permutation_p": float(perm_p),
                    "permutation_z": float(perm_z),
                }
                print(f"    {name:<10s} L{L}: MMD={obs:.4f}  cohort z={z:7.2f}  "
                      f"cohort p={p:.4f}  perm z={perm_z:8.1f}  perm p={perm_p:.4f}")
            entry["generators"][name] = g_entry

        results["by_cohort_size"][str(n_sub)] = entry

    path = os.path.join(args.output_dir, "cohort_heterogeneity_null.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  results -> {path}")

    # ---- figure ---------------------------------------------------------
    sizes = sorted(int(k) for k in results["by_cohort_size"])
    if not sizes:
        return

    fig_path = os.path.join(args.output_dir, "cohort_heterogeneity_null.png")
    try:
        from tools.regen_paper_figures import fig_cohort_null
        from experiments._plot_style import apply_paper_style
        apply_paper_style()
        fig_cohort_null(results, fig_path)
        return
    except Exception as exc:      # pragma: no cover - fallback path
        print(f"  [WARN] styled plotter unavailable ({exc}); using basic figure")

    fig, axes = plt.subplots(1, len(sizes), figsize=(4.6 * len(sizes), 4.2), squeeze=False)
    L_plot = 12
    for ax, n_sub in zip(axes[0], sizes):
        e = results["by_cohort_size"][str(n_sub)]
        vals = e["null"][f"L{L_plot}"]["values"]
        ax.hist(vals, bins=20, color="#bdc3c7", edgecolor="white",
                label=f"real-vs-real null\n({e['n_null_pairs']} cohort pairs)")
        colors = {"ddpm": "#2980b9", "wdm3d": "#e67e22",
                  "lidc_ct": "#c0392b", "retinal": "#8e44ad"}
        for name, g in e["generators"].items():
            v = g[f"L{L_plot}"]["mmd"]
            ax.axvline(v, color=colors.get(name, "#333"), lw=1.8,
                       label=f"{name} ({v:.3f})")
        ax.set_title(f"{n_sub} subjects per cohort")
        ax.set_xlabel(f"MMD$^2$ at L{L_plot}")
        ax.set_ylabel("cohort pairs")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.suptitle("Generator distances against the real cohort-heterogeneity null", y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, "cohort_heterogeneity_null.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {fig_path}")


if __name__ == "__main__":
    main()
