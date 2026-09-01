"""
Reference-set construction as a confound in generative medical image metrics (EXP 35).
======================================================================================

Distributional metrics compare a generated sample against a *finite real
reference*.  In natural-image benchmarks that reference is a set of independent
photographs, so its composition is rarely questioned.  Medical volumes break
that assumption: neighbouring axial slices from one subject are near-duplicates,
so a reference of N slices can span anywhere from a handful of subjects to N of
them, and the number of subjects -- not N -- controls how much of the real
distribution the reference actually represents.

This script holds N fixed and sweeps only the number of distinct subjects the N
slices are drawn from.  Everything else -- the generated set, the backbone, the
estimator, the seed -- is held constant, so any movement in the reported metric
is attributable to reference composition alone.

Three quantities are tracked at each diversity level:

  MMD^2(reference, generated)   the reported score for a fixed generator
  MMD^2(reference, real probe)  the real-vs-real null, which should be ~0
  FID(reference, generated)     the same sweep for the standard baseline, to
                                show whether the effect is specific to the
                                medical backbone or general to the setting

The practical question is whether a reader can compare two published numbers
without knowing the subject count behind each reference.

Usage
-----
    python experiments/reference_set_construction.py \\
        --real_dir data_mri/brats_axial_multislice \\
        --gen_dir  output/generated_500_standard \\
        --num_images 500 --n_seeds 3 \\
        --output_dir results/reference_set_construction \\
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

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.per_axis_validation import _glob_images, _load_tensor
from experiments.depth_task_grid import _patient_id, split_real_by_patient

# Depths reported: the a-priori final layer and the empirical fidelity optimum.
DEPTHS = [8, 12]


def draw_reference(
    by_patient: Dict[str, List[str]],
    pool: List[str],
    n_subjects: int,
    n_images: int,
    seed: int,
) -> List[str]:
    """N slices drawn from exactly n_subjects subjects, balanced across them."""
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(pool, size=n_subjects, replace=False))
    buckets = [list(rng.permutation(by_patient[q])) for q in chosen]

    picked: List[str] = []
    depth = 0
    while len(picked) < n_images:
        progressed = False
        for b in buckets:
            if depth < len(b):
                picked.append(b[depth])
                progressed = True
                if len(picked) == n_images:
                    break
        if not progressed:
            break
        depth += 1
    return picked


def fid_from_uint8(real_u8: torch.Tensor, gen_u8: torch.Tensor, device: str) -> float:
    """torchmetrics FID, matching evaluation/eval_pipeline.py."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.update(real_u8.to(device), real=True)
    fid.update(gen_u8.to(device), real=False)
    val = float(fid.compute().item())
    del fid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val


def load_uint8(paths: List[str]) -> torch.Tensor:
    from PIL import Image
    arrs = []
    for p in paths:
        a = np.array(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.uint8)
        arrs.append(torch.from_numpy(a.transpose(2, 0, 1)))
    return torch.stack(arrs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", default="data_mri/brats_axial_multislice")
    ap.add_argument("--gen_dir", default="output/generated_500_standard")
    ap.add_argument("--num_images", type=int, default=500)
    ap.add_argument("--subject_counts", type=int, nargs="+",
                    default=[5, 10, 17, 25, 50, 100, 200, 400])
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--output_dir", default="results/reference_set_construction")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone", default="Snarcy/RadioDino-s16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with_fid", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metric = M3EntropyMetric(device=args.device, backbone_id=args.backbone,
                             single_layer=None, seed=args.seed)

    all_real = _glob_images(args.real_dir)
    by_patient: Dict[str, List[str]] = {}
    for p in all_real:
        by_patient.setdefault(_patient_id(p), []).append(p)
    print(f"[reference_set_construction] {len(all_real)} slices, "
          f"{len(by_patient)} subjects")

    # A fixed, maximally diverse real probe supplies the real-vs-real null.
    # It is drawn from a disjoint subject pool so the null is never inflated by
    # subject overlap with the reference.
    probe_paths, _ = split_real_by_patient(all_real, args.num_images, args.seed + 777)
    probe_subjects = {_patient_id(p) for p in probe_paths}
    ref_pool = [q for q in by_patient if q not in probe_subjects]
    print(f"  probe: {len(probe_subjects)} subjects; "
          f"reference pool: {len(ref_pool)} subjects")

    gen_paths = _glob_images(args.gen_dir, args.num_images)
    gen_t = _load_tensor(gen_paths)
    gen_f = metric._extract_features_and_entropy(gen_t, layers_to_keep=set(DEPTHS))[0]
    probe_t = _load_tensor(probe_paths)
    probe_f = metric._extract_features_and_entropy(probe_t, layers_to_keep=set(DEPTHS))[0]

    gen_u8 = load_uint8(gen_paths) if args.with_fid else None

    rows: List[Dict] = []
    for n_sub in args.subject_counts:
        if n_sub > len(ref_pool):
            print(f"  [skip] {n_sub} subjects > pool of {len(ref_pool)}")
            continue
        for s in range(args.n_seeds):
            paths = draw_reference(by_patient, ref_pool, n_sub,
                                   args.num_images, args.seed + 1000 * s)
            if len(paths) < args.num_images:
                print(f"  [skip] {n_sub} subjects: only {len(paths)} slices available")
                continue
            t = _load_tensor(paths)
            f = metric._extract_features_and_entropy(t, layers_to_keep=set(DEPTHS))[0]

            row: Dict = {"n_subjects": n_sub, "seed_idx": s,
                         "n_images": args.num_images}
            for L in DEPTHS:
                rf = f[L - 1].float().to(metric.device)
                row[f"mmd_gen_L{L}"] = float(
                    metric._compute_mmd2_rbf(rf, gen_f[L - 1].float().to(metric.device)).item()
                )
                row[f"mmd_null_L{L}"] = float(
                    metric._compute_mmd2_rbf(rf, probe_f[L - 1].float().to(metric.device)).item()
                )
            if args.with_fid:
                row["fid_gen"] = fid_from_uint8(load_uint8(paths), gen_u8, args.device)

            rows.append(row)
            print(
                f"  subjects={n_sub:<4d} seed={s}  "
                + "  ".join(f"MMD_L{L}={row[f'mmd_gen_L{L}']:.4f} "
                            f"(null {row[f'mmd_null_L{L}']:.4f})" for L in DEPTHS)
                + (f"  FID={row['fid_gen']:.2f}" if args.with_fid else "")
            )

    # ---- aggregate ------------------------------------------------------
    agg: Dict[str, Dict] = {}
    for n_sub in sorted({r["n_subjects"] for r in rows}):
        sub = [r for r in rows if r["n_subjects"] == n_sub]
        e: Dict[str, float] = {"n_runs": len(sub)}
        for key in [k for k in sub[0] if k.startswith(("mmd_", "fid_"))]:
            vals = [r[key] for r in sub]
            e[f"{key}_mean"] = float(np.mean(vals))
            e[f"{key}_std"] = float(np.std(vals))
        agg[str(n_sub)] = e

    counts = sorted(int(k) for k in agg)
    lo, hi = str(counts[0]), str(counts[-1])
    summary = {
        "min_subjects": counts[0],
        "max_subjects": counts[-1],
        "inflation_factor": {
            f"L{L}": agg[lo][f"mmd_gen_L{L}_mean"] / max(agg[hi][f"mmd_gen_L{L}_mean"], 1e-12)
            for L in DEPTHS
        },
        "null_inflation": {
            f"L{L}": agg[lo][f"mmd_null_L{L}_mean"] - agg[hi][f"mmd_null_L{L}_mean"]
            for L in DEPTHS
        },
    }
    if args.with_fid:
        summary["inflation_factor"]["FID"] = (
            agg[lo]["fid_gen_mean"] / max(agg[hi]["fid_gen_mean"], 1e-12)
        )

    out = {
        "experiment": "reference_set_construction",
        "backbone": args.backbone,
        "num_images": args.num_images,
        "depths": DEPTHS,
        "n_seeds": args.n_seeds,
        "rows": rows,
        "aggregate": agg,
        "summary": summary,
    }
    path = os.path.join(args.output_dir, "reference_set_construction.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  results -> {path}")
    print(f"  inflation from {counts[-1]} to {counts[0]} subjects: "
          + ", ".join(f"{k} x{v:.2f}" for k, v in summary["inflation_factor"].items()))

    # ---- figure ---------------------------------------------------------
    fig_path = os.path.join(args.output_dir, "reference_set_construction.png")
    try:
        from tools.regen_paper_figures import fig_reference_sweep
        from experiments._plot_style import apply_paper_style
        apply_paper_style()
        fig_reference_sweep(out, fig_path)
        return
    except Exception as exc:      # pragma: no cover - fallback path
        print(f"  [WARN] styled plotter unavailable ({exc}); using basic figure")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = counts

    ax = axes[0]
    for L, c in zip(DEPTHS, ["#2c3e50", "#c0392b"]):
        mu = [agg[str(n)][f"mmd_gen_L{L}_mean"] for n in x]
        sd = [agg[str(n)][f"mmd_gen_L{L}_std"] for n in x]
        ax.errorbar(x, mu, yerr=sd, fmt="o-", color=c, capsize=3, label=f"MMD$^2$ L{L} (vs DDPM)")
        nu = [agg[str(n)][f"mmd_null_L{L}_mean"] for n in x]
        ax.plot(x, nu, "s--", ms=4, color=c, alpha=0.5, label=f"null L{L} (real vs real)")
    ax.set_xscale("log")
    ax.set_xlabel("distinct subjects in the real reference ($N$ = 500 slices)")
    ax.set_ylabel("MMD$^2$")
    ax.set_title("(a) Medical backbone (RadioDINO-s16)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if args.with_fid:
        mu = [agg[str(n)]["fid_gen_mean"] for n in x]
        sd = [agg[str(n)]["fid_gen_std"] for n in x]
        ax.errorbar(x, mu, yerr=sd, fmt="o-", color="#16a085", capsize=3, label="FID (InceptionV3)")
        ax.set_xscale("log")
        ax.set_ylabel("FID")
        ax.legend(fontsize=8)
    ax.set_xlabel("distinct subjects in the real reference ($N$ = 500 slices)")
    ax.set_title("(b) Natural-image backbone")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, "reference_set_construction.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"  figure -> {fig_path}")


if __name__ == "__main__":
    main()
