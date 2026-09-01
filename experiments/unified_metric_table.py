"""
Unified cross-metric comparison on a single reference protocol (EXP 37).
=======================================================================

Earlier runs computed the same nominal comparison -- N real BraTS slices
against N DDPM samples -- in several scripts that each built the real reference
differently and, for FID, used different Inception implementations.  The
resulting numbers disagreed by up to a factor of two while carrying captions
that described them as identical.  This script exists so there is exactly one
place where the headline cross-metric table is produced: one reference
protocol, one implementation per metric, one output file.

Protocol
--------
Real reference: N slices drawn balanced across a subject block, with the block
sampled at random from the cohort rather than taken from the head of the sorted
filename list.  Two real probes are reported alongside the generators:

  real_probe   a disjoint, subject-diverse real block -- the value a metric
               should return when nothing is wrong.
  real_headblk the lowest-ID subject block, i.e. what the legacy
               first-N-filenames protocol selects -- reported to show what a
               site-homogeneous reference does to every metric at once.

Metrics: M3 (RadioDINO-s16 MMD^2 at L8 and L12), FID and KID (torchmetrics,
matching evaluation/eval_pipeline.py), CMMD (CLIP ViT-L/14).  Bootstrap 95%
intervals are reported for the MMD-family estimators, which are non-degenerate
under the alternative; no interval is claimed under the null, where the
statistic is degenerate and the ordinary bootstrap is not consistent.

Usage
-----
    python experiments/unified_metric_table.py \\
        --num_images 500 --n_boot 500 \\
        --output_dir results/unified_metric_table --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.per_axis_validation import _glob_images, _load_tensor
from experiments.depth_task_grid import _patient_id
from experiments.cohort_heterogeneity_null import balanced_draw

DEPTHS = [8, 12]


def load_uint8(paths: List[str]) -> torch.Tensor:
    from PIL import Image
    arrs = []
    for p in paths:
        a = np.array(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.uint8)
        arrs.append(torch.from_numpy(a.transpose(2, 0, 1)))
    return torch.stack(arrs)


def fid_kid(real_u8: torch.Tensor, gen_u8: torch.Tensor, device: str, subset: int) -> tuple:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    f = FrechetInceptionDistance(feature=2048).to(device)
    f.update(real_u8.to(device), real=True)
    f.update(gen_u8.to(device), real=False)
    fid_v = float(f.compute().item())
    del f
    k = KernelInceptionDistance(feature=2048, subset_size=subset).to(device)
    k.update(real_u8.to(device), real=True)
    k.update(gen_u8.to(device), real=False)
    kid_m, _ = k.compute()
    kid_v = float(kid_m.item())
    del k
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fid_v, kid_v


def _cluster_index(groups: Optional[List[str]], n: int) -> List[np.ndarray]:
    """Row indices grouped by cluster id, or one cluster per row if unlabelled."""
    if groups is None:
        return [np.array([i]) for i in range(n)]
    order: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    return [np.array(v) for v in order.values()]


def bootstrap_ci(
    fn,
    a: torch.Tensor,
    b: torch.Tensor,
    n_boot: int,
    seed: int,
    a_groups: Optional[List[str]] = None,
    b_groups: Optional[List[str]] = None,
) -> tuple:
    """Percentile bootstrap interval for a two-sample statistic.

    Resampling is done over CLUSTERS, not rows. For the real side a cluster is
    a subject: slices from one patient are near-duplicates, so resampling
    slices independently treats correlated observations as independent and
    produces intervals that are too narrow. This is the same non-independence
    the rest of the paper is about, so applying a row-level bootstrap here
    would contradict its own argument. When no grouping is supplied -- as for
    generated images, which have no subject structure -- each row is its own
    cluster and the procedure reduces to the ordinary bootstrap.

    Valid here because every interval reported is computed under the
    alternative, where MMD^2 is a non-degenerate U-statistic; under the null it
    is degenerate and no interval is reported.
    """
    rng = np.random.default_rng(seed)
    ca = _cluster_index(a_groups, a.shape[0])
    cb = _cluster_index(b_groups, b.shape[0])

    vals = []
    for _ in range(n_boot):
        ia = np.concatenate(
            [ca[k] for k in rng.integers(0, len(ca), size=len(ca))])
        ib = np.concatenate(
            [cb[k] for k in rng.integers(0, len(cb), size=len(cb))])
        vals.append(fn(a[torch.from_numpy(ia)], b[torch.from_numpy(ib)]))
    arr = np.array(vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", default="data_mri/brats_axial_multislice")
    ap.add_argument("--num_images", type=int, default=500)
    ap.add_argument("--reference_subjects", type=int, default=100)
    ap.add_argument("--output_dir", default="results/unified_metric_table")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--no_cmmd", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metric = M3EntropyMetric(device=args.device, single_layer=None, seed=args.seed)
    keep = set(DEPTHS)

    all_real = _glob_images(args.real_dir)
    by_patient: Dict[str, List[str]] = {}
    for p in all_real:
        by_patient.setdefault(_patient_id(p), []).append(p)
    pids = sorted(by_patient)

    rng = np.random.default_rng(args.seed)
    shuffled = list(rng.permutation(pids))
    n = args.reference_subjects
    ref_paths = balanced_draw(by_patient, shuffled[:n], args.num_images, seed=args.seed)
    probe_paths = balanced_draw(by_patient, shuffled[n:2 * n], args.num_images, seed=args.seed + 1)
    head_paths = balanced_draw(by_patient, pids[:17], args.num_images, seed=args.seed + 2)

    sets: Dict[str, List[str]] = {
        "real_probe": probe_paths,
        "real_headblk": head_paths,
        "ddpm": _glob_images("output/generated_500_standard", args.num_images),
        "wdm3d_mri": _glob_images("output/generated_wdm3d/brats", args.num_images),
        "lidc_ct": _glob_images("output/generated_wdm3d/lidc", args.num_images),
        "retinal": _glob_images("output/generated_retinal", args.num_images),
    }

    print(f"[unified_metric_table] reference: {n} subjects, {len(ref_paths)} slices")
    for k, v in sets.items():
        print(f"  {k:<14s} n={len(v)}")

    def m3_feats(paths: List[str]) -> Dict[int, torch.Tensor]:
        f = metric._extract_features_and_entropy(_load_tensor(paths), layers_to_keep=keep)[0]
        return {L: f[L - 1].float().to(metric.device) for L in DEPTHS}

    ref_f = m3_feats(ref_paths)
    ref_u8 = load_uint8(ref_paths)

    cmmd = None
    cmmd_ref = None
    if not args.no_cmmd:
        try:
            from evaluation.cmmd_metric import CMMDMetric
            cmmd = CMMDMetric(device=args.device)
            cmmd_ref = cmmd.extract_features(ref_paths, desc="CLIP [ref]")
        except Exception as exc:
            print(f"  [WARN] CMMD unavailable: {exc}")
            cmmd = None

    # Subject labels for the reference, so its side of the bootstrap resamples
    # patients rather than slices. Generated sets have no subject structure and
    # are resampled per image.
    ref_groups = [_patient_id(p) for p in ref_paths]
    real_sets = {"real_probe", "real_headblk"}

    rows: Dict[str, Dict] = {}
    for name, paths in sets.items():
        if not paths:
            continue
        print(f"\n  -- {name}")
        row: Dict = {"n": len(paths)}
        gf = m3_feats(paths)
        other_groups = [_patient_id(p) for p in paths] if name in real_sets else None
        row["n_subjects"] = len(set(other_groups)) if other_groups else None
        for L in DEPTHS:
            val = float(metric._compute_mmd2_rbf(ref_f[L], gf[L]).item())
            lo, hi = bootstrap_ci(
                lambda x, y: float(metric._compute_mmd2_rbf(x, y).item()),
                ref_f[L], gf[L], args.n_boot, args.seed,
                a_groups=ref_groups, b_groups=other_groups
            )
            row[f"m3_L{L}"] = val
            row[f"m3_L{L}_ci95"] = [lo, hi]
            print(f"     M3 L{L:<2d} = {val:.4f}  [{lo:.4f}, {hi:.4f}]")

        gen_u8 = load_uint8(paths)
        f_v, k_v = fid_kid(ref_u8, gen_u8, args.device, subset=min(100, len(paths)))
        row["fid"], row["kid"] = f_v, k_v
        print(f"     FID    = {f_v:.2f}   KID = {k_v:.5f}")

        if cmmd is not None:
            gfeat = cmmd.extract_features(paths, desc=f"CLIP [{name}]")
            c_v = float(cmmd.compute_from_features(cmmd_ref, gfeat))
            lo, hi = bootstrap_ci(
                lambda x, y: float(cmmd.compute_from_features(x.numpy(), y.numpy())),
                torch.from_numpy(cmmd_ref), torch.from_numpy(gfeat), 200, args.seed
            )
            row["cmmd"] = c_v
            row["cmmd_ci95"] = [lo, hi]
            print(f"     CMMD   = {c_v:.4f}  [{lo:.4f}, {hi:.4f}]")

        rows[name] = row

    # ---- ordering checks -------------------------------------------------
    # A same-modality generator of clearly lower quality should sit closer to
    # the reference than a cross-modality shift.  This is the ordering CMMD was
    # previously reported to invert; it is re-checked here on the corrected
    # reference protocol.
    checks: Dict[str, Dict] = {}
    if "wdm3d_mri" in rows and "lidc_ct" in rows:
        for key in ["m3_L8", "m3_L12", "fid", "kid", "cmmd"]:
            if key in rows["wdm3d_mri"] and key in rows["lidc_ct"]:
                mri = rows["wdm3d_mri"][key]
                ct = rows["lidc_ct"][key]
                checks[key] = {
                    "wdm3d_mri": mri,
                    "lidc_ct": ct,
                    "delta_ct_minus_mri": ct - mri,
                    "correct_order": bool(ct > mri),
                }

    out = {
        "experiment": "unified_metric_table",
        "num_images": args.num_images,
        "reference_subjects": args.reference_subjects,
        "reference_protocol": "balanced draw across a random subject block",
        "seed": args.seed,
        "n_boot": args.n_boot,
        "rows": rows,
        "ordering_checks": checks,
    }
    path = os.path.join(args.output_dir, "unified_metric_table.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  results -> {path}")
    if checks:
        print("  ordering (cross-modality CT should exceed weak same-modality MRI):")
        for k, v in checks.items():
            print(f"    {k:<8s} delta={v['delta_ct_minus_mri']:+.4f}  "
                  f"correct={v['correct_order']}")


if __name__ == "__main__":
    main()
