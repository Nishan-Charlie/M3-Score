"""
Does generator/reference subject overlap change the reported distance? (EXP 39)
==============================================================================

Every generator evaluated elsewhere in this project was fine-tuned on the same
slice pool the evaluation references are drawn from, so every reference image
was also a training image. That leaves the central real-versus-generated
comparison open to the objection that the generated distribution sits close to
the reference because the generator was fitted to it, rather than because of
anything the paper claims.

This measures the size of that effect directly, and needs only one generator to
do it. A model trained on a subject-disjoint split is scored against two
references that differ in exactly one respect:

    D(G, R_seen)     reference drawn from the generator's OWN training subjects
    D(G, R_unseen)   reference drawn from held-out subjects it never saw

Both references are matched in image count, subject count and draw procedure,
so the gap between them is attributable to subject overlap alone:

    contamination gap = D(G, R_unseen) - D(G, R_seen)

A gap near zero means overlap is not moving the metric and the existing
results stand. A large positive gap means distances measured against seen
subjects are optimistic, and every real-versus-generated number has to be
restated on held-out references.

The kernel is frozen to a bandwidth calibrated once on a held-out cohort, so
that the comparison cannot be confounded by the median heuristic re-estimating
itself between the two conditions.

Usage
-----
    python experiments/contamination_check.py \\
        --gen_dir output/generated_train70 \\
        --manifest data_mri/brats_split/split_manifest.json \\
        --n_draws 20 --num_images 300 --device cuda:0
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

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.per_axis_validation import _glob_images, _load_tensor
from experiments.depth_task_grid import _patient_id
from experiments.reference_uncertainty import (
    mmd2, median_sigma2, balanced_draw_idx, describe,
)

DEPTHS = [8, 12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", default="data_mri/brats_axial_multislice")
    ap.add_argument("--gen_dir", required=True,
                    help="samples from the model trained on the split's TRAIN subjects")
    ap.add_argument("--manifest", default="data_mri/brats_split/split_manifest.json")
    ap.add_argument("--num_images", type=int, default=300)
    ap.add_argument("--cohort_subjects", type=int, default=100)
    ap.add_argument("--n_draws", type=int, default=20)
    ap.add_argument("--output_dir", default="results/contamination_check")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(args.threads)

    with open(args.manifest) as f:
        man = json.load(f)
    seen_ids = set(man["train_subjects"])
    unseen_ids = set(man["heldout_subjects"])
    assert not (seen_ids & unseen_ids), "split manifest is not subject-disjoint"
    print(f"[contamination_check] {len(seen_ids)} train (seen) / "
          f"{len(unseen_ids)} held-out (unseen) subjects")

    metric = M3EntropyMetric(device=args.device, single_layer=None, seed=args.seed)
    keep = set(DEPTHS)

    all_real = _glob_images(args.real_dir)
    by_subj: Dict[str, List[str]] = {}
    for p in all_real:
        by_subj.setdefault(_patient_id(p), []).append(p)

    def extract(paths: List[str]) -> Dict[int, torch.Tensor]:
        out: Dict[int, List[torch.Tensor]] = {L: [] for L in DEPTHS}
        B = 250
        for i in range(0, len(paths), B):
            f = metric._extract_features_and_entropy(
                _load_tensor(paths[i:i + B]), layers_to_keep=keep)[0]
            for L in DEPTHS:
                out[L].append(f[L - 1].float())
        return {L: torch.cat(out[L], 0) for L in DEPTHS}

    gen_paths = _glob_images(args.gen_dir, args.num_images)
    if not gen_paths:
        raise SystemExit(f"no images in {args.gen_dir}")
    print(f"  generated: {len(gen_paths)} images from {args.gen_dir}")
    gen_f = extract(gen_paths)

    # Pool each side, capped so both conditions draw from comparable material.
    def pool(ids: List[str]) -> tuple:
        paths, owner = [], []
        for q in sorted(ids):
            for p in by_subj.get(q, []):
                paths.append(p)
                owner.append(q)
        return paths, owner

    n_pool = min(len(seen_ids), len(unseen_ids))
    rng0 = np.random.default_rng(args.seed)
    seen_sub = list(rng0.choice(sorted(seen_ids), n_pool, replace=False))
    unseen_sub = sorted(unseen_ids)
    print(f"  pooling {n_pool} subjects per side")

    pools = {}
    for label, ids in (("seen", seen_sub), ("unseen", unseen_sub)):
        paths, owner = pool(ids)
        print(f"  caching {label}: {len(paths)} slices from {len(set(owner))} subjects")
        feats = extract(paths)
        idx_by_subj: Dict[str, List[int]] = {}
        for i, q in enumerate(owner):
            idx_by_subj.setdefault(q, []).append(i)
        pools[label] = {"feats": feats, "idx": idx_by_subj, "ids": sorted(set(owner))}

    # Freeze the kernel on the unseen side, so it is not tuned to either
    # condition by the comparison itself.
    calib_rng = np.random.default_rng(args.seed + 99)
    calib_subs = list(calib_rng.choice(pools["unseen"]["ids"],
                                       min(40, len(pools["unseen"]["ids"])),
                                       replace=False))
    calib_idx = balanced_draw_idx(pools["unseen"]["idx"], calib_subs,
                                  args.num_images, calib_rng)
    fixed_sigma2 = {
        L: median_sigma2(pools["unseen"]["feats"][L][torch.from_numpy(calib_idx)],
                         gen_f[L])
        for L in DEPTHS
    }
    print("  fixed bandwidth: " + ", ".join(
        f"L{L}: {fixed_sigma2[L]:.5f}" for L in DEPTHS))

    results: Dict = {
        "experiment": "contamination_check",
        "gen_dir": args.gen_dir,
        "num_images": args.num_images,
        "cohort_subjects": args.cohort_subjects,
        "n_draws": args.n_draws,
        "depths": DEPTHS,
        "fixed_sigma2": fixed_sigma2,
        "n_train_subjects": len(seen_ids),
        "n_heldout_subjects": len(unseen_ids),
        "conditions": {},
    }

    for label in ("seen", "unseen"):
        P = pools[label]
        rng = np.random.default_rng(args.seed + (0 if label == "seen" else 1))
        vals: Dict[int, List[float]] = {L: [] for L in DEPTHS}
        m = min(args.cohort_subjects, len(P["ids"]))
        for _ in range(args.n_draws):
            subs = list(rng.choice(P["ids"], size=m, replace=False))
            idx = balanced_draw_idx(P["idx"], subs, args.num_images, rng)
            if idx.size < args.num_images:
                continue
            for L in DEPTHS:
                R = P["feats"][L][torch.from_numpy(idx)]
                vals[L].append(mmd2(R, gen_f[L], fixed_sigma2[L]))
        results["conditions"][label] = {
            f"L{L}": describe(np.array(vals[L])) for L in DEPTHS
        }
        for L in DEPTHS:
            s = results["conditions"][label][f"L{L}"]
            print(f"  {label:<7s} L{L}: median={s['median']:.4f} "
                  f"[{s['p2_5']:.4f}, {s['p97_5']:.4f}]  n={s['n']}")

    # ---- the contamination gap ------------------------------------------
    print("\n  contamination gap = D(G, unseen) - D(G, seen)")
    gaps: Dict[str, Dict] = {}
    for L in DEPTHS:
        a = results["conditions"]["seen"][f"L{L}"]
        b = results["conditions"]["unseen"][f"L{L}"]
        gap = b["median"] - a["median"]
        rel = gap / a["median"] if a["median"] > 0 else None
        # Non-overlapping 95% intervals is the conservative read; overlapping
        # intervals mean the gap is not resolved at this number of draws.
        separated = a["p97_5"] < b["p2_5"] or b["p97_5"] < a["p2_5"]
        gaps[f"L{L}"] = {
            "seen_median": a["median"], "unseen_median": b["median"],
            "gap": gap, "relative_gap": rel,
            "intervals_disjoint": bool(separated),
        }
        print(f"    L{L}: {gap:+.4f} ({100 * rel:+.1f}% of seen)   "
              f"95% intervals disjoint: {separated}")

    results["contamination_gap"] = gaps
    path = os.path.join(args.output_dir, "contamination_check.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  results -> {path}")


if __name__ == "__main__":
    main()
