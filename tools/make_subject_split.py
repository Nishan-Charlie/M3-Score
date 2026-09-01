"""
Build a subject-disjoint train/held-out split of the BraTS slice directory.
===========================================================================

The generators evaluated in this project were fine-tuned on the same slice
directory the evaluation references are drawn from, so every reference image was
also a training image. That makes the real-versus-generated comparison
vulnerable to the objection that the generated distribution is unusually close
to the reference because the generator was fitted to it.

This script produces the split needed to answer that objection: a training
directory containing only TRAIN subjects, and a manifest recording exactly which
subjects fall on each side. A generator fine-tuned on the training directory can
then be scored against references drawn from held-out subjects, and the
difference between its distance to seen and unseen cohorts is a direct estimate
of the contamination effect.

Slices are hard-linked rather than copied, so the split costs no additional disk
space and no read time. Hard links work on NTFS without elevation; if the
filesystem refuses them the script falls back to copying.

Usage
-----
    python tools/make_subject_split.py \\
        --src data_mri/brats_axial_multislice \\
        --out_dir data_mri/brats_split \\
        --train_frac 0.7 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.depth_task_grid import _patient_id


def link_or_copy(src: str, dst: str) -> str:
    """Hard-link src to dst; fall back to copying if the filesystem refuses."""
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data_mri/brats_axial_multislice")
    ap.add_argument("--out_dir", default="data_mri/brats_split")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.src, "**", "*.png"), recursive=True))
    if not paths:
        raise SystemExit(f"no PNGs under {args.src}")

    by_patient: dict[str, list[str]] = {}
    for p in paths:
        by_patient.setdefault(_patient_id(p), []).append(p)
    pids = sorted(by_patient)

    rng = np.random.default_rng(args.seed)
    shuffled = list(rng.permutation(pids))
    n_train = int(round(args.train_frac * len(shuffled)))
    train_ids, held_ids = sorted(shuffled[:n_train]), sorted(shuffled[n_train:])

    n_train_img = sum(len(by_patient[q]) for q in train_ids)
    n_held_img = sum(len(by_patient[q]) for q in held_ids)
    print(f"[make_subject_split] {len(paths)} slices, {len(pids)} subjects")
    print(f"  train:    {len(train_ids):>5d} subjects, {n_train_img:>6d} slices")
    print(f"  held-out: {len(held_ids):>5d} subjects, {n_held_img:>6d} slices")
    assert not (set(train_ids) & set(held_ids)), "subject overlap between splits"

    if args.dry_run:
        print("  (dry run; nothing written)")
        return

    train_dir = os.path.join(args.out_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    modes = {"link": 0, "copy": 0, "exists": 0}
    for q in train_ids:
        for src in by_patient[q]:
            dst = os.path.join(train_dir, os.path.basename(src))
            modes[link_or_copy(src, dst)] += 1
    print(f"  wrote {train_dir}: {modes}")

    manifest = {
        "source": args.src,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "n_subjects_total": len(pids),
        "n_train_subjects": len(train_ids),
        "n_heldout_subjects": len(held_ids),
        "n_train_slices": n_train_img,
        "n_heldout_slices": n_held_img,
        "train_subjects": train_ids,
        "heldout_subjects": held_ids,
    }
    mpath = os.path.join(args.out_dir, "split_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest -> {mpath}")
    print(f"\n  next: fine-tune on {train_dir}, then evaluate against references")
    print("  drawn from heldout_subjects in the manifest.")


if __name__ == "__main__":
    main()
