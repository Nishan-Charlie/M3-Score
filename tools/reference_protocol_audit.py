"""
Literature audit: how do published medical generative papers build their real
reference set?
==============================================================================

The central claim of the manuscript is that reference-set composition confounds
distributional evaluation in medical imaging. A reviewer can reasonably answer:
"you constructed a deliberately poor selection rule and then showed it is
poor." The rebuttal has to be empirical -- what do published papers actually
do?

This module holds the audit protocol and the recorded findings, so the numbers
quoted in the paper are reproducible and every entry is attributable to a
specific claim in a specific paper.

Protocol
--------
For each paper that reports a distributional metric (FID, KID, MMD, FRD, CMMD)
on medical images, we record five fields, each answerable from the text alone:

  n_images_reported     Does it state how many real images form the reference?
  n_subjects_reported   Does it state how many distinct patients/subjects those
                        images come from?
  selection_described   Does it describe HOW those images were selected from
                        the pool (random, all, first-N, per-subject quota)?
  subject_level_split   Does it state that train/eval splits are patient-level
                        rather than image-level?
  reference_from_train  Is the reference drawn from data the generator was
                        trained on?

Values are one of: "yes", "no", "partial", "n/a" (single-subject-per-image
datasets, where the subject question does not arise).

An entry of "no" is not a criticism of the paper. Most of these papers are not
about evaluation protocol, and the omission is the field's convention rather
than an individual lapse -- which is exactly the point being measured.

Usage
-----
    python tools/reference_protocol_audit.py            # summary table
    python tools/reference_protocol_audit.py --latex    # LaTeX table for paper
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

FIELDS = [
    "n_images_reported",
    "n_subjects_reported",
    "selection_described",
    "subject_level_split",
    "reference_from_train",
]

# ---------------------------------------------------------------------------
# Recorded findings. Each entry cites the specific evidence used to score it.
# Populated by reading the papers' evaluation sections; "quote" holds the text
# the judgement rests on, or a note when the field is simply absent.
# ---------------------------------------------------------------------------
AUDIT: list[dict] = []


def load(path: str = "results/literature_audit/audit.json") -> list[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return AUDIT


def summarise(rows: list[dict]) -> dict:
    out = {"n_papers": len(rows)}
    for f in FIELDS:
        c = Counter(r[f] for r in rows)
        denom = sum(v for k, v in c.items() if k != "n/a")
        out[f] = {
            "counts": dict(c),
            "yes_rate": (c.get("yes", 0) / denom) if denom else None,
            "denominator": denom,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", default="results/literature_audit/audit.json")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    rows = load(args.audit)
    if not rows:
        raise SystemExit(f"no audit records at {args.audit}")

    s = summarise(rows)
    print(f"[reference_protocol_audit] {s['n_papers']} papers\n")
    for f in FIELDS:
        e = s[f]
        rate = "n/a" if e["yes_rate"] is None else f"{100 * e['yes_rate']:.0f}%"
        print(f"  {f:<22s} yes={rate:>5s}  of {e['denominator']:>2d}   {e['counts']}")

    if args.latex:
        print("\n% ---- LaTeX ----")
        print("\\begin{tabular}{lccccc}")
        print("\\toprule")
        print("Paper & $N$ imgs & $N$ subj & Selection & Subj.\\ split & Ref.\\ from train \\\\")
        print("\\midrule")
        mark = {"yes": "\\cmark", "no": "\\xmark", "partial": "$\\sim$", "n/a": "---"}
        for r in rows:
            cells = " & ".join(mark.get(r[f], "?") for f in FIELDS)
            print(f"{r['short']} & {cells} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
