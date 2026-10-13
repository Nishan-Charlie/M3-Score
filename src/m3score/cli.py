"""Command line: ``m3score REAL GEN``."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence

from ._errors import M3ScoreError, M3ScoreWarning
from ._version import __version__
from .metric import M3Score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m3score",
        description=(
            "Score generated radiology images against real ones on the three M3 axes: "
            "fidelity, memorization and coverage."
        ),
    )
    parser.add_argument(
        "real", metavar="REAL", help="folder or file of real images: PNG/JPG/TIFF, NIfTI or DICOM"
    )
    parser.add_argument("gen", metavar="GEN", help="folder or file of generated images")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="images to score per set, drawn at random (default: all, up to 5000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling and statistics")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32, metavar="B")
    parser.add_argument(
        "--include",
        metavar="GLOB",
        help=(
            'only read files whose path inside REAL matches, e.g. "*_flair.nii.gz"; '
            "applied to GEN too when it matches files there"
        ),
    )
    parser.add_argument(
        "--group-pattern",
        metavar="REGEX",
        help=(
            "regular expression whose first capture group names the subject of each real "
            'file, e.g. "(BraTS2021_\\d+)"; enables subject-balanced sampling'
        ),
    )
    parser.add_argument(
        "--slice-band",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="fraction of each volume's depth to keep (default: 0.355 0.555)",
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help=(
            "intensity window for volumes, e.g. -1000 400 for CT "
            "(default: the stored window, else the 0.5-99.5 percentile range)"
        ),
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="skip the confidence interval, permutation test and real-vs-real floor",
    )
    parser.add_argument("--json", metavar="FILE", help="also write the result as JSON")
    parser.add_argument("--quiet", action="store_true", help="no progress bars")
    parser.add_argument("--version", action="version", version=f"m3-score {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metric = M3Score(
            None if args.device == "auto" else args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            progress=not args.quiet,
        )
        with warnings.catch_warnings():
            # warnings are listed under the result instead of interleaving with it
            warnings.simplefilter("ignore", M3ScoreWarning)
            result = metric.score(
                args.real,
                args.gen,
                n=args.n,
                stats=not args.no_stats,
                include=args.include,
                group_pattern=args.group_pattern,
                slice_band=tuple(args.slice_band) if args.slice_band else None,
                window=tuple(args.window) if args.window else None,
            )
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"m3score: error: {exc}", file=sys.stderr)
        return 2
    except M3ScoreError as exc:
        print(f"m3score: error: {exc}", file=sys.stderr)
        return 1

    print(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"m3score_version": __version__, **result.to_dict()}, handle, indent=2)
            handle.write("\n")
    return 0
