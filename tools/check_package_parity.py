"""Compare the m3score package with the research pipeline, end to end, on two image folders.

The research pipeline resizes each 8-bit image with Pillow, which rounds the resized
image back to whole grey levels. The package resizes in floating point and keeps the
fractions; its resize is otherwise identical to Pillow's bilinear filter. This script
measures what that rounding does to the three axes and exits non-zero when the gap
exceeds the tolerances in the design spec: fidelity within 2% relative, memorization
and coverage within 0.02 absolute.

Measured on 2026-10-12 with the local generated sets: fidelity -0.06% (DDPM against
retinal, 500/500) and -1.19% (DDPM against DDPM-train70, 500/300); memorization and
coverage identical in both.

    python tools/check_package_parity.py REAL_DIR GEN_DIR [--device cuda]
"""

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import m3score  # noqa: E402
from evaluation.m3_score_v2 import COVERAGE_LAYER, M3EntropyMetric  # noqa: E402

FIDELITY_RELATIVE = 0.02
RATE_ABSOLUTE = 0.02


def research_load(directory: str) -> torch.Tensor:
    """Every PNG under ``directory``, as the research evaluation pipeline loads them."""
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.png"), recursive=True))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("real")
    parser.add_argument("gen")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    real, gen = research_load(args.real), research_load(args.gen)
    research = M3EntropyMetric(device=args.device)
    axes = research.compute_axes(real, gen, n_permutations=1, n_bootstrap=1)
    real_l4 = research._extract_cls_features(real, {COVERAGE_LAYER})[COVERAGE_LAYER - 1]
    gen_l4 = research._extract_cls_features(gen, {COVERAGE_LAYER})[COVERAGE_LAYER - 1]
    research_coverage = research._compute_coverage(real_l4, gen_l4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", m3score.M3ScoreWarning)
        package = m3score.score(args.real, args.gen, device=args.device, stats=False, progress=False)

    rows = [
        ("fidelity", axes["fidelity/mmd2"], package.fidelity),
        ("memorization", axes["memorization/rate"], package.memorization),
        ("coverage", research_coverage, package.coverage),
    ]
    print(f"\n{len(real)} real, {len(gen)} generated images; package scored "
          f"{package.n_real} and {package.n_gen}")
    print(f"{'axis':<14}{'research':>12}{'package':>12}{'difference':>14}")
    failed = False
    for name, expected, actual in rows:
        difference = actual - expected
        if name == "fidelity":
            within = abs(difference) <= FIDELITY_RELATIVE * max(abs(expected), 1e-12)
        else:
            within = abs(difference) <= RATE_ABSOLUTE
        failed = failed or not within
        print(f"{name:<14}{expected:>12.6f}{actual:>12.6f}{difference:>+14.6f}"
              f"{'' if within else '   OUTSIDE TOLERANCE'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
