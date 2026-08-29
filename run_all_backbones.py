"""
run_all_backbones.py
====================
Runs the full experiment suite (run_all_experiments.py) sequentially for all
six backbone models.  Results land in per-backbone subdirectories under
results/ so comparisons can be made across backbones.

Usage:
    python run_all_backbones.py
    python run_all_backbones.py --device cuda:0 --seed 42
    python run_all_backbones.py --backbones rad-dino dinov2   # subset
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

_ALL_BACKBONES: dict[str, str] = {
    "rad-dino":    "microsoft/rad-dino",
    "dinov2":      "facebook/dinov2-base",
    "radiodino-s16": "Snarcy/RadioDino-s16",
    "pubmedclip":  "flaviagiammarino/pubmed-clip-vit-base-patch32",
    "biomedclip":  "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "resnet50":    "resnet50",   # loaded via timm with pretrained=True
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run all experiments for all backbones")
parser.add_argument("--device",   type=str, default=None,
                    help="Torch device (default: auto-detect cuda/cpu)")
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--n_subset", type=int, default=500)
parser.add_argument("--run_id",   type=int, default=None,
                    help="Numeric run ID appended to each output directory")
parser.add_argument("--backbones", nargs="+",
                    choices=list(_ALL_BACKBONES.keys()),
                    default=list(_ALL_BACKBONES.keys()),
                    help="Which backbones to run (default: all)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT     = os.path.dirname(os.path.abspath(__file__))
RUNNER   = os.path.join(ROOT, "run_all_experiments.py")
PYTHON   = sys.executable

def _hms(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

results: dict[str, str] = {}  # backbone_tag → "OK" | "FAILED"
total_start = time.time()

selected = {k: v for k, v in _ALL_BACKBONES.items() if k in args.backbones}

print(f"\n{'='*64}")
print(f"  run_all_backbones.py — {len(selected)} backbone(s) queued")
print(f"{'='*64}")
for tag, bid in selected.items():
    print(f"  - {tag:20s} -> {bid}")
print()

for tag, backbone_id in selected.items():
    print(f"\n{'='*64}")
    print(f"  BACKBONE: {tag}  ({backbone_id})")
    print(f"{'='*64}\n")

    cmd = [PYTHON, RUNNER,
           "--backbone", backbone_id,
           "--seed",     str(args.seed),
           "--n_subset", str(args.n_subset)]

    if args.device:
        cmd += ["--device", args.device]
    if args.run_id is not None:
        cmd += ["--run_id", str(args.run_id)]

    t0 = time.time()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"   # use only cached models

    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    results[tag] = status
    print(f"\n  [{tag}] finished in {_hms(elapsed)} — {status}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total_elapsed = time.time() - total_start
print(f"\n{'='*64}")
print(f"  ALL BACKBONES COMPLETE — total time {_hms(total_elapsed)}")
print(f"{'='*64}")
for tag, status in results.items():
    icon = "OK" if status == "OK" else "FAIL"
    print(f"  [{icon}] {tag:20s} {status}")
print()
