"""
notebooks/nb_config.py
======================
Shared configuration and helper functions for the MRI-Diffuser notebooks.

Import this at the top of every notebook:

    import nb_config as C
    C.setup()           # adds project root to sys.path, prints a summary

Everything is designed to work whether the kernel is launched from the
project root or from inside the ``notebooks/`` directory.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_project_root() -> str:
    """Walk upward until we find the repo marker (CLAUDE.md / train.py)."""
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(6):
        if os.path.exists(os.path.join(cur, "train.py")) and \
           os.path.exists(os.path.join(cur, "evaluation")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback: parent of notebooks/
    return os.path.dirname(here)


PROJECT_ROOT = _find_project_root()

# ---------------------------------------------------------------------------
# Default data / output locations  (edit these to point at your own dirs)
# ---------------------------------------------------------------------------

REAL_DIR    = os.path.join(PROJECT_ROOT, "data_mri", "brats_axial_multislice")
GEN_DIR     = os.path.join(PROJECT_ROOT, "output", "generated_500_best")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
EXP_DIR     = os.path.join(PROJECT_ROOT, "experiments")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup(verbose: bool = True) -> str:
    """Put the project root on sys.path and (optionally) print a summary."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if verbose:
        print(f"Project root : {PROJECT_ROOT}")
        print(f"Real dir     : {REAL_DIR}   ({count_images(REAL_DIR)} imgs)")
        print(f"Gen dir      : {GEN_DIR}   ({count_images(GEN_DIR)} imgs)")
        print(f"Results dir  : {RESULTS_DIR}")
        print(f"Device       : {detect_device()}")
    return PROJECT_ROOT


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Image discovery / loading
# ---------------------------------------------------------------------------

def list_images(directory: str, n: Optional[int] = None, recursive: bool = True) -> List[str]:
    """Return sorted image paths under *directory*."""
    paths: List[str] = []
    if recursive:
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(directory, "**", "*" + ext), recursive=True))
    else:
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(directory, "*" + ext)))
    paths = sorted(paths)
    return paths[:n] if n else paths


def count_images(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return len(list_images(directory))


def load_pil_batch(directory: str, n: int = 16, recursive: bool = True):
    """Load up to *n* PIL RGB images from *directory*."""
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in list_images(directory, n, recursive)]


def show_collage(directory: str, n: int = 16, ncols: int = 4, title: str = "", cmap: str = "gray"):
    """Display a grid of the first *n* images from *directory*."""
    import math
    import matplotlib.pyplot as plt
    from PIL import Image

    paths = list_images(directory, n)
    if not paths:
        print(f"[!] No images found in {directory}")
        return
    nrows = math.ceil(len(paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        ax.imshow(Image.open(p).convert("L"), cmap=cmap)
        ax.set_title(os.path.basename(p)[:14], fontsize=7)
    if title:
        fig.suptitle(f"{title}  ({len(paths)} of {count_images(directory)})", fontsize=12)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Results / report discovery
# ---------------------------------------------------------------------------

def find_reports(root: Optional[str] = None) -> List[str]:
    """Find every *_report.json / results.json under the results tree."""
    root = root or RESULTS_DIR
    out: List[str] = []
    for pattern in ("**/*_report.json", "**/results.json", "**/*results*.json"):
        out.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    return sorted(set(out))


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_pngs(subdir: str, root: Optional[str] = None) -> List[str]:
    """Return PNGs inside a given results subdirectory."""
    root = root or RESULTS_DIR
    return sorted(glob.glob(os.path.join(root, subdir, "**", "*.png"), recursive=True))


def show_pngs(paths: List[str], ncols: int = 2, max_imgs: int = 8):
    """Render a list of PNG paths inline."""
    import math
    import matplotlib.pyplot as plt
    from PIL import Image

    paths = paths[:max_imgs]
    if not paths:
        print("[!] No PNGs to show.")
        return
    nrows = math.ceil(len(paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        ax.imshow(Image.open(p))
        ax.set_title(os.path.relpath(p, RESULTS_DIR), fontsize=8)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Running experiment scripts as subprocesses
# ---------------------------------------------------------------------------

def run_script(rel_path: str, args: Optional[List[str]] = None, live: bool = True) -> int:
    """
    Run a project python script (e.g. 'experiments/noise_robustness.py') with the
    given CLI args, streaming output into the notebook. Returns the exit code.
    """
    args = args or []
    script = os.path.join(PROJECT_ROOT, rel_path)
    cmd = [sys.executable, script, *map(str, args)]
    print("$ " + " ".join(os.path.basename(c) if i == 1 else c for i, c in enumerate(cmd)))
    print("-" * 70)
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    if live:
        for line in proc.stdout:           # type: ignore[union-attr]
            print(line, end="")
    proc.wait()
    print("-" * 70)
    print(f"[exit code {proc.returncode}]")
    return proc.returncode
