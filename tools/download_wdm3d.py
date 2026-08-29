"""
Download WDM-3D pretrained weights and source code
====================================================
Wavelet Diffusion Model (WDM) — pfriedri/wdm-3d
  - brats_unet_128_1200k.pt   : BraTS 2023 brain MRI  (128³)
  - lidc-idri_unet_128_1200k.pt: LIDC-IDRI lung CT    (128³)

Downloads to:  <project_root>/pretrained/wdm3d/
Clones source: <project_root>/external/wdm-3d/   (needed for custom modules)

Usage
-----
    python tools/download_wdm3d.py
    python tools/download_wdm3d.py --model all       # both weights (default)
    python tools/download_wdm3d.py --model brats
    python tools/download_wdm3d.py --model lidc
    python tools/download_wdm3d.py --skip_clone      # skip git clone if already done
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO_ID       = "pfriedri/wdm-3d"
WEIGHTS_DIR   = os.path.join(_ROOT, "pretrained", "wdm3d")
EXTERNAL_DIR  = os.path.join(_ROOT, "external", "wdm-3d")
GIT_URL       = "https://github.com/pfriedri/wdm-3d.git"

WEIGHT_FILES = {
    "brats": "brats_unet_128_1200k.pt",
    "lidc":  "lidc-idri_unet_128_1200k.pt",
}


def _download_weights(model: str = "all") -> None:
    """Download model weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    targets = list(WEIGHT_FILES.items()) if model == "all" else [(model, WEIGHT_FILES[model])]

    for key, filename in targets:
        dest = os.path.join(WEIGHTS_DIR, filename)
        if os.path.isfile(dest):
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  [{key}] Already downloaded: {dest}  ({size_mb:.0f} MB)")
            continue

        print(f"  [{key}] Downloading {filename} from {REPO_ID} ...")
        try:
            path = hf_hub_download(
                repo_id  = REPO_ID,
                filename = filename,
                local_dir= WEIGHTS_DIR,
                local_dir_use_symlinks=False,
            )
            size_mb = os.path.getsize(path) / 1e6
            print(f"  [{key}] Saved: {path}  ({size_mb:.0f} MB)")
        except Exception as e:
            print(f"  [{key}] FAILED: {e}")


def _clone_source(skip_clone: bool = False) -> None:
    """Clone the WDM-3D source repo (needed for custom guided_diffusion module)."""
    if skip_clone:
        print("  Skipping git clone (--skip_clone set)")
        return

    if os.path.isdir(os.path.join(EXTERNAL_DIR, ".git")):
        print(f"  Source already cloned at {EXTERNAL_DIR}")
        # Pull latest just in case
        try:
            subprocess.run(["git", "-C", EXTERNAL_DIR, "pull", "--quiet"],
                           check=False, capture_output=True)
            print("  Pulled latest changes.")
        except Exception:
            pass
        return

    os.makedirs(os.path.dirname(EXTERNAL_DIR), exist_ok=True)
    print(f"  Cloning {GIT_URL} -> {EXTERNAL_DIR} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", GIT_URL, EXTERNAL_DIR],
                       check=True)
        print(f"  Cloned to {EXTERNAL_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"  git clone failed: {e}")
        print("  Install git or clone manually and pass --skip_clone")
        sys.exit(1)


def _write_path_file() -> None:
    """Write a .pth file so external/wdm-3d is always on sys.path."""
    try:
        import site
        site_dir = site.getusersitepackages()
        os.makedirs(site_dir, exist_ok=True)
        pth = os.path.join(site_dir, "wdm3d.pth")
        with open(pth, "w") as f:
            f.write(EXTERNAL_DIR + "\n")
        print(f"  Added to sys.path via: {pth}")
    except Exception as e:
        print(f"  Could not write .pth (add {EXTERNAL_DIR} to PYTHONPATH manually): {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=["all", "brats", "lidc"],
                   help="Which weights to download (default: all)")
    p.add_argument("--skip_clone", action="store_true",
                   help="Skip cloning the source repo (use if already cloned)")
    args = p.parse_args()

    print("\n=== WDM-3D Download ===")
    print(f"Weights -> {WEIGHTS_DIR}")
    print(f"Source  -> {EXTERNAL_DIR}\n")

    print("Step 1: Clone WDM-3D source (custom guided_diffusion module) ...")
    _clone_source(args.skip_clone)

    print("\nStep 2: Download pretrained weights ...")
    _download_weights(args.model)

    _write_path_file()

    print("\n=== Done ===")
    print(f"Weights:  {WEIGHTS_DIR}/")
    print(f"Source:   {EXTERNAL_DIR}/")
    print(f"\nNext step:")
    print(f"  python tools/generate_wdm3d.py --model brats --n_volumes 10 --output_dir output/generated_wdm3d")


if __name__ == "__main__":
    main()
