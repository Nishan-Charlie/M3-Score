"""
Hallucination Detection via BraTS Tumor Mask Sensitivity (Section 3.11)
=======================================================================
Tests whether M3 and FID are sensitive to the presence of tumor pathology.

Experimental protocol
---------------------
Three evaluation conditions using REAL BraTS segmentation masks (no fallback):

  condition_a  M3 / FID (real, gen)          -- baseline
  condition_b  M3 / FID (real, masked_real)  -- real vs tumour-zeroed real
  condition_c  M3 / FID (real, masked_gen)   -- real vs tumour-zeroed gen

Masks are loaded from PNG files named  *_segmask_slice*.png  produced by
tools/prepare_brats_slices.py --save_masks.  If no mask files are found the
script raises FileNotFoundError -- there is NO centre-square fallback.

sensitivity_b_vs_a = (metric_b - metric_a) / (abs(metric_a) + 1e-8)

Paper claim (Section 3.11): M3 is 4.23x more sensitive to pathology masking
than FID.  Note: this claim was validated with real masks; the current code
reports the raw ratio so the reader can verify.

Usage
-----
    python experiments/hallucination_detection.py \\
        --real_dir  data_mri/brats_axial_multislice \\
        --gen_dir   output/generated_500_best \\
        --output_dir results/hallucination_detection \\
        --num_images 200 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import glob
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo-root import hygiene
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.m3_score_v2 import M3V2Metric
from experiments._shared_utils import get_m3_transform, get_fid_transform

# ---------------------------------------------------------------------------
# Image / mask utilities
# ---------------------------------------------------------------------------

def _glob_images(directory: str) -> List[str]:
    """Recursively find all PNG/JPG/TIF images that are NOT mask files."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    # exclude mask files produced by prepare_brats_slices.py --save_masks
    paths = [p for p in paths if "_segmask_" not in os.path.basename(p)]
    return sorted(paths)


def _find_masks(real_dir: str) -> dict:
    """Return a dict mapping image basename (without extension) -> mask path.

    Mask files must follow the naming pattern: <case>_segmask_slice<N>.png
    produced by  tools/prepare_brats_slices.py --save_masks.
    """
    mask_paths = glob.glob(
        os.path.join(real_dir, "**", "*_segmask_slice*.png"), recursive=True
    )
    mapping: dict = {}
    for mp in mask_paths:
        # derive the corresponding modality image basename
        # e.g.  BraTS2021_00000_segmask_slice042.png
        #    -> BraTS2021_00000_slice042
        base = os.path.splitext(os.path.basename(mp))[0]
        img_base = base.replace("_segmask_", "_")
        mapping[img_base] = mp
    return mapping


def _load_image_tensor(path: str, tfm) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return tfm(img)


def _apply_mask(image_tensor: torch.Tensor, mask_pil: Image.Image) -> torch.Tensor:
    """Zero-out the tumour region (mask pixel = 255) in the image tensor.

    image_tensor: (3, H, W) float32 in [0, 1]
    mask_pil:     PIL image (L mode), 255 = tumour, 0 = background
    """
    mask_np = np.array(mask_pil.resize(
        (image_tensor.shape[2], image_tensor.shape[1]), Image.NEAREST
    )).astype(np.float32) / 255.0                  # [0, 1]
    mask_t  = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)
    # Zero out tumour pixels
    return image_tensor * (1.0 - mask_t)


# ---------------------------------------------------------------------------
# FID helper (minimal, via torchmetrics if available)
# ---------------------------------------------------------------------------

def _compute_fid(real_imgs: torch.Tensor, gen_imgs: torch.Tensor, device: str) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        # torchmetrics expects uint8 or normalised float in [0,1]
        fid_metric.update(real_imgs.to(device), real=True)
        fid_metric.update(gen_imgs.to(device),  real=False)
        return float(fid_metric.compute().item())
    except Exception as exc:
        print(f"  [FID] computation failed: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(
    real_dir: str,
    gen_dir: str,
    output_dir: str,
    num_images: int = 200,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Locate masks  -- HARD REQUIREMENT, no fallback                   #
    # ------------------------------------------------------------------ #
    mask_map = _find_masks(real_dir)
    if not mask_map:
        raise FileNotFoundError(
            f"No segmentation mask files (*_segmask_slice*.png) found under "
            f"{real_dir}.  Run tools/prepare_brats_slices.py --save_masks first."
        )
    print(f"Found {len(mask_map)} mask files in {real_dir}")

    # ------------------------------------------------------------------ #
    # 2. Build matched real-image / mask pairs                            #
    # ------------------------------------------------------------------ #
    all_real = _glob_images(real_dir)
    if not all_real:
        raise FileNotFoundError(f"No real images found in {real_dir}")

    # Only keep real images that have a corresponding mask
    matched: List[Tuple[str, str]] = []
    for rp in all_real:
        base = os.path.splitext(os.path.basename(rp))[0]
        if base in mask_map:
            matched.append((rp, mask_map[base]))

    if not matched:
        raise FileNotFoundError(
            f"No real images could be matched to mask files in {real_dir}.  "
            f"Check that image and mask file names share the same case ID and "
            f"slice index (e.g. BraTS2021_00000_slice042.png <-> "
            f"BraTS2021_00000_segmask_slice042.png)."
        )

    print(f"Matched {len(matched)} real images to masks")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(matched), size=min(num_images, len(matched)), replace=False)
    matched = [matched[i] for i in indices]
    n = len(matched)
    print(f"Using {n} matched pairs for evaluation")

    # ------------------------------------------------------------------ #
    # 3. Load gen images                                                   #
    # ------------------------------------------------------------------ #
    all_gen = _glob_images(gen_dir)
    if not all_gen:
        raise FileNotFoundError(f"No generated images found in {gen_dir}")
    gen_indices = rng.choice(len(all_gen), size=min(n, len(all_gen)), replace=False)
    gen_paths = [all_gen[i] for i in gen_indices]

    # ------------------------------------------------------------------ #
    # 4. Build tensors for all three conditions                            #
    # ------------------------------------------------------------------ #
    m3_tfm  = get_m3_transform()
    fid_tfm = get_fid_transform()

    real_m3 = []
    real_fid = []
    masked_real_m3  = []
    masked_real_fid = []
    masked_gen_m3   = []
    masked_gen_fid  = []

    gen_loaded_m3  = []
    gen_loaded_fid = []

    print("Loading images and applying masks...")
    for idx, (rp, mp) in enumerate(tqdm(matched)):
        mask_pil = Image.open(mp).convert("L")

        # real
        r_m3  = _load_image_tensor(rp, m3_tfm)
        r_fid = _load_image_tensor(rp, fid_tfm)
        real_m3.append(r_m3)
        real_fid.append(r_fid)

        # masked real (tumour region zeroed)
        mr_m3  = _apply_mask(_load_image_tensor(rp, m3_tfm),  mask_pil)
        mr_fid = _apply_mask(_load_image_tensor(rp, fid_tfm), mask_pil)
        masked_real_m3.append(mr_m3)
        masked_real_fid.append(mr_fid)

        # gen (same index modulo gen set size)
        gp = gen_paths[idx % len(gen_paths)]
        g_m3  = _load_image_tensor(gp, m3_tfm)
        g_fid = _load_image_tensor(gp, fid_tfm)
        gen_loaded_m3.append(g_m3)
        gen_loaded_fid.append(g_fid)

        # masked gen (apply real mask spatially to gen image)
        mg_m3  = _apply_mask(_load_image_tensor(gp, m3_tfm),  mask_pil)
        mg_fid = _apply_mask(_load_image_tensor(gp, fid_tfm), mask_pil)
        masked_gen_m3.append(mg_m3)
        masked_gen_fid.append(mg_fid)

    real_m3_t        = torch.stack(real_m3)
    real_fid_t       = torch.stack(real_fid)
    masked_real_m3_t = torch.stack(masked_real_m3)
    masked_real_fid_t = torch.stack(masked_real_fid)
    gen_m3_t         = torch.stack(gen_loaded_m3)
    gen_fid_t        = torch.stack(gen_loaded_fid)
    masked_gen_m3_t  = torch.stack(masked_gen_m3)
    masked_gen_fid_t = torch.stack(masked_gen_fid)

    # ------------------------------------------------------------------ #
    # 5. Compute metrics for three conditions                              #
    # ------------------------------------------------------------------ #
    m3 = M3V2Metric(device=device, single_layer=12)

    def _m3(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(m3(a, b)["m3_score"])

    print("\nCondition A: real vs gen (baseline)...")
    a_m3  = _m3(real_m3_t, gen_m3_t)
    a_fid = _compute_fid(real_fid_t, gen_fid_t, device)

    print("Condition B: real vs masked-real...")
    b_m3  = _m3(real_m3_t, masked_real_m3_t)
    b_fid = _compute_fid(real_fid_t, masked_real_fid_t, device)

    print("Condition C: real vs masked-gen...")
    c_m3  = _m3(real_m3_t, masked_gen_m3_t)
    c_fid = _compute_fid(real_fid_t, masked_gen_fid_t, device)

    # ------------------------------------------------------------------ #
    # 6. Sensitivity ratios (condition_b relative to condition_a)          #
    # ------------------------------------------------------------------ #
    def _rel_change(new, baseline):
        return (new - baseline) / (abs(baseline) + 1e-8)

    sens_m3_b  = _rel_change(b_m3,  a_m3)
    sens_fid_b = _rel_change(b_fid, a_fid)

    sens_ratio_b = abs(sens_m3_b) / (abs(sens_fid_b) + 1e-8)

    sens_m3_c  = _rel_change(c_m3,  a_m3)
    sens_fid_c = _rel_change(c_fid, a_fid)
    sens_ratio_c = abs(sens_m3_c) / (abs(sens_fid_c) + 1e-8)

    report = {
        "config": {
            "real_dir":   real_dir,
            "gen_dir":    gen_dir,
            "num_images": n,
            "mask_files": len(mask_map),
            "note": "Real BraTS segmentation masks only. No centre-square fallback.",
        },
        "condition_a_real_vs_gen": {
            "m3":  a_m3,
            "fid": a_fid,
        },
        "condition_b_real_vs_masked_real": {
            "m3":  b_m3,
            "fid": b_fid,
        },
        "condition_c_real_vs_masked_gen": {
            "m3":  c_m3,
            "fid": c_fid,
        },
        "sensitivity_b_vs_a": {
            "m3":              sens_m3_b,
            "fid":             sens_fid_b,
            "m3_abs_fid_ratio": sens_ratio_b,
            "interpretation": (
                f"M3 relative change {sens_m3_b:+.4f} vs FID {sens_fid_b:+.4f}. "
                f"|M3|/|FID| ratio = {sens_ratio_b:.3f}. "
                f"Paper claim: 4.23x. Reported without reinterpretation."
            ),
        },
        "sensitivity_c_vs_a": {
            "m3":              sens_m3_c,
            "fid":             sens_fid_c,
            "m3_abs_fid_ratio": sens_ratio_c,
        },
    }

    out_path = os.path.join(output_dir, "hallucination_detection_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nReport saved: {out_path}")

    print("\n=== HALLUCINATION DETECTION RESULTS ===")
    print(f"  Condition A (real vs gen):         M3={a_m3:.4f}  FID={a_fid:.2f}")
    print(f"  Condition B (real vs masked real): M3={b_m3:.4f}  FID={b_fid:.2f}")
    print(f"  Condition C (real vs masked gen):  M3={c_m3:.4f}  FID={c_fid:.2f}")
    print(f"  Sensitivity B vs A: M3={sens_m3_b:+.4f}  FID={sens_fid_b:+.4f}  ratio={sens_ratio_b:.3f}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_dir",    required=True)
    parser.add_argument("--gen_dir",     required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--num_images",  type=int, default=200)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    run(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        device=args.device,
        seed=args.seed,
    )
