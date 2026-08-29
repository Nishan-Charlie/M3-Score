"""
Progressive Improvement Experiment -- Section 3.x
===================================================
Tests whether M3-Score correctly tracks progressive quality improvement
as DDIM inference steps increase, and whether it detects this improvement
in cases where FID fails to be monotonic.

Protocol:
  1. Load a trained DDPM checkpoint and switch its scheduler to DDIM.
  2. For each step count in steps_to_test, generate num_gen images.
  3. Compute M3-Score, FRD, FID, KID, SSIM, PSNR, MS-SSIM, and LPIPS
     between real and generated sets at each step count.
  4. Report whether each metric is strictly monotonically improving.
  5. Plot all metrics vs step count.

The key claim: if M3-Score is monotonically decreasing while FID is not,
M3 is capturing progressive improvement that FID misses.
All other metrics are tested on the same condition for full comparability.

Seed note:
  A distinct seed is used for each step count (base_seed + step_index) so
  that different step counts do not all generate from identical noise.
  Using the same seed for all step counts would mean every condition
  starts from the same latent, making the comparison trivially about
  denoising from a fixed starting point rather than distribution quality.

Usage:
    python progressive_improvement.py \\
        --real_dir <path> \\
        --checkpoint_dir <path> \\
        --output_dir <path> \\
        --num_gen 50 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import glob
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.m3_score_v2 import M3V2Metric

# Reuse shared metric helpers from noise_robustness to avoid duplication
from experiments.noise_robustness import (
    _compute_ssim_psnr,
    _compute_msssim_lpips,
    _compute_frd,
)


# ---------------------------------------------------------------------------
# Image loading helpers (inline, no _shared_utils dependency)
# ---------------------------------------------------------------------------

def _load_real_paths(directory: str, n: int) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))[:n]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}.")
    return paths


def _load_real_images(
    paths: list[str],
    m3_size: int = 224,
    fid_size: int = 299,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load real images in two formats needed by the experiment:
      - m3_batch:  (N, 3, m3_size, m3_size) float32 in [0, 1] for M3V2Metric.
      - fid_batch: (N, 3, fid_size, fid_size) uint8 for FrechetInceptionDistance.
    """
    m3_transform = transforms.Compose([
        transforms.Resize((m3_size, m3_size)),
        transforms.ToTensor(),
    ])
    fid_transform = transforms.Compose([
        transforms.Resize((fid_size, fid_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ])
    m3_imgs:  list[torch.Tensor] = []
    fid_imgs: list[torch.Tensor] = []
    for p in tqdm(paths, desc="Loading real images", leave=False):
        img = Image.open(p).convert("RGB")
        m3_imgs.append(m3_transform(img))
        fid_imgs.append(fid_transform(img))
    return torch.stack(m3_imgs), torch.stack(fid_imgs)


def _load_real_float(paths: list[str], size: int = 256) -> torch.Tensor:
    """
    Load real images as (N, 1, H, W) float32 tensors in [0, 1] for
    SSIM, PSNR, MS-SSIM, and LPIPS computation.
    """
    imgs = []
    for p in paths:
        arr = np.array(
            Image.open(p).convert("L").resize((size, size)), dtype=np.float32
        ) / 255.0
        imgs.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.stack(imgs)


def _pil_to_fid_tensor(pil_img: Image.Image, size: int = 299) -> torch.Tensor:
    """Convert a PIL image to a (3, H, W) uint8 tensor for FID update."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ])
    return transform(pil_img.convert("RGB"))


# ---------------------------------------------------------------------------
# FID helper
# ---------------------------------------------------------------------------

def _build_fid_with_real(
    real_fid_batch: torch.Tensor,
    device:         str,
) -> object:
    """
    Instantiate FrechetInceptionDistance and pre-load real features.
    Real features are extracted once and cached by torchmetrics internally.
    Returns the metric object ready for fake updates.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(device)
    batch_size = 32
    for i in range(0, len(real_fid_batch), batch_size):
        fid.update(real_fid_batch[i:i + batch_size].to(device), real=True)
    return fid


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def _load_pipeline(checkpoint_dir: str, device: str):
    """
    Load a DDPM pipeline from a checkpoint directory and switch its
    scheduler to DDIM to allow variable num_inference_steps.

    Handles the common 1-channel grayscale UNet mismatch gracefully.
    """
    from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel

    print(f"Loading pipeline from {checkpoint_dir} ...")
    try:
        pipeline = DDPMPipeline.from_pretrained(checkpoint_dir)
    except ValueError as exc:
        if "conv_in.weight" in str(exc) or "in_channels" in str(exc):
            print("  Detected channel mismatch. Loading UNet with 1-channel override ...")
            unet = UNet2DModel.from_pretrained(
                checkpoint_dir,
                subfolder="unet",
                in_channels=1,
                out_channels=1,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
            pipeline = DDPMPipeline.from_pretrained(checkpoint_dir, unet=unet)
        else:
            raise

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    return pipeline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_progressive_improvement(
    real_dir:       str,
    checkpoint_dir: str,
    output_dir:     str           = "./progressive_improvement_output",
    steps_to_test:  Optional[list] = None,
    num_gen:        int           = 50,
    batch_size:     int           = 10,
    num_real:       int           = 100,
    device:         Optional[str]  = None,
    seed:           int           = 42,
) -> dict:
    """
    Run the progressive improvement experiment.

    Args:
        real_dir:       Directory of real images.
        checkpoint_dir: Path to a trained DDPM/DDIM checkpoint directory.
        output_dir:     Destination for plots, CSV, and JSON report.
        steps_to_test:  DDIM inference step counts to evaluate.
        num_gen:        Number of images to generate per step count.
        batch_size:     Generation batch size.
        num_real:       Number of real images to use as reference.
        device:         Torch device string (None = auto-detect).
        seed:           Base random seed. Each step count uses seed + index
                        to ensure independent noise across conditions.

    Returns:
        dict with per-step results, monotonicity flags, and plot paths,
        compatible with master_report in run_experiments.py.
    """
    if steps_to_test is None:
        steps_to_test = [5, 10, 25, 50, 100, 250]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load real images (once) ───────────────────────────────────────────────
    real_paths = _load_real_paths(real_dir, num_real)
    real_m3_batch, real_fid_batch = _load_real_images(real_paths)
    print(f"Loaded {len(real_paths)} real images.")

    # ── Initialise M3-Score (once) ───────────────────────────────────────────
    print("Initialising M3-Score and running CKA layer selection ...")
    m3_metric = M3V2Metric(device=device)
    m3_metric.prune_layers_via_cka(real_m3_batch[:20])
    print(f"  Active layers: {m3_metric.active_layers}")

    # ── Pre-load real FID features (once, reused per step count) ─────────────
    # A fresh FID metric is built once with real features cached. For each step
    # count the fake side is reset by re-instantiating from the cached real state.
    # torchmetrics does not expose a "reset fake only" API, so we rebuild the
    # metric per step and re-insert real features. This is still faster than
    # re-running InceptionV3 on the real set because torchmetrics caches nothing
    # internally between instances — we must re-run, but we do so efficiently
    # in a single batched pass rather than per-image.
    print("Pre-extracting real InceptionV3 features for FID ...")

    # ── Load DDPM pipeline ────────────────────────────────────────────────────
    pipeline = _load_pipeline(checkpoint_dir, device)

    # ── M3 transform for generated PIL images ────────────────────────────────
    m3_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Real float batch for SSIM/PSNR/MS-SSIM/LPIPS (loaded once)
    real_float_batch = _load_real_float(real_paths)

    # LPIPS (instantiated once)
    lpips_fn = None
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
    except ImportError:
        print("  [WARN] lpips not installed; LPIPS skipped.")

    # ── Step sweep ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    for step_idx, steps in enumerate(steps_to_test):
        print(f"\n[Progressive] inference steps = {steps}")

        # Use a distinct seed per step count so conditions are independent
        step_seed = seed + step_idx
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        np.random.seed(step_seed)

        # Build FID and KID metrics with real features for this step count
        fid_metric = _build_fid_with_real(real_fid_batch, device)
        kid_metric = None
        try:
            from torchmetrics.image.kid import KernelInceptionDistance
            kid_metric = KernelInceptionDistance(
                subset_size=min(50, num_gen)
            ).to(device)
            for i in range(0, len(real_fid_batch), 32):
                kid_metric.update(
                    real_fid_batch[i:i + 32].to(device), real=True
                )
        except ImportError:
            pass

        gen_m3_tensors:    list[torch.Tensor] = []
        gen_float_tensors: list[torch.Tensor] = []
        gen_disk_dir = os.path.join(output_dir, f"gen_steps_{steps}")
        os.makedirs(gen_disk_dir, exist_ok=True)
        gen_img_count = 0
        n_batches = num_gen // batch_size

        for b in tqdm(range(n_batches), desc=f"Generating (steps={steps})", leave=False):
            outputs = pipeline(
                batch_size=batch_size,
                num_inference_steps=steps,
            )
            for img in outputs.images:
                img_rgb = img.convert("RGB")
                gen_m3_tensors.append(m3_transform(img_rgb))
                # Float grayscale for SSIM/PSNR/MS-SSIM/LPIPS
                arr_gray = np.array(
                    img.convert("L").resize((256, 256)), dtype=np.float32
                ) / 255.0
                gen_float_tensors.append(
                    torch.from_numpy(arr_gray).unsqueeze(0)
                )
                # Save to disk for FRD
                img_gray = img.convert("L")
                img_gray.save(
                    os.path.join(gen_disk_dir, f"gen_{gen_img_count:04d}.png")
                )
                gen_img_count += 1
                # FID / KID
                fid_tensor = (
                    _pil_to_fid_tensor(img_rgb, size=299)
                    .unsqueeze(0).to(device)
                )
                fid_metric.update(fid_tensor, real=False)
                if kid_metric is not None:
                    kid_metric.update(fid_tensor, real=False)

        # Save one sample per step count for visual inspection
        if outputs.images:
            outputs.images[0].save(
                os.path.join(sample_dir, f"sample_steps_{steps}.png")
            )

        gen_batch_m3 = torch.stack(gen_m3_tensors)
        with torch.no_grad():
            m3_result = m3_metric(real_m3_batch, gen_batch_m3)

        fid_val = float(fid_metric.compute().item())
        m3_val  = float(m3_result["m3_v2_final_score"])

        # KID
        kid_val = float("nan")
        if kid_metric is not None:
            try:
                kid_mean, _ = kid_metric.compute()
                kid_val = float(kid_mean.item())
            except Exception:
                pass

        # FRD
        frd_val = float("nan")
        try:
            frd_val = _compute_frd(real_dir, gen_disk_dir)
        except Exception as e:
            print(f"  FRD failed: {e}")

        # SSIM / PSNR
        ssim_val = psnr_val = float("nan")
        gen_float = torch.stack(gen_float_tensors)
        try:
            ssim_val, psnr_val = _compute_ssim_psnr(real_float_batch, gen_float)
        except Exception as e:
            print(f"  SSIM/PSNR failed: {e}")

        # MS-SSIM / LPIPS
        ms_ssim_val = lpips_val = float("nan")
        try:
            ext = _compute_msssim_lpips(
                real_float_batch, gen_float, device, lpips_fn
            )
            ms_ssim_val = ext.get("ms_ssim", float("nan"))
            lpips_val   = ext.get("lpips",   float("nan"))
        except Exception as e:
            print(f"  MS-SSIM/LPIPS failed: {e}")

        # Clean up generated images from disk
        import shutil
        shutil.rmtree(gen_disk_dir, ignore_errors=True)

        print(
            f"  steps={steps:4d}  M3={m3_val:.4f}  FRD={frd_val:.4f}  "
            f"FID={fid_val:.4f}  KID={kid_val:.4f}  "
            f"SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}"
        )
        rows.append({
            "steps":    steps,
            "m3_score": round(m3_val,    6),
            "frd":      round(frd_val,   4) if not (isinstance(frd_val, float) and __import__('math').isnan(frd_val)) else frd_val,
            "fid":      round(fid_val,   4),
            "kid":      round(kid_val,   6) if not (isinstance(kid_val, float) and __import__('math').isnan(kid_val)) else kid_val,
            "ssim":     round(ssim_val,  6) if not (isinstance(ssim_val, float) and __import__('math').isnan(ssim_val)) else ssim_val,
            "psnr":     round(psnr_val,  4) if not (isinstance(psnr_val, float) and __import__('math').isnan(psnr_val)) else psnr_val,
            "ms_ssim":  round(ms_ssim_val, 6) if not (isinstance(ms_ssim_val, float) and __import__('math').isnan(ms_ssim_val)) else ms_ssim_val,
            "lpips":    round(lpips_val,   6) if not (isinstance(lpips_val, float) and __import__('math').isnan(lpips_val)) else lpips_val,
        })

    # ── Monotonicity analysis ─────────────────────────────────────────────────
    # Strict monotonicity: each value must be strictly less than the previous
    # (FID and M3 should decrease as inference quality improves).
    import math
    metric_keys = ["m3_score", "frd", "fid", "kid", "ssim", "psnr", "ms_ssim", "lpips"]
    # Metrics that should DECREASE as quality improves
    decreasing = {"m3_score", "frd", "fid", "kid", "lpips"}
    # Metrics that should INCREASE as quality improves
    increasing = {"ssim", "psnr", "ms_ssim"}

    monotonicity: dict = {}
    for key in metric_keys:
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        if len(vals) < 2:
            monotonicity[key] = None
            continue
        if key in decreasing:
            monotonicity[key] = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        else:
            monotonicity[key] = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    print("\n  Strict monotonicity by metric:")
    for key, mono in monotonicity.items():
        print(f"    {key:12s}: {mono}")

    fid_monotonic = monotonicity.get("fid", False)
    m3_monotonic  = monotonicity.get("m3_score", False)
    if not fid_monotonic and m3_monotonic:
        print("  M3-Score captured progressive improvement where FID did not.")

    # ── Multi-panel plot ──────────────────────────────────────────────────────
    step_vals = [r["steps"] for r in rows]
    colors = {
        "m3_score": "#ef5350", "frd": "#ff9800",
        "fid":      "#4fc3f7", "kid": "#66bb6a",
        "ssim":     "#ffe082", "psnr": "#ce93d8",
        "ms_ssim":  "#ff8a65", "lpips": "#80cbc4",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)

    def _ax_plot(ax, keys, title):
        for key in keys:
            vals = [r.get(key, float("nan")) for r in rows]
            ax.plot(step_vals, vals, "o-", color=colors[key],
                    label=key.upper(), lw=2, markersize=6)
        ax.set_xlabel("DDIM inference steps")
        ax.set_ylabel("Metric value")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    _ax_plot(axes[0, 0], ["m3_score", "frd"],       "Domain-specific metrics vs steps")
    _ax_plot(axes[0, 1], ["fid", "kid"],             "Inception-based metrics vs steps")
    _ax_plot(axes[1, 0], ["ssim", "psnr", "ms_ssim"], "Structural metrics vs steps")
    _ax_plot(axes[1, 1], ["lpips"],                  "Perceptual (LPIPS) vs steps")

    mono_str = "  ".join(
        f"{k}={'Y' if v else 'N'}"
        for k, v in monotonicity.items() if v is not None
    )
    plt.suptitle(f"Progressive improvement: all metrics vs DDIM steps\n{mono_str}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "progressive_improvement.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # ── CSV output ────────────────────────────────────────────────────────────
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "progressive_improvement.csv"), index=False
        )
    except ImportError:
        pass  # pandas optional for CSV; JSON is the primary output

    # ── JSON report ───────────────────────────────────────────────────────────
    results = {
        "steps_to_test":    steps_to_test,
        "num_gen":          num_gen,
        "num_real":         num_real,
        "rows":             rows,
        "monotonicity":     monotonicity,
        "active_layers":    m3_metric.active_layers,
        "plot":             plot_path,
    }
    report_path = os.path.join(output_dir, "progressive_improvement_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\nReport saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Progressive improvement experiment: FID vs M3 vs DDIM steps"
    )
    parser.add_argument("--real_dir",       required=True,
                        help="Directory of real images")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to trained DDPM checkpoint directory")
    parser.add_argument("--output_dir",     default="./progressive_improvement_output")
    parser.add_argument("--steps",          nargs="+", type=int,
                        default=[5, 10, 25, 50, 100, 250],
                        help="DDIM inference step counts to evaluate")
    parser.add_argument("--num_gen",        type=int, default=50,
                        help="Images to generate per step count")
    parser.add_argument("--batch_size",     type=int, default=10,
                        help="Generation batch size")
    parser.add_argument("--num_real",       type=int, default=100,
                        help="Real images used as reference")
    parser.add_argument("--device",         default=None)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()
    run_progressive_improvement(
        real_dir       = args.real_dir,
        checkpoint_dir = args.checkpoint_dir,
        output_dir     = args.output_dir,
        steps_to_test  = args.steps,
        num_gen        = args.num_gen,
        batch_size     = args.batch_size,
        num_real       = args.num_real,
        device         = args.device,
        seed           = args.seed,
    )


if __name__ == "__main__":
    main()