"""
Denoising Steps Quality Ladder (Section 3.x)
=============================================
Tests whether M3, FID, and CMMD correctly order generated images by the
number of DDPM denoising steps used during generation.

Known quality ordering (more steps = higher quality):
    T=10 < T=20 < T=50 < T=100 < T=1000

All images are generated from the same pre-trained checkpoint; only the
number of reverse-diffusion steps differs.  This gives a ground-truth quality
ordering we can use to verify metric monotonicity.

Protocol
--------
1. Load a pre-trained DDPM checkpoint from --checkpoint_dir.
2. Generate --num_images images at each T in [1000, 100, 50, 20, 10].
3. Compute M3 (RadioDINO-s16 L12), FID, and CMMD vs real images.
4. Report whether each metric recovers the known ordering.

Usage
-----
    python experiments/denoising_steps_ladder.py \\
        --real_dir      data_mri/brats_axial_multislice \\
        --checkpoint_dir output/output_unet/checkpoints/best \\
        --output_dir    results/denoising_steps_ladder \\
        --num_images    100 \\
        --device        cuda

Notes
-----
- Generation at T=1000 is slow (~minutes per batch on CPU). Use --device cuda.
- If no checkpoint is available, use --use_noise_proxy to substitute Gaussian
  noise at decreasing sigma levels (equivalent to checkpoint_ranking.py but
  with the known-ordering framing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from scipy.linalg import sqrtm
from tqdm import tqdm

from evaluation.m3_score_v2 import M3V2Metric

# ---------------------------------------------------------------------------
# Shared transform / loading
# ---------------------------------------------------------------------------

_TFM_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_TFM_299 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

_INC_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INC_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _glob_images(directory: str, n: Optional[int] = None) -> List[str]:
    import glob
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))
    return paths[:n] if n else paths


def _load_tensor(paths: List[str], tfm) -> torch.Tensor:
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])


# ---------------------------------------------------------------------------
# Inception / FID helpers
# ---------------------------------------------------------------------------

def _inception_model(device: str):
    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


def _inception_features(t: torch.Tensor, model, device: str, bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(t), bs):
        b = t[i:i + bs].to(device)
        b = F.interpolate(b, (299, 299), mode="bilinear", align_corners=False)
        b = (b - _INC_MEAN.to(device)) / _INC_STD.to(device)
        with torch.no_grad():
            out.append(model(b).cpu().numpy())
    return np.concatenate(out, 0)


def _fid(fr: np.ndarray, fg: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    sr = np.cov(fr, rowvar=False) + eps * np.eye(fr.shape[1])
    sg = np.cov(fg, rowvar=False) + eps * np.eye(fg.shape[1])
    cm = sqrtm(sr @ sg)
    if np.iscomplexobj(cm):
        cm = cm.real
    diff = mu_r - mu_g
    return float(np.dot(diff, diff) + np.trace(sr + sg - 2 * cm))


def _cmmd(fr: np.ndarray, fg: np.ndarray) -> float:
    try:
        from evaluation.cmmd_metric import CMMDMetric
        return float(CMMDMetric.gaussian_mmd2_unbiased(fr, fg))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# DDPM generation at T steps
# ---------------------------------------------------------------------------

def _generate_at_steps(
    checkpoint_dir: str,
    n_images: int,
    t_steps: int,
    device: str,
    seed: int,
    output_dir: str,
) -> torch.Tensor:
    """Generate n_images using a DDPM pipeline at t_steps denoising steps.

    Saves PNG files to output_dir/T{t_steps}/ and returns (N, 3, 224, 224)
    float32 tensor.
    """
    from diffusers import DDPMPipeline, DDPMScheduler

    step_dir = os.path.join(output_dir, f"T{t_steps:04d}")
    png_files = _glob_images(step_dir)
    if len(png_files) >= n_images:
        print(f"  T={t_steps}: loading {n_images} cached images from {step_dir}")
        return _load_tensor(png_files[:n_images], _TFM_224)

    os.makedirs(step_dir, exist_ok=True)
    print(f"  T={t_steps}: generating {n_images} images...")

    pipe = DDPMPipeline.from_pretrained(checkpoint_dir).to(device)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    generator = torch.Generator(device=device).manual_seed(seed)
    imgs = []
    batch = 8
    for start in range(0, n_images, batch):
        n_batch = min(batch, n_images - start)
        result = pipe(
            batch_size=n_batch,
            num_inference_steps=t_steps,
            generator=generator,
        )
        for i, img in enumerate(result.images):
            img_rgb = img.convert("RGB")
            img_rgb.save(os.path.join(step_dir, f"gen_{start + i:04d}.png"))
            imgs.append(_TFM_224(img_rgb))

    return torch.stack(imgs)


# ---------------------------------------------------------------------------
# Noise-proxy fallback
# ---------------------------------------------------------------------------

def _noise_proxy_at_steps(
    base_tensor: torch.Tensor,
    t_steps: int,
    t_max: int,
    seed: int,
) -> torch.Tensor:
    """Proxy for generation quality: add noise inversely proportional to T.

    sigma = (1 - t_steps / t_max) * 0.5  so T=t_max -> sigma=0 (clean),
    T=1 -> sigma~0.5 (noisy).
    """
    sigma = (1.0 - t_steps / t_max) * 0.5
    if sigma < 1e-6:
        return base_tensor.clone()
    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.randn(base_tensor.shape, generator=g)
    return (base_tensor + noise * sigma).clamp(0, 1)


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------

def _is_monotone_decreasing(vals: List[float]) -> bool:
    """Check if vals[0] >= vals[1] >= ... (metric decreases as quality improves)."""
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def _is_monotone_increasing(vals: List[float]) -> bool:
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    real_dir: str,
    output_dir: str,
    checkpoint_dir: Optional[str] = None,
    gen_dir: Optional[str] = None,
    num_images: int = 100,
    t_steps_list: Optional[List[int]] = None,
    device: str = "cpu",
    seed: int = 42,
    use_noise_proxy: bool = False,
) -> dict:
    if t_steps_list is None:
        t_steps_list = [10, 20, 50, 100, 1000]  # ascending quality order
    t_steps_list = sorted(t_steps_list)           # low T first

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    needs_checkpoint = (not use_noise_proxy) and checkpoint_dir
    needs_gen_dir    = use_noise_proxy or (not checkpoint_dir)

    # Real images
    real_paths = _glob_images(real_dir, num_images)
    if not real_paths:
        raise FileNotFoundError(f"No real images in {real_dir}")
    real_t = _load_tensor(real_paths[:num_images], _TFM_224)
    N = len(real_t)
    print(f"Real images loaded: {N}")

    # Metric objects
    m3_metric = M3V2Metric(device=device, single_layer=12, seed=seed)
    inc_model  = _inception_model(device)
    f_real     = _inception_features(real_t, inc_model, device)

    # Base gen tensor for noise proxy
    base_gen_t = None
    if use_noise_proxy or not checkpoint_dir:
        if not gen_dir:
            raise ValueError("--gen_dir required when --use_noise_proxy is set or --checkpoint_dir is absent")
        gen_paths = _glob_images(gen_dir, N)
        if not gen_paths:
            raise FileNotFoundError(f"No images in gen_dir={gen_dir}")
        base_gen_t = _load_tensor(gen_paths[:N], _TFM_224)
        print(f"Base gen images loaded: {len(base_gen_t)} (noise proxy mode)")

    # ---------------------------------------------------------------
    gen_dir_for_cache = os.path.join(output_dir, "generated")
    results: List[dict] = []

    for t in t_steps_list:
        print(f"\nEvaluating T={t}...")

        if use_noise_proxy or not checkpoint_dir:
            t_max = max(t_steps_list)
            gen_t = _noise_proxy_at_steps(base_gen_t, t, t_max, seed)
        else:
            gen_t = _generate_at_steps(
                checkpoint_dir, N, t, device, seed, gen_dir_for_cache
            )

        n = min(len(real_t), len(gen_t))
        rt, gt = real_t[:n], gen_t[:n]

        m3_val  = float(m3_metric(rt, gt)["m3_score"])
        f_gen   = _inception_features(gt, inc_model, device)
        fid_val = _fid(f_real[:n], f_gen)
        cmd_val = _cmmd(f_real[:n], f_gen)

        print(f"  T={t:4d}  M3={m3_val:.4f}  FID={fid_val:.2f}  CMMD={cmd_val:.4f}")
        results.append({"t_steps": t, "m3": m3_val, "fid": fid_val, "cmmd": cmd_val, "n": n})

    # ---------------------------------------------------------------
    # Monotonicity: metrics should increase as T decreases (lower quality)
    # i.e. results are sorted ascending T, so quality is ascending -> metrics descending
    t_vals   = [r["t_steps"] for r in results]
    m3_vals  = [r["m3"]      for r in results]
    fid_vals = [r["fid"]     for r in results]
    cmd_vals = [r["cmmd"]    for r in results]

    # T is ascending -> quality ascending -> good metric value should decrease
    m3_monotone  = _is_monotone_decreasing(m3_vals)
    fid_monotone = _is_monotone_decreasing(fid_vals)
    cmd_monotone = _is_monotone_decreasing(cmd_vals)

    print(f"\nMonotonicity (value decreases as T increases / quality improves):")
    print(f"  M3:   {m3_monotone}  ({[f'{v:.4f}' for v in m3_vals]})")
    print(f"  FID:  {fid_monotone}  ({[f'{v:.2f}' for v in fid_vals]})")
    print(f"  CMMD: {cmd_monotone}  ({[f'{v:.4f}' for v in cmd_vals]})")

    # ---------------------------------------------------------------
    mode = "noise_proxy" if (use_noise_proxy or not checkpoint_dir) else "ddpm_checkpoint"
    report = {
        "config": {
            "real_dir":       real_dir,
            "checkpoint_dir": checkpoint_dir,
            "gen_dir":        gen_dir,
            "num_images":     N,
            "t_steps":        t_steps_list,
            "mode":           mode,
            "mode_note": (
                "Images generated at each T from real DDPM checkpoint." if mode == "ddpm_checkpoint"
                else "Gaussian noise proxy (sigma inversely proportional to T). NOT real DDPM steps."
            ),
        },
        "results":      results,
        "monotonicity": {
            "m3":   m3_monotone,
            "fid":  fid_monotone,
            "cmmd": cmd_monotone,
            "note": "True means metric correctly orders the T ladder (higher T = lower metric).",
        },
    }

    out_path = os.path.join(output_dir, "denoising_steps_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")

    # Plot
    _plot(results, output_dir)
    return report


def _plot(results: List[dict], output_dir: str) -> None:
    t_vals   = [r["t_steps"] for r in results]
    m3_vals  = [r["m3"]      for r in results]
    fid_vals = [r["fid"]     for r in results]
    cmd_vals = [r["cmmd"]    for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    for ax, vals, label, color in zip(
        axes,
        [m3_vals, fid_vals, cmd_vals],
        ["M3-Score (L12 RBF)", "FID", "CMMD"],
        ["#1f77b4", "#d62728", "#2ca02c"],
    ):
        ax.plot(t_vals, vals, "o-", color=color, lw=2, ms=7)
        ax.set_xlabel("Denoising steps T")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle(
        "Denoising Steps Quality Ladder\n"
        "Each metric should decrease monotonically as T increases (quality improves)",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "denoising_steps_ladder.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_dir",        required=True,
                        help="Directory of real reference images")
    parser.add_argument("--checkpoint_dir",  default=None,
                        help="Pre-trained DDPM checkpoint (diffusers format)")
    parser.add_argument("--gen_dir",         default=None,
                        help="Pre-generated images (required for noise proxy mode)")
    parser.add_argument("--output_dir",      default="results/denoising_steps_ladder")
    parser.add_argument("--num_images",      type=int, default=100)
    parser.add_argument("--t_steps",         type=int, nargs="+",
                        default=[10, 20, 50, 100, 1000])
    parser.add_argument("--device",          default="cpu")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--use_noise_proxy", action="store_true",
                        help="Use Gaussian noise proxy instead of real DDPM generation")
    args = parser.parse_args()

    run(
        real_dir=args.real_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        gen_dir=args.gen_dir,
        num_images=args.num_images,
        t_steps_list=args.t_steps,
        device=args.device,
        seed=args.seed,
        use_noise_proxy=args.use_noise_proxy,
    )
