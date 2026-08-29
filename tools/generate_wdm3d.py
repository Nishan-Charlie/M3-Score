"""
WDM-3D Generator - 2D Slice Extraction
========================================
Loads the pretrained Wavelet Diffusion Model (pfriedri/wdm-3d), generates
3D medical volumes, and extracts 2D slices compatible with the M3-Score
evaluation pipeline.

Supported models
----------------
  brats  : BraTS 2023 brain MRI   (T1w, 128x128x128)
  lidc   : LIDC-IDRI lung CT      (128x128x128)

Slice extraction
----------------
  axial      : slices along Z-axis (default)
  coronal    : slices along Y-axis
  sagittal   : slices along X-axis
  all        : all three planes (3x more slices)

Output format
-------------
  PNG, 8-bit grayscale, 256x256 (bicubic upsampled from 128)
  Filename: {model}_{volume:04d}_{plane}_{slice:03d}.png

Usage
-----
  python tools/generate_wdm3d.py --model brats --n_volumes 20 --output_dir output/generated_wdm3d

  # Select middle third of axial slices (excludes blank edges)
  python tools/generate_wdm3d.py --model brats --n_volumes 20 --slice_frac 0.33 0.66

  # Generate from both models
  python tools/generate_wdm3d.py --model all --n_volumes 10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.path.join(_ROOT, "external", "wdm-3d")
WEIGHTS_DIR  = os.path.join(_ROOT, "pretrained", "wdm3d")

# Add WDM-3D source to path
if EXTERNAL_DIR not in sys.path:
    sys.path.insert(0, EXTERNAL_DIR)

WEIGHT_FILES = {
    "brats": os.path.join(WEIGHTS_DIR, "brats_unet_128_1200k.pt"),
    "lidc":  os.path.join(WEIGHTS_DIR, "lidc-idri_unet_128_1200k.pt"),
}

# Exact config from run.sh for model 'ours_unet_128'
# Checkpoints: brats_unet_128_1200k.pt, lidc-idri_unet_128_1200k.pt
# Key facts: image_size=128 (full res); noise shape is (B,8,64,64,64) = image_size//2
_SHARED_CFG = dict(
    image_size=128,
    num_channels=64,
    num_res_blocks=2,
    channel_mult="1,2,2,4,4",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="",   # no attention in unet variant
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=False,
    use_new_attention_order=False,
    dims=3,
    num_groups=32,
    in_channels=8,              # 8 haar-3D wavelet sub-bands
    out_channels=8,
    bottleneck_attention=False,
    resample_2d=False,
    additive_skips=True,        # unet variant
    mode="default",
    use_freq=False,             # unet variant: no frequency-domain processing
    diffusion_steps=1000,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=True,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
)

MODEL_CONFIGS = {
    "brats": {**_SHARED_CFG, "dataset": "brats"},
    "lidc":  {**_SHARED_CFG, "dataset": "lidc-idri"},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _check_source():
    if not os.path.isdir(EXTERNAL_DIR):
        print(f"ERROR: WDM-3D source not found at {EXTERNAL_DIR}")
        print("Run first:  python tools/download_wdm3d.py")
        sys.exit(1)


def _check_weights(model_key: str):
    wp = WEIGHT_FILES[model_key]
    if not os.path.isfile(wp):
        print(f"ERROR: Weights not found: {wp}")
        print(f"Run first:  python tools/download_wdm3d.py --model {model_key}")
        sys.exit(1)
    return wp


def _load_model(model_key: str, device: str):
    """Load WDM-3D model + diffusion sampler from checkpoint."""
    _check_source()
    wp = _check_weights(model_key)

    try:
        from guided_diffusion.script_util import (
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )
    except ImportError as e:
        print(f"ERROR: Cannot import guided_diffusion from {EXTERNAL_DIR}: {e}")
        print("Make sure WDM-3D was cloned correctly.")
        sys.exit(1)

    cfg = {**model_and_diffusion_defaults(), **MODEL_CONFIGS[model_key]}
    model, diffusion = create_model_and_diffusion(**cfg)

    state = torch.load(wp, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"  Loaded {model_key} checkpoint: {wp}")
    return model, diffusion


# ---------------------------------------------------------------------------
# IDWT reconstruction
# ---------------------------------------------------------------------------

def _idwt_reconstruct(sample: torch.Tensor) -> torch.Tensor:
    """
    Apply 3D inverse Haar wavelet transform to recover spatial volume.

    sample: (B, 8, D/2, H/2, W/2)
    IDWT_3D.forward takes 8 separate (B,1,D/2,H/2,W/2) tensors.
    LL sub-band (channel 0) is scaled by 3.0 as per the official sample script.
    Returns: (B, 1, D, H, W) in original wavelet-coefficient range.
    """
    try:
        from DWT_IDWT.DWT_IDWT_layer import IDWT_3D
    except ImportError as e:
        print(f"WARNING: IDWT not available ({e}). Returning LL sub-band only.")
        return sample[:, 0:1]

    idwt = IDWT_3D("haar")
    B, _, D, H, W = sample.shape

    def _sub(c, scale=1.0):
        return sample[:, c, :, :, :].view(B, 1, D, H, W) * scale

    vol = idwt(
        _sub(0, 3.0),  # LLL -- scaled by 3 per official script
        _sub(1),       # LLH
        _sub(2),       # LHL
        _sub(3),       # LHH
        _sub(4),       # HLL
        _sub(5),       # HLH
        _sub(6),       # HHL
        _sub(7),       # HHH
    )
    return vol  # (B, 1, D, H, W)


# ---------------------------------------------------------------------------
# Slice extraction
# ---------------------------------------------------------------------------

def _extract_slices(
    vol: np.ndarray,
    plane: str,
    slice_frac: tuple,
    out_size: int = 256,
) -> list:
    """Extract 2D slices from a (D, H, W) float [0,1] volume."""
    D, H, W = vol.shape

    if plane == "axial":
        n = D
        getter = lambda i: vol[i, :, :]
    elif plane == "coronal":
        n = H
        getter = lambda i: vol[:, i, :]
    elif plane == "sagittal":
        n = W
        getter = lambda i: vol[:, :, i]
    else:
        raise ValueError(f"Unknown plane: {plane}")

    lo = int(n * slice_frac[0])
    hi = int(n * slice_frac[1])
    slices = []
    for i in range(lo, hi):
        sl = getter(i)
        sl_t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).float()
        sl_t = F.interpolate(sl_t, size=(out_size, out_size), mode="bicubic",
                             align_corners=False).clamp(0, 1)
        sl_np = (sl_t.squeeze().numpy() * 255).astype(np.uint8)
        slices.append(sl_np)
    return slices


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_slices(
    model_key   = "brats",
    n_volumes   = 10,
    output_dir  = "output/generated_wdm3d",
    plane       = "axial",
    slice_frac  = (0.30, 0.70),
    out_size    = 256,
    batch_size  = 1,
    device      = None,
    seed        = 42,
    ddim_steps  = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    planes = ["axial", "coronal", "sagittal"] if plane == "all" else [plane]

    print(f"\n{'='*55}")
    print(f"  WDM-3D Generator")
    print(f"  Model:   {model_key}")
    print(f"  Volumes: {n_volumes}  |  Plane: {plane}")
    print(f"  Device:  {device}")
    print(f"{'='*55}\n")

    model, diffusion = _load_model(model_key, device)

    cfg        = MODEL_CONFIGS[model_key]
    img_size   = cfg["image_size"]    # 128 (full spatial resolution)
    half_size  = img_size // 2        # 64  (wavelet-domain resolution)
    n_channels = cfg["in_channels"]   # 8

    total_slices = 0
    vol_idx = 0

    with tqdm(total=n_volumes, desc="Generating volumes") as pbar:
        while vol_idx < n_volumes:
            bs = min(batch_size, n_volumes - vol_idx)

            noise = torch.randn(bs, n_channels, half_size, half_size, half_size,
                                device=device)

            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model, noise.shape, noise=noise,
                    clip_denoised=True, device=device,
                    progress=False,
                )

            # IDWT -> spatial domain  (B, 1, D, H, W)
            # Keep on same device as IDWT_3D matrices (CUDA when available)
            vol_spatial = _idwt_reconstruct(sample)
            vol_spatial = (vol_spatial + 1.0) / 2.0   # [-1,1] -> [0,1]
            vol_spatial = vol_spatial.clamp(0, 1).cpu()

            for b in range(bs):
                vol_np = vol_spatial[b, 0].numpy()   # (D, H, W)

                for p in planes:
                    slices = _extract_slices(vol_np, p, slice_frac, out_size)
                    for sl_idx, sl in enumerate(slices):
                        fname = f"{model_key}_{vol_idx:04d}_{p}_{sl_idx:03d}.png"
                        Image.fromarray(sl, mode="L").save(
                            os.path.join(output_dir, fname)
                        )
                        total_slices += 1
                vol_idx += 1

            pbar.update(bs)

    print(f"\n  Volumes generated : {n_volumes}")
    print(f"  2D slices saved   : {total_slices}")
    print(f"  Output dir        : {output_dir}")

    import json
    report = {
        "model":         model_key,
        "n_volumes":     n_volumes,
        "total_slices":  total_slices,
        "plane":         plane,
        "slice_frac":    list(slice_frac),
        "out_size":      out_size,
        "output_dir":    output_dir,
    }
    with open(os.path.join(output_dir, "wdm3d_generation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="brats", choices=["brats", "lidc", "all"])
    p.add_argument("--n_volumes",   type=int,   default=10)
    p.add_argument("--output_dir",  default="output/generated_wdm3d")
    p.add_argument("--plane",       default="axial",
                   choices=["axial", "coronal", "sagittal", "all"])
    p.add_argument("--slice_frac",  nargs=2, type=float, default=[0.30, 0.70],
                   metavar=("LO", "HI"))
    p.add_argument("--out_size",    type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int,   default=42)
    a = p.parse_args()

    models = ["brats", "lidc"] if a.model == "all" else [a.model]
    for m in models:
        out = a.output_dir if len(models) == 1 else os.path.join(a.output_dir, m)
        generate_slices(
            model_key  = m,
            n_volumes  = a.n_volumes,
            output_dir = out,
            plane      = a.plane,
            slice_frac = tuple(a.slice_frac),
            out_size   = a.out_size,
            batch_size = a.batch_size,
            device     = a.device,
            seed       = a.seed,
        )
