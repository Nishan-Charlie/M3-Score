"""
tools/generate_step_ladder.py
==============================
Generates BraTS images at multiple denoising step counts to create
a controlled quality ladder with a known ground-truth ordering.

Why:
  Fewer DDPM denoising steps → more noise in output → lower quality.
  The ordering 1000 > 500 > 200 > 100 > 50 > 20 > 10 steps is known a priori.
  This gives 7 generated sets whose true quality ranking is ground-truth,
  enabling a valid Spearman rho experiment at n=7 (p < 0.001 even with 1 error).

This is the fix for the TSTR experiment: replaces the degenerate
4-generator set (random noise + wrong-domain model) with a principled
quality spectrum from the same model.

Usage:
    python tools/generate_step_ladder.py \
        --checkpoint_dir output/output_unet/checkpoints/best \
        --num_images 500 \
        --device cuda \
        --output_root output/step_ladder
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import DDPMPipeline, UNet2DModel
from PIL import Image
from tqdm import tqdm

# Quality ladder: step counts in descending quality order
STEP_LADDER = [1000, 500, 200, 100, 50, 20, 10]


def load_pipeline(checkpoint_dir: str, device: str) -> DDPMPipeline:
    """Load the BraTS DDPM pipeline from a checkpoint directory."""
    ckpt = Path(checkpoint_dir)
    try:
        # Try loading the full saved pipeline first
        pipe = DDPMPipeline.from_pretrained(str(ckpt))
        print(f"  Loaded full DDPMPipeline from {ckpt}")
    except Exception:
        # Fall back to loading UNet + default scheduler
        from diffusers import DDPMScheduler
        unet = UNet2DModel.from_pretrained(str(ckpt / "unet"))
        scheduler = DDPMScheduler.from_pretrained(str(ckpt / "scheduler"))
        pipe = DDPMPipeline(unet=unet, scheduler=scheduler)
        print(f"  Loaded UNet + Scheduler separately from {ckpt}")

    pipe = pipe.to(device)
    return pipe


def generate_at_steps(
    pipe: DDPMPipeline,
    num_steps: int,
    num_images: int,
    output_dir: Path,
    batch_size: int,
    device: str,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the number of inference steps in the scheduler
    pipe.scheduler.set_timesteps(num_steps)

    generator = torch.Generator(device=device).manual_seed(seed)
    n_generated = 0
    pbar = tqdm(total=num_images, desc=f"  steps={num_steps:4d}", leave=False)

    while n_generated < num_images:
        current_batch = min(batch_size, num_images - n_generated)
        with torch.no_grad():
            output = pipe(
                batch_size=current_batch,
                generator=generator,
                num_inference_steps=num_steps,
            )

        for img in output.images:
            # Save as grayscale PNG (BraTS pipeline output is grayscale)
            if img.mode != "L":
                img = img.convert("L")
            fname = output_dir / f"generated_{n_generated:04d}.png"
            img.save(fname)
            n_generated += 1
            pbar.update(1)
            if n_generated >= num_images:
                break

    pbar.close()
    return n_generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to the saved BraTS DDPM pipeline checkpoint")
    parser.add_argument("--num_images", type=int, default=500,
                        help="Number of images per step count")
    parser.add_argument("--step_counts", nargs="+", type=int, default=STEP_LADDER,
                        help="List of denoising step counts to evaluate")
    parser.add_argument("--output_root", default="output/step_ladder",
                        help="Root directory; subdirs named steps_N/ are created")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.checkpoint_dir} ...")
    pipe = load_pipeline(args.checkpoint_dir, args.device)
    pipe.set_progress_bar_config(disable=True)

    print(f"\nGenerating {args.num_images} images at each step count:")
    print(f"Step counts: {args.step_counts}")
    print(f"Output root: {root.resolve()}\n")

    results = {}
    for steps in sorted(args.step_counts, reverse=True):
        out_dir = root / f"steps_{steps:04d}"
        print(f"[steps={steps}] → {out_dir}/")
        n = generate_at_steps(
            pipe, steps, args.num_images, out_dir,
            args.batch_size, args.device, args.seed,
        )
        results[steps] = {"n_images": n, "dir": str(out_dir)}
        print(f"  Saved {n} images.")

    # Write a manifest file
    import json
    manifest = {
        "checkpoint_dir": args.checkpoint_dir,
        "num_images": args.num_images,
        "seed": args.seed,
        "step_ladder": {
            str(s): results[s] for s in sorted(results.keys(), reverse=True)
        },
        "known_quality_ordering": "descending by step count (1000=best, 10=worst)",
        "purpose": "Controlled quality ladder for TSTR Spearman rho experiment (n=7)",
    }
    mpath = root / "step_ladder_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {mpath}")

    print("\nDone. Run experiments/tstr_utility.py with these directories.")
    print("Expected TSTR Spearman rho ≈ 1.0 for M3 if it recovers the step ordering.")


if __name__ == "__main__":
    main()
