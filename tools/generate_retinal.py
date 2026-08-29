"""
tools/generate_retinal.py
==========================
Runs inference on `GS-23/ddpm-unet-retinal-fundus-image-generator`
(a pretrained DDPM UNet for retinal fundus images, 128×128 RGB)
and saves 500 generated images as grayscale PNG for use with
the M3-Score evaluation pipeline.

The model is structurally identical to the BraTS UNet pipeline
(both use DDPMPipeline from diffusers) so the same generate.py
code could load it — but this script handles the RGB→grayscale
conversion and resolution difference explicitly.

Usage:
    python tools/generate_retinal.py \
        --num_images 500 \
        --num_inference_steps 1000 \
        --output_dir output/generated_retinal \
        --device cuda
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm

MODEL_ID = "GS-23/ddpm-unet-retinal-fundus-image-generator"


def generate_retinal(
    num_images: int = 500,
    num_inference_steps: int = 1000,
    output_dir: str = "output/generated_retinal",
    batch_size: int = 8,
    device: str = "cuda",
    seed: int = 42,
    save_rgb: bool = False,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    pipe = DDPMPipeline.from_pretrained(MODEL_ID)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # Override scheduler timesteps for step-count experiments
    if num_inference_steps != pipe.scheduler.config.num_train_timesteps:
        pipe.scheduler.set_timesteps(num_inference_steps)
        print(f"  num_inference_steps overridden to {num_inference_steps}")

    generator = torch.Generator(device=device).manual_seed(seed)

    n_generated = 0
    pbar = tqdm(total=num_images, desc="Generating retinal images")

    while n_generated < num_images:
        current_batch = min(batch_size, num_images - n_generated)
        with torch.no_grad():
            output = pipe(
                batch_size=current_batch,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )

        for img in output.images:
            # img is a PIL RGB Image (128×128)
            if save_rgb:
                rgb_path = out / f"retinal_rgb_{n_generated:04d}.png"
                img.save(rgb_path)

            # Convert to grayscale for consistent evaluation with BraTS pipeline
            gray = img.convert("L")
            fname = out / f"retinal_{n_generated:04d}.png"
            gray.save(fname)
            n_generated += 1
            pbar.update(1)
            if n_generated >= num_images:
                break

    pbar.close()
    print(f"Saved {n_generated} grayscale images to {out.resolve()}")
    return str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                        help="DDPM denoising steps (1000=full quality, lower=faster/worse)")
    parser.add_argument("--output_dir", default="output/generated_retinal")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_rgb", action="store_true",
                        help="Also save RGB copies alongside grayscale")
    args = parser.parse_args()

    generate_retinal(
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        save_rgb=args.save_rgb,
    )


if __name__ == "__main__":
    main()
