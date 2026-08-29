import argparse
import os
import torch
import numpy as np
import json
from diffusers import DDPMPipeline
from tqdm.auto import tqdm
from PIL import Image
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from pytorch_msssim import ssim as pt_ssim
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
import glob


from utils.reward_system import build_reward_fn as _build_reward_fn

# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------

def load_images_as_tensors(directory, num_images=None, img_size=(256, 256), use_tqdm=True, mode="RGB"):
    """Loads images from a directory as a batched torch tensor [N, C, H, W]."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    
    image_paths = sorted(image_paths)
    if num_images:
        image_paths = image_paths[:num_images]
    
    tensors = []
    for path in tqdm(image_paths, desc=f"Loading images {mode} from {os.path.basename(directory)}", disable=not use_tqdm):
        img = Image.open(path).convert(mode).resize(img_size)
        img_array = np.array(img)
        if mode == "L":
            img_array = np.expand_dims(img_array, axis=0)
        else:
            img_array = img_array.transpose(2, 0, 1) 
        tensors.append(torch.from_numpy(img_array))
    
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# PIL image → normalised float tensor (for reward scoring)
# ---------------------------------------------------------------------------

def _pil_batch_to_tensor(pil_images, device: str) -> torch.Tensor:
    """Convert a list of PIL images to a float tensor [N, C, H, W] in [0, 1]."""
    arrays = []
    for img in pil_images:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:          # grayscale → (H,W) → (1,H,W)
            arr = arr[np.newaxis]
        else:                       # (H,W,3) → (3,H,W)
            arr = arr.transpose(2, 0, 1)
        arrays.append(arr)
    return torch.from_numpy(np.stack(arrays)).to(device)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_images(
    checkpoint_dir,
    output_dir,
    num_images=100,
    batch_size=16,
    data_dir=None,
    calculate_metrics=False,
    device="cuda:0",
    use_tqdm=True,
    rescale_from_01=False,
    # ---- Rotation fix -------------------------------------------------------
    apply_rotation_fix: bool = True,
    # ---- Reward / Best-of-N ------------------------------------------------
    reward_type: str = "none",
    best_of_n: int = 1,
    reward_scale: float = 1.0,
    # ---- Step count (quality ladder) ---------------------------------------
    num_inference_steps: int = None,
):
    """
    Loads a pretrained diffusion pipeline, generates images, and optionally
    applies Best-of-N reward-guided rejection sampling.

    Best-of-N sampling:
      When reward_type != 'none' and best_of_n > 1, each desired output image
      is chosen as the highest-reward sample from `best_of_n` candidates.
      This requires no model updates — it is purely an inference-time filter.

    apply_rotation_fix:
      Rotate generated images 90° CCW to undo the (W, H) transposition caused
      by MONAI's ITKReader during training. Set to False once the model is
      retrained with the fixed dataloader (reader="PILReader").
    """
    print(f"Loading pipeline from {checkpoint_dir}...")
    
    from diffusers import UNet2DModel
    try:
        pipeline = DDPMPipeline.from_pretrained(checkpoint_dir)
    except ValueError as e:
        if "conv_in.weight" in str(e):
            print("Detected channel mismatch. Forcing 1-channel UNet loading...")
            unet = UNet2DModel.from_pretrained(
                checkpoint_dir, 
                subfolder="unet", 
                in_channels=1, 
                out_channels=1, 
                low_cpu_mem_usage=False, 
                ignore_mismatched_sizes=True
            )
            pipeline = DDPMPipeline.from_pretrained(checkpoint_dir, unet=unet)
        else:
            raise e

    pipeline.to(device)

    # Override denoising step count (quality ladder experiment)
    if num_inference_steps is not None:
        pipeline.scheduler.set_timesteps(num_inference_steps)
        print(f"num_inference_steps set to {num_inference_steps}")

    # ------------------------------------------------------------------
    # Build reward function (only if requested)
    # ------------------------------------------------------------------
    reward_fn = None
    use_reward = (reward_type != "none") and (best_of_n > 1)
    if use_reward:
        print(f"[Reward] Initialising '{reward_type}' reward (Best-of-{best_of_n})...")
        try:
            reward_fn = _build_reward_fn(reward_type, reward_scale, device)
            print(f"[Reward] Ready: {reward_fn.__class__.__name__}")
        except Exception as e:
            print(f"[Reward] WARNING: could not build reward ({e}). Falling back to standard sampling.")
            use_reward = False

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} images{'  [Best-of-N=' + str(best_of_n) + ']' if use_reward else ''}...")
    
    num_batches = (num_images + batch_size - 1) // batch_size
    generated_count = 0
    generated_images_list = []
    reward_log = []   # track per-image rewards for logging

    for i in tqdm(range(num_batches), desc="Generating", disable=not use_tqdm):
        current_batch_size = min(batch_size, num_images - generated_count)

        # ----------------------------------------------------------------
        # Best-of-N: generate N candidate sets, score, pick winners
        # ----------------------------------------------------------------
        if use_reward:
            n_candidates = best_of_n
            # Collect all candidate images first
            all_candidates = [[] for _ in range(current_batch_size)]   # [img_idx][candidate_idx]

            for _cand in range(n_candidates):
                outputs = pipeline(batch_size=current_batch_size)
                cand_images = outputs.images  # list of PIL
                for img_idx, img in enumerate(cand_images):
                    all_candidates[img_idx].append(img)

            # Score each candidate with the reward function
            selected_images = []
            for img_idx in range(current_batch_size):
                candidates = all_candidates[img_idx]          # List[PIL], len = n_candidates
                # Stack into tensor [n_candidates, C, H, W]
                cand_tensor = _pil_batch_to_tensor(candidates, device)

                with torch.no_grad():
                    rewards = reward_fn(cand_tensor)          # shape: (n_candidates,)

                best_idx = int(rewards.argmax().item())
                selected_images.append(candidates[best_idx])
                reward_log.append(float(rewards[best_idx].item()))
            images = selected_images

        else:
            # ----------------------------------------------------------------
            # Standard generation (no reward)
            # ----------------------------------------------------------------
            call_kwargs = {"batch_size": current_batch_size}
            if num_inference_steps is not None:
                call_kwargs["num_inference_steps"] = num_inference_steps
            outputs = pipeline(**call_kwargs)
            images = outputs.images

        # ------------------------------------------------------------------
        # Optional rescaling for [0,1]-trained models
        # ------------------------------------------------------------------
        if rescale_from_01:
            fixed_images = []
            for img in images:
                img_arr = np.array(img).astype(np.float32) / 255.0
                img_arr = (img_arr - 0.5) * 2.0
                img_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
                fixed_images.append(Image.fromarray(img_arr))
            images = fixed_images
        
        # Save images
        for img in images:
            img_path = os.path.join(output_dir, f"generated_{generated_count:04d}.png")
            if apply_rotation_fix:
                # Rotate 90° CCW to undo the (W,H) transposition that MONAI's
                # ITKReader introduced during training. Disable once the model
                # is retrained with the fixed PILReader dataloader.
                img = img.rotate(90, expand=False)
            img.save(img_path)
            generated_images_list.append(img_path)
            generated_count += 1

    print(f"Generation complete. Images saved to {output_dir}")

    # Print reward summary if used
    if use_reward and reward_log:
        avg_r = float(np.mean(reward_log))
        print(f"[Reward] Average selected reward: {avg_r:.4f}  (over {len(reward_log)} images)")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if calculate_metrics and data_dir:
        if not METRICS_AVAILABLE:
            print("Warning: torchmetrics or skimage not installed. Skipping metrics.")
            return

        print("Calculating metrics...")
        metrics = {}
        
        # 1. FID & KID Calculation
        fid = FrechetInceptionDistance(feature=2048).to(device)
        kid_subset_size = min(50, num_images)
        kid = KernelInceptionDistance(subset_size=kid_subset_size).to(device)
        
        real_tensors_rgb = load_images_as_tensors(data_dir, num_images=num_images, use_tqdm=use_tqdm, mode="RGB").to(device)
        fid.update(real_tensors_rgb, real=True)
        kid.update(real_tensors_rgb, real=True)
        
        gen_tensors_rgb = load_images_as_tensors(output_dir, num_images=num_images, use_tqdm=use_tqdm, mode="RGB").to(device)
        fid.update(gen_tensors_rgb, real=False)
        kid.update(gen_tensors_rgb, real=False)
        
        metrics["fid"] = float(fid.compute().item())
        kid_mean, kid_std = kid.compute()
        metrics["kid_mean"] = float(kid_mean.item())
        metrics["kid_std"] = float(kid_std.item())
        
        # 2. SSIM and PSNR
        ssim_values = []
        psnr_values = []
        
        real_tensors_gray = load_images_as_tensors(data_dir, num_images=num_images, use_tqdm=use_tqdm, mode="L").to(device)
        gen_tensors_gray = load_images_as_tensors(output_dir, num_images=num_images, use_tqdm=use_tqdm, mode="L").to(device)
        
        real_tensors_gray = real_tensors_gray.float()
        gen_tensors_gray = gen_tensors_gray.float()
        
        num_real = len(real_tensors_gray)
        num_comp_samples = min(100, num_real)
        
        def pt_psnr(img1, img2, data_range):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100.0
            return 20 * torch.log10(torch.tensor(data_range, device=img1.device)) - 10 * torch.log10(mse)
            
        print("Calculating SSIM/PSNR (Nearest Neighbour Estimation)...")
        for i in tqdm(range(len(gen_tensors_gray)), disable=not use_tqdm):
            gen_img = gen_tensors_gray[i:i+1] 
            
            subset_indices = torch.randperm(num_real)[:num_comp_samples]
            real_subset = real_tensors_gray[subset_indices]
            
            best_ssim = -1.0
            best_psnr = 0.0
            
            for j in range(num_comp_samples):
                real_img = real_subset[j:j+1]
                
                img_min = real_img.min().item()
                img_max = real_img.max().item()
                drange = float(img_max - img_min)
                if drange == 0:
                    drange = 1.0
                
                curr_ssim = pt_ssim(gen_img, real_img, data_range=drange, size_average=True).item()
                
                if curr_ssim > best_ssim:
                    best_ssim = curr_ssim
                    best_psnr = pt_psnr(gen_img, real_img, drange).item()
            
            ssim_values.append(best_ssim)
            psnr_values.append(best_psnr)
            
        metrics["ssim"] = float(np.mean(ssim_values))
        metrics["psnr"] = float(np.mean(psnr_values))

        # Include reward info in metrics if available
        if use_reward and reward_log:
            metrics["reward_type"] = reward_type
            metrics["best_of_n"] = best_of_n
            metrics["avg_selected_reward"] = float(np.mean(reward_log))
        
        print("\n--- Generation Metrics ---")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v}")
        
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Generate images using a trained Diffusion pipeline.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the saved pipeline checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output/generated_images", help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--data_dir", type=str, help="Path to real images for metric calculation")
    parser.add_argument("--calculate_metrics", action="store_true", help="Whether to calculate FID, SSIM, PSNR")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--rescale_from_01", action="store_true", default=False, help="Rescale output if model was trained on [0, 1] data (usually not needed)")
    parser.add_argument("--no_rotation_fix", action="store_true", default=False, help="Disable the 90° CCW rotation fix (only use after retraining with the fixed PILReader dataloader)")
    # --- Reward / Best-of-N -----------------------------------------------
    parser.add_argument(
        "--reward_type",
        type=str,
        default="none",
        choices=["none", "trajectory_efficiency", "deep_cosine_diversity"],
        help=(
            "Optional reward function for Best-of-N rejection sampling. "
            "'none' disables reward (default). "
            "'trajectory_efficiency' requires no external model. "
            "'deep_cosine_diversity' requires torchvision (auto-loads VGG-16)."
        ),
    )
    parser.add_argument(
        "--best_of_n",
        type=int,
        default=1,
        help=(
            "Number of candidate images to generate per output image. "
            "The candidate with the highest reward score is kept. "
            "Only active when --reward_type != 'none'. Default: 1 (disabled)."
        ),
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="Scaling factor applied to the reward signal. Default: 1.0.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help=(
            "Number of DDPM denoising steps at inference. Default: use model's "
            "training timestep count (usually 1000). Set lower (e.g. 50, 20, 10) "
            "for the quality-ladder experiment — fewer steps = lower quality."
        ),
    )

    args = parser.parse_args()
    generate_images(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        calculate_metrics=args.calculate_metrics,
        device=args.device,
        use_tqdm=not args.no_tqdm,
        rescale_from_01=args.rescale_from_01,
        apply_rotation_fix=not args.no_rotation_fix,
        reward_type=args.reward_type,
        best_of_n=args.best_of_n,
        reward_scale=args.reward_scale,
        num_inference_steps=args.num_inference_steps,
    )
