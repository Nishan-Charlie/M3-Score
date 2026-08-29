"""
finetune_rl.py
===============
Entry-point for DDPO reinforcement-learning fine-tuning of a pre-trained
DDPMPipeline checkpoint.

This is an *optional* post-training step — it requires an already-trained
checkpoint (from train.py / run_dit.sh) and refines the model to maximise
a chosen reward function without overwriting the base checkpoints.

Research basis:
  - Black et al. (2023) DDPO: RL fine-tuning using policy-gradient
    denoised trajectories is most stable when started from a converged
    base policy.
  - Fan et al. (2023) DPOK: KL regularisation prevents reward collapse.

Usage
-----
    python finetune_rl.py \\
        --checkpoint_dir  ./output/output_dit/checkpoints/epoch_299 \\
        --reward_type     deep_cosine_diversity \\
        --rl_epochs       50 \\
        --batch_size      4 \\
        --lr              1e-6 \\
        --device          cuda:1

    # Faster / shorter run for a smoke test
    python finetune_rl.py \\
        --checkpoint_dir  ./output/output_dit/checkpoints/epoch_299 \\
        --reward_type     trajectory_efficiency \\
        --rl_epochs       2 \\
        --batch_size      4 \\
        --device          cuda:1

Available --reward_type options
--------------------------------
  trajectory_efficiency   No external model needed. Rewards diverse, fast generation.
  deep_cosine_diversity   Uses a VGG-16 backbone (torchvision). Rewards feature diversity.
"""

import argparse
import os

from diffusers import DDPMPipeline

from utils.reward_system import build_reward_fn as _build_reward_fn

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DDPO RL fine-tuning for a pre-trained MRI diffusion model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to a trained DDPMPipeline checkpoint (output of train.py).",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        required=True,
        choices=["trajectory_efficiency", "deep_cosine_diversity"],
        help=(
            "Reward function to optimise:\n"
            "  trajectory_efficiency   — No external model needed.\n"
            "  deep_cosine_diversity   — VGG-16 feature diversity (auto-loaded).\n"
        ),
    )

    # --- Training ---
    parser.add_argument("--rl_epochs",       type=int,   default=50,   help="Number of RL gradient steps.")
    parser.add_argument("--batch_size",      type=int,   default=4,    help="Images generated per update step.")
    parser.add_argument("--lr",              type=float, default=1e-6, help="RL learning rate.")
    parser.add_argument("--kl_coeff",        type=float, default=0.05, help="KL penalty coefficient (prevents collapse).")
    parser.add_argument("--clip_epsilon",    type=float, default=0.2,  help="PPO-clip epsilon range.")
    parser.add_argument("--grad_timesteps",  type=int,   default=10,   help="Number of final denoising steps to differentiate.")
    parser.add_argument("--reward_scale",    type=float, default=1.0,  help="Reward scaling factor.")
    parser.add_argument("--reward_clip",     type=float, default=10.0, help="Max absolute reward value (stability).")
    parser.add_argument("--save_every",      type=int,   default=10,   help="Save RL checkpoint every N epochs.")

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save RL checkpoints. "
            "Defaults to <checkpoint_dir>/../rl_checkpoints."
        ),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="CSV file for per-epoch reward/loss stats. Defaults to <output_dir>/rl_log.csv.",
    )

    # --- Misc ---
    parser.add_argument("--device",  type=str,  default="cuda:0", help="Device (e.g. cuda:0, cpu).")
    parser.add_argument("--no_tqdm", action="store_true",         help="Disable tqdm progress bar.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.checkpoint_dir)),
            "rl_checkpoints",
        )

    if args.log_file is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.log_file = os.path.join(args.output_dir, "rl_log.csv")

    # ------------------------------------------------------------------
    # Print banner
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  DDPO RL Fine-tuning — MRI Diffuser")
    print("=" * 60)
    print(f"  Base checkpoint : {args.checkpoint_dir}")
    print(f"  Reward          : {args.reward_type} (scale={args.reward_scale})")
    print(f"  RL epochs       : {args.rl_epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  LR              : {args.lr}")
    print(f"  KL coeff        : {args.kl_coeff}")
    print(f"  Grad timesteps  : {args.grad_timesteps}")
    print(f"  Device          : {args.device}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Log file        : {args.log_file}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    print(f"\nLoading pipeline from: {args.checkpoint_dir}")
    try:
        pipeline = DDPMPipeline.from_pretrained(args.checkpoint_dir)
    except ValueError as e:
        if "conv_in.weight" in str(e):
            print("Detected channel mismatch — forcing 1-channel UNet loading...")
            from diffusers import UNet2DModel
            unet = UNet2DModel.from_pretrained(
                args.checkpoint_dir,
                subfolder="unet",
                in_channels=1,
                out_channels=1,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
            pipeline = DDPMPipeline.from_pretrained(args.checkpoint_dir, unet=unet)
        else:
            raise e

    pipeline.to(args.device)
    print("Pipeline loaded.\n")

    # ------------------------------------------------------------------
    # Build reward
    # ------------------------------------------------------------------
    print(f"Initialising reward function: {args.reward_type}")
    reward_fn = _build_reward_fn(args.reward_type, args.reward_scale, args.device)
    print(f"Reward ready: {reward_fn.__class__.__name__}\n")

    # ------------------------------------------------------------------
    # Run RL fine-tuning
    # ------------------------------------------------------------------
    from utils.rl_trainer import run_rl_finetuning

    history = run_rl_finetuning(
        pipeline=pipeline,
        reward_fn=reward_fn,
        checkpoint_dir=args.output_dir,
        device=args.device,
        rl_epochs=args.rl_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_coeff=args.kl_coeff,
        clip_epsilon=args.clip_epsilon,
        grad_timesteps=args.grad_timesteps,
        reward_clip=args.reward_clip,
        save_every=args.save_every,
        log_file=args.log_file,
        use_tqdm=not args.no_tqdm,
    )

    final_stats = history[-1] if history else {}
    print("\n" + "=" * 60)
    print("  RL Fine-tuning Complete!")
    print("=" * 60)
    print(f"  Final reward mean : {final_stats.get('reward_mean', 'N/A'):.4f}")
    print(f"  Final loss        : {final_stats.get('loss', 'N/A'):.4f}")
    print(f"  RL checkpoints    : {args.output_dir}")
    print(f"  Training log      : {args.log_file}")
    print("=" * 60)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
