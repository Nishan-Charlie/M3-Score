import argparse
import torch
from diffusers import DDPMScheduler

from data.dataloader import get_mri_2d_dataloader
from models.unet import get_model, get_default_scheduler_model
from utils.trainer import run_training

import os

def main():
    parser = argparse.ArgumentParser(description="Train a 2D Diffusion model on MRI images.")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing images (JPG, PNG, etc.)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Spatial size to resize images to")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="HuggingFace pretrained model ID. Defaults to the "
                             "architecture-specific best checkpoint:\n"
                             "  unet           → google/ddpm-celebahq-256\n"
                             "  attention_unet → benetraco/brain_ddpm_256\n"
                             "  dit            → facebook/DiT-XL-2-256")
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet", "attention_unet", "dit"], help="Type of model architecture")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels (1 for grayscale)")
    parser.add_argument("--out_channels", type=int, default=1, help="Number of output channels")
    parser.add_argument("--attention_head_dim", type=int, default=8, help="Dimension of attention heads")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_file", type=str, default="mri_training_log.csv", help="CSV file for training logs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training (if using multi-GPU environment)")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--early_stopping_patience", type=int, default=100, help="Number of epochs to wait for improvement before stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum change in loss to qualify as an improvement")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--save_periodic", action="store_true", help="Save periodic checkpoints every 10 epochs")
    parser.add_argument("--eval_images", type=int, default=0, help="Number of images to generate for evaluation every epoch (0 to disable)")
    parser.add_argument("--eval_freq", type=int, default=1, help="Frequency (in epochs) to run evaluation")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode for Accelerate (default: fp16). "
                             "Use 'no' for CPU or older GPUs without fp16 support.")
    
    args = parser.parse_args()

    # Reorganize output directories into output/output_[ModelName]
    base_output_dir = os.path.join("output", f"output_{args.model_type}")
    
    # Update checkpoint directory
    if args.checkpoint_dir == "./checkpoints": # Only update if it's the default
        args.checkpoint_dir = os.path.join(base_output_dir, "checkpoints")
    elif not f"output_{args.model_type}" in args.checkpoint_dir:
        # If user provided a custom path, we still try to respect the naming if it doesn't match
        args.checkpoint_dir = f"{args.checkpoint_dir}_{args.model_type}"

    # Update log file path
    log_dir = os.path.join(base_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.basename(args.log_file)
    if args.log_file == "mri_training_log.csv": # Only update if it's the default
        args.log_file = os.path.join(log_dir, f"mri_training_log_{args.model_type}.csv")
    else:
        # If custom log name, place it in the logs dir
        args.log_file = os.path.join(log_dir, log_filename)

    # 1. Setup Data Loader
    print(f"Loading data from {args.data_dir}...")
    loader = get_mri_2d_dataloader(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        spatial_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )

    # 2. Setup Model
    print(f"Setting up model type: {args.model_type}...")
    model = get_model(
        model_type=args.model_type,
        pretrained_model_name=args.pretrained_model,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=args.img_size,
        attention_head_dim=args.attention_head_dim
    )

    # 3. Resolve scheduler source — use architecture-specific model unless
    #    the user has explicitly overridden --pretrained_model.
    scheduler_source = (
        args.pretrained_model                        # user override
        if args.pretrained_model is not None
        else get_default_scheduler_model(args.model_type)
    )
    print(f"Loading DDPM scheduler from: {scheduler_source}")
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(scheduler_source)
    except Exception:
        # Some model repos don't expose a standalone scheduler config.
        # Fall back to the standard celebahq DDPM scheduler.
        fallback = "google/ddpm-celebahq-256"
        print(f"  WARNING: scheduler not found at '{scheduler_source}', "
              f"falling back to '{fallback}'.")
        noise_scheduler = DDPMScheduler.from_pretrained(fallback)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Handle Resumption
    start_epoch = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        # Load weights
        if os.path.exists(os.path.join(args.resume_from, "unet")): # diffusers format
            from diffusers import UNet2DConditionModel, UNet2DModel
            if hasattr(model, "from_pretrained"):
                 # For diffusers models, we can load weights into the existing model
                 # or unwrap it. The simplest is loading the state dict.
                 # But get_model already created the model.
                 pass
            
            # Simple way: load state dict from the .safetensors or .bin
            checkpoint_file = os.path.join(args.resume_from, "unet", "diffusion_pytorch_model.safetensors")
            if os.path.exists(checkpoint_file):
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_file)
                model.load_state_dict(state_dict)
            else:
                checkpoint_file = os.path.join(args.resume_from, "unet", "diffusion_pytorch_model.bin")
                if os.path.exists(checkpoint_file):
                    model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))
        elif os.path.exists(os.path.join(args.resume_from, "model.pt")):
            model.load_state_dict(torch.load(os.path.join(args.resume_from, "model.pt"), map_location="cpu"))
        
        # Try to load start_epoch from metrics.json
        metrics_file = os.path.join(args.resume_from, "metrics.json")
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                start_epoch = metrics_data.get("epoch", 0) + 1
                print(f"  Detected start epoch: {start_epoch}")

    # 5. Run Training
    print("Starting training...")
    run_training(
        model=model,
        loader=loader,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler,
        num_epochs=args.epochs,
        start_epoch=start_epoch,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        device=args.device,
        num_gpus=args.num_gpus,
        use_tqdm=not args.no_tqdm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_periodic=args.save_periodic,
        eval_images=args.eval_images,
        eval_freq=args.eval_freq,
        mixed_precision=args.mixed_precision,
    )
    print("Training complete.")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()