import os
import torch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDIMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

def run_training(
    model,
    loader,
    optimizer,
    noise_scheduler,
    num_epochs=50,
    start_epoch=0,
    checkpoint_dir="./checkpoints",
    log_file="mri_training_log.csv",
    device="cuda:0",
    num_gpus=1,
    use_tqdm=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.0,
    save_periodic=True,
    eval_images=0,
    eval_freq=1,
    mixed_precision="fp16",
):
    """
    Executes the training loop for the diffusion model.
    Supports multi-GPU via HuggingFace Accelerate (DDP).
    """
    accelerator = Accelerator(mixed_precision=mixed_precision)
    is_main = accelerator.is_main_process  # Only rank 0 prints/saves

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if is_main:
        print(f"[Trainer] Distributed: {accelerator.distributed_type}, "
              f"Num processes: {accelerator.num_processes}, "
              f"Device: {accelerator.device}")

    # Reference batch for metrics (KID/FID)
    ref_images = None
    if eval_images > 0:
        if is_main:
            print(f"Preparing reference images for evaluation (n={eval_images})...")
        ref_batches = []
        collected = 0
        for batch in loader:
            ref_batches.append(batch.detach().cpu())
            collected += batch.shape[0]
            if collected >= eval_images:
                break
        ref_images = torch.cat(ref_batches, dim=0)[:eval_images]
        # Convert grayscale to RGB for Inception-based metrics
        if ref_images.shape[1] == 1:
            ref_images = ref_images.repeat(1, 3, 1, 1)
        # Rescale to [0, 255] uint8 for torchmetrics
        ref_images = ((ref_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

    history = []
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early stopping state
    best_loss = float('inf')
    epochs_no_improve = 0
    
    if is_main:
        print(f"Starting training from epoch {start_epoch} to {num_epochs}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        if use_tqdm and is_main:
            progress_bar = tqdm(loader, desc=f"Epoch {epoch}")
        else:
            if is_main:
                print(f"Epoch {epoch}/{num_epochs} starting...")
            progress_bar = loader
        
        for batch in progress_bar:
            clean_images = batch
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (clean_images.shape[0],), 
                device=clean_images.device
            ).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                num_batches += 1
                
            if use_tqdm and is_main:
                progress_bar.set_postfix({"loss": loss.item()})

        # Gather the average loss across all processes for consistent reporting
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_loss_tensor = torch.tensor([avg_loss], device=accelerator.device)
        avg_loss_tensor = accelerator.reduce(avg_loss_tensor, reduction="mean")
        avg_loss = avg_loss_tensor.item()

        # Wait for all processes before proceeding
        accelerator.wait_for_everyone()

        # ---- Everything below runs ONLY on rank 0 ----
        if is_main:
            # Generation Metrics
            gen_metrics = {}
            if eval_images > 0 and (epoch % eval_freq == 0 or epoch == num_epochs - 1):
                gen_metrics = evaluate_generation(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    accelerator=accelerator,
                    ref_images=ref_images,
                    num_samples=eval_images,
                    img_size=ref_images.shape[2] if ref_images is not None else 256,
                    device=accelerator.device
                )
                print(f"  Generation Metrics - FID: {gen_metrics.get('fid', 'N/A')}, KID: {gen_metrics.get('kid', 'N/A')}")

            history_item = {"epoch": epoch, "loss": avg_loss}
            history_item.update(gen_metrics)
            history.append(history_item)
            pd.DataFrame(history).to_csv(log_file, index=False)
            
            # Early stopping and best model check (use FID if available, otherwise loss)
            primary_metric = gen_metrics.get("fid", avg_loss)
            is_best = False
            # If we use FID, lower is better. If we use loss, lower is better.
            if primary_metric < (best_loss - early_stopping_min_delta):
                best_loss = primary_metric
                epochs_no_improve = 0
                is_best = True
                print(f"  New best { 'FID' if 'fid' in gen_metrics else 'loss' }: {best_loss:.6f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}. Metric has not improved for {early_stopping_patience} epochs.")
                    save_checkpoint(model, noise_scheduler, accelerator, checkpoint_dir, epoch, avg_loss, gen_metrics, tag="last")
                    if is_best:
                         save_checkpoint(model, noise_scheduler, accelerator, checkpoint_dir, epoch, avg_loss, gen_metrics, tag="best")
                    break

            # Save checkpoints
            # 1. Periodic checkpoint (every 10 epochs)
            if save_periodic and (epoch % 10 == 0 or epoch == num_epochs - 1):
                save_checkpoint(model, noise_scheduler, accelerator, checkpoint_dir, epoch, avg_loss, gen_metrics)
                
            # 2. Always save "best" and "last"
            save_checkpoint(model, noise_scheduler, accelerator, checkpoint_dir, epoch, avg_loss, gen_metrics, tag="last")
            if is_best:
                save_checkpoint(model, noise_scheduler, accelerator, checkpoint_dir, epoch, avg_loss, gen_metrics, tag="best")

        # Sync all processes after checkpointing before next epoch
        accelerator.wait_for_everyone()

    return history

def evaluate_generation(model, noise_scheduler, accelerator, ref_images, num_samples, img_size, device):
    """Generates images and computes FID/KID scores."""
    model.eval()
    unet = accelerator.unwrap_model(model)
    
    # Use DDIM for faster sampling during evaluation if possible
    eval_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    eval_scheduler.set_timesteps(50) # Use 50 steps for speed
    
    print(f"  Generating {num_samples} images for evaluation ({img_size}x{img_size})...")
    with torch.no_grad():
        # Generate in batches to avoid OOM
        batch_size = 16
        generated = []
        for i in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - i)
            # Manual sampling loop
            channels = unet.conv_in.in_channels
            shape = (n, channels, img_size, img_size)
            image = torch.randn(shape, device=device)
            
            for t in eval_scheduler.timesteps:
                model_output = unet(image, t).sample
                image = eval_scheduler.step(model_output, t, image).prev_sample
            
            # Map back to [0, 1] for metric processing
            image = (image / 2 + 0.5).clamp(0, 1)
            generated.append(image.cpu())
        
        gen_images = torch.cat(generated, dim=0) # [N, C, H, W]
        if gen_images.shape[1] == 1:
            gen_images = gen_images.repeat(1, 3, 1, 1)
        gen_images = (gen_images * 255).to(torch.uint8)

    # Compute FID
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    fid_metric.update(ref_images.to(device), real=True)
    fid_metric.update(gen_images.to(device), real=False)
    fid_score = fid_metric.compute().item()

    # Compute KID
    kid_metric = KernelInceptionDistance(subset_size=min(num_samples, 50)).to(device)
    kid_metric.update(ref_images.to(device), real=True)
    kid_metric.update(gen_images.to(device), real=False)
    kid_mean, _ = kid_metric.compute()
    
    return {
        "fid": round(float(fid_score), 4),
        "kid": round(float(kid_mean.item()), 6)
    }

def save_checkpoint(model, noise_scheduler, accelerator, checkpoint_dir, epoch, loss, metrics=None, tag=None):
    """
    Saves the model checkpoint. If tag is provided, saves to checkpoint_dir/tag (e.g. 'best', 'last').
    Otherwise saves to checkpoint_dir/epoch_{epoch}.
    Only called from the main process (rank 0).
    """
    if tag:
        save_path = os.path.join(checkpoint_dir, tag)
    else:
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Attempt to save as standard pipeline if it's a UNet
        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model), 
            scheduler=noise_scheduler
        )
        pipeline.save_pretrained(save_path)
    except Exception as e:
        # Fallback: save model state dict directly if pipeline fails
        print(f"Note: Standard pipeline save failed ({e}). Saving model state dict instead.")
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_path, "model.pt"))
    
    # Save metrics metadata
    metrics_to_save = {
        "epoch": epoch,
        "loss": loss,
        "is_best": (tag == "best")
    }
    if metrics:
        metrics_to_save.update(metrics)
        
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=4)
        
    print(f"Saved checkpoint to {save_path} (Loss: {loss:.6f})")
