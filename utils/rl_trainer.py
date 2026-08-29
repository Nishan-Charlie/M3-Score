"""
utils/rl_trainer.py
====================
Implements a lightweight DDPO (Denoising Diffusion Policy Optimization)
fine-tuner for an already-trained DDPMPipeline.

Research basis:
  - Black et al. (2023) "Training Diffusion Models with Reinforcement Learning"
    (DDPO paper): treats each denoising step as an MDP action, uses policy
    gradient (REINFORCE with importance sampling) to maximize a reward.
  - We apply a KL-divergence penalty (via log-ratio clipping, PPO-clip style)
    to prevent reward overoptimization / collapse — a known failure mode in
    small medical dataset settings (Fan et al., DPOK 2023).

Usage (via finetune_rl.py):
    python finetune_rl.py \\
        --checkpoint_dir  ./output/output_dit/checkpoints/epoch_299 \\
        --reward_type     deep_cosine_diversity \\
        --rl_epochs       50 \\
        --device          cuda:1
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler
from tqdm.auto import tqdm

from utils.reward_system import BaseReward


class DDPOTrainer:
    """
    Lightweight DDPO fine-tuner for an unconditional DDPMPipeline.

    Strategy
    --------
    1. Perform a full forward denoising rollout (T steps) to obtain x0.
    2. Score x0 with a reward function r(x0).
    3. Compute a policy-gradient loss (REINFORCE) against log p_θ(x0),
       accumulated across the last `grad_timesteps` denoising steps.
    4. Apply a PPO-clip style KL penalty to prevent drifting too far
       from the original (frozen) policy — crucial for small MRI datasets.
    5. Save RL checkpoints to a separate sub-directory.

    Parameters
    ----------
    pipeline          : DDPMPipeline  — already-trained base pipeline.
    reward_fn         : BaseReward    — callable torch Module producing (B,) rewards.
    device            : str
    lr                : float         — RL fine-tune learning rate (typically ~1e-6).
    kl_coeff          : float         — KL penalty coefficient (default 0.05).
    clip_epsilon      : float         — PPO clip range (default 0.2).
    grad_timesteps    : int           — How many final denoising steps to differentiate.
    reward_clip       : float         — Clip reward magnitude to prevent unstable updates.
    """

    def __init__(
        self,
        pipeline: DDPMPipeline,
        reward_fn: BaseReward,
        device: str = "cuda:0",
        lr: float = 1e-6,
        kl_coeff: float = 0.05,
        clip_epsilon: float = 0.2,
        grad_timesteps: int = 10,
        reward_clip: float = 10.0,
    ):
        self.pipeline = pipeline
        self.reward_fn = reward_fn
        self.device = device
        self.kl_coeff = kl_coeff
        self.clip_epsilon = clip_epsilon
        self.grad_timesteps = grad_timesteps
        self.reward_clip = reward_clip

        self.unet = pipeline.unet.to(device)
        self.scheduler: DDPMScheduler = pipeline.scheduler

        # Freeze a reference copy of the initial (base) theta for KL penalty
        import copy
        self._ref_unet = copy.deepcopy(self.unet)
        self._ref_unet.eval()
        for p in self._ref_unet.parameters():
            p.requires_grad = False

        # Only the UNet is trainable
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout_frozen(self, batch_size: int) -> torch.Tensor:
        """
        Full denoising rollout with gradients disabled (used to get a clean
        starting noise sample x_T that we will later re-denoise with grads).
        Returns x_T (pure Gaussian noise) — the noise that starts the trajectory.
        """
        num_channels = self.unet.config.in_channels
        img_size = self.unet.config.sample_size
        if isinstance(img_size, (list, tuple)):
            h, w = img_size
        else:
            h = w = img_size

        x_T = torch.randn(
            batch_size, num_channels, h, w,
            device=self.device, dtype=torch.float32
        )
        return x_T

    def _denoise_with_grads(
        self,
        x_T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Re-run the denoising trajectory from x_T, accumulating log-probs for
        the last `self.grad_timesteps` steps (with gradients enabled).

        Returns
        -------
        x0          : torch.Tensor  — final generated image (B, C, H, W) in [-1, 1]
        log_prob    : torch.Tensor  — sum of log-probs over last grad_timesteps (B,)
        """
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        timesteps = self.scheduler.timesteps  # descending order T → 0

        T_total = len(timesteps)
        T_grad = min(self.grad_timesteps, T_total)

        x = x_T.clone()
        log_prob = torch.zeros(x.shape[0], device=self.device)

        for step_idx, t in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            use_grad = step_idx >= (T_total - T_grad)

            ctx = torch.enable_grad() if use_grad else torch.no_grad()
            with ctx:
                noise_pred = self.unet(x, t_batch, return_dict=False)[0]

                # Scheduler step (closed-form update)
                out = self.scheduler.step(noise_pred, t, x)
                x_prev = out.prev_sample          # x_{t-1}

                if use_grad:
                    # Approximate log p_θ(x_{t-1} | x_t) as -||noise_pred - ε||²
                    # where ε is the "target" noise (posterior mean direction).
                    # This is the standard DDPO log-prob surrogate.
                    alpha_prod = self.scheduler.alphas_cumprod[t].to(self.device)
                    # Recover "effective noise" that was removed
                    pred_noise_target = (x - alpha_prod.sqrt() * x_prev) / (
                        (1 - alpha_prod).sqrt() + 1e-8
                    )
                    # Gaussian log-prob: -0.5 * ||noise_pred - target||²
                    lp = -0.5 * F.mse_loss(noise_pred, pred_noise_target, reduction="none")
                    lp = lp.view(x.shape[0], -1).sum(dim=1)   # (B,)
                    log_prob = log_prob + lp

            x = x_prev.detach() if not use_grad else x_prev

        return x, log_prob

    @torch.no_grad()
    def _ref_log_prob(
        self,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-prob under the frozen reference policy (for KL penalty)."""
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        timesteps = self.scheduler.timesteps
        T_total = len(timesteps)
        T_grad = min(self.grad_timesteps, T_total)

        x = x_T.clone()
        log_prob = torch.zeros(x.shape[0], device=self.device)

        for step_idx, t in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self._ref_unet(x, t_batch, return_dict=False)[0]
            out = self.scheduler.step(noise_pred, t, x)
            x_prev = out.prev_sample

            if step_idx >= (T_total - T_grad):
                alpha_prod = self.scheduler.alphas_cumprod[t].to(self.device)
                pred_noise_target = (x - alpha_prod.sqrt() * x_prev) / (
                    (1 - alpha_prod).sqrt() + 1e-8
                )
                lp = -0.5 * F.mse_loss(noise_pred, pred_noise_target, reduction="none")
                lp = lp.view(x.shape[0], -1).sum(dim=1)
                log_prob = log_prob + lp

            x = x_prev

        return log_prob

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self, batch_size: int) -> dict:
        """
        Run one RL training step (one batch).

        Returns a dict with 'loss', 'reward_mean', 'reward_std', 'kl_penalty'.
        """
        self.unet.train()
        self.optimizer.zero_grad()

        # 1. Sample starting noise
        x_T = self._rollout_frozen(batch_size)

        # 2. Denoise with gradients (policy rollout)
        x0, log_prob = self._denoise_with_grads(x_T)

        # 3. Score generated images with reward
        with torch.no_grad():
            # Convert x0 from [-1, 1] → [0, 1] for reward functions
            x0_01 = (x0.clamp(-1, 1) + 1.0) / 2.0
            # If grayscale (1 ch), some rewards expect 3ch — handled inside reward
            rewards = self.reward_fn(x0_01)          # (B,)
            rewards = rewards.clamp(-self.reward_clip, self.reward_clip)

        # 4. KL penalty via log-ratio (PPO-clip style)
        ref_log_prob = self._ref_log_prob(x_T)
        log_ratio = log_prob - ref_log_prob          # (B,)

        # PPO-clip: ratio = exp(log_ratio), clip to [1-ε, 1+ε]
        ratio = log_ratio.exp()
        ratio_clipped = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # REINFORCE objective: maximise E[r * log π_θ]
        # = maximise E[r * ratio]  (importance-weighted)
        # PPO-clip: take the min of clipped / unclipped
        pg_objective = torch.min(
            rewards * ratio,
            rewards * ratio_clipped,
        ).mean()

        # KL penalty (penalise large divergence from ref policy)
        kl_penalty = self.kl_coeff * log_ratio.pow(2).mean()

        # Total loss (negate because we want to maximise reward)
        loss = -pg_objective + kl_penalty
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "kl_penalty": kl_penalty.item(),
            "pg_objective": pg_objective.item(),
        }

    def save(self, checkpoint_dir: str, epoch: int):
        """Save fine-tuned pipeline to checkpoint_dir/rl_epoch_{epoch}/."""
        save_path = os.path.join(checkpoint_dir, f"rl_epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        pipeline_to_save = DDPMPipeline(
            unet=self.unet,
            scheduler=self.scheduler,
        )
        pipeline_to_save.save_pretrained(save_path)
        print(f"[DDPOTrainer] Saved RL checkpoint → {save_path}")
        return save_path


def run_rl_finetuning(
    pipeline: DDPMPipeline,
    reward_fn: BaseReward,
    checkpoint_dir: str,
    device: str = "cuda:0",
    rl_epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-6,
    kl_coeff: float = 0.05,
    clip_epsilon: float = 0.2,
    grad_timesteps: int = 10,
    reward_clip: float = 10.0,
    save_every: int = 10,
    log_file: Optional[str] = None,
    use_tqdm: bool = True,
) -> list[dict]:
    """
    Top-level function to run DDPO fine-tuning.

    Parameters
    ----------
    pipeline        : Pre-loaded DDPMPipeline (trained DDPM).
    reward_fn       : Instantiated reward from reward_system.py.
    checkpoint_dir  : Where to save RL checkpoints.
    rl_epochs       : Number of gradient updates (one batch each).
    batch_size      : Images per update step.
    lr              : Learning rate for RL updates.
    kl_coeff        : KL divergence penalty weight.
    clip_epsilon    : PPO clip range.
    grad_timesteps  : How many final denoising steps to differentiate.
    reward_clip     : Max absolute reward value.
    save_every      : Save checkpoint every N epochs.
    log_file        : Optional CSV path to save training history.
    use_tqdm        : Show progress bar.

    Returns
    -------
    history : List of per-epoch stat dicts.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = DDPOTrainer(
        pipeline=pipeline,
        reward_fn=reward_fn,
        device=device,
        lr=lr,
        kl_coeff=kl_coeff,
        clip_epsilon=clip_epsilon,
        grad_timesteps=grad_timesteps,
        reward_clip=reward_clip,
    )

    history = []
    pbar = tqdm(range(rl_epochs), desc="RL Fine-tuning", disable=not use_tqdm)

    for epoch in pbar:
        stats = trainer.train_epoch(batch_size)
        stats["epoch"] = epoch
        history.append(stats)

        pbar.set_postfix({
            "reward": f"{stats['reward_mean']:.4f}",
            "loss":   f"{stats['loss']:.4f}",
            "kl":     f"{stats['kl_penalty']:.4f}",
        })

        # Save checkpoints
        if epoch % save_every == 0 or epoch == rl_epochs - 1:
            trainer.save(checkpoint_dir, epoch)

    # Save log to CSV
    if log_file:
        pd.DataFrame(history).to_csv(log_file, index=False)
        print(f"[DDPOTrainer] Log saved → {log_file}")

    return history
