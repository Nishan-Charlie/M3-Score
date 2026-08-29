"""
M3 Patch Scorer
================
PatchCore-style spatial quality auditor for generated MRI.

Pipeline
--------
1. Divide each image into non-overlapping P×P patches.
2. Extract RadioDino-s16 L12 features for every patch via a sliding
   window forward pass (no fine-tuning).
3. Build a memory bank of real-image patch features (subsample if large).
4. Score each generated patch as its L2 distance to the nearest neighbour
   in the real memory bank — high score = unusual / out-of-distribution.
5. Aggregate patch scores to a global scalar (mean or max).
6. Upsample the P×P score map back to the original image resolution
   via bilinear interpolation → per-pixel deviation heatmap.

Usage
-----
    scorer = M3PatchScorer(device="cuda")
    scorer.fit(real_image_tensors)          # build memory bank
    result = scorer.score(gen_image_tensor) # single image
    # result["score"]   — scalar quality score
    # result["heatmap"] — (H, W) numpy array, high = anomalous

    batch_result = scorer.score_batch(gen_tensors)
    # batch_result["scores"]   — (N,) array
    # batch_result["heatmaps"] — (N, H, W) array
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_DEFAULT_BACKBONE = "Snarcy/RadioDino-s16"
_DEFAULT_LAYER    = 12


class M3PatchScorer:
    """
    Spatial quality auditor using RadioDino-s16 patch features.

    Parameters
    ----------
    patch_size      : int   — pixels per patch (input images are 224×224 by
                              default, so patch_size=16 → 14×14 = 196 patches)
    layer           : int   — which ViT block's output to extract (default 12)
    backbone_id     : str   — HuggingFace / timm model ID
    device          : str
    max_memory_bank : int   — cap on real patches stored (subsample if exceeded)
    seed            : int   — for memory-bank subsampling
    """

    def __init__(
        self,
        patch_size:      int  = 16,
        layer:           int  = _DEFAULT_LAYER,
        backbone_id:     str  = _DEFAULT_BACKBONE,
        device:          str  = "cuda",
        max_memory_bank: int  = 50_000,
        seed:            int  = 42,
    ):
        self.patch_size      = patch_size
        self.layer           = layer
        self.device          = device
        self.max_memory_bank = max_memory_bank
        self.seed            = seed
        self._memory_bank: Optional[torch.Tensor] = None   # (M, D)
        self._img_size       = 224

        import timm
        self._model = timm.create_model(
            f"hf_hub:{backbone_id}", pretrained=True, img_size=self._img_size
        ).to(device).eval()
        self._num_layers = len(self._model.blocks)
        assert 1 <= layer <= self._num_layers, (
            f"layer={layer} out of [1, {self._num_layers}]"
        )

        self._norm = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
        # Number of patches per side after ViT patch embedding
        self._n_patches_side = self._img_size // self._model.patch_embed.patch_size[0]

    # ------------------------------------------------------------------
    # Feature extraction — returns (N, n_patches, D) patch tokens
    # ------------------------------------------------------------------

    def _extract_patch_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs : (N, 3, 224, 224) float [0, 1]
        returns: (N, n_patches, D)  — patch tokens from self.layer
        """
        imgs = self._norm(imgs).to(self.device)
        all_feats: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(len(imgs)):
                x = imgs[i:i+1]          # (1, 3, 224, 224)
                # forward through patch embed + positional embed
                x = self._model.patch_embed(x)
                if self._model.cls_token is not None:
                    cls = self._model.cls_token.expand(1, -1, -1)
                    x   = torch.cat([cls, x], dim=1)
                x = x + self._model.pos_embed
                x = self._model.pos_drop(x)
                # run through blocks up to self.layer
                for blk in self._model.blocks[:self.layer]:
                    x = blk(x)
                # drop CLS token → (1, n_patches, D)
                patch_tokens = x[:, 1:, :]
                all_feats.append(patch_tokens.cpu())

        return torch.cat(all_feats, dim=0)   # (N, n_patches, D)

    # ------------------------------------------------------------------
    # Memory bank
    # ------------------------------------------------------------------

    def fit(self, real_imgs: torch.Tensor, batch_size: int = 32) -> None:
        """
        Build the real-patch memory bank.

        real_imgs : (N, 3, 224, 224) float [0, 1]
        """
        all_patches: list[torch.Tensor] = []
        for start in range(0, len(real_imgs), batch_size):
            batch = real_imgs[start:start + batch_size]
            feats = self._extract_patch_features(batch)  # (B, P, D)
            B, P, D = feats.shape
            all_patches.append(feats.reshape(B * P, D))

        bank = torch.cat(all_patches, dim=0)   # (N*P, D)

        # Subsample if too large
        if len(bank) > self.max_memory_bank:
            rng  = torch.Generator().manual_seed(self.seed)
            idx  = torch.randperm(len(bank), generator=rng)[:self.max_memory_bank]
            bank = bank[idx]

        # L2-normalise for cosine-distance scoring
        bank = F.normalize(bank, dim=-1)
        self._memory_bank = bank.to(self.device)
        print(f"Memory bank: {len(self._memory_bank)} patches, dim={bank.shape[1]}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _patch_distances(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """
        patch_feats : (n_patches, D) on device
        returns     : (n_patches,)  L2 distance to nearest real patch
        """
        assert self._memory_bank is not None, "Call fit() first."
        pf = F.normalize(patch_feats, dim=-1)        # (P, D)
        # Cosine similarity → (P, M), then convert to distance
        sim   = pf @ self._memory_bank.t()            # (P, M)
        dist  = 1.0 - sim.max(dim=1).values           # (P,)  ∈ [0, 2]
        return dist

    def score(self, img: torch.Tensor) -> dict:
        """
        Score a single generated image.

        img : (3, H, W) or (1, 3, H, W) float [0, 1]
        Returns dict with keys: score (float), heatmap (H×W ndarray)
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = F.interpolate(img, size=(self._img_size, self._img_size),
                            mode="bilinear", align_corners=False)
        feats = self._extract_patch_features(img)     # (1, P, D)
        patch_feats = feats[0].to(self.device)        # (P, D)
        dist = self._patch_distances(patch_feats)     # (P,)

        P_side = self._n_patches_side
        score_map = dist.reshape(P_side, P_side)      # (√P, √P)

        # Upsample to original resolution
        heatmap = F.interpolate(
            score_map.unsqueeze(0).unsqueeze(0),
            size=(img.shape[-2], img.shape[-1]),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()

        return {
            "score":   float(dist.mean().item()),
            "score_max": float(dist.max().item()),
            "heatmap": heatmap,
            "patch_scores": dist.cpu().numpy(),
        }

    def score_batch(self, imgs: torch.Tensor, batch_size: int = 16) -> dict:
        """
        Score a batch of generated images.

        imgs : (N, 3, H, W) float [0, 1]
        Returns dict: scores (N,), heatmaps (N, H, W)
        """
        H, W   = imgs.shape[-2], imgs.shape[-1]
        scores: list[float]       = []
        heatmaps: list[np.ndarray] = []

        for start in range(0, len(imgs), batch_size):
            batch = imgs[start:start + batch_size]
            for img in batch:
                r = self.score(img)
                scores.append(r["score"])
                heatmaps.append(r["heatmap"])

        return {
            "scores":   np.array(scores),
            "heatmaps": np.stack(heatmaps, axis=0),
        }

    # ------------------------------------------------------------------
    # Convenience: compute global M3-patch score (mean NN-dist)
    # ------------------------------------------------------------------

    def global_score(self, real_imgs: torch.Tensor, gen_imgs: torch.Tensor) -> float:
        """
        Fit on real_imgs, score gen_imgs, return mean patch distance.
        Drop-in alternative to MMD-based M3 for ablation.
        """
        self.fit(real_imgs)
        result = self.score_batch(gen_imgs)
        return float(result["scores"].mean())
