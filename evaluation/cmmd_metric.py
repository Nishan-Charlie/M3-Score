"""
CMMD — CLIP Maximum Mean Discrepancy
Jayasumana et al., "Rethinking FID: Towards a Better Evaluation Metric for Image Generation", 2024.

Uses CLIP ViT-L/14 CLS embeddings with an unbiased Gaussian-RBF MMD² estimator
and median-heuristic bandwidth, exactly as in the original paper.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from typing import List, Optional, Union


_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


class CMMDMetric:
    """
    Computes CMMD (CLIP-MMD²) between two image collections.

    Usage:
        metric = CMMDMetric(device="cuda")
        score  = metric.compute_from_paths(real_paths, gen_paths)
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = _CLIP_MODEL_NAME,
        batch_size: int = 64,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device     = device
        self.batch_size = batch_size
        self._model     = CLIPModel.from_pretrained(model_name)
        self._model     = self._model.vision_model.to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def extract_features(
        self,
        paths: List[str],
        desc: str = "CLIP",
    ) -> np.ndarray:
        """Extract L2-normalised CLIP ViT-L/14 CLS features. Returns (N, 1024) float32."""
        feats: list[np.ndarray] = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc=desc, leave=False):
            batch_paths = paths[i : i + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            cls_feats = outputs.pooler_output  # (B, 1024) – already after LayerNorm
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            feats.append(cls_feats.cpu().float().numpy())
        return np.concatenate(feats, axis=0)

    # ------------------------------------------------------------------
    # Core MMD² estimator
    # ------------------------------------------------------------------

    @staticmethod
    def gaussian_mmd2_unbiased(
        X: np.ndarray,
        Y: np.ndarray,
        sigma: Optional[float] = None,
    ) -> float:
        """
        Unbiased Gaussian-RBF MMD² with median-heuristic bandwidth.

        Matches the estimator in Jayasumana et al. (2024):
            MMD²(P,Q) = E[k(x,x')] - 2 E[k(x,y)] + E[k(y,y')]
        where diagonal terms are excluded from the self-kernel expectations.
        """
        X_t = torch.from_numpy(X).float()
        Y_t = torch.from_numpy(Y).float()

        if sigma is None:
            # Median heuristic on the joint sample (standard practice)
            Z = torch.cat([X_t, Y_t], dim=0)
            sq_dists = torch.cdist(Z, Z, p=2) ** 2
            # Exclude zero diagonal
            mask = ~torch.eye(len(Z), dtype=torch.bool)
            sigma = (sq_dists[mask].median().item() / 2.0) ** 0.5
            sigma = max(sigma, 1e-6)

        gamma = 1.0 / (2.0 * sigma ** 2)

        def rbf(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return (-gamma * torch.cdist(A, B, p=2) ** 2).exp()

        K_XX = rbf(X_t, X_t)
        K_YY = rbf(Y_t, Y_t)
        K_XY = rbf(X_t, Y_t)

        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = (
            (K_XX.sum() - K_XX.diagonal().sum()) / (m * (m - 1))
            + (K_YY.sum() - K_YY.diagonal().sum()) / (n * (n - 1))
            - 2.0 * K_XY.mean()
        )
        return float(mmd2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_from_paths(
        self,
        real_paths: List[str],
        gen_paths:  List[str],
    ) -> float:
        """Compute CMMD between two path lists. Returns MMD² (lower = better)."""
        real_feats = self.extract_features(real_paths, desc="CLIP [real]")
        gen_feats  = self.extract_features(gen_paths,  desc="CLIP [gen]")
        return self.gaussian_mmd2_unbiased(real_feats, gen_feats)

    def compute_from_features(
        self,
        real_feats: np.ndarray,
        gen_feats:  np.ndarray,
    ) -> float:
        """Compute CMMD from pre-extracted feature arrays."""
        return self.gaussian_mmd2_unbiased(real_feats, gen_feats)
