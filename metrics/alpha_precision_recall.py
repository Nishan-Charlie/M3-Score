"""
α-Precision, β-Recall, Authenticity & Improved Precision/Recall
================================================================
Based on:
  • Naeem et al. "Reliable Fidelity and Diversity Metrics for Generative Models" (NeurIPS 2020)
  • Kynkäänniemi et al. "Improved Precision and Recall Metric for Assessing Generative Models" (NeurIPS 2019)

Usage
-----
    from metrics.alpha_precision_recall import compute_manifold_metrics

    results = compute_manifold_metrics(
        real_dir="path/to/real",
        gen_dir="path/to/generated",
        num_images=1000,
        k=5,
        device="cuda:0",
        use_tqdm=True,
    )
    # results is a dict with keys:
    #   alpha_precision, beta_recall, authenticity,
    #   improved_precision, improved_recall
"""

from __future__ import annotations

import os
import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import models, transforms
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 truncated to the pool3 feature layer (2048-D)."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # Keep everything up to (and including) avgpool
        self.feature_net = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.feature_net.eval()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)  float [0,1] → features (B, 2048)"""
        x = x.to(self.device)
        feats = self.feature_net(x)
        return feats.squeeze(-1).squeeze(-1)  # (B, 2048)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

_INCEPTION_TRANSFORMS = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_image_paths(directory: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = [p for p in paths if "evaluation_results" not in p and "experiments_output" not in p]
    return sorted(paths)


def _extract_features(
    image_paths: list[str],
    extractor: InceptionFeatureExtractor,
    batch_size: int = 32,
    use_tqdm: bool = True,
    label: str = "Extracting",
) -> np.ndarray:
    """Return (N, 2048) float32 feature matrix."""
    all_feats: list[np.ndarray] = []

    iterator = range(0, len(image_paths), batch_size)
    if use_tqdm:
        iterator = tqdm(iterator, desc=label)

    for start in iterator:
        batch_paths = image_paths[start: start + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(_INCEPTION_TRANSFORMS(img))
        batch = torch.stack(imgs)
        feats = extractor(batch)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core manifold metrics
# ---------------------------------------------------------------------------

def _build_knn(features: np.ndarray, k: int) -> tuple[np.ndarray, NearestNeighbors]:
    """Fit k-NN and return (radii, fitted_model).

    radii[i] = distance to (k+1)-th nearest neighbour of sample i
    (the sample itself counts as 1st, so we query k+1).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="euclidean")
    nbrs.fit(features)
    distances, _ = nbrs.kneighbors(features)
    radii = distances[:, -1]  # distance to k-th *other* neighbour
    return radii, nbrs


def alpha_precision(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    real_radii: np.ndarray,
    real_knn: NearestNeighbors,
) -> float:
    """Fraction of generated samples inside the real manifold (fidelity)."""
    distances, _ = real_knn.kneighbors(gen_features)
    # A generated sample is inside the real manifold if its distance
    # to its nearest real neighbour ≤ that real neighbour's radius.
    nearest_real_dist = distances[:, 0]
    # Retrieve the index of the nearest real neighbour to get its radius
    _, nearest_idx = real_knn.kneighbors(gen_features)
    nearest_idx = nearest_idx[:, 0]
    inside = nearest_real_dist <= real_radii[nearest_idx]
    return float(inside.mean())


def beta_recall(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    gen_radii: np.ndarray,
    gen_knn: NearestNeighbors,
) -> float:
    """Fraction of real samples covered by the generated manifold (diversity)."""
    distances, nearest_idx = gen_knn.kneighbors(real_features)
    nearest_gen_dist = distances[:, 0]
    nearest_idx = nearest_idx[:, 0]
    covered = nearest_gen_dist <= gen_radii[nearest_idx]
    return float(covered.mean())


def authenticity(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    real_knn: NearestNeighbors,
) -> float:
    """
    Mean authenticity score.

    A generated sample g is authentic if:
        d(g, nearest_real) < d(nearest_real, nearest_other_gen)
    i.e., g is closer to a real sample than any other generated sample is.
    """
    n_gen = len(gen_features)

    # Distance from each generated sample to its nearest real sample
    dist_gen_to_real, nearest_real_idx = real_knn.kneighbors(gen_features)
    dist_gen_to_real = dist_gen_to_real[:, 0]

    # For each real sample, find the nearest generated sample (excluding itself)
    gen_knn_for_auth = NearestNeighbors(n_neighbors=2, algorithm="ball_tree", metric="euclidean")
    gen_knn_for_auth.fit(gen_features)

    scores = np.zeros(n_gen, dtype=np.float32)
    for i in range(n_gen):
        g = gen_features[i: i + 1]
        # nearest gen OTHER than g itself → query k=2, take index 1
        dist_other_gen, _ = gen_knn_for_auth.kneighbors(g)
        dist_nearest_other_gen = dist_other_gen[0, 1] if dist_other_gen.shape[1] > 1 else np.inf
        scores[i] = float(dist_gen_to_real[i] < dist_nearest_other_gen)

    return float(scores.mean())


def improved_precision_recall(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    k: int = 5,
) -> tuple[float, float]:
    """Kynkäänniemi et al. Improved Precision & Recall."""
    real_radii, real_knn = _build_knn(real_features, k)
    gen_radii, gen_knn = _build_knn(gen_features, k)

    precision = alpha_precision(real_features, gen_features, real_radii, real_knn)
    recall = beta_recall(real_features, gen_features, gen_radii, gen_knn)
    return precision, recall


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_manifold_metrics(
    real_dir: str,
    gen_dir: str,
    num_images: Optional[int] = None,
    k: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
    use_tqdm: bool = True,
) -> dict:
    """
    Compute α-precision, β-recall, Authenticity, Improved Precision & Recall.

    Parameters
    ----------
    real_dir    : Directory of real images.
    gen_dir     : Directory of generated images.
    num_images  : If set, limit to first N images per split.
    k           : k for k-NN manifold estimation.
    batch_size  : Batch size for feature extraction.
    device      : Torch device string.
    use_tqdm    : Show progress bars.

    Returns
    -------
    dict with keys: alpha_precision, beta_recall, authenticity,
                    improved_precision, improved_recall
    """
    extractor = InceptionFeatureExtractor(device=device)

    real_paths = _load_image_paths(real_dir)
    gen_paths = _load_image_paths(gen_dir)

    if num_images:
        real_paths = real_paths[:num_images]
        gen_paths = gen_paths[:num_images]

    # Balance datasets! Manifold metrics fail unconditionally if N_real >> N_gen
    min_len = min(len(real_paths), len(gen_paths))
    if min_len > 0 and len(real_paths) != len(gen_paths):
        np.random.seed(42)
        real_paths = np.random.choice(real_paths, min_len, replace=False).tolist()
        gen_paths = np.random.choice(gen_paths, min_len, replace=False).tolist()

    if not real_paths:
        raise FileNotFoundError(f"No images found in real_dir: {real_dir}")
    if not gen_paths:
        raise FileNotFoundError(f"No images found in gen_dir: {gen_dir}")

    print(f"[α/β metrics] Real images : {len(real_paths)}")
    print(f"[α/β metrics] Gen  images : {len(gen_paths)}")

    real_feats = _extract_features(real_paths, extractor, batch_size, use_tqdm, "Real features")
    gen_feats = _extract_features(gen_paths, extractor, batch_size, use_tqdm, "Gen  features")

    # Build manifolds
    real_radii, real_knn = _build_knn(real_feats, k)
    gen_radii, gen_knn = _build_knn(gen_feats, k)

    alpha_p = alpha_precision(real_feats, gen_feats, real_radii, real_knn)
    beta_r = beta_recall(real_feats, gen_feats, gen_radii, gen_knn)
    auth = authenticity(real_feats, gen_feats, real_knn)
    imp_prec, imp_rec = improved_precision_recall(real_feats, gen_feats, k)

    results = {
        "alpha_precision": round(alpha_p, 6),
        "beta_recall": round(beta_r, 6),
        "authenticity": round(auth, 6),
        "improved_precision": round(imp_prec, 6),
        "improved_recall": round(imp_rec, 6),
    }

    print("\n[α/β metrics] Results:")
    for key, val in results.items():
        print(f"  {key:<25} {val:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Compute α-precision, β-recall, Authenticity metrics.")
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--output_dir", default="./metrics_output")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no_tqdm", action="store_true")
    args = parser.parse_args()

    results = compute_manifold_metrics(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        num_images=args.num_images,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
        use_tqdm=not args.no_tqdm,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "manifold_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {out_path}")
