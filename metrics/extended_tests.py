"""
Extended Perceptual & Statistical Tests
=========================================
Computes additional image quality metrics beyond FID/SSIM:

  • MS-SSIM        — Multi-Scale Structural Similarity (pytorch_msssim)
  • LPIPS           — Learned Perceptual Image Patch Similarity (lpips)
  • Inception Score (IS) — torchmetrics
  • Coverage        — k-NN manifold coverage (Naeem et al.)
  • Density         — k-NN manifold density  (Naeem et al.)

Usage
-----
    from metrics.extended_tests import compute_extended_metrics

    results = compute_extended_metrics(
        real_dir   = "path/to/real",
        gen_dir    = "path/to/generated",
        output_dir = "path/to/results",
        device     = "cuda:0",
        use_tqdm   = True,
    )
"""

from __future__ import annotations

import os
import glob
from typing import Optional

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from pytorch_msssim import ms_ssim as _ms_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False
    print("[ExtMetrics] WARNING: pytorch_msssim not installed. MS-SSIM will be skipped.")

try:
    import lpips as _lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[ExtMetrics] WARNING: lpips not installed. LPIPS will be skipped.")

try:
    from torchmetrics.image.inception import InceptionScore
    IS_AVAILABLE = True
except ImportError:
    IS_AVAILABLE = False
    print("[ExtMetrics] WARNING: torchmetrics not installed. IS will be skipped.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_paths(directory: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = [p for p in paths if "evaluation_results" not in p and "experiments_output" not in p]
    return sorted(paths)


def _load_as_gray_tensor(paths: list[str], img_size: tuple = (256, 256)) -> torch.Tensor:
    """→ float32 tensor (N, 1, H, W) scaled [0, 1]."""
    imgs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("L").resize(img_size), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.stack(imgs)   # (N, 1, H, W)


def _load_as_rgb_tensor(paths: list[str], img_size: tuple = (256, 256)) -> torch.Tensor:
    """→ uint8 tensor (N, 3, H, W) in [0, 255] — required by IS and LPIPS."""
    imgs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB").resize(img_size), dtype=np.uint8)
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs)   # (N, 3, H, W) uint8


def _load_as_rgb_float(paths: list[str], img_size: tuple = (256, 256)) -> torch.Tensor:
    """→ float32 tensor (N, 3, H, W) in [-1, 1] for LPIPS."""
    imgs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB").resize(img_size), dtype=np.float32) / 127.5 - 1.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs)   # (N, 3, H, W)


# ---------------------------------------------------------------------------
# ResNet-18 embeddings for Coverage / Density
# ---------------------------------------------------------------------------

_EMBED_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class _EmbedNet(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.features.eval()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.to(self.device)).squeeze(-1).squeeze(-1)


def _extract_embeddings(
    paths: list[str],
    net: _EmbedNet,
    batch_size: int = 64,
    use_tqdm: bool = True,
    label: str = "",
) -> np.ndarray:
    all_feats: list[np.ndarray] = []
    itr = range(0, len(paths), batch_size)
    if use_tqdm:
        itr = tqdm(itr, desc=label)
    for s in itr:
        batch = [_EMBED_TRANSFORM(Image.open(p).convert("RGB"))
                 for p in paths[s: s + batch_size]]
        feats = net(torch.stack(batch))
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Coverage & Density (Naeem et al. NeurIPS 2020)
# ---------------------------------------------------------------------------

def _coverage_density(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
    k: int = 5,
) -> tuple[float, float]:
    """
    Coverage: fraction of real images for which at least one generated
              image falls within its k-NN ball.
    Density : mean number of generated images per real k-NN ball, normalised.
    """
    # Fit on real
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nbrs.fit(real_feats)
    real_dists, _ = nbrs.kneighbors(real_feats)
    real_radii = real_dists[:, -1]                   # (N_real,)

    # For each real sample, count generated samples inside its ball
    gen_dists, _ = nbrs.kneighbors(gen_feats)        # (N_gen, k+1)
    nearest_real_dist = gen_dists[:, 0]              # distance to nearest real

    _, nearest_real_idx = nbrs.kneighbors(gen_feats)
    nearest_idx = nearest_real_idx[:, 0]

    # Coverage: any gen inside real's ball
    inside_mask = nearest_real_dist <= real_radii[nearest_idx]

    covered_real = np.zeros(len(real_feats), dtype=bool)
    counts = np.zeros(len(real_feats), dtype=int)
    for i, (inside, idx) in enumerate(zip(inside_mask, nearest_idx)):
        if inside:
            covered_real[idx] = True
            counts[idx] += 1

    coverage = float(covered_real.mean())
    density  = float(counts.mean()) / k              # normalise by k

    return coverage, density


# ---------------------------------------------------------------------------
# MS-SSIM
# ---------------------------------------------------------------------------

def _compute_msssim(
    real_paths: list[str],
    gen_paths: list[str],
    device: str = "cpu",
    batch_size: int = 8,
    use_tqdm: bool = True,
) -> float:
    if not MSSSIM_AVAILABLE:
        return float("nan")

    real_t = _load_as_gray_tensor(real_paths[:len(gen_paths)]).to(device)
    gen_t  = _load_as_gray_tensor(gen_paths).to(device)
    num = min(len(real_t), len(gen_t))
    scores: list[float] = []
    itr = range(0, num, batch_size)
    if use_tqdm:
        itr = tqdm(itr, desc="MS-SSIM")
    for s in itr:
        r = real_t[s: s + batch_size]
        g = gen_t[s:  s + batch_size]
        val = _ms_ssim(g, r, data_range=1.0, size_average=True).item()
        scores.append(val)
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def _compute_lpips(
    real_paths: list[str],
    gen_paths: list[str],
    device: str = "cpu",
    batch_size: int = 16,
    use_tqdm: bool = True,
) -> float:
    if not LPIPS_AVAILABLE:
        return float("nan")

    loss_fn = _lpips_lib.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    real_t = _load_as_rgb_float(real_paths[:len(gen_paths)]).to(device)
    gen_t  = _load_as_rgb_float(gen_paths).to(device)
    num = min(len(real_t), len(gen_t))
    scores: list[float] = []
    itr = range(0, num, batch_size)
    if use_tqdm:
        itr = tqdm(itr, desc="LPIPS")
    for s in itr:
        with torch.no_grad():
            val = loss_fn(real_t[s: s + batch_size], gen_t[s: s + batch_size])
        scores.append(val.mean().item())
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------

def _compute_is(
    gen_paths: list[str],
    device: str = "cpu",
    batch_size: int = 32,
    use_tqdm: bool = True,
) -> tuple[float, float]:
    if not IS_AVAILABLE:
        return float("nan"), float("nan")

    metric = InceptionScore(normalize=True).to(device)
    gen_t_uint8 = _load_as_rgb_tensor(gen_paths)   # (N, 3, H=256, W=256) uint8

    itr = range(0, len(gen_t_uint8), batch_size)
    if use_tqdm:
        itr = tqdm(itr, desc="IS")
    for s in itr:
        batch = gen_t_uint8[s: s + batch_size].to(device)
        metric.update(batch)

    is_mean, is_std = metric.compute()
    return float(is_mean.item()), float(is_std.item())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_extended_metrics(
    real_dir: str,
    gen_dir: str,
    output_dir: str = "./extended_metrics_output",
    num_images: Optional[int] = None,
    k: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
    use_tqdm: bool = True,
) -> dict:
    """
    Compute MS-SSIM, LPIPS, Inception Score, Coverage, and Density.

    Returns
    -------
    dict with keys: ms_ssim, lpips, is_mean, is_std, coverage, density
    """
    os.makedirs(output_dir, exist_ok=True)

    real_paths = _load_paths(real_dir)
    gen_paths  = _load_paths(gen_dir)

    if num_images:
        real_paths = real_paths[:num_images]
        gen_paths  = gen_paths[:num_images]

    # Balance datasets! Manifold metrics fail unconditionally if N_real >> N_gen
    min_len = min(len(real_paths), len(gen_paths))
    if min_len > 0 and len(real_paths) != len(gen_paths):
        np.random.seed(42)
        real_paths = np.random.choice(real_paths, min_len, replace=False).tolist()
        gen_paths = np.random.choice(gen_paths, min_len, replace=False).tolist()

    print(f"[ExtMetrics] Real: {len(real_paths)} | Gen: {len(gen_paths)}")
    results: dict = {}

    # MS-SSIM
    print("[ExtMetrics] Computing MS-SSIM...")
    results["ms_ssim"] = _compute_msssim(real_paths, gen_paths, device, batch_size, use_tqdm)

    # LPIPS
    print("[ExtMetrics] Computing LPIPS...")
    results["lpips"] = _compute_lpips(real_paths, gen_paths, device, batch_size, use_tqdm)

    # IS
    print("[ExtMetrics] Computing Inception Score...")
    is_m, is_s = _compute_is(gen_paths, device, batch_size, use_tqdm)
    results["is_mean"] = is_m
    results["is_std"]  = is_s

    # Coverage & Density (needs embeddings)
    print("[ExtMetrics] Computing Coverage & Density...")
    embed_net = _EmbedNet(device)
    real_feats = _extract_embeddings(real_paths, embed_net, batch_size, use_tqdm, "Real embed")
    gen_feats  = _extract_embeddings(gen_paths,  embed_net, batch_size, use_tqdm, "Gen  embed")
    cov, den = _coverage_density(real_feats, gen_feats, k)
    results["coverage"] = round(cov, 6)
    results["density"]  = round(den, 6)

    # Round numeric results
    for key in ["ms_ssim", "lpips", "is_mean", "is_std"]:
        if results[key] is not None and not (isinstance(results[key], float) and np.isnan(results[key])):
            results[key] = round(results[key], 6)

    # Bar chart summary
    _save_metrics_bar(results, output_dir)

    # JSON
    out_path = os.path.join(output_dir, "extended_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n[ExtMetrics] Results:")
    for k_, v in results.items():
        print(f"  {k_:<20} {v}")
    print(f"[ExtMetrics] Saved → {out_path}")
    return results


def _save_metrics_bar(results: dict, output_dir: str):
    """Save a simple bar chart for the scalar metrics."""
    keys = [k for k, v in results.items()
            if isinstance(v, float) and not np.isnan(v) and k != "is_std"]
    vals = [results[k] for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.4), 5), dpi=120)
    colors = ["#4fc3f7", "#ef9a9a", "#a5d6a7", "#ffe082", "#ce93d8", "#80cbc4"]
    bars = ax.bar(keys, vals, color=colors[:len(keys)], edgecolor="#333", linewidth=0.7)
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=4)
    ax.set_title("Extended Metrics Summary", fontsize=13)
    ax.set_ylabel("Score")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "extended_metrics_bar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended metrics: MS-SSIM, LPIPS, IS, Coverage, Density.")
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="./extended_metrics_output")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--k",          type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--no_tqdm",    action="store_true")
    args = parser.parse_args()

    compute_extended_metrics(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
        use_tqdm=not args.no_tqdm,
    )
