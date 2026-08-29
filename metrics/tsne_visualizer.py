"""
t-SNE Feature-Space Visualizer
================================
Extracts deep features (ResNet-18) from real and generated images and
renders a 2-D t-SNE scatter plot coloured by source (real vs generated).

Usage
-----
    from metrics.tsne_visualizer import plot_tsne

    plot_tsne(
        real_dir   = "path/to/real",
        gen_dir    = "path/to/generated",
        output_dir = "path/to/results",
        num_images = 500,
        perplexity = 30,
        n_iter     = 1000,
        device     = "cuda:0",
        use_tqdm   = True,
    )
"""

from __future__ import annotations

import os
import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import models, transforms
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Lightweight feature extractor (ResNet-18, pool layer → 512-D)
# ---------------------------------------------------------------------------

class ResNetFeatureExtractor(nn.Module):
    """ResNet-18 feature extractor (global avg-pool → 512 dims)."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Drop the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.features.eval()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.features(x)
        return out.squeeze(-1).squeeze(-1)   # (B, 512)


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_paths(directory: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(paths)


def _extract_features(
    paths: list[str],
    extractor: ResNetFeatureExtractor,
    batch_size: int = 64,
    use_tqdm: bool = True,
    label: str = "Features",
) -> np.ndarray:
    all_feats: list[np.ndarray] = []
    itr = range(0, len(paths), batch_size)
    if use_tqdm:
        itr = tqdm(itr, desc=label)
    for start in itr:
        batch_paths = paths[start: start + batch_size]
        imgs = [_TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths]
        feats = extractor(torch.stack(imgs))
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# t-SNE plot
# ---------------------------------------------------------------------------

def plot_tsne(
    real_dir: str,
    gen_dir: str,
    output_dir: str = "./tsne_output",
    num_images: Optional[int] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    batch_size: int = 64,
    device: str = "cpu",
    use_tqdm: bool = True,
    random_state: int = 42,
) -> str:
    """
    Generate and save a t-SNE scatter plot of real vs generated images.

    Returns
    -------
    str : Absolute path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = ResNetFeatureExtractor(device=device)

    real_paths = _load_paths(real_dir)
    gen_paths = _load_paths(gen_dir)

    if num_images:
        real_paths = real_paths[:num_images]
        gen_paths = gen_paths[:num_images]

    print(f"[t-SNE] Real: {len(real_paths)} | Gen: {len(gen_paths)}")

    real_feats = _extract_features(real_paths, extractor, batch_size, use_tqdm, "Real feats")
    gen_feats = _extract_features(gen_paths, extractor, batch_size, use_tqdm, "Gen feats ")

    n_real, n_gen = len(real_feats), len(gen_feats)
    all_feats = np.concatenate([real_feats, gen_feats], axis=0)
    labels = np.array(["Real"] * n_real + ["Generated"] * n_gen)

    print(f"[t-SNE] Running t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=n_iter,          # sklearn >= 1.5
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    except TypeError:
        # Older sklearn uses n_iter
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    embedding = tsne.fit_transform(all_feats)

    # ---- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    palette = {
        "Real":      ("#4fc3f7", "o", 45),   # sky blue circles
        "Generated": ("#ef9a9a", "^", 40),   # rose triangles
    }

    for grp, (color, marker, size) in palette.items():
        mask = labels == grp
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            marker=marker,
            s=size,
            alpha=0.70,
            linewidths=0,
            label=grp,
        )

    # Kernel-density contours
    try:
        from scipy.stats import gaussian_kde
        for grp, (color, _, _) in palette.items():
            mask = labels == grp
            pts = embedding[mask].T
            if pts.shape[1] > 2:
                kde = gaussian_kde(pts)
                xmin, xmax = embedding[:, 0].min() - 5, embedding[:, 0].max() + 5
                ymin, ymax = embedding[:, 1].min() - 5, embedding[:, 1].max() + 5
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors=color, alpha=0.3, linewidths=0.8)
    except Exception:
        pass  # Skip contours if KDE fails

    legend = ax.legend(
        framealpha=0.15,
        facecolor="#1a1a1a",
        edgecolor="#444",
        fontsize=12,
        markerscale=1.5,
        labelcolor="white",
    )

    ax.set_title("t-SNE: Real vs Generated MRI Feature Space",
                 color="white", fontsize=15, pad=14)
    ax.set_xlabel("t-SNE dim 1", color="#aaa", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", color="#aaa", fontsize=11)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Annotation with counts
    ax.text(
        0.01, 0.99,
        f"Real n={n_real} | Gen n={n_gen}\nperplexity={perplexity}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9, color="#aaa",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "tsne_plot.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[t-SNE] Plot saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="t-SNE visualisation of real vs generated images.")
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--output_dir", default="./tsne_output")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plot_tsne(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        device=args.device,
        use_tqdm=not args.no_tqdm,
        random_state=args.seed,
    )
