"""
Anomaly Detection for Generated MRI Images
============================================
Two complementary detectors:

1. **Isolation Forest** (sklearn) — unsupervised, no training required.
   Fits on real image embeddings, scores generated images,
   flags high anomaly scores as suspicious.

2. **Convolutional Autoencoder** — trained on real images.
   High reconstruction MSE on generated images indicates distribution shift.

Usage
-----
    from metrics.anomaly_detection import run_anomaly_detection

    results = run_anomaly_detection(
        real_dir   = "path/to/real",
        gen_dir    = "path/to/generated",
        output_dir = "path/to/output",
        method     = "both",          # "iforest" | "autoencoder" | "both"
        device     = "cuda:0",
        top_n      = 10,              # Save top-N most anomalous images
        use_tqdm   = True,
    )
"""

from __future__ import annotations

import os
import glob
import shutil
from typing import Optional

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import IsolationForest
from torchvision import models, transforms
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EMBED_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_AE_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _load_paths(directory: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(paths)


class _EmbedNet(nn.Module):
    """ResNet-18 feature extractor (512-D) for Isolation Forest."""

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
    label: str = "Embed",
) -> np.ndarray:
    all_feats: list[np.ndarray] = []
    itr = range(0, len(paths), batch_size)
    if use_tqdm:
        itr = tqdm(itr, desc=label)
    for start in itr:
        batch = [_EMBED_TRANSFORM(Image.open(p).convert("RGB"))
                 for p in paths[start: start + batch_size]]
        feats = net(torch.stack(batch))
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Isolation Forest detector
# ---------------------------------------------------------------------------

def _run_isolation_forest(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
    contamination: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Fit IsolationForest on real, score generated.
    Returns anomaly scores, fraction flagged, and sorted indices.
    """
    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(real_feats)

    # decision_function: negative = more anomalous, positive = more normal
    raw_scores = iforest.decision_function(gen_feats)   # (N,)
    # Invert so higher = more anomalous
    anomaly_scores = -raw_scores
    predictions = iforest.predict(gen_feats)            # -1 = anomaly, 1 = normal
    is_anomaly = (predictions == -1)

    sorted_idx = np.argsort(anomaly_scores)[::-1]       # most anomalous first

    return {
        "anomaly_scores": anomaly_scores,
        "is_anomaly": is_anomaly,
        "sorted_indices": sorted_idx,
        "mean_score": float(anomaly_scores.mean()),
        "pct_anomalous": float(is_anomaly.mean()) * 100.0,
    }


# ---------------------------------------------------------------------------
# 2. Convolutional Autoencoder detector
# ---------------------------------------------------------------------------

class _ConvAutoEncoder(nn.Module):
    """Tiny convolutional autoencoder for 1-channel 64×64 images."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16,  1, 3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.Tanh(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

    @torch.no_grad()
    def reconstruction_mse(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-image MSE, shape (B,)."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=[1, 2, 3])


def _run_autoencoder(
    real_paths: list[str],
    gen_paths: list[str],
    device: str,
    epochs: int = 15,
    batch_size: int = 32,
    use_tqdm: bool = True,
) -> dict:
    """Train AE on real images; measure reconstruction error on generated."""
    from torch.utils.data import TensorDataset, DataLoader

    def _load_tensor(paths):
        imgs = [_AE_TRANSFORM(Image.open(p).convert("L")) for p in paths]
        return torch.stack(imgs)

    print("[AE] Loading real images for autoencoder training...")
    real_tensor = _load_tensor(real_paths)
    real_ds = TensorDataset(real_tensor)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=2)

    model = _ConvAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"[AE] Training autoencoder for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        itr = tqdm(real_loader, desc=f"  AE epoch {epoch:02d}/{epochs}", disable=not use_tqdm)
        for (x,) in itr:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        if use_tqdm:
            print(f"  AE epoch {epoch:02d}/{epochs} | loss={ep_loss/len(real_loader):.5f}")

    # Score generated images
    print("[AE] Scoring generated images...")
    gen_tensor = _load_tensor(gen_paths)
    all_mse: list[float] = []
    model.eval()
    for start in range(0, len(gen_tensor), batch_size):
        chunk = gen_tensor[start: start + batch_size].to(device)
        mse = model.reconstruction_mse(chunk)
        all_mse.extend(mse.cpu().tolist())

    # Score real images for reference threshold
    real_mse_list: list[float] = []
    for start in range(0, len(real_tensor), batch_size):
        chunk = real_tensor[start: start + batch_size].to(device)
        mse = model.reconstruction_mse(chunk)
        real_mse_list.extend(mse.cpu().tolist())

    gen_mse = np.array(all_mse)
    real_mse = np.array(real_mse_list)

    # Threshold = mean + 2*std of real reconstruction error
    threshold = float(real_mse.mean() + 2 * real_mse.std())
    is_anomaly = gen_mse > threshold
    sorted_idx = np.argsort(gen_mse)[::-1]

    return {
        "anomaly_scores": gen_mse,
        "is_anomaly": is_anomaly,
        "sorted_indices": sorted_idx,
        "mean_score": float(gen_mse.mean()),
        "pct_anomalous": float(is_anomaly.mean()) * 100.0,
        "threshold": threshold,
        "real_mean_mse": float(real_mse.mean()),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_anomalous_grid(
    gen_paths: list[str],
    sorted_indices: np.ndarray,
    anomaly_scores: np.ndarray,
    output_dir: str,
    top_n: int,
    prefix: str,
) -> str:
    """Save a grid of the top-N most anomalous generated images."""
    top_n = min(top_n, len(sorted_indices))
    ncols = min(top_n, 5)
    nrows = (top_n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5), dpi=100)
    axes = np.array(axes).flatten()

    for j, ax in enumerate(axes):
        if j < top_n:
            idx = sorted_indices[j]
            img = Image.open(gen_paths[idx]).convert("L")
            ax.imshow(img, cmap="gray")
            ax.set_title(f"score={anomaly_scores[idx]:.3f}", fontsize=7)
        ax.axis("off")

    plt.suptitle(f"Top-{top_n} Most Anomalous Generated Images ({prefix})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f"anomaly_samples_{prefix}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _plot_score_distribution(
    scores: np.ndarray,
    is_anomaly: np.ndarray,
    prefix: str,
    output_dir: str,
    threshold: Optional[float] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.hist(scores[~is_anomaly], bins=40, alpha=0.7, color="#4fc3f7", label="Normal")
    ax.hist(scores[ is_anomaly], bins=40, alpha=0.7, color="#ef5350", label="Anomalous")
    if threshold is not None:
        ax.axvline(threshold, color="orange", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Count")
    ax.set_title(f"Anomaly Score Distribution — {prefix}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"anomaly_score_dist_{prefix}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_anomaly_detection(
    real_dir: str,
    gen_dir: str,
    output_dir: str = "./anomaly_output",
    method: str = "both",   # "iforest" | "autoencoder" | "both"
    num_images: Optional[int] = None,
    batch_size: int = 64,
    contamination: float = 0.05,
    ae_epochs: int = 15,
    top_n: int = 10,
    device: str = "cpu",
    use_tqdm: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run anomaly detection on generated images relative to the real distribution.

    Parameters
    ----------
    method      : "iforest", "autoencoder", or "both"
    top_n       : Number of most anomalous images to save to disk.

    Returns
    -------
    dict : per-method mean_score, pct_anomalous, and output paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    real_paths = _load_paths(real_dir)
    gen_paths  = _load_paths(gen_dir)

    if num_images:
        real_paths = real_paths[:num_images]
        gen_paths  = gen_paths[:num_images]

    print(f"[Anomaly] Real: {len(real_paths)} | Gen: {len(gen_paths)}")

    results: dict = {}

    # -- Isolation Forest --------------------------------------------------
    if method in ("iforest", "both"):
        print("[Anomaly] Running Isolation Forest detector...")
        embed_net = _EmbedNet(device)
        real_feats = _extract_embeddings(real_paths, embed_net, batch_size, use_tqdm, "Real embed")
        gen_feats  = _extract_embeddings(gen_paths,  embed_net, batch_size, use_tqdm, "Gen  embed")

        iforest_res = _run_isolation_forest(real_feats, gen_feats, contamination, seed)
        grid_path  = _save_anomalous_grid(
            gen_paths, iforest_res["sorted_indices"],
            iforest_res["anomaly_scores"], output_dir, top_n, "iforest")
        dist_path  = _plot_score_distribution(
            iforest_res["anomaly_scores"], iforest_res["is_anomaly"],
            "iforest", output_dir)

        results["isolation_forest"] = {
            "mean_anomaly_score": round(iforest_res["mean_score"], 6),
            "pct_anomalous": round(iforest_res["pct_anomalous"], 2),
            "sample_grid": grid_path,
            "score_distribution": dist_path,
        }
        print(f"  IForest | mean_score={iforest_res['mean_score']:.4f} "
              f"| pct_anomalous={iforest_res['pct_anomalous']:.1f}%")

    # -- Autoencoder -------------------------------------------------------
    if method in ("autoencoder", "both"):
        print("[Anomaly] Running Autoencoder detector...")
        ae_res = _run_autoencoder(
            real_paths, gen_paths, device, ae_epochs, batch_size, use_tqdm)
        grid_path = _save_anomalous_grid(
            gen_paths, ae_res["sorted_indices"],
            ae_res["anomaly_scores"], output_dir, top_n, "autoencoder")
        dist_path = _plot_score_distribution(
            ae_res["anomaly_scores"], ae_res["is_anomaly"],
            "autoencoder", output_dir, ae_res.get("threshold"))

        results["autoencoder"] = {
            "mean_reconstruction_mse": round(ae_res["mean_score"], 6),
            "pct_anomalous": round(ae_res["pct_anomalous"], 2),
            "threshold": round(ae_res["threshold"], 6),
            "real_mean_mse": round(ae_res["real_mean_mse"], 6),
            "sample_grid": grid_path,
            "score_distribution": dist_path,
        }
        print(f"  AutoEncoder | mean_mse={ae_res['mean_score']:.5f} "
              f"| pct_anomalous={ae_res['pct_anomalous']:.1f}%")

    # Save combined report
    report_path = os.path.join(output_dir, "anomaly_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[Anomaly] Report saved → {report_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomaly detection on generated MRI images.")
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="./anomaly_output")
    parser.add_argument("--method",     choices=["iforest", "autoencoder", "both"], default="both")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--ae_epochs",  type=int, default=15)
    parser.add_argument("--top_n",      type=int, default=10)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--no_tqdm",    action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_anomaly_detection(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir,
        method=args.method,
        num_images=args.num_images,
        batch_size=args.batch_size,
        contamination=args.contamination,
        ae_epochs=args.ae_epochs,
        top_n=args.top_n,
        device=args.device,
        use_tqdm=not args.no_tqdm,
        seed=args.seed,
    )
