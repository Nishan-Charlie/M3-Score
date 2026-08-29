"""
rad_fid.py
============
Computes the RaDDINO-based Fréchet Distance (Rad-FID) between real and generated
MRI distributions using a fine-tuned medical foundation model (RaDDINO).

Design:
  - Extracts CLS token embeddings (768-dim) from fine-tuned RaDDINO backbone
  - Applies PCA dimensionality reduction before Fréchet distance (avoids
    covariance underestimation when N_samples < feature_dim)
  - Reports both the Rad-FID and a real-vs-real baseline split (lower bound)

Why PCA?
  The 768-dim covariance matrix needs >> 768 samples to be well-determined.
  With 500–1000 images, raw covariance is rank-deficient → inflated scores.
  PCA to n_components (default 64) follows the original FID paper's convention
  of using the 2048→pooled InceptionV3 embedding, and is the standard approach
  for high-dim FID variants.

Interpretation:
  - Rad-FID baseline (real vs real split): ~0–10 typical
  - Rad-FID generated vs real: lower is better
  - "Good" generated Rad-FID: within 2–3× of the real-vs-real baseline
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy import linalg
from tqdm.auto import tqdm
from transformers import AutoModel, AutoImageProcessor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, directory, processor, n=None):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        self.paths = []
        for ext in exts:
            self.paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        self.paths = sorted(self.paths)
        if n:
            self.paths = self.paths[:n]
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            return pixel_values
        except Exception:
            return torch.zeros(3, 518, 518)


# ---------------------------------------------------------------------------
# Fréchet Distance (raw)
# ---------------------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} in FID calculation")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_embeddings(directory, model, processor, num_images, batch_size, device, use_tqdm=True):
    model.eval()
    dataset = ImageFolderDataset(directory, processor, n=num_images)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    embeddings = []
    pbar = tqdm(loader, desc=f"Rad-FID features ({os.path.basename(directory)})", disable=not use_tqdm)
    for batch in pbar:
        batch = batch.to(device)
        outputs = model(batch)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        embeddings.append(cls_token.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def fit_pca(X, n_components):
    """Fit PCA on X (N, D) and return (components, mean)."""
    mu = X.mean(axis=0)
    X_c = X - mu
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components]          # (n_components, D)
    return components, mu


def apply_pca(X, components, mu):
    """Project X onto PCA components."""
    return (X - mu) @ components.T          # (N, n_components)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rad_fid(
    real_dir: str,
    gen_dir: str,
    checkpoint_path: str,
    num_images: int = 1000,
    batch_size: int = 16,
    device: str = "cuda",
    use_tqdm: bool = True,
    n_pca_components: int = 64,
) -> dict:
    """
    Compute Rad-FID using the fine-tuned RaDDINO backbone.

    Returns a dict with:
      - rad_fid          : Rad-FID of generated vs real (PCA-reduced)
      - rad_fid_baseline : Rad-FID of real-vs-real split (lower bound for reference)
      - n_real           : number of real images used
      - n_gen            : number of generated images used
      - n_pca_components : PCA dimension used
    """
    print(f"  [Rad-FID] Loading fine-tuned backbone from: {checkpoint_path}")
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 1. Extract embeddings
    print("  [Rad-FID] Extracting real embeddings...")
    real_embeds = get_embeddings(real_dir, model, processor, num_images, batch_size, device, use_tqdm)
    print("  [Rad-FID] Extracting generated embeddings...")
    gen_embeds  = get_embeddings(gen_dir,  model, processor, num_images, batch_size, device, use_tqdm)

    n_real = len(real_embeds)
    n_gen  = len(gen_embeds)
    print(f"  [Rad-FID] Embeddings: real={n_real}, gen={n_gen}, dim={real_embeds.shape[1]}")

    # 2. PCA — fit on real, apply to both
    #    Cap n_pca_components to avoid overdetermination
    n_pca = min(n_pca_components, n_real - 1, n_gen - 1, real_embeds.shape[1])
    print(f"  [Rad-FID] Applying PCA: {real_embeds.shape[1]}-dim → {n_pca}-dim")
    pca_components, pca_mean = fit_pca(real_embeds, n_pca)

    real_proj = apply_pca(real_embeds, pca_components, pca_mean)
    gen_proj  = apply_pca(gen_embeds,  pca_components, pca_mean)

    # 3. Rad-FID: generated vs real
    mu_real    = real_proj.mean(axis=0)
    sigma_real = np.cov(real_proj, rowvar=False)
    mu_gen     = gen_proj.mean(axis=0)
    sigma_gen  = np.cov(gen_proj, rowvar=False)

    rad_fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    # 4. Real-vs-real baseline: split real into two halves
    half = n_real // 2
    real_a = real_proj[:half]
    real_b = real_proj[half:]
    mu_a, sig_a = real_a.mean(axis=0), np.cov(real_a, rowvar=False)
    mu_b, sig_b = real_b.mean(axis=0), np.cov(real_b, rowvar=False)
    baseline = calculate_frechet_distance(mu_a, sig_a, mu_b, sig_b)

    print(f"  [Rad-FID] Real-vs-Real baseline : {baseline:.4f}")
    print(f"  [Rad-FID] Generated vs Real      : {rad_fid_value:.4f}")
    print(f"  [Rad-FID] Ratio (gen/baseline)   : {rad_fid_value / max(baseline, 1e-6):.2f}×")

    return {
        "rad_fid": round(rad_fid_value, 4),
        "rad_fid_baseline_real_vs_real": round(baseline, 4),
        "rad_fid_ratio_vs_baseline": round(rad_fid_value / max(baseline, 1e-6), 3),
        "n_real": n_real,
        "n_gen": n_gen,
        "n_pca_components": n_pca,
    }
