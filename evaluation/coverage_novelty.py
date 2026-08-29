"""
coverage_novelty.py  --  Axes 2 & 3 of the M3-Score tri-axial framework
=========================================================================

Axis 2 -- Coverage (recall-style manifold support)
    For each real sample, find the nearest generated sample.
    Coverage = fraction of real samples that have at least one generated
    sample within an adaptive radius r_k (the k-th NN distance within the
    real-real set).  Equivalently: the fraction of the real manifold that
    is "covered" by the generator.  Low coverage => mode dropping.

Axis 3 -- Calibrated Novelty  (memorization detector)
    For each generated sample, find its nearest real sample (gen->real NN
    distance d_gr).  Compare to the real-real intra-set NN distribution
    (d_rr).  A generated sample is flagged as memorised when
        d_gr < pct_memorize-th percentile of d_rr   (default 5th pct)
    Novelty = fraction of gen samples that are NOT memorised.
    Memorization rate = 1 - Novelty.  Values near 0 => safe; near 1 =>
    generator copies training data.

Both metrics operate entirely in RadioDino-s16 L12 feature space
(cosine distance) and share a single feature-extraction pass.

Public API
----------
    result = compute_coverage_novelty(
        real_imgs, gen_imgs,
        backbone_id="Snarcy/RadioDino-s16",
        device="cuda",
        k_neighbors=5,
        pct_memorize=5,
        batch_size=32,
    )
    # result keys:
    #   coverage          -- float [0,1], higher is better
    #   memorization_rate -- float [0,1], lower is better
    #   novelty           -- float [0,1]  = 1 - memorization_rate
    #   nn_dists_rr       -- np.ndarray (n_real,)  real->real NN distances
    #   nn_dists_gr       -- np.ndarray (n_gen,)   gen->real NN distances
    #   nn_dists_rg       -- np.ndarray (n_real,)  real->gen NN distances
    #   radius_rr_k       -- float  adaptive radius (k-th pct of rr dists)

    CoverageNoveltyMetric   -- class wrapper (stateful, same backbone reused)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Optional

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_backbone(backbone_id: str, device: str):
    """Load RadioDino-s16 (timm) or any transformers ViT backbone."""
    bl = backbone_id.lower()
    if "radiodino-s16" in bl or "snarcy" in bl:
        import timm
        model = timm.create_model(f"hf_hub:{backbone_id}", pretrained=True, img_size=224)
        model = model.to(device).eval()
        return model, "timm"
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            backbone_id, output_hidden_states=True
        ).to(device).eval()
        return model, "transformers"


def _transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def _extract_features(
    images: List,          # list of PIL images or (C,H,W) tensors
    backbone,
    backend: str,
    device: str,
    batch_size: int = 32,
    layer_idx: int = -1,   # -1 => CLS from last layer
) -> np.ndarray:
    """Return L2-normalised (N, D) feature matrix."""
    tf = _transform()
    feats = []
    for i in tqdm(range(0, len(images), batch_size),
                  desc="Extracting features", leave=False):
        batch = images[i : i + batch_size]
        # accept PIL or tensor
        tensors = []
        for img in batch:
            if hasattr(img, "convert"):   # PIL
                tensors.append(tf(img.convert("RGB")))
            else:
                tensors.append(img)
        x = torch.stack(tensors).to(device)

        if backend == "timm":
            feat = backbone.forward_features(x)          # (B, tokens, D) or (B, D)
            if feat.dim() == 3:
                feat = feat[:, 0]                         # CLS token
        else:
            out  = backbone(pixel_values=x)
            feat = out.last_hidden_state[:, 0]            # CLS token

        feat = F.normalize(feat, dim=-1)
        feats.append(feat.cpu().numpy())

    return np.concatenate(feats, axis=0)   # (N, D)


# ---------------------------------------------------------------------------
# Efficient pairwise distance helpers (cosine -> L2-normalised -> L2)
# ---------------------------------------------------------------------------

def _pairwise_cosine_dists(A: np.ndarray, B: np.ndarray,
                            chunk: int = 512) -> np.ndarray:
    """
    Compute cosine distance matrix (N_A, N_B) in chunked fashion.
    A, B must be L2-normalised (unit vectors).
    cosine_dist = 1 - dot(a, b)
    """
    n_a, n_b = len(A), len(B)
    D = np.empty((n_a, n_b), dtype=np.float32)
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    for i in range(0, n_a, chunk):
        a_chunk = A_t[i : i + chunk]          # (c, d)
        sims = a_chunk @ B_t.T                # (c, n_b)
        D[i : i + len(a_chunk)] = (1.0 - sims.numpy()).clip(0)
    return D


def _knn_distances(A: np.ndarray, B: np.ndarray, k: int = 1) -> np.ndarray:
    """
    For each row in A, return the distance to its k-th nearest neighbour in B.
    Returns shape (n_A,).
    """
    D = _pairwise_cosine_dists(A, B)
    # sort each row, take k-th smallest (index k-1; 0-indexed)
    k = min(k, D.shape[1])
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def _nn1_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row in A, return distance to nearest neighbour in B (k=1)."""
    return _knn_distances(A, B, k=1)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_coverage_novelty(
    real_imgs,
    gen_imgs,
    backbone_id: str = "Snarcy/RadioDino-s16",
    device: str = "cuda",
    k_neighbors: int = 5,
    pct_memorize: float = 5.0,
    batch_size: int = 32,
    real_feats: Optional[np.ndarray] = None,
    gen_feats:  Optional[np.ndarray] = None,
) -> dict:
    """
    Compute Coverage and Calibrated Novelty metrics.

    Parameters
    ----------
    real_imgs, gen_imgs : list of PIL images (skipped if *_feats are provided)
    backbone_id         : backbone to use for feature extraction
    device              : torch device
    k_neighbors         : k for adaptive radius r_k (coverage)
    pct_memorize        : percentile of rr-dist used as memorization threshold
    real_feats, gen_feats : pre-computed (N,D) L2-normed features (optional)

    Returns
    -------
    dict with keys: coverage, novelty, memorization_rate,
                    nn_dists_rr, nn_dists_gr, nn_dists_rg, radius_rr_k
    """
    device = device if torch.cuda.is_available() else "cpu"

    # --- Feature extraction ---
    if real_feats is None or gen_feats is None:
        backbone, backend = _load_backbone(backbone_id, device)
        if real_feats is None:
            real_feats = _extract_features(real_imgs, backbone, backend,
                                           device, batch_size)
        if gen_feats is None:
            gen_feats  = _extract_features(gen_imgs,  backbone, backend,
                                           device, batch_size)

    n_real, n_gen = len(real_feats), len(gen_feats)

    # --- Real-Real k-NN distances (adaptive radius) ---
    # Exclude self-distances by computing with the full set and ignoring dist=0
    D_rr = _pairwise_cosine_dists(real_feats, real_feats)
    np.fill_diagonal(D_rr, np.inf)
    nn_dists_rr = np.sort(D_rr, axis=1)[:, k_neighbors - 1]   # k-th NN dist

    # Adaptive radius: k-th percentile of rr NN distances
    # Each real sample has its OWN radius = its k-th NN dist to other reals
    # (per-sample adaptive radius from Kynkaanniemi Precision-Recall paper)
    radii_rr = nn_dists_rr   # (n_real,)  per-sample adaptive radius

    # --- Real-Gen distances (coverage: real->gen nearest) ---
    D_rg = _pairwise_cosine_dists(real_feats, gen_feats)
    nn_dists_rg = D_rg.min(axis=1)   # (n_real,) min distance to any gen sample

    # Coverage = fraction of real samples with at least 1 gen sample inside radius
    coverage = float(np.mean(nn_dists_rg <= radii_rr))

    # --- Gen-Real distances (novelty: gen->real nearest) ---
    nn_dists_gr = D_rr_ignored = None  # reuse D_rg transpose
    D_gr = D_rg.T                       # (n_gen, n_real)
    nn_dists_gr = D_gr.min(axis=1)      # (n_gen,) min distance to any real sample

    # Memorization threshold: pct_memorize-th percentile of real-real NN dists
    # Use k=1 for the calibration distribution (nearest real-real neighbour)
    D_rr_k1 = D_rr.copy()
    D_rr_k1[D_rr_k1 == np.inf] = 0
    np.fill_diagonal(D_rr_k1, np.inf)
    nn_dists_rr_k1 = D_rr_k1.min(axis=1)   # (n_real,) k=1 real-real dists

    memorize_threshold = np.percentile(nn_dists_rr_k1, pct_memorize)
    memorization_rate  = float(np.mean(nn_dists_gr <= memorize_threshold))
    novelty            = 1.0 - memorization_rate

    # Scalar radius for reporting
    radius_rr_k = float(np.mean(radii_rr))

    return {
        "coverage":          coverage,
        "novelty":           novelty,
        "memorization_rate": memorization_rate,
        "nn_dists_rr":       nn_dists_rr_k1,
        "nn_dists_gr":       nn_dists_gr,
        "nn_dists_rg":       nn_dists_rg,
        "radius_rr_k":       radius_rr_k,
        "memorize_threshold": memorize_threshold,
        "n_real":            n_real,
        "n_gen":             n_gen,
        "k_neighbors":       k_neighbors,
        "pct_memorize":      pct_memorize,
        "backbone_id":       backbone_id,
    }


# ---------------------------------------------------------------------------
# Stateful class wrapper
# ---------------------------------------------------------------------------

class CoverageNoveltyMetric:
    """
    Stateful wrapper that loads the backbone once and caches real features.

    Usage:
        m = CoverageNoveltyMetric(device="cuda")
        m.set_real(real_imgs)                   # extract + cache real feats
        result = m.evaluate(gen_imgs)           # evaluate against cached real
    """

    def __init__(
        self,
        backbone_id: str = "Snarcy/RadioDino-s16",
        device: str = "cuda",
        k_neighbors: int = 5,
        pct_memorize: float = 5.0,
        batch_size: int = 32,
    ):
        self.backbone_id  = backbone_id
        self.device       = device if torch.cuda.is_available() else "cpu"
        self.k_neighbors  = k_neighbors
        self.pct_memorize = pct_memorize
        self.batch_size   = batch_size

        self._backbone, self._backend = _load_backbone(backbone_id, self.device)
        self._real_feats: Optional[np.ndarray] = None

    def _extract(self, imgs) -> np.ndarray:
        return _extract_features(imgs, self._backbone, self._backend,
                                 self.device, self.batch_size)

    def set_real(self, real_imgs) -> np.ndarray:
        """Extract and cache real features. Returns feature matrix."""
        self._real_feats = self._extract(real_imgs)
        return self._real_feats

    def evaluate(self, gen_imgs, gen_feats: Optional[np.ndarray] = None) -> dict:
        """Evaluate coverage + novelty for a set of generated images."""
        if self._real_feats is None:
            raise RuntimeError("Call set_real() before evaluate().")
        if gen_feats is None:
            gen_feats = self._extract(gen_imgs)
        return compute_coverage_novelty(
            real_imgs=None, gen_imgs=None,
            real_feats=self._real_feats,
            gen_feats=gen_feats,
            backbone_id=self.backbone_id,
            device=self.device,
            k_neighbors=self.k_neighbors,
            pct_memorize=self.pct_memorize,
            batch_size=self.batch_size,
        )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, os
    from experiments._shared_utils import load_pils_recursive

    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",    required=True)
    p.add_argument("--gen_dir",     required=True)
    p.add_argument("--output_dir",  default="results/coverage_novelty")
    p.add_argument("--n_images",    type=int, default=500)
    p.add_argument("--k_neighbors", type=int, default=5)
    p.add_argument("--pct_memorize",type=float, default=5.0)
    p.add_argument("--device",      default=None)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.output_dir, exist_ok=True)

    real_imgs = load_pils_recursive(a.real_dir, n=a.n_images)
    gen_imgs  = load_pils_recursive(a.gen_dir,  n=a.n_images)
    print(f"Real: {len(real_imgs)}  Gen: {len(gen_imgs)}")

    result = compute_coverage_novelty(
        real_imgs, gen_imgs,
        backbone_id=a.backbone_id,
        device=device,
        k_neighbors=a.k_neighbors,
        pct_memorize=a.pct_memorize,
    )

    # Save report (strip numpy arrays for JSON)
    report = {k: (float(v) if np.isscalar(v) else v)
              for k, v in result.items()
              if not isinstance(v, np.ndarray)}
    with open(os.path.join(a.output_dir, "coverage_novelty_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nCoverage          : {result['coverage']:.4f}")
    print(f"Novelty           : {result['novelty']:.4f}")
    print(f"Memorization rate : {result['memorization_rate']:.4f}")
    print(f"Saved -> {a.output_dir}")
