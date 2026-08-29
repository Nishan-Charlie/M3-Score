"""
Bootstrap OOD AUC for M3 (BraTS real vs LIDC CT) on CPU.
Caches ref features to avoid re-extracting 500 images per split.
"""
import json, random, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.m3_score_v2 import M3EntropyMetric

RANDOM_SEED = 42
N_SPLITS    = 20
SPLIT_SIZE  = 50
IMG_SIZE    = 224
N_REF       = 200   # ref images (cached once)
N_SAMPLE    = 500
DEVICE      = "cpu"


def _collect(d, max_n=N_SAMPLE):
    d = Path(d)
    paths = sorted(p for p in d.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"})
    if max_n and len(paths) > max_n:
        rng = random.Random(RANDOM_SEED)
        paths = rng.sample(paths, max_n)
        paths = sorted(paths)
    return paths


def _load(paths):
    return [Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            for p in tqdm(paths, desc="Load", leave=False)]


def _tensor(pils):
    arrs = [np.array(p).astype(np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)


def _mmd2_rbf(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Inline unbiased RBF MMD² with 5 bandwidths."""
    XY = torch.cat([X, Y], dim=0)
    D = torch.cdist(XY, XY, p=2).pow(2)
    n, m = len(X), len(Y)
    mask = ~torch.eye(len(XY), dtype=torch.bool)
    sigma2 = float(D[mask].median().clamp(min=1e-10))
    bandwidths = [sigma2 / 2, sigma2, sigma2 * 2, sigma2 * 4, sigma2 * 8]
    mmd = 0.0
    for bw in bandwidths:
        K = torch.exp(-D / (2 * bw))
        kxx = (K[:n, :n].sum() - K[:n, :n].diagonal().sum()) / (n * (n - 1))
        kyy = (K[n:, n:].sum() - K[n:, n:].diagonal().sum()) / (m * (m - 1))
        kxy = K[:n, n:].mean()
        mmd += float(kxx - 2 * kxy + kyy)
    return mmd / len(bandwidths)


def _extract_feats(m3, tensor, batch=32):
    """Extract RadioDino L12 features, returns (N, D) on CPU."""
    m3.eval()
    parts = []
    for i in range(0, len(tensor), batch):
        chunk = tensor[i:i+batch].to(DEVICE)
        with torch.no_grad():
            raw = m3._extract_raw_features(chunk,
                    layers_to_keep=set(m3.active_layers))
        last = max(m3.active_layers)
        parts.append(raw[last - 1].cpu())
    return torch.cat(parts, dim=0)


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng_np = np.random.default_rng(RANDOM_SEED)

    print(f"Loading M3 (device={DEVICE})...")
    m3 = M3EntropyMetric(device=DEVICE)

    brats_all   = _collect(ROOT / "data_mri" / "brats_axial_multislice", max_n=N_SAMPLE * 2)
    lidc_paths  = _collect(ROOT / "output" / "generated_wdm3d" / "lidc", max_n=N_SAMPLE)
    in_paths    = brats_all[N_SAMPLE:N_SAMPLE + 200]

    print("Loading images...")
    ref_pils = _load(brats_all[:N_REF])
    in_pils  = _load(in_paths)
    ood_pils = _load(lidc_paths[:200])

    print(f"Extracting ref features ({N_REF} images, cached)...")
    ref_feats = _extract_feats(m3, _tensor(ref_pils))
    print(f"  ref_feats: {ref_feats.shape}")

    scores_in  = []
    scores_ood = []

    for i in tqdm(range(N_SPLITS), desc="Bootstrap splits"):
        in_idx  = rng_np.choice(len(in_pils),  SPLIT_SIZE, replace=False)
        ood_idx = rng_np.choice(len(ood_pils), SPLIT_SIZE, replace=False)

        in_t  = _tensor([in_pils[j]  for j in in_idx])
        ood_t = _tensor([ood_pils[j] for j in ood_idx])

        try:
            in_feats  = _extract_feats(m3, in_t)
            ood_feats = _extract_feats(m3, ood_t)
            s_in  = _mmd2_rbf(ref_feats, in_feats)
            s_ood = _mmd2_rbf(ref_feats, ood_feats)
            scores_in.append(s_in)
            scores_ood.append(s_ood)
            print(f"  split {i+1:02d}: in={s_in:.4f}  ood={s_ood:.4f}")
        except Exception as e:
            print(f"  split {i+1:02d}: FAILED — {e}")

    labels = [0] * len(scores_in) + [1] * len(scores_ood)
    all_scores = scores_in + scores_ood

    if len(set(labels)) == 2 and len(labels) >= 4:
        auc = float(roc_auc_score(labels, all_scores))
        print(f"\nOOD AUC (M3, BraTS ref vs LIDC CT):")
        print(f"  AUC         = {auc:.4f}")
        print(f"  In-dist M3  = {np.mean(scores_in):.5f} +/- {np.std(scores_in):.5f}")
        print(f"  OOD M3      = {np.mean(scores_ood):.5f} +/- {np.std(scores_ood):.5f}")

        out_path = ROOT / "results" / "ood_detection_comparison" / "results.json"
        with open(out_path) as f:
            data = json.load(f)
        # remove stale bootstrap_auc entries
        data = [r for r in data if r.get("name") not in {"bootstrap_auc", "bootstrap_auc_cpu"}]
        data.append({
            "name": "bootstrap_auc_cpu",
            "m3_auc": auc,
            "in_mean": float(np.mean(scores_in)),
            "ood_mean": float(np.mean(scores_ood)),
            "n_splits": len(scores_in),
            "split_size": SPLIT_SIZE,
        })
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved to {out_path}")
    else:
        print("Insufficient valid splits to compute AUC.")


if __name__ == "__main__":
    main()
