"""
experiments/multi_metric_comparison.py
========================================
Comprehensive multi-metric comparison: M3-Score vs FID vs KID vs CMMD
across multiple dataset pairs (BraTS, retinal, OOD).

Comparisons evaluated:
  1. BraTS_DDPM        : BraTS real  vs DDPM UNet generated (best-of-4)
  2. BraTS_WDM3D       : BraTS real  vs WDM3D generated (BraTS)
  3. BraTS_vs_LIDC_OOD : BraTS real  vs WDM3D LIDC CT (cross-modality OOD)
  4. Retinal_DDPM      : RetinaMNIST vs GS-23 retinal DDPM generated
  5. Retinal_Noise     : RetinaMNIST vs Gaussian-corrupted retinal (sigma=0.3)
  6. CXR_Noise         : PneumoniaMNIST vs Gaussian-corrupted CXR (sigma=0.3)

Metrics computed:
  - M3   : RadioDino-s16 L12 MMD² (unbiased, multi-bandwidth RBF)
  - FID  : InceptionV3 pool3 Fréchet distance
  - KID  : InceptionV3 polynomial-kernel MMD × 100
  - CMMD : CLIP ViT-L/14 unbiased MMD²

Statistical analysis (M3):
  - 500-permutation p-value
  - 200-bootstrap 95% CI
  - Effect-size Z = (M3 - mu_null) / sigma_null

Output: results/multi_metric_comparison/results.json + summary table (stdout)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.m3_score_v2 import M3EntropyMetric
from evaluation.cmmd_metric import CMMDMetric

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
N_SAMPLE      = 500
N_PERMUTE     = 500
N_BOOTSTRAP   = 200
SIGMA_NOISE   = 0.30
RANDOM_SEED   = 42
IMG_SIZE      = 224


# ─────────────────────────────────────────────────────────────────────────────
# Image loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_paths(directory: str | Path, max_n: int | None = None) -> List[Path]:
    d = Path(directory)
    paths = sorted(
        p for p in d.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if max_n and len(paths) > max_n:
        rng = random.Random(RANDOM_SEED)
        paths = rng.sample(paths, max_n)
        paths = sorted(paths)
    return paths


def _load_pils(paths: List[Path]) -> List[Image.Image]:
    imgs = []
    for p in tqdm(paths, desc="Loading", leave=False):
        imgs.append(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return imgs


def _pils_to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    """PIL list → (N, 3, H, W) float32 in [0, 1]."""
    arrs = [np.array(p).astype(np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)


def _add_gaussian_noise(pils: List[Image.Image], sigma: float, seed: int = 0) -> List[Image.Image]:
    rng = np.random.default_rng(seed)
    noisy = []
    for img in pils:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape).astype(np.float32), 0, 1)
        noisy.append(Image.fromarray((arr * 255).astype(np.uint8)))
    return noisy


# ─────────────────────────────────────────────────────────────────────────────
# FID + KID
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fid_kid(
    real_pils: List[Image.Image],
    gen_pils:  List[Image.Image],
    device:    str,
) -> Dict[str, float]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        return {"fid": float("nan"), "kid_mean": float("nan"), "kid_std": float("nan")}

    def _to_uint8(pils: List[Image.Image]) -> torch.Tensor:
        arrs = [np.array(p.convert("RGB").resize((299, 299))) for p in pils]
        return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(torch.uint8)

    subset = min(50, len(real_pils), len(gen_pils))
    fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_m = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)

    real_t = _to_uint8(real_pils).to(device)
    gen_t  = _to_uint8(gen_pils).to(device)

    BATCH = 64
    for i in range(0, len(real_t), BATCH):
        fid_m.update(real_t[i:i+BATCH], real=True)
        kid_m.update(real_t[i:i+BATCH], real=True)
    for i in range(0, len(gen_t), BATCH):
        fid_m.update(gen_t[i:i+BATCH], real=False)
        kid_m.update(gen_t[i:i+BATCH], real=False)

    fid_val       = float(fid_m.compute().item())
    kid_mean, kid_std = kid_m.compute()
    return {
        "fid":      fid_val,
        "kid_mean": float(kid_mean.item()) * 100,
        "kid_std":  float(kid_std.item()) * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M3 feature extraction (batch-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_m3_feats(m3: M3EntropyMetric, tensor: torch.Tensor, batch: int = 32) -> np.ndarray:
    """Extract L12 CLS features from (N,3,H,W) tensor. Returns (N, D) float32."""
    active = set(m3.active_layers)
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tensor), batch), desc="Extracting feats", leave=False):
            chunk = tensor[i:i+batch]
            raw = m3._extract_raw_features(chunk, layers_to_keep=active)
            # raw is list[Tensor] indexed 0..num_layers-1; active_layers are 1-indexed
            last_layer = m3.active_layers[-1]
            feat = raw[last_layer - 1]   # (B, D)
            all_feats.append(feat.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Unbiased RBF MMD² (standalone, for permutation / bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def _mmd2_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()
    joint = torch.cat([Xt, Yt])
    dists = torch.cdist(joint, joint, p=2)
    sigma = float(torch.median(dists[dists > 0]).item())
    bandwidths = [sigma / 2, sigma, sigma * 2, sigma * 4, sigma * 8]
    K = torch.zeros(len(joint), len(joint))
    for bw in bandwidths:
        K += torch.exp(-dists ** 2 / (2 * bw ** 2))
    K /= len(bandwidths)
    nx, ny = len(X), len(Y)
    Kxx, Kyy, Kxy = K[:nx, :nx], K[nx:, nx:], K[:nx, nx:]
    xx = (Kxx.sum() - Kxx.diag().sum()) / (nx * (nx - 1))
    yy = (Kyy.sum() - Kyy.diag().sum()) / (ny * (ny - 1))
    xy = Kxy.sum() / (nx * ny)
    return float((xx + yy - 2 * xy).item())


def _permutation_test(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
    n_perm: int = N_PERMUTE,
) -> Dict[str, float]:
    all_f = np.concatenate([real_feats, gen_feats])
    nx    = len(real_feats)
    obs   = _mmd2_rbf(real_feats, gen_feats)
    rng   = np.random.default_rng(RANDOM_SEED)
    null  = []
    for _ in tqdm(range(n_perm), desc="Permuting", leave=False):
        perm = rng.permutation(len(all_f))
        null.append(_mmd2_rbf(all_f[perm[:nx]], all_f[perm[nx:]]))
    null = np.array(null)
    p    = float(max((null >= obs).mean(), 1.0 / n_perm))
    z    = float((obs - null.mean()) / (null.std() + 1e-12))
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_std": float(null.std()), "p_value": p, "z_score": z}


def _bootstrap_ci(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
    n_boot: int = N_BOOTSTRAP,
) -> Dict[str, float]:
    rng    = np.random.default_rng(RANDOM_SEED + 1)
    nr, ng = len(real_feats), len(gen_feats)
    scores = []
    for _ in tqdm(range(n_boot), desc="Bootstrap", leave=False):
        ri = rng.integers(0, nr, nr)
        gi = rng.integers(0, ng, ng)
        scores.append(_mmd2_rbf(real_feats[ri], gen_feats[gi]))
    s = np.array(scores)
    return {"ci_low": float(np.percentile(s, 2.5)), "ci_high": float(np.percentile(s, 97.5)),
            "mean": float(s.mean()), "std": float(s.std())}


# ─────────────────────────────────────────────────────────────────────────────
# Single comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    name:      str,
    real_pils: List[Image.Image],
    gen_pils:  List[Image.Image],
    m3:        M3EntropyMetric,
    cmmd:      CMMDMetric,
    device:    str,
    run_stats: bool = True,
    cmmd_real_paths: Optional[List[str]] = None,
    cmmd_gen_paths:  Optional[List[str]] = None,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  {name}  (n_real={len(real_pils)}, n_gen={len(gen_pils)})")
    print(f"{'='*60}")

    result = {"name": name, "n_real": len(real_pils), "n_gen": len(gen_pils)}

    # ── M3 ───────────────────────────────────────────────────────────────────
    print("  [M3] Computing score...")
    t0 = time.time()
    try:
        real_t = _pils_to_tensor(real_pils)
        gen_t  = _pils_to_tensor(gen_pils)
        m3_out = m3(real_t, gen_t)
        m3_val = float(m3_out.get("m3_score", m3_out.get("m3_v2_final_score", float("nan"))))
        result["m3"] = m3_val
        print(f"  M3  = {m3_val:.6f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  M3 failed: {e}")
        m3_val = float("nan")
        result["m3"] = m3_val
        real_t = gen_t = None

    # ── M3 stats ─────────────────────────────────────────────────────────────
    if run_stats and not np.isnan(m3_val) and real_t is not None:
        print("  [M3] Extracting features for stats...")
        real_feats = _extract_m3_feats(m3, real_t)
        gen_feats  = _extract_m3_feats(m3, gen_t)
        result["m3_feat_dim"] = int(real_feats.shape[1])

        print(f"  [M3] Permutation test ({N_PERMUTE} perms)...")
        perm = _permutation_test(real_feats, gen_feats)
        result["m3_permutation"] = perm
        print(f"    p={perm['p_value']:.4f}  Z={perm['z_score']:.1f}")

        print(f"  [M3] Bootstrap CI ({N_BOOTSTRAP} resamples)...")
        boot = _bootstrap_ci(real_feats, gen_feats)
        result["m3_bootstrap"] = boot
        print(f"    CI=[{boot['ci_low']:.6f}, {boot['ci_high']:.6f}]")

    # ── FID + KID ─────────────────────────────────────────────────────────────
    print("  [FID/KID] Computing...")
    t0 = time.time()
    try:
        fk = _compute_fid_kid(real_pils, gen_pils, device)
        result.update(fk)
        print(f"  FID = {fk['fid']:.2f}  KID×100 = {fk['kid_mean']:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FID/KID failed: {e}")
        result.update({"fid": float("nan"), "kid_mean": float("nan"), "kid_std": float("nan")})

    # ── CMMD ─────────────────────────────────────────────────────────────────
    print("  [CMMD] Computing...")
    t0 = time.time()
    try:
        if cmmd_real_paths and cmmd_gen_paths:
            cmmd_val = cmmd.compute_from_paths(cmmd_real_paths, cmmd_gen_paths)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                td_p = Path(td)
                rp, gp = [], []
                for i, img in enumerate(real_pils):
                    fp = td_p / f"r_{i:04d}.png"; img.save(fp); rp.append(str(fp))
                for i, img in enumerate(gen_pils):
                    fp = td_p / f"g_{i:04d}.png"; img.save(fp); gp.append(str(fp))
                cmmd_val = cmmd.compute_from_paths(rp, gp)
        result["cmmd"] = float(cmmd_val)
        print(f"  CMMD = {cmmd_val:.6f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  CMMD failed: {e}")
        result["cmmd"] = float("nan")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table + cross-metric Spearman
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: List[Dict]) -> None:
    hdr = f"{'Comparison':<25} {'M3':>10} {'FID':>8} {'KID*100':>10} {'CMMD':>12}  {'p(M3)':>8} {'Z(M3)':>8}"
    sep = "=" * 85
    print(f"\n{sep}\nMULTI-METRIC COMPARISON SUMMARY\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        p = r.get("m3_permutation", {}).get("p_value", float("nan"))
        z = r.get("m3_permutation", {}).get("z_score",  float("nan"))
        print(
            f"{r['name']:<25} "
            f"{r.get('m3',float('nan')):>10.5f} "
            f"{r.get('fid',float('nan')):>8.2f} "
            f"{r.get('kid_mean',float('nan')):>10.4f} "
            f"{r.get('cmmd',float('nan')):>12.6f}  "
            f"{p:>8.4f} {z:>8.1f}"
        )
    print(sep)

    valid = [r for r in results
             if not any(np.isnan(r.get(k, float("nan")))
                        for k in ("m3", "fid", "kid_mean", "cmmd"))]
    if len(valid) >= 3:
        from scipy.stats import spearmanr
        m3s  = [r["m3"]       for r in valid]
        fids = [r["fid"]      for r in valid]
        kids = [r["kid_mean"] for r in valid]
        cmds = [r["cmmd"]     for r in valid]
        print(f"\nSpearman cross-metric rank correlations (n={len(valid)}):")
        for nm, vals in [("FID", fids), ("KID", kids), ("CMMD", cmds)]:
            rho, pv = spearmanr(m3s, vals)
            print(f"  M3 vs {nm:<5}: rho={rho:+.3f}  p={pv:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    device = args.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initialising M3 (RadioDino-s16, L12)...")
    m3 = M3EntropyMetric(device=device)

    print("Initialising CMMD (CLIP ViT-L/14)...")
    cmmd = CMMDMetric(device=device, batch_size=64)

    def _load_dir(d: str | None, max_n: int = N_SAMPLE):
        if not d or not Path(d).exists():
            return None, None
        paths = _collect_paths(d, max_n=max_n)
        return paths, _load_pils(paths)

    # ─── BraTS DDPM ──────────────────────────────────────────────────────────
    brats_r_paths, brats_r_pils = _load_dir(args.brats_real_dir)
    comparisons = []

    if brats_r_pils:
        # 1. BraTS vs DDPM best-of-4
        ddpm_paths, ddpm_pils = _load_dir(args.brats_ddpm_dir)
        if ddpm_pils:
            comparisons.append({
                "name": "BraTS_DDPM",
                "real_pils": brats_r_pils, "gen_pils": ddpm_pils,
                "cmmd_real": [str(p) for p in brats_r_paths],
                "cmmd_gen":  [str(p) for p in ddpm_paths],
            })

        # 2. BraTS vs WDM3D BraTS
        wdm_paths, wdm_pils = _load_dir(args.brats_wdm3d_dir)
        if wdm_pils:
            comparisons.append({
                "name": "BraTS_WDM3D",
                "real_pils": brats_r_pils, "gen_pils": wdm_pils,
                "cmmd_real": [str(p) for p in brats_r_paths],
                "cmmd_gen":  [str(p) for p in wdm_paths],
            })

        # 3. BraTS vs LIDC CT OOD
        lidc_paths, lidc_pils = _load_dir(args.lidc_wdm3d_dir)
        if lidc_pils:
            comparisons.append({
                "name": "BraTS_vs_LIDC_OOD",
                "real_pils": brats_r_pils, "gen_pils": lidc_pils,
                "cmmd_real": [str(p) for p in brats_r_paths],
                "cmmd_gen":  [str(p) for p in lidc_paths],
            })

    # ─── Retinal ─────────────────────────────────────────────────────────────
    ret_r_paths, ret_r_pils = _load_dir(args.retina_real_dir)
    if ret_r_pils:
        # 4. Retinal DDPM (if generated)
        ret_g_paths, ret_g_pils = _load_dir(args.retina_gen_dir)
        if ret_g_pils:
            comparisons.append({
                "name": "Retinal_DDPM",
                "real_pils": ret_r_pils, "gen_pils": ret_g_pils,
                "cmmd_real": [str(p) for p in ret_r_paths],
                "cmmd_gen":  [str(p) for p in ret_g_paths],
            })

        # 5. Retinal Gaussian noise proxy
        noisy_ret = _add_gaussian_noise(ret_r_pils, sigma=SIGMA_NOISE, seed=0)
        comparisons.append({
            "name": "Retinal_GaussNoise",
            "real_pils": ret_r_pils, "gen_pils": noisy_ret,
            "cmmd_real": None, "cmmd_gen": None,
        })

    # ─── CXR / Pneumonia ─────────────────────────────────────────────────────
    pneu_paths, pneu_pils = _load_dir(args.pneumonia_real_dir)
    if pneu_pils:
        noisy_pneu = _add_gaussian_noise(pneu_pils, sigma=SIGMA_NOISE, seed=1)
        comparisons.append({
            "name": "CXR_GaussNoise",
            "real_pils": pneu_pils, "gen_pils": noisy_pneu,
            "cmmd_real": None, "cmmd_gen": None,
        })

    if not comparisons:
        print("No valid comparisons — check directory arguments.")
        return

    print(f"\nRunning {len(comparisons)} comparisons with M3 + FID + KID + CMMD...")
    all_results = []

    for cfg in comparisons:
        res = run_comparison(
            name=cfg["name"],
            real_pils=cfg["real_pils"],
            gen_pils=cfg["gen_pils"],
            m3=m3,
            cmmd=cmmd,
            device=device,
            run_stats=args.run_stats,
            cmmd_real_paths=cfg.get("cmmd_real"),
            cmmd_gen_paths=cfg.get("cmmd_gen"),
        )
        all_results.append(res)
        with open(out_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  [saved]")

    _print_summary(all_results)
    print(f"\nAll results: {out_dir / 'results.json'}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats_real_dir",     default="data_mri/brats_axial_multislice")
    parser.add_argument("--brats_ddpm_dir",     default="output/generated_500_best")
    parser.add_argument("--brats_wdm3d_dir",    default="output/generated_wdm3d/brats")
    parser.add_argument("--lidc_wdm3d_dir",     default="output/generated_wdm3d/lidc")
    parser.add_argument("--retina_real_dir",    default="data_mri/medmnist/retinamnist/all")
    parser.add_argument("--retina_gen_dir",     default="output/generated_retinal")
    parser.add_argument("--pneumonia_real_dir", default="data_mri/medmnist/pneumoniamnist/all")
    parser.add_argument("--output_dir",         default="results/multi_metric_comparison")
    parser.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_stats",          action="store_true", default=True)
    parser.add_argument("--no_stats",           dest="run_stats", action="store_false")
    args = parser.parse_args()
    main(args)
