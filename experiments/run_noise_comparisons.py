"""
experiments/run_noise_comparisons.py
======================================
Runs only the Gaussian-noise proxy comparisons (Retinal and CXR)
that failed due to CUDA state corruption in the main run.
Computes M3, FID, KID, CMMD and merges results into
results/multi_metric_comparison/results.json.

Run after retinal generation completes (output/generated_retinal/).
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.m3_score_v2 import M3EntropyMetric
from evaluation.cmmd_metric import CMMDMetric

N_SAMPLE    = 500
SIGMA_NOISE = 0.30
RANDOM_SEED = 42
IMG_SIZE    = 224


def _collect(d, max_n=N_SAMPLE):
    d = Path(d)
    paths = sorted(p for p in d.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if max_n and len(paths) > max_n:
        rng = random.Random(RANDOM_SEED)
        paths = rng.sample(paths, max_n)
        paths = sorted(paths)
    return paths


def _load(paths):
    return [Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            for p in tqdm(paths, desc="Load", leave=False)]


def _noise(pils, sigma, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for img in pils:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape).astype(np.float32), 0, 1)
        out.append(Image.fromarray((arr * 255).astype(np.uint8)))
    return out


def _tensor(pils):
    arrs = [np.array(p).astype(np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)


def _uint8_299(pils):
    arrs = [np.array(p.convert("RGB").resize((299, 299))) for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(torch.uint8)


def run_one(name, real_pils, gen_pils, m3, cmmd, device):
    print(f"\n{'='*55}\n  {name}  (n={len(real_pils)})\n{'='*55}")
    result = {"name": name, "n_real": len(real_pils), "n_gen": len(gen_pils)}

    # M3 + stats
    t0 = time.time()
    try:
        real_t = _tensor(real_pils)
        gen_t  = _tensor(gen_pils)
        out = m3(real_t, gen_t)
        result["m3"] = float(out.get("m3_score", out.get("m3_v2_final_score", float("nan"))))

        # permutation test
        from experiments.multi_metric_comparison import (
            _extract_m3_feats, _permutation_test, _bootstrap_ci
        )
        rf = _extract_m3_feats(m3, real_t)
        gf = _extract_m3_feats(m3, gen_t)
        result["m3_permutation"] = _permutation_test(rf, gf)
        result["m3_bootstrap"]   = _bootstrap_ci(rf, gf)
        print(f"  M3={result['m3']:.5f}  p={result['m3_permutation']['p_value']:.4f}  Z={result['m3_permutation']['z_score']:.1f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  M3 failed: {e}")
        result["m3"] = float("nan")

    # FID + KID
    t0 = time.time()
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        kid_m = KernelInceptionDistance(subset_size=min(50, len(real_pils)), normalize=False).to(device)
        rt = _uint8_299(real_pils).to(device)
        gt = _uint8_299(gen_pils).to(device)
        BATCH = 64
        for i in range(0, len(rt), BATCH): fid_m.update(rt[i:i+BATCH], real=True);  kid_m.update(rt[i:i+BATCH], real=True)
        for i in range(0, len(gt), BATCH): fid_m.update(gt[i:i+BATCH], real=False); kid_m.update(gt[i:i+BATCH], real=False)
        result["fid"] = float(fid_m.compute().item())
        km, ks = kid_m.compute()
        result["kid_mean"] = float(km.item()) * 100
        result["kid_std"]  = float(ks.item()) * 100
        print(f"  FID={result['fid']:.2f}  KID*100={result['kid_mean']:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FID/KID failed: {e}")
        result.update({"fid": float("nan"), "kid_mean": float("nan"), "kid_std": float("nan")})

    # CMMD
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            rp, gp = [], []
            for i, (r, g) in enumerate(zip(real_pils, gen_pils)):
                rf = tp / f"r_{i:04d}.png"; r.save(rf); rp.append(str(rf))
                gf = tp / f"g_{i:04d}.png"; g.save(gf); gp.append(str(gf))
            result["cmmd"] = float(cmmd.compute_from_paths(rp, gp))
        print(f"  CMMD={result['cmmd']:.6f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  CMMD failed: {e}")
        result["cmmd"] = float("nan")

    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading M3...")
    m3 = M3EntropyMetric(device=device)
    print("Loading CMMD...")
    cmmd = CMMDMetric(device=device, batch_size=64)

    results_path = ROOT / "results" / "multi_metric_comparison" / "results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    name_to_idx = {r["name"]: i for i, r in enumerate(all_results) if isinstance(r.get("name"), str)}

    # ── Retinal DDPM (if generated) ───────────────────────────────────────────
    ret_real_dir = ROOT / "data_mri" / "medmnist" / "retinamnist" / "all"
    ret_gen_dir  = ROOT / "output" / "generated_retinal"
    if ret_real_dir.exists() and ret_gen_dir.exists():
        gen_paths = _collect(ret_gen_dir)
        if len(gen_paths) >= 50:
            print(f"Retinal_DDPM: {len(gen_paths)} generated images found")
            real_paths = _collect(ret_real_dir)
            real_pils  = _load(real_paths)
            gen_pils   = _load(gen_paths)
            res = run_one("Retinal_DDPM", real_pils, gen_pils, m3, cmmd, device)
            if "Retinal_DDPM" in name_to_idx:
                all_results[name_to_idx["Retinal_DDPM"]] = res
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        else:
            print(f"Retinal_DDPM: only {len(gen_paths)} images — skipping (run generate_retinal.py first)")

    # ── Retinal Gaussian noise proxy ──────────────────────────────────────────
    if ret_real_dir.exists():
        real_paths = _collect(ret_real_dir)
        real_pils  = _load(real_paths)
        noisy_pils = _noise(real_pils, SIGMA_NOISE, seed=0)
        res = run_one("Retinal_GaussNoise", real_pils, noisy_pils, m3, cmmd, device)
        if "Retinal_GaussNoise" in name_to_idx:
            all_results[name_to_idx["Retinal_GaussNoise"]] = res
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── CXR / Pneumonia Gaussian noise proxy ──────────────────────────────────
    pneu_dir = ROOT / "data_mri" / "medmnist" / "pneumoniamnist" / "all"
    if pneu_dir.exists():
        real_paths = _collect(pneu_dir)
        real_pils  = _load(real_paths)
        noisy_pils = _noise(real_pils, SIGMA_NOISE, seed=1)
        res = run_one("CXR_GaussNoise", real_pils, noisy_pils, m3, cmmd, device)
        if "CXR_GaussNoise" in name_to_idx:
            all_results[name_to_idx["CXR_GaussNoise"]] = res
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL MULTI-METRIC COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Name':<25} {'M3':>10} {'FID':>8} {'KID*100':>9} {'CMMD':>12}  {'p':>7} {'Z':>7}")
    print("-" * 80)
    for r in all_results:
        if not isinstance(r.get("name"), str):
            continue
        perm = r.get("m3_permutation", {})
        print(
            f"{r['name']:<25} "
            f"{r.get('m3',float('nan')):>10.5f} "
            f"{r.get('fid',float('nan')):>8.2f} "
            f"{r.get('kid_mean',float('nan')):>9.4f} "
            f"{r.get('cmmd',float('nan')):>12.6f}  "
            f"{perm.get('p_value',float('nan')):>7.4f} "
            f"{perm.get('z_score',float('nan')):>7.1f}"
        )
    print("=" * 80)
    print(f"\nSaved -> {results_path}")


if __name__ == "__main__":
    main()
