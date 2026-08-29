"""
experiments/ood_detection_comparison.py
=========================================
Compares M3, FID, KID, and CMMD as OOD detectors by measuring their
discrimination between in-distribution (real BraTS) samples and
progressively more out-of-distribution generated / corrupted sets.

Experimental design:
  In-distribution reference:  BraTS real images (n=500)

  Test sets (ordered by expected OOD severity):
    1. in_dist       : BraTS real held-out (should give ~0 distance)
    2. brats_ddpm    : BraTS DDPM generated (in-domain, same modality)
    3. brats_wdm3d   : BraTS WDM3D generated (in-domain, different model)
    4. retinal_ddpm  : Retinal fundus generated (different anatomy, same modality)
    5. lidc_ct       : LIDC CT lung images (different modality + anatomy — strongest OOD)

For each (metric, test_set) pair we compute the metric score.
A good metric should give monotonically increasing scores from in_dist → lidc_ct.

Output:
  - OOD discrimination table (metric vs test set)
  - Kendall's τ for monotonicity preservation per metric
  - AUROC for binary OOD classification (in_dist vs each OOD set)
  - results/ood_detection_comparison/results.json
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
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.m3_score_v2 import M3EntropyMetric
from evaluation.cmmd_metric import CMMDMetric

N_SAMPLE    = 500
N_SPLITS    = 20      # bootstrap sub-samples for AUC estimation
RANDOM_SEED = 42
IMG_SIZE    = 224


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with multi_metric_comparison.py, copy kept small)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_paths(d: str | Path, max_n: int | None = None) -> List[Path]:
    d = Path(d)
    paths = sorted(p for p in d.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if max_n and len(paths) > max_n:
        rng = random.Random(RANDOM_SEED)
        paths = rng.sample(paths, max_n)
        paths = sorted(paths)
    return paths


def _load_pils(paths: List[Path]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            for p in tqdm(paths, desc="Loading", leave=False)]


def _pils_to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    arrs = [np.array(p).astype(np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)


def _to_uint8_299(pils: List[Image.Image]) -> torch.Tensor:
    arrs = [np.array(p.convert("RGB").resize((299, 299))) for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(torch.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-metric score functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_m3(m3: M3EntropyMetric, real_pils, gen_pils) -> float:
    real_t = _pils_to_tensor(real_pils)
    gen_t  = _pils_to_tensor(gen_pils)
    out = m3(real_t, gen_t)
    return float(out.get("m3_score", out.get("m3_v2_final_score", float("nan"))))


def _score_fid(real_pils, gen_pils, device) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return float("nan")
    fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    real_t = _to_uint8_299(real_pils).to(device)
    gen_t  = _to_uint8_299(gen_pils).to(device)
    BATCH = 64
    for i in range(0, len(real_t), BATCH):
        fid_m.update(real_t[i:i+BATCH], real=True)
    for i in range(0, len(gen_t), BATCH):
        fid_m.update(gen_t[i:i+BATCH], real=False)
    return float(fid_m.compute().item())


def _score_kid(real_pils, gen_pils, device) -> float:
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        return float("nan")
    subset = min(50, len(real_pils), len(gen_pils))
    kid_m = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)
    real_t = _to_uint8_299(real_pils).to(device)
    gen_t  = _to_uint8_299(gen_pils).to(device)
    BATCH = 64
    for i in range(0, len(real_t), BATCH):
        kid_m.update(real_t[i:i+BATCH], real=True)
    for i in range(0, len(gen_t), BATCH):
        kid_m.update(gen_t[i:i+BATCH], real=False)
    mean, _ = kid_m.compute()
    return float(mean.item()) * 100


def _score_cmmd(cmmd: CMMDMetric, real_paths, gen_paths,
                real_pils, gen_pils) -> float:
    try:
        if real_paths and gen_paths:
            return float(cmmd.compute_from_paths(
                [str(p) for p in real_paths],
                [str(p) for p in gen_paths],
            ))
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            rp, gp = [], []
            for i, img in enumerate(real_pils):
                fp = tp / f"r_{i:04d}.png"; img.save(fp); rp.append(str(fp))
            for i, img in enumerate(gen_pils):
                fp = tp / f"g_{i:04d}.png"; img.save(fp); gp.append(str(fp))
            return float(cmmd.compute_from_paths(rp, gp))
    except Exception as e:
        print(f"  CMMD error: {e}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap AUC for binary OOD classification
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _bootstrap_auc(
    m3: M3EntropyMetric,
    ref_pils:     List[Image.Image],
    in_pils:      List[Image.Image],
    ood_pils:     List[Image.Image],
    n_splits:     int = N_SPLITS,
    split_size:   int = 100,
) -> Dict[str, float]:
    """
    Estimate OOD detection AUC for each metric by bootstrap sub-sampling.

    Approach:
      In each bootstrap round, draw split_size images from (in_pils) and
      split_size images from (ood_pils). Compute metric score for
      (ref_pils, in_sample) and (ref_pils, ood_sample). The metric that
      gives higher score for OOD than in-dist is a good detector.

      AUROC is estimated from 2*n_splits (score, label) pairs.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    results = {k: {"scores_in": [], "scores_ood": []}
               for k in ("m3",)}

    ref_t = _pils_to_tensor(ref_pils)

    for _ in tqdm(range(n_splits), desc="Bootstrap AUC", leave=False):
        in_idx  = rng.choice(len(in_pils),  split_size, replace=False)
        ood_idx = rng.choice(len(ood_pils), split_size, replace=False)
        in_sample  = [in_pils[i]  for i in in_idx]
        ood_sample = [ood_pils[i] for i in ood_idx]

        in_t   = _pils_to_tensor(in_sample)
        ood_t  = _pils_to_tensor(ood_sample)

        out_in  = m3(ref_t, in_t)
        out_ood = m3(ref_t, ood_t)

        results["m3"]["scores_in"].append(
            float(out_in.get("m3_score", out_in.get("m3_v2_final_score", 0))))
        results["m3"]["scores_ood"].append(
            float(out_ood.get("m3_score", out_ood.get("m3_v2_final_score", 0))))

    aucs = {}
    for metric, d in results.items():
        scores = np.array(d["scores_in"] + d["scores_ood"])
        labels = np.array([0] * n_splits + [1] * n_splits)  # 0=in, 1=ood
        try:
            aucs[metric] = float(roc_auc_score(labels, scores))
        except Exception:
            aucs[metric] = float("nan")
    return aucs


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

    print("Initialising M3...")
    m3 = M3EntropyMetric(device=device)
    print("Initialising CMMD...")
    cmmd = CMMDMetric(device=device, batch_size=64)

    def _load(d, max_n=N_SAMPLE):
        if not d or not Path(d).exists():
            return None, None
        paths = _collect_paths(d, max_n=max_n)
        return paths, _load_pils(paths)

    # Reference: real BraTS (split into ref pool + in-dist test pool)
    brats_paths_all = _collect_paths(args.brats_real_dir, max_n=N_SAMPLE * 2)
    ref_paths = brats_paths_all[:N_SAMPLE]
    in_paths  = brats_paths_all[N_SAMPLE:N_SAMPLE * 2]
    ref_pils  = _load_pils(ref_paths)
    in_pils   = _load_pils(in_paths)

    # OOD sets ordered by expected distance from BraTS real
    ood_sets = []

    # 1. In-distribution held-out BraTS (should be ~0)
    ood_sets.append({
        "name": "BraTS_held_out",
        "severity": 0,
        "paths": in_paths,
        "pils": in_pils,
        "expected": "near-zero (same distribution)",
    })

    # 2. DDPM BraTS generated
    dp, dpils = _load(args.brats_ddpm_dir)
    if dpils:
        ood_sets.append({
            "name": "BraTS_DDPM_gen",
            "severity": 1,
            "paths": dp,
            "pils": dpils,
            "expected": "small (same modality, trained on BraTS)",
        })

    # 3. WDM3D BraTS generated
    wp, wpils = _load(args.brats_wdm3d_dir)
    if wpils:
        ood_sets.append({
            "name": "BraTS_WDM3D_gen",
            "severity": 2,
            "paths": wp,
            "pils": wpils,
            "expected": "moderate (same modality, different model)",
        })

    # 4. Retinal fundus (different anatomy, same modality type — 2D optical)
    rp, rpils = _load(args.retina_real_dir)
    if rpils:
        ood_sets.append({
            "name": "Retinal_real",
            "severity": 3,
            "paths": rp,
            "pils": rpils,
            "expected": "large (different anatomy: fundus vs brain)",
        })

    # 5. LIDC CT — strongest OOD (different modality + anatomy)
    lp, lpils = _load(args.lidc_wdm3d_dir)
    if lpils:
        ood_sets.append({
            "name": "LIDC_CT_gen",
            "severity": 4,
            "paths": lp,
            "pils": lpils,
            "expected": "very large (CT lung — cross-modality OOD)",
        })

    print(f"\nReference: BraTS real (n={len(ref_pils)})")
    print(f"Test sets: {[s['name'] for s in ood_sets]}")

    all_results = []

    for ood in ood_sets:
        name = ood["name"]
        print(f"\n── {name} ─────────────────────────────────────────────────")
        gen_pils  = ood["pils"]
        gen_paths = ood.get("paths")

        row = {
            "name":      name,
            "severity":  ood["severity"],
            "expected":  ood["expected"],
            "n_ref":     len(ref_pils),
            "n_gen":     len(gen_pils),
        }

        t0 = time.time()
        try:
            row["m3"]  = _score_m3(m3, ref_pils, gen_pils)
            print(f"  M3   = {row['m3']:.5f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  M3 failed: {e}"); row["m3"] = float("nan")

        t0 = time.time()
        try:
            row["fid"] = _score_fid(ref_pils, gen_pils, device)
            print(f"  FID  = {row['fid']:.2f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FID failed: {e}"); row["fid"] = float("nan")

        t0 = time.time()
        try:
            row["kid"] = _score_kid(ref_pils, gen_pils, device)
            print(f"  KID  = {row['kid']:.4f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  KID failed: {e}"); row["kid"] = float("nan")

        t0 = time.time()
        try:
            row["cmmd"] = _score_cmmd(cmmd, ref_paths, gen_paths,
                                      ref_pils, gen_pils)
            print(f"  CMMD = {row['cmmd']:.6f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  CMMD failed: {e}"); row["cmmd"] = float("nan")

        all_results.append(row)
        with open(out_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Monotonicity (Kendall τ) ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("OOD DETECTION SUMMARY")
    print("="*70)
    sevs = [r["severity"] for r in all_results]

    header = f"{'Set':<22} {'Sev':>4} {'M3':>10} {'FID':>8} {'KID×100':>9} {'CMMD':>10}"
    print(header)
    print("-"*70)
    for r in all_results:
        print(
            f"{r['name']:<22} {r['severity']:>4} "
            f"{r.get('m3',float('nan')):>10.5f} "
            f"{r.get('fid',float('nan')):>8.2f} "
            f"{r.get('kid',float('nan')):>9.4f} "
            f"{r.get('cmmd',float('nan')):>10.6f}"
        )
    print("="*70)

    print("\nKendall τ (monotonicity with OOD severity ordering):")
    for metric in ("m3", "fid", "kid", "cmmd"):
        vals = [r.get(metric, float("nan")) for r in all_results]
        valid = [(s, v) for s, v in zip(sevs, vals) if not np.isnan(v)]
        if len(valid) >= 3:
            s_arr = [x[0] for x in valid]
            v_arr = [x[1] for x in valid]
            tau, pv = kendalltau(s_arr, v_arr)
            print(f"  {metric.upper():<5}: tau={tau:+.3f}  p={pv:.4f}  "
                  f"({'MONOTONE' if tau > 0.6 else 'partial' if tau > 0 else 'FAILS'})")

    # AUC estimation for in vs OOD binary detection
    if len(ood_sets) >= 2 and in_pils and len(in_pils) >= 100:
        print("\nOOD AUC (M3, bootstrap, ref vs strongest OOD = LIDC CT):")
        ood_strong = next((s for s in ood_sets if "LIDC" in s["name"]), None)
        if ood_strong and len(ood_strong["pils"]) >= 100:
            aucs = _bootstrap_auc(
                m3,
                ref_pils=ref_pils[:200],
                in_pils=in_pils[:min(200, len(in_pils))],
                ood_pils=ood_strong["pils"][:200],
                n_splits=N_SPLITS,
                split_size=50,
            )
            for metric, auc in aucs.items():
                print(f"  {metric.upper()}: AUC={auc:.4f}")
            all_results.append({"name": "bootstrap_auc", **aucs})

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir / 'results.json'}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats_real_dir",  default="data_mri/brats_axial_multislice")
    parser.add_argument("--brats_ddpm_dir",  default="output/generated_500_best")
    parser.add_argument("--brats_wdm3d_dir", default="output/generated_wdm3d/brats")
    parser.add_argument("--lidc_wdm3d_dir",  default="output/generated_wdm3d/lidc")
    parser.add_argument("--retina_real_dir", default="data_mri/medmnist/retinamnist/all")
    parser.add_argument("--output_dir",      default="results/ood_detection_comparison")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
