"""
experiments/noise_quality_ladder.py
======================================
TSTR fix: builds a known-quality-ordering ladder using Gaussian corruption
of BraTS real images at 7 noise levels.

This replaces the degenerate 4-generator TSTR set with a principled n=7
experiment where ground-truth quality ordering is σ=0 > σ=0.05 > ... > σ=0.5.

For each noise level we:
  1. Generate 500 corrupted BraTS images
  2. Compute M3, FID, KID, CMMD vs real BraTS reference
  3. Fit a ResNet-18 classifier on corrupted images, test on real BraTS
     (TSTR accuracy as downstream quality proxy)
  4. Compute Spearman ρ between metric scores and TSTR accuracy

A metric with ρ ≈ 1.0 reliably tracks downstream task quality.
M3 should outperform FID here because it operates in a medical feature space.

Output: results/noise_quality_ladder/results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.m3_score_v2 import M3EntropyMetric
from evaluation.cmmd_metric import CMMDMetric

SIGMA_LADDER = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
N_SAMPLE     = 500
RANDOM_SEED  = 42
IMG_SIZE     = 224


def _collect(d: str | Path, max_n: int = N_SAMPLE) -> List[Path]:
    d = Path(d)
    paths = sorted(p for p in d.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if max_n and len(paths) > max_n:
        rng = random.Random(RANDOM_SEED)
        paths = rng.sample(paths, max_n)
        paths = sorted(paths)
    return paths


def _load(paths: List[Path]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            for p in tqdm(paths, desc="Load", leave=False)]


def _corrupt(pils: List[Image.Image], sigma: float, seed: int = 0) -> List[Image.Image]:
    if sigma == 0.0:
        return pils
    rng = np.random.default_rng(seed)
    out = []
    for img in pils:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape).astype(np.float32), 0, 1)
        out.append(Image.fromarray((arr * 255).astype(np.uint8)))
    return out


def _to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    arrs = [np.array(p).astype(np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)


def _to_uint8_299(pils: List[Image.Image]) -> torch.Tensor:
    arrs = [np.array(p.convert("RGB").resize((299, 299))) for p in pils]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(torch.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# TSTR: train ResNet-18 on corrupted, test on real (binary: normal vs corrupted)
# ─────────────────────────────────────────────────────────────────────────────

def _tstr_accuracy(
    real_pils:  List[Image.Image],
    noisy_pils: List[Image.Image],
    device:     str,
    n_epochs:   int = 5,
) -> float:
    """
    Train a linear probe (frozen ResNet-18 pool → 2-class) on
    [real=0, corrupted=1], evaluate on held-out real images.
    Returns accuracy on held-out real (higher = corrupted closer to real).
    """
    from torchvision import models, transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Split 80/20
    n = min(len(real_pils), len(noisy_pils))
    n_train = int(0.8 * n)

    def _to_batch(pils, label, indices):
        ts = torch.stack([tfm(pils[i]) for i in indices])
        ls = torch.full((len(indices),), label, dtype=torch.long)
        return ts, ls

    train_idx = list(range(n_train))
    val_idx   = list(range(n_train, n))

    X_train = torch.cat([_to_batch(real_pils, 0, train_idx)[0],
                         _to_batch(noisy_pils, 1, train_idx)[0]])
    y_train = torch.cat([_to_batch(real_pils, 0, train_idx)[1],
                         _to_batch(noisy_pils, 1, train_idx)[1]])
    X_val = torch.cat([_to_batch(real_pils, 0, val_idx)[0],
                       _to_batch(noisy_pils, 1, val_idx)[0]])
    y_val = torch.cat([_to_batch(real_pils, 0, val_idx)[1],
                       _to_batch(noisy_pils, 1, val_idx)[1]])

    # Extract features using frozen ResNet-18
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    backbone.fc = nn.Identity()
    backbone.eval()

    with torch.no_grad():
        BATCH = 64
        train_feats = torch.cat([
            backbone(X_train[i:i+BATCH].to(device)).cpu()
            for i in range(0, len(X_train), BATCH)
        ])
        val_feats = torch.cat([
            backbone(X_val[i:i+BATCH].to(device)).cpu()
            for i in range(0, len(X_val), BATCH)
        ])

    # Linear probe
    clf = nn.Linear(512, 2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_dataset = torch.utils.data.TensorDataset(train_feats, y_train)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    clf.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(clf(xb.to(device)), yb.to(device)).backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        preds = clf(val_feats.to(device)).argmax(dim=1).cpu()
    acc = float((preds == y_val).float().mean().item())
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = args.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reference BraTS real images...")
    ref_paths = _collect(args.real_dir, max_n=N_SAMPLE * 2)
    # Split: first half = reference for metric computation; second half = TSTR test
    ref_pils  = _load(ref_paths[:N_SAMPLE])
    tstr_test = _load(ref_paths[N_SAMPLE:min(N_SAMPLE * 2, len(ref_paths))])
    ref_t     = _to_tensor(ref_pils)

    print("Initialising M3...")
    m3   = M3EntropyMetric(device=device)
    print("Initialising CMMD...")
    cmmd = CMMDMetric(device=device, batch_size=64)

    rows = []

    for sigma in SIGMA_LADDER:
        print(f"\n=== sigma={sigma:.2f} ===")
        noisy = _corrupt(ref_pils, sigma, seed=int(sigma * 100))

        row = {"sigma": sigma, "n": N_SAMPLE}

        # M3
        try:
            noisy_t = _to_tensor(noisy)
            out = m3(ref_t, noisy_t)
            row["m3"] = float(out.get("m3_score", out.get("m3_v2_final_score", float("nan"))))
            print(f"  M3   = {row['m3']:.6f}")
        except Exception as e:
            print(f"  M3 failed: {e}"); row["m3"] = float("nan")

        # FID
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            rt = _to_uint8_299(ref_pils).to(device)
            nt = _to_uint8_299(noisy).to(device)
            BATCH = 64
            for i in range(0, len(rt), BATCH):
                fid_m.update(rt[i:i+BATCH], real=True)
            for i in range(0, len(nt), BATCH):
                fid_m.update(nt[i:i+BATCH], real=False)
            row["fid"] = float(fid_m.compute().item())
            print(f"  FID  = {row['fid']:.2f}")
        except Exception as e:
            print(f"  FID failed: {e}"); row["fid"] = float("nan")

        # KID
        try:
            from torchmetrics.image.kid import KernelInceptionDistance
            kid_m = KernelInceptionDistance(subset_size=min(50, N_SAMPLE), normalize=False).to(device)
            rt = _to_uint8_299(ref_pils).to(device)
            nt = _to_uint8_299(noisy).to(device)
            BATCH = 64
            for i in range(0, len(rt), BATCH):
                kid_m.update(rt[i:i+BATCH], real=True)
            for i in range(0, len(nt), BATCH):
                kid_m.update(nt[i:i+BATCH], real=False)
            kmean, kstd = kid_m.compute()
            row["kid"] = float(kmean.item()) * 100
            print(f"  KID  = {row['kid']:.4f}")
        except Exception as e:
            print(f"  KID failed: {e}"); row["kid"] = float("nan")

        # CMMD
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                tp = Path(td)
                rp, np_ = [], []
                for i, (ri, ni) in enumerate(zip(ref_pils, noisy)):
                    rfp = tp / f"r_{i:04d}.png"; ri.save(rfp); rp.append(str(rfp))
                    nfp = tp / f"n_{i:04d}.png"; ni.save(nfp); np_.append(str(nfp))
                row["cmmd"] = float(cmmd.compute_from_paths(rp, np_))
            print(f"  CMMD = {row['cmmd']:.6f}")
        except Exception as e:
            print(f"  CMMD failed: {e}"); row["cmmd"] = float("nan")

        # TSTR
        if args.run_tstr:
            try:
                row["tstr_acc"] = _tstr_accuracy(
                    tstr_test, _corrupt(tstr_test, sigma, seed=int(sigma * 100) + 1),
                    device=device,
                )
                print(f"  TSTR_acc = {row['tstr_acc']:.4f}")
            except Exception as e:
                print(f"  TSTR failed: {e}"); row["tstr_acc"] = float("nan")

        rows.append(row)
        with open(out_dir / "results.json", "w") as f:
            json.dump(rows, f, indent=2, default=str)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("NOISE QUALITY LADDER SUMMARY")
    print("=" * 65)
    print(f"{'sigma':>6} {'M3':>10} {'FID':>8} {'KID*100':>9} {'CMMD':>10}" +
          ("  {'TSTR_acc':>9}" if args.run_tstr else ""))
    print("-" * 65)
    for r in rows:
        line = (f"{r['sigma']:>6.2f} {r.get('m3',float('nan')):>10.5f} "
                f"{r.get('fid',float('nan')):>8.2f} {r.get('kid',float('nan')):>9.4f} "
                f"{r.get('cmmd',float('nan')):>10.6f}")
        if args.run_tstr:
            line += f"  {r.get('tstr_acc',float('nan')):>9.4f}"
        print(line)
    print("=" * 65)

    # Spearman correlations: metrics vs sigma (known quality ordering)
    sigmas  = [r["sigma"] for r in rows]
    print("\nSpearman rho vs noise sigma (should be high for good metric):")
    for metric in ("m3", "fid", "kid", "cmmd"):
        vals = [r.get(metric, float("nan")) for r in rows]
        valid = [(s, v) for s, v in zip(sigmas, vals) if not np.isnan(v)]
        if len(valid) >= 4:
            s_arr = [x[0] for x in valid]
            v_arr = [x[1] for x in valid]
            rho, pv = spearmanr(s_arr, v_arr)
            print(f"  {metric.upper():<5}: rho={rho:+.4f}  p={pv:.4f}")

    if args.run_tstr:
        tstr_vals = [r.get("tstr_acc", float("nan")) for r in rows]
        for metric in ("m3", "fid", "kid", "cmmd"):
            m_vals = [r.get(metric, float("nan")) for r in rows]
            valid  = [(t, m) for t, m in zip(tstr_vals, m_vals)
                      if not np.isnan(t) and not np.isnan(m)]
            if len(valid) >= 4:
                t_arr = [x[0] for x in valid]
                m_arr = [x[1] for x in valid]
                rho, pv = spearmanr(t_arr, m_arr)
                print(f"  {metric.upper():<5} vs TSTR_acc: rho={rho:+.4f}  p={pv:.4f}")

    print(f"\nResults: {out_dir / 'results.json'}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir",    default="data_mri/brats_axial_multislice")
    parser.add_argument("--output_dir",  default="results/noise_quality_ladder")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_tstr",    action="store_true", default=True)
    parser.add_argument("--no_tstr",     dest="run_tstr", action="store_false")
    args = parser.parse_args()
    main(args)
