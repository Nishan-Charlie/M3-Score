"""
Noise Degradation Quality Ladder (replaces "Checkpoint Ranking")
================================================================
Validates M3, FID, and CMMD by showing they each correctly order images
by known quality level.

IMPORTANT: When no real checkpoint directories are provided, this experiment
does NOT simulate training checkpoints.  It applies Gaussian noise at known
sigma levels to a fixed generated image set.  The ordering sigma=0.60 (worst)
-> sigma=0.00 (best) is ground-truth quality order; a valid metric must
recover it.  This is NOT a substitute for ranking real training checkpoints.
Report labels say "sigma=X" -- NOT "epoch_NNN" -- to avoid the misleading
implication that these represent real training states.

Two modes
---------
1. **Real checkpoints** (preferred): Pass --checkpoint_gen_dir pointing to a
   directory where each sub-folder contains generated images from one checkpoint.

2. **Noise degradation ladder** (default, no real checkpoints needed):
   Applies Gaussian noise at sigma = [0.60, 0.35, 0.15, 0.00] to the
   generated image set.  Known ordering: sigma=0.60 is lowest quality,
   sigma=0.00 is highest quality.

Medical FID paradox check
-------------------------
Constructs blurred (spatially smooth, high-frequency detail removed) generated
images that look "cleaner" and may score better on FID than the originals while
clearly losing anatomical texture.  M3 should detect the regression.

Outputs
-------
  checkpoint_ranking_report.json
  checkpoint_ranking_plot.png
  fid_paradox_plot.png
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms, models

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.m3_score_v2 import M3V2Metric

_TFM_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _glob_imgs(d: str, n: Optional[int] = None) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif"):
        paths.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    paths = sorted(set(paths))
    return paths[:n] if n else paths


def _load_t(paths: list[str]) -> torch.Tensor:
    return torch.stack([_TFM_224(Image.open(p).convert("RGB")) for p in paths])


def _inc(device: str):
    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


def _inc_feats(t: torch.Tensor, inc, device: str, bs: int = 32) -> np.ndarray:
    out = []
    mn = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    st = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for i in range(0, len(t), bs):
        b = t[i:i + bs]
        b = F.interpolate(b, (299, 299), mode="bilinear", align_corners=False)
        b = (b - mn) / st
        with torch.no_grad():
            out.append(inc(b.to(device)).cpu().numpy())
    return np.concatenate(out, 0)


def _fid(fr: np.ndarray, fg: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    sr = np.cov(fr, rowvar=False) + eps * np.eye(fr.shape[1])
    sg = np.cov(fg, rowvar=False) + eps * np.eye(fg.shape[1])
    cm = sqrtm(sr @ sg)
    if np.iscomplexobj(cm): cm = cm.real
    diff = mu_r - mu_g
    return float(np.dot(diff, diff) + np.trace(sr + sg - 2 * cm))


def _cmmd(fr: np.ndarray, fg: np.ndarray) -> float:
    try:
        from evaluation.cmmd_metric import CMMDMetric
        return float(CMMDMetric.gaussian_mmd2_unbiased(fr, fg))
    except Exception:
        return float("nan")


def _m3_score(real_t: torch.Tensor, gen_t: torch.Tensor, metric: M3V2Metric) -> float:
    with torch.no_grad():
        return float(metric(real_t, gen_t)["m3_score"])


def _add_noise(t: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.randn(t.shape, generator=g)
    return (t + noise * sigma).clamp(0, 1)


def _blur(t: torch.Tensor, kernel_size: int) -> torch.Tensor:
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    sigma = ks / 6.0
    from torchvision.transforms.functional import gaussian_blur
    return torch.stack([gaussian_blur(img, [ks, ks], [sigma, sigma]) for img in t])


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_checkpoint_ranking(
    real_dir:            str,
    gen_dir:             str,
    output_dir:          str           = "./checkpoint_ranking",
    checkpoint_gen_dir:  Optional[str] = None,
    n_images:            int           = 200,
    device:              Optional[str] = None,
    seed:                int           = 42,
    backbone_id:         str           = "Snarcy/RadioDino-s16",
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    metric = M3V2Metric(device=device, backbone_id=backbone_id,
                        kernel="rbf", single_layer=12, seed=seed)
    inc = _inc(device)

    real_paths = _glob_imgs(real_dir, n_images)
    real_t = _load_t(real_paths)
    f_real = _inc_feats(real_t, inc, device)
    print(f"Real images: {len(real_t)}")

    # ------------------------------------------------------------------
    # 1. Checkpoint ranking
    # ------------------------------------------------------------------
    if checkpoint_gen_dir and os.path.isdir(checkpoint_gen_dir):
        cp_dirs = sorted(d for d in os.listdir(checkpoint_gen_dir)
                         if os.path.isdir(os.path.join(checkpoint_gen_dir, d)))
        checkpoints = [{"name": d, "paths": _glob_imgs(os.path.join(checkpoint_gen_dir, d), n_images)}
                       for d in cp_dirs]
        print(f"Using {len(checkpoints)} real checkpoint directories")
    else:
        # Simulate with noise-corrupted gen at 4 levels: early, mid, late, best
        gen_paths = _glob_imgs(gen_dir, n_images)
        if not gen_paths:
            raise FileNotFoundError(f"No images found in gen_dir={gen_dir!r}")
        gen_t_base = _load_t(gen_paths)
        N = min(len(real_t), len(gen_t_base))
        gen_t_base = gen_t_base[:N]; real_t = real_t[:N]; f_real = f_real[:N]
        print(f"Noise degradation ladder (N={N}): sigma=[0.60, 0.35, 0.15, 0.00]")
        print("  NOTE: these are NOT real training checkpoints.")
        simulated = [
            ("sigma=0.60 (heavy noise)",  _add_noise(gen_t_base, 0.60, seed)),
            ("sigma=0.35 (medium noise)", _add_noise(gen_t_base, 0.35, seed)),
            ("sigma=0.15 (light noise)",  _add_noise(gen_t_base, 0.15, seed)),
            ("sigma=0.00 (no noise)",     gen_t_base),
        ]
        checkpoints = [{"name": n, "tensor": t} for n, t in simulated]

    cp_results: list[dict] = []
    for cp in checkpoints:
        name = cp["name"]
        if "tensor" in cp:
            gen_t = cp["tensor"]
        else:
            if not cp["paths"]:
                continue
            gen_t = _load_t(cp["paths"])
        N = min(len(real_t), len(gen_t))
        if N < 10:
            continue
        rt, gt = real_t[:N], gen_t[:N]
        fr, fg = f_real[:N], _inc_feats(gt, inc, device)
        m3  = _m3_score(rt, gt, metric)
        fid = _fid(fr, fg)
        cmd = _cmmd(fr, fg)
        print(f"  {name:30s}  M3={m3:.4f}  FID={fid:.2f}  CMMD={cmd:.4f}")
        cp_results.append({"checkpoint": name, "m3": m3, "fid": fid, "cmmd": cmd, "n": N})

    # ------------------------------------------------------------------
    # 2. Medical FID paradox
    # ------------------------------------------------------------------
    print("\nMedical FID-paradox test (blur vs. originals) ...")
    gen_paths = _glob_imgs(gen_dir, n_images)
    if not gen_paths:
        raise FileNotFoundError(f"No images found in gen_dir={gen_dir!r} for FID-paradox test")
    gen_t_base = _load_t(gen_paths)
    N = min(len(real_t), len(gen_t_base))
    real_t = real_t[:N]; gen_t_base = gen_t_base[:N]; f_real = f_real[:N]

    paradox: list[dict] = []
    blur_sizes = [0, 3, 7, 11, 17]
    for ks in blur_sizes:
        gt = gen_t_base if ks == 0 else _blur(gen_t_base, ks)
        fg = _inc_feats(gt, inc, device)
        m3  = _m3_score(real_t, gt, metric)
        fid = _fid(f_real, fg)
        cmd = _cmmd(f_real, fg)
        print(f"  blur ks={ks:2d}  M3={m3:.4f}  FID={fid:.2f}  CMMD={cmd:.4f}")
        paradox.append({"blur_kernel": ks, "m3": m3, "fid": fid, "cmmd": cmd})

    # Check: does FID decrease (paradox) at mild blur while M3 increases?
    m3_0, fid_0 = paradox[0]["m3"], paradox[0]["fid"]
    m3_1, fid_1 = paradox[1]["m3"], paradox[1]["fid"]
    fid_paradox_detected = fid_1 < fid_0 and m3_1 >= m3_0
    print(f"\n  FID-paradox detected: {fid_paradox_detected} "
          f"(FID: {fid_0:.2f}->{fid_1:.2f}, M3: {m3_0:.4f}->{m3_1:.4f})")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_ranking(cp_results, output_dir)
    _plot_paradox(paradox, output_dir)

    mode = "real_checkpoints" if checkpoint_gen_dir and os.path.isdir(checkpoint_gen_dir) \
           else "noise_degradation_ladder"
    report = {
        "checkpoint_results": cp_results,
        "paradox_results":    paradox,
        "fid_paradox_detected": fid_paradox_detected,
        "n_checkpoints": len(cp_results),
        "mode": mode,
        "mode_note": (
            "Real checkpoint directories." if mode == "real_checkpoints"
            else "Gaussian noise degradation at sigma=[0.60, 0.35, 0.15, 0.00]. "
                 "NOT real training checkpoints."
        ),
    }
    rp = os.path.join(output_dir, "checkpoint_ranking_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {rp}")
    return report


def _plot_ranking(cp_results: list[dict], output_dir: str) -> None:
    if not cp_results:
        return
    names = [r["checkpoint"] for r in cp_results]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    for ax, key, label, color in zip(axes,
            ["m3", "fid", "cmmd"],
            ["M3-Score (RBF L12)", "FID (scipy)", "CMMD"],
            ["#1f77b4", "#d62728", "#2ca02c"]):
        vals = [r[key] for r in cp_results]
        ax.plot(x, vals, "o-", color=color, lw=2.0, ms=7)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Noise Degradation Ladder -- metrics should increase with sigma (lower quality)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "checkpoint_ranking_plot.png")
    fig.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  Checkpoint plot saved: {out}")


def _plot_paradox(paradox: list[dict], output_dir: str) -> None:
    if not paradox:
        return
    ks_vals = [r["blur_kernel"] for r in paradox]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    for ax, key, label, color, note in zip(axes,
            ["m3",    "fid",  "cmmd"],
            ["M3",    "FID",  "CMMD"],
            ["#1f77b4", "#d62728", "#2ca02c"],
            ["should rise with blur",
             "may FALL at mild blur (paradox)",
             "should rise with blur"]):
        vals = [r[key] for r in paradox]
        ax.plot(ks_vals, vals, "s-", color=color, lw=2.0, ms=7)
        ax.set_xlabel("Blur kernel size (px)")
        ax.set_title(f"{label} vs blur\n({note})", fontsize=10)
        ax.set_ylabel(label); ax.grid(alpha=0.3)
    fig.suptitle("Medical FID-Paradox: mild blur can lower FID while degrading anatomy", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "fid_paradox_plot.png")
    fig.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  FID-paradox plot saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",           required=True)
    p.add_argument("--gen_dir",            required=True)
    p.add_argument("--output_dir",         default="results/checkpoint_ranking")
    p.add_argument("--checkpoint_gen_dir", default=None)
    p.add_argument("--n_images",           type=int, default=200)
    p.add_argument("--device",             default=None)
    p.add_argument("--seed",               type=int, default=42)
    a = p.parse_args()
    run_checkpoint_ranking(**vars(a))
