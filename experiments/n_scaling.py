"""
N-Scaling / Sample Efficiency Experiment
==========================================
Shows M3 is usable at sample sizes where FID becomes unreliable or undefined.

Protocol
--------
For N in [25, 50, 100, 200, 500, 1000]:
  - Compute M3 (RBF, L12), FID (scipy), CMMD on random subsets.
  - Repeat R=10 times per N to estimate mean and std (CV).
  - Plot metric value ± 1 std and CV (std/mean × 100%) vs N.

Expected results (from the paper)
----------------------------------
- FID: unstable at N < 500, denominator-singular covariance, CV > 50% at N=50
- M3:  usable down to N ~ 50, CV < 20% (lower variance RBF kernel)
- CMMD: similar to M3

Outputs
-------
  n_scaling_report.json
  n_scaling_metrics.png   (mean ± std)
  n_scaling_cv.png        (coefficient of variation %)
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
    if len(fr) < 3 or len(fg) < 3:
        return float("nan")
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    sr = np.cov(fr, rowvar=False) + eps * np.eye(fr.shape[1])
    sg = np.cov(fg, rowvar=False) + eps * np.eye(fg.shape[1])
    try:
        cm = sqrtm(sr @ sg)
        if np.iscomplexobj(cm): cm = cm.real
        diff = mu_r - mu_g
        return float(np.dot(diff, diff) + np.trace(sr + sg - 2 * cm))
    except Exception:
        return float("nan")


def _cmmd(fr: np.ndarray, fg: np.ndarray) -> float:
    try:
        from evaluation.cmmd_metric import CMMDMetric
        return float(CMMDMetric.gaussian_mmd2_unbiased(fr, fg))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_n_scaling(
    real_dir:    str,
    gen_dir:     str,
    output_dir:  str           = "./n_scaling",
    n_list:      Optional[list[int]] = None,
    n_repeats:   int           = 10,
    device:      Optional[str] = None,
    seed:        int           = 42,
    backbone_id: str           = "Snarcy/RadioDino-s16",
) -> dict:
    device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_list  = n_list or [25, 50, 100, 200, 500]
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    metric = M3V2Metric(device=device, backbone_id=backbone_id,
                        kernel="rbf", single_layer=12, seed=seed)
    inc    = _inc(device)

    # Load all images up-front
    max_n = max(n_list) * 2   # need 2× for real/gen split per repeat
    real_paths = _glob_imgs(real_dir, max_n)
    gen_paths  = _glob_imgs(gen_dir, max_n)
    N_avail = min(len(real_paths), len(gen_paths))
    print(f"Available: real={len(real_paths)}  gen={len(gen_paths)}")

    orig_n_list = list(n_list)
    n_list = [n for n in n_list if n <= N_avail]
    if not n_list:
        raise ValueError(f"Not enough images (need at least {min(orig_n_list)}, have {N_avail})")

    real_t_all = _load_t(real_paths[:N_avail])
    gen_t_all  = _load_t(gen_paths[:N_avail])
    f_real_all = _inc_feats(real_t_all, inc, device)
    f_gen_all  = _inc_feats(gen_t_all,  inc, device)
    print(f"Feature extraction done (N={N_avail})")

    results: dict[int, dict[str, list]] = {
        n: {"m3": [], "fid": [], "cmmd": []} for n in n_list
    }

    for n in n_list:
        print(f"\nN = {n}  ({n_repeats} repeats)")
        for rep in range(n_repeats):
            ri = rng.choice(N_avail, size=n, replace=False)
            gi = rng.choice(N_avail, size=n, replace=False)
            rt  = real_t_all[ri];  gt  = gen_t_all[gi]
            fr  = f_real_all[ri];  fg  = f_gen_all[gi]

            with torch.no_grad():
                m3_val = float(metric(rt, gt)["m3_score"])
            fid_val  = _fid(fr, fg)
            cmmd_val = _cmmd(fr, fg)

            results[n]["m3"].append(m3_val)
            results[n]["fid"].append(fid_val)
            results[n]["cmmd"].append(cmmd_val)
            print(f"  rep {rep+1:2d}: M3={m3_val:.4f}  FID={fid_val:.2f}  CMMD={cmmd_val:.4f}")

    # Compute stats
    def _stats(vals: list[float]) -> dict:
        a = np.array([v for v in vals if not np.isnan(v)])
        if len(a) == 0:
            return {"mean": float("nan"), "std": float("nan"), "cv": float("nan")}
        mu, sd = float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0)
        cv = sd / (abs(mu) + 1e-12) * 100
        return {"mean": round(mu, 6), "std": round(sd, 6), "cv": round(cv, 2),
                "n_valid": len(a)}

    summary: dict[str, dict] = {}
    for n in n_list:
        summary[str(n)] = {k: _stats(v) for k, v in results[n].items()}

    _plot_metrics(summary, n_list, output_dir)
    _plot_cv(summary, n_list, output_dir)

    report = {"n_list": n_list, "n_repeats": n_repeats, "summary": summary}
    rp = os.path.join(output_dir, "n_scaling_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")
    return report


def _plot_metrics(summary: dict, n_list: list[int], output_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    for ax, key, label, color in zip(axes,
            ["m3",    "fid",  "cmmd"],
            ["M3-Score (RBF L12)", "FID (scipy)", "CMMD"],
            ["#1f77b4", "#d62728", "#2ca02c"]):
        means = [summary[str(n)][key]["mean"] for n in n_list]
        stds  = [summary[str(n)][key]["std"]  for n in n_list]
        ax.errorbar(n_list, means, yerr=stds, fmt="o-", color=color, lw=2, ms=7,
                    capsize=5, elinewidth=1.5, label=label)
        ax.set_xlabel("N (sample size)"); ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xscale("log"); ax.grid(alpha=0.3)
    fig.suptitle("N-Scaling: metric value ± std vs sample size", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "n_scaling_metrics.png")
    fig.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  Metrics plot: {out}")


def _plot_cv(summary: dict, n_list: list[int], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for key, label, color, ls in [
        ("m3",   "M3 (RBF L12)", "#1f77b4", "-"),
        ("fid",  "FID",          "#d62728", "--"),
        ("cmmd", "CMMD",         "#2ca02c", ":"),
    ]:
        cvs = [summary[str(n)][key]["cv"] for n in n_list]
        ax.plot(n_list, cvs, marker="o", color=color, ls=ls, lw=2, ms=7, label=label)
    ax.axhline(20, color="grey", lw=1, ls="--", label="CV=20% threshold")
    ax.set_xlabel("N (sample size)"); ax.set_ylabel("CV (%)")
    ax.set_title("Coefficient of variation (lower = more stable)", fontsize=12, fontweight="bold")
    ax.set_xscale("log"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(output_dir, "n_scaling_cv.png")
    fig.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  CV plot: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",   required=True)
    p.add_argument("--gen_dir",    required=True)
    p.add_argument("--output_dir", default="results/n_scaling")
    p.add_argument("--n_list",     nargs="+", type=int, default=[25, 50, 100, 200, 500])
    p.add_argument("--n_repeats",  type=int, default=10)
    p.add_argument("--device",     default=None)
    p.add_argument("--seed",       type=int, default=42)
    a = p.parse_args()
    run_n_scaling(**vars(a))
