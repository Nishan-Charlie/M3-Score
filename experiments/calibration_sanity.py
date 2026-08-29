"""
Calibration / Sanity Panel
============================
Four checks that any metric must pass before it can be trusted:

  Check 1 – Real-vs-real ≈ 0
      M3 on two independent halves of the real set should be near zero.
      FID and CMMD are included for comparison.

  Check 2 – Monotonic degradation
      M3, FID, CMMD all increase as we apply progressively stronger
      Gaussian blur (σ = 0, 1, 2, 3, 4, 5) to the generated set.

  Check 3 – Model ranking (DDPM > GAN > constant)
      Three synthetic "generators" are produced from real images:
        * DDPM-proxy  : real images + tiny σ=0.05 Gaussian noise  (best)
        * GAN-proxy   : real images + stronger σ=0.20 noise       (medium)
        * Constant    : all images set to mean-intensity           (worst)
      Each metric should rank them DDPM < GAN < Constant
      (lower score = better quality).

  Check 4 – CV stability across N sub-samples
      Sub-sample sizes [50, 100, 200, 500].  CV = std/mean across 5 bootstrap
      rounds per size.  M3 should have lower CV than FID at small N.

Output (in output_dir/):
  calibration_sanity_report.json
  calibration_sanity.png          (2×2 panel)
"""

from __future__ import annotations

import json
import os
import sys
import glob
import random
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms, models
from scipy.linalg import sqrtm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.m3_score_v2 import M3V2Metric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_paths(directory: str, n: int, recursive: bool = True) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        if recursive:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        else:
            paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(set(paths))
    random.shuffle(paths)
    return paths[:n]


_IMG_TFM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def _pil_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma)) if sigma > 0 else img


def _load_tensors(paths: list[str], blur_sigma: float = 0.0) -> torch.Tensor:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if blur_sigma > 0:
            img = _pil_blur(img, blur_sigma)
        imgs.append(_IMG_TFM(img))
    return torch.stack(imgs)


# ---------------------------------------------------------------------------
# FID helper (scipy, eps-regularised)
# ---------------------------------------------------------------------------

_INC_TFM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _build_inception(device: str):
    inc = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inc.fc = torch.nn.Identity()
    inc.eval().to(device)
    return inc


def _extract_inception_feats(paths: list[str], inc_model, device: str,
                              blur_sigma: float = 0.0) -> np.ndarray:
    feats = []
    batch: list[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if blur_sigma > 0:
            img = _pil_blur(img, blur_sigma)
        batch.append(_INC_TFM(img))
        if len(batch) == 32:
            with torch.no_grad():
                feats.append(inc_model(torch.stack(batch).to(device)).cpu().numpy())
            batch = []
    if batch:
        with torch.no_grad():
            feats.append(inc_model(torch.stack(batch).to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _fid(f_r: np.ndarray, f_g: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, mu_g = f_r.mean(0), f_g.mean(0)
    sig_r = np.cov(f_r, rowvar=False) + eps * np.eye(f_r.shape[1])
    sig_g = np.cov(f_g, rowvar=False) + eps * np.eye(f_g.shape[1])
    diff = mu_r - mu_g
    covmean = sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.dot(diff, diff) + np.trace(sig_r + sig_g - 2 * covmean))


# ---------------------------------------------------------------------------
# CMMD helper (Gaussian MMD² on CLIP features)
# ---------------------------------------------------------------------------

def _cmmd(f_r: np.ndarray, f_g: np.ndarray) -> float:
    from evaluation.cmmd_metric import CMMDMetric
    return float(CMMDMetric.gaussian_mmd2_unbiased(f_r, f_g))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_calibration_sanity(
    real_dir:    str,
    gen_dir:     str,
    output_dir:  str           = "./calibration_sanity",
    n_images:    int           = 500,
    device:      Optional[str] = None,
    seed:        int           = 42,
    backbone_id: str           = "Snarcy/RadioDino-s16",
    layers_cache: Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Load paths ─────────────────────────────────────────────────────────
    real_paths = _load_paths(real_dir, n_images)
    gen_paths  = _load_paths(gen_dir,  n_images, recursive=False)
    cap = min(len(real_paths), len(gen_paths), n_images)
    real_paths, gen_paths = real_paths[:cap], gen_paths[:cap]
    half = cap // 2
    real_A_paths, real_B_paths = real_paths[:half], real_paths[half:2 * half]

    print(f"Using {cap} real + {cap} gen images (half={half})")

    # ── Initialise M3 metric ───────────────────────────────────────────────
    metric = M3V2Metric(device=device, backbone_id=backbone_id, seed=seed)
    cka_ref_n   = min(200, len(real_paths))
    gen_seed    = torch.Generator().manual_seed(seed)
    cka_ref_all = _load_tensors(real_paths)
    cka_ref_idx = torch.randperm(len(cka_ref_all), generator=gen_seed)[:cka_ref_n]
    metric.prune_layers_via_cka(cka_ref_all[cka_ref_idx], cache_path=layers_cache, seed=seed)

    def _m3(t_r: torch.Tensor, t_g: torch.Tensor) -> float:
        with torch.no_grad():
            return float(metric(t_r, t_g)["m3_score"])

    # ── Inception model (for FID) ──────────────────────────────────────────
    inc = _build_inception(device)

    # ── CMMD model ─────────────────────────────────────────────────────────
    try:
        from evaluation.cmmd_metric import CMMDMetric
        cmmd_model = CMMDMetric(device=device)
        have_cmmd  = True
    except Exception as e:
        print(f"CMMD unavailable: {e}; skipping CMMD checks")
        have_cmmd  = False

    def _get_clip_feats(paths: list[str], blur_sigma: float = 0.0) -> Optional[np.ndarray]:
        if not have_cmmd:
            return None
        if blur_sigma == 0:
            return cmmd_model.extract_features(paths)
        # save blurred images to temp files so extract_features gets paths
        _tmp = os.path.join(output_dir, "_tmp_clip_blur")
        os.makedirs(_tmp, exist_ok=True)
        tmp_paths = []
        for i, p in enumerate(paths):
            img = _pil_blur(Image.open(p).convert("RGB"), blur_sigma)
            tp = os.path.join(_tmp, f"blur{blur_sigma}_{i}.png")
            img.save(tp)
            tmp_paths.append(tp)
        return cmmd_model.extract_features(tmp_paths)

    # ══════════════════════════════════════════════════════════════════════
    # Check 1 — real-vs-real ≈ 0
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Check 1] Real-vs-real ...")
    tA = _load_tensors(real_A_paths)
    tB = _load_tensors(real_B_paths)
    c1_m3 = _m3(tA, tB)
    c1_fid = _fid(
        _extract_inception_feats(real_A_paths, inc, device),
        _extract_inception_feats(real_B_paths, inc, device),
    )
    cf_A = _get_clip_feats(real_A_paths)
    cf_B = _get_clip_feats(real_B_paths)
    c1_cmmd = _cmmd(cf_A, cf_B) if have_cmmd else None
    print(f"  M3={c1_m3:.6f}  FID={c1_fid:.3f}  CMMD={c1_cmmd}")

    # ══════════════════════════════════════════════════════════════════════
    # Check 2 — monotonic degradation under blur
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Check 2] Monotonic degradation ...")
    blur_sigmas  = [0, 1, 2, 3, 4, 5]
    c2_m3_vals   = []
    c2_fid_vals  = []
    c2_cmmd_vals = []
    tR = _load_tensors(real_paths[:cap])
    inc_real_f = _extract_inception_feats(real_paths[:cap], inc, device)
    clip_real_f = _get_clip_feats(real_paths[:cap])

    for sigma in blur_sigmas:
        tG_blur   = _load_tensors(gen_paths[:cap], blur_sigma=sigma)
        m3_v      = _m3(tR, tG_blur)
        inc_gen_f = _extract_inception_feats(gen_paths[:cap], inc, device, blur_sigma=sigma)
        fid_v     = _fid(inc_real_f, inc_gen_f)
        clip_gen_f = _get_clip_feats(gen_paths[:cap], blur_sigma=sigma)
        cmmd_v    = _cmmd(clip_real_f, clip_gen_f) if have_cmmd else None
        c2_m3_vals.append(m3_v)
        c2_fid_vals.append(fid_v)
        c2_cmmd_vals.append(cmmd_v)
        print(f"  sigma={sigma:>2}  M3={m3_v:.6f}  FID={fid_v:.3f}  CMMD={cmmd_v}")

    # monotonicity: each subsequent value ≥ previous (allow small ties)
    c2_m3_mono   = all(c2_m3_vals[i+1] >= c2_m3_vals[i] * 0.99   for i in range(len(c2_m3_vals)-1))
    c2_fid_mono  = all(c2_fid_vals[i+1] >= c2_fid_vals[i] * 0.99  for i in range(len(c2_fid_vals)-1))
    c2_cmmd_mono = (
        all(c2_cmmd_vals[i+1] >= c2_cmmd_vals[i] * 0.99 for i in range(len(c2_cmmd_vals)-1))
        if have_cmmd else None
    )

    # ══════════════════════════════════════════════════════════════════════
    # Check 3 — model ranking
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Check 3] Model ranking ...")
    tR_rank = _load_tensors(real_paths[:cap])
    tG_rank = _load_tensors(gen_paths[:cap])

    # DDPM-proxy: gen + tiny noise
    gen_arr = tG_rank.clone()
    torch.manual_seed(seed)
    t_ddpm = gen_arr + 0.05 * torch.randn_like(gen_arr)
    t_ddpm.clamp_(0, 1)

    # GAN-proxy: gen + larger noise
    torch.manual_seed(seed)
    t_gan = gen_arr + 0.20 * torch.randn_like(gen_arr)
    t_gan.clamp_(0, 1)

    # Constant: mean-value image
    mean_val = tR_rank.mean(dim=(0, 2, 3), keepdim=True)
    t_const  = mean_val.expand_as(tR_rank)

    c3_scores: dict[str, dict] = {}
    for name, t_g in [("ddpm_proxy", t_ddpm), ("gan_proxy", t_gan), ("constant", t_const)]:
        m3_v = _m3(tR_rank, t_g)
        # FID requires image paths; use tensor-to-PIL conversion
        from torchvision.transforms.functional import to_pil_image
        tmp_paths = []
        _tmp_dir = os.path.join(output_dir, "_tmp_rank")
        os.makedirs(_tmp_dir, exist_ok=True)
        for idx, img_t in enumerate(t_g[:half]):
            p = os.path.join(_tmp_dir, f"{name}_{idx}.png")
            to_pil_image(img_t.clamp(0, 1)).save(p)
            tmp_paths.append(p)
        inc_g_f = _extract_inception_feats(tmp_paths, inc, device)
        inc_r_f = _extract_inception_feats(real_paths[:half], inc, device)
        fid_v   = _fid(inc_r_f, inc_g_f)
        clip_g_f = None
        cmmd_v  = None
        if have_cmmd:
            clip_g_f = cmmd_model.extract_features(tmp_paths)
            clip_r_f = _get_clip_feats(real_paths[:half])
            cmmd_v   = _cmmd(clip_r_f, clip_g_f)
        c3_scores[name] = {"m3": m3_v, "fid": fid_v, "cmmd": cmmd_v}
        print(f"  {name:<14}  M3={m3_v:.6f}  FID={fid_v:.3f}  CMMD={cmmd_v}")

    c3_m3_ok  = c3_scores["ddpm_proxy"]["m3"] < c3_scores["gan_proxy"]["m3"] < c3_scores["constant"]["m3"]
    c3_fid_ok = c3_scores["ddpm_proxy"]["fid"] < c3_scores["gan_proxy"]["fid"] < c3_scores["constant"]["fid"]
    c3_cmmd_ok = (
        c3_scores["ddpm_proxy"]["cmmd"] < c3_scores["gan_proxy"]["cmmd"] < c3_scores["constant"]["cmmd"]
        if have_cmmd else None
    )

    # ══════════════════════════════════════════════════════════════════════
    # Check 4 — CV stability across N
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Check 4] CV stability ...")
    n_list  = [50, 100, 200, min(500, cap)]
    n_boots = 5
    c4_m3_cv   = []
    c4_fid_cv  = []
    inc_full_r = _extract_inception_feats(real_paths, inc, device)
    inc_full_g = _extract_inception_feats(gen_paths,  inc, device)
    real_full  = _load_tensors(real_paths)
    gen_full   = _load_tensors(gen_paths)

    for n in n_list:
        m3_vals_  = []
        fid_vals_ = []
        for b in range(n_boots):
            rng = np.random.default_rng(seed + b)
            ri  = rng.choice(cap, size=n, replace=False)
            gi  = rng.choice(cap, size=n, replace=False)
            m3_vals_.append(_m3(real_full[ri], gen_full[gi]))
            fid_vals_.append(_fid(inc_full_r[ri], inc_full_g[gi]))
        m3_arr = np.array(m3_vals_)
        fid_arr = np.array(fid_vals_)
        cv_m3  = float(m3_arr.std()  / (m3_arr.mean()  + 1e-9))
        cv_fid = float(fid_arr.std() / (fid_arr.mean() + 1e-9))
        c4_m3_cv.append(cv_m3)
        c4_fid_cv.append(cv_fid)
        print(f"  N={n:>4}  CV(M3)={cv_m3:.4f}  CV(FID)={cv_fid:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Plot: 2×2 panel
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    # — Check 1: bar chart
    ax = axes[0, 0]
    names1 = ["M3", "FID/100", "CMMD×10" if have_cmmd else "CMMD (N/A)"]
    vals1  = [c1_m3, c1_fid / 100, (c1_cmmd * 10 if c1_cmmd else 0)]
    colors1 = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(names1, vals1, color=colors1, edgecolor="black")
    for bar, v in zip(bars, vals1):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Check 1: Real-vs-real (should be ≈ 0)", fontsize=11)
    ax.set_ylabel("Score (rescaled for visibility)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # — Check 2: monotonic degradation
    ax = axes[0, 1]
    m3_norm  = np.array(c2_m3_vals) / (max(c2_m3_vals) + 1e-9)
    fid_norm = np.array(c2_fid_vals) / (max(c2_fid_vals) + 1e-9)
    ax.plot(blur_sigmas, m3_norm,  "o-", label="M3 (norm)", color="#1f77b4")
    ax.plot(blur_sigmas, fid_norm, "s-", label="FID (norm)", color="#ff7f0e")
    if have_cmmd and any(v is not None for v in c2_cmmd_vals):
        cmmd_arr = np.array([v if v is not None else 0 for v in c2_cmmd_vals])
        cmmd_norm = cmmd_arr / (cmmd_arr.max() + 1e-9)
        ax.plot(blur_sigmas, cmmd_norm, "^-", label="CMMD (norm)", color="#2ca02c")
    mono_sym = {True: "PASS", False: "FAIL"}
    ax.set_title(f"Check 2: Monotonic under blur\n"
                 f"M3 {mono_sym[c2_m3_mono]}  FID {mono_sym[c2_fid_mono]}", fontsize=11)
    ax.set_xlabel("Blur σ"); ax.set_ylabel("Normalised score")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # — Check 3: model ranking
    ax = axes[1, 0]
    models_  = ["DDPM-proxy\n(σ=0.05)", "GAN-proxy\n(σ=0.20)", "Constant\n(mean)"]
    m3_r  = [c3_scores["ddpm_proxy"]["m3"],  c3_scores["gan_proxy"]["m3"],  c3_scores["constant"]["m3"]]
    fid_r = [c3_scores["ddpm_proxy"]["fid"],  c3_scores["gan_proxy"]["fid"],  c3_scores["constant"]["fid"]]
    x3 = np.arange(3)
    ax.bar(x3 - 0.2, np.array(m3_r)  / (max(m3_r)  + 1e-9), width=0.35,
           color="#1f77b4", label="M3 (norm)", edgecolor="black")
    ax.bar(x3 + 0.2, np.array(fid_r) / (max(fid_r) + 1e-9), width=0.35,
           color="#ff7f0e", label="FID (norm)", edgecolor="black")
    ax.set_xticks(x3); ax.set_xticklabels(models_, fontsize=9)
    ok_sym = {True: "PASS", False: "FAIL", None: "N/A"}
    ax.set_title(f"Check 3: Model ranking\n"
                 f"M3 {ok_sym[c3_m3_ok]}  FID {ok_sym[c3_fid_ok]}  CMMD {ok_sym[c3_cmmd_ok]}", fontsize=11)
    ax.set_ylabel("Normalised score (lower = better)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # — Check 4: CV stability
    ax = axes[1, 1]
    ax.plot(n_list, c4_m3_cv,  "o-", color="#1f77b4", label="CV(M3)")
    ax.plot(n_list, c4_fid_cv, "s-", color="#ff7f0e", label="CV(FID)")
    ax.set_title("Check 4: CV stability vs N\n(lower = more stable)", fontsize=11)
    ax.set_xlabel("N samples"); ax.set_ylabel("CV (std/mean)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("M3-Score Calibration / Sanity Panel", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "calibration_sanity.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # ── Report ─────────────────────────────────────────────────────────────
    report = {
        "check1_real_vs_real": {
            "m3":  round(c1_m3,  6),
            "fid": round(c1_fid, 4),
            "cmmd": round(float(c1_cmmd), 6) if c1_cmmd is not None else None,
        },
        "check2_monotonic": {
            "blur_sigmas": blur_sigmas,
            "m3":    [round(v, 6) for v in c2_m3_vals],
            "fid":   [round(v, 4) for v in c2_fid_vals],
            "cmmd":  [round(float(v), 6) if v is not None else None for v in c2_cmmd_vals],
            "m3_monotonic":   c2_m3_mono,
            "fid_monotonic":  c2_fid_mono,
            "cmmd_monotonic": c2_cmmd_mono,
        },
        "check3_model_ranking": {
            **{k: {mk: (round(v, 6) if v is not None else None) for mk, v in d.items()}
               for k, d in c3_scores.items()},
            "m3_rank_ok":   c3_m3_ok,
            "fid_rank_ok":  c3_fid_ok,
            "cmmd_rank_ok": c3_cmmd_ok,
        },
        "check4_cv_stability": {
            "n_list":     n_list,
            "cv_m3":      [round(v, 4) for v in c4_m3_cv],
            "cv_fid":     [round(v, 4) for v in c4_fid_cv],
            "m3_lower_cv_than_fid": [m <= f for m, f in zip(c4_m3_cv, c4_fid_cv)],
        },
        "plots": {"calibration_sanity": plot_path},
    }
    report_path = os.path.join(output_dir, "calibration_sanity_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",    required=True)
    p.add_argument("--gen_dir",     required=True)
    p.add_argument("--output_dir",  default="results/calibration_sanity")
    p.add_argument("--n_images",    type=int,   default=500)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    a = p.parse_args()
    run_calibration_sanity(**vars(a))
