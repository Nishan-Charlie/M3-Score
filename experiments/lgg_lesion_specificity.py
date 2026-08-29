"""
experiments/lgg_lesion_specificity.py
=====================================
Workstream B: does M3 respond specifically to *pathology* damage, or just to pixel-area
change anywhere? This is the one property a radiology-trained backbone should buy and that
FID (natural-image InceptionV3) should not.

Design — within-image, area-matched contrast on the Kaggle/Buda LGG dataset (real expert
FLAIR-abnormality masks):

  For each image we build two perturbations of identical shape/area:
    LESION  : perturb ONLY the annotated tumour region.
    CONTROL : perturb an equal-area region in HEALTHY tissue (the tumour mask mirrored to
              the contralateral hemisphere; brain is ~bilaterally symmetric, tumours are
              usually unilateral). Images where the mirrored region is not on brain, or
              overlaps the tumour, are skipped so the control is clean.

  We then measure the distributional shift each perturbation induces:
    real  vs  lesion-perturbed      -> Delta_lesion
    real  vs  control-perturbed     -> Delta_control
  for both M3 (single-layer L12 unbiased multi-bandwidth RBF MMD2, the config validated in
  Workstream A) and FID (torchmetrics, InceptionV3).

  Specificity ratio = Delta_lesion / Delta_control.
    ratio > 1  => the metric reacts more to tumour damage than to equal-area healthy damage.
  Claim to test:  ratio_M3 > ratio_FID  (M3 is more lesion-specific than FID).

Honest scoping: no LGG-trained *generator* exists here, so this is a metric-SENSITIVITY test
on real images with controlled perturbations — exactly what a metric paper needs — not a
generator-quality test. If the ratios come out equal, we report the null.

Usage:
    python experiments/lgg_lesion_specificity.py --lgg_dir data_mri/lgg_with_masks \
        --n 300 --perturb erase --device cuda --out results/lgg_pathology
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.m3_score_v2 import M3EntropyMetric


# ---------------------------------------------------------------------------
# Load image + mask pairs
# ---------------------------------------------------------------------------

def load_pairs(lgg_dir: str, n: int):
    """Yield (rgb uint8 HxWx3, mask bool HxW) for lgg_*_slice*.png + *_segmask_*.png."""
    imgs = sorted(glob.glob(os.path.join(lgg_dir, "**", "*_slice*.png"), recursive=True))
    imgs = [p for p in imgs if "_segmask_" not in os.path.basename(p)]
    pairs = []
    for ip in imgs:
        base = ip.replace("_slice", "_segmask_slice")
        if not os.path.isfile(base):
            continue
        rgb = np.array(Image.open(ip).convert("RGB"))
        m = np.array(Image.open(base).convert("L")) > 127
        if m.sum() < 20:  # need a real tumour
            continue
        pairs.append((rgb, m))
        if len(pairs) >= n:
            break
    return pairs


# ---------------------------------------------------------------------------
# Control region + perturbation
# ---------------------------------------------------------------------------

def brain_mask(rgb: np.ndarray, thr: int = 12) -> np.ndarray:
    return rgb.max(axis=2) > thr


def mirrored_control(mask: np.ndarray, brain: np.ndarray, tumour: np.ndarray):
    """Horizontal mirror of the tumour mask -> contralateral region. Returns (ctrl_mask, ok)."""
    ctrl = mask[:, ::-1].copy()
    area = mask.sum()
    if area == 0:
        return ctrl, False
    on_brain = (ctrl & brain).sum() / area
    on_tumour = (ctrl & tumour).sum() / max(ctrl.sum(), 1)
    ok = (on_brain >= 0.7) and (on_tumour <= 0.10)
    return ctrl, ok


def perturb(rgb: np.ndarray, region: np.ndarray, mode: str) -> np.ndarray:
    out = rgb.copy()
    if mode == "erase":
        out[region] = 0
    elif mode == "blur":
        blurred = np.array(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=6)))
        out[region] = blurred[region]
    elif mode == "noise":
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 40, rgb.shape)
        tmp = rgb.astype(np.float32) + noise
        out[region] = np.clip(tmp, 0, 255).astype(np.uint8)[region]
    else:
        raise ValueError(mode)
    return out


# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------

def to_m3(arrs, device):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return torch.stack([t(Image.fromarray(a)) for a in arrs]).to(device)


def to_fid01(arrs, device):
    t = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    return torch.stack([t(Image.fromarray(a)) for a in arrs]).to(device)


_INC = {"net": None}


def _inception(device):
    """Cache a torchvision InceptionV3 (2048-d pool features) — same as Workstream A3."""
    if _INC["net"] is None:
        from torchvision import models
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                  aux_logits=True)
        net.fc = torch.nn.Identity()
        _INC["net"] = net.eval().to(device)
    return _INC["net"]


def inception_feats(imgs01, device, batch=32):
    net = _inception(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs01), batch):
            x = ((imgs01[i:i + batch].to(device) - mean) / std)
            out.append(net(x).cpu())
    return torch.cat(out, 0).numpy()


def kid_mmd2(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """Unbiased KID = polynomial-kernel MMD^2 on Inception features (Binkowski 2018).

    k(x,y) = (x.y/d + 1)^3.  Valid at small N (unlike Frechet/FID, which needs N > 2048
    for a full-rank 2048-d covariance). Same estimator family as M3 (MMD), so the only
    difference from M3 is the backbone -> isolates the backbone's contribution.
    """
    import torch as _t
    x = _t.from_numpy(feat_a).double()
    y = _t.from_numpy(feat_b).double()
    d = x.shape[1]
    Kxx = (x @ x.t() / d + 1.0) ** 3
    Kyy = (y @ y.t() / d + 1.0) ** 3
    Kxy = (x @ y.t() / d + 1.0) ** 3
    m, n = x.shape[0], y.shape[0]
    t_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    t_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    t_xy = Kxy.mean()
    return float((t_xx + t_yy - 2.0 * t_xy).clamp(min=0.0).item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgg_dir", default=os.path.join(ROOT, "data_mri", "lgg_with_masks"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--perturb", default="all", choices=["all", "erase", "blur", "noise"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lgg_pathology"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_perm", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    modes = ["erase", "blur", "noise"] if args.perturb == "all" else [args.perturb]

    pairs = load_pairs(args.lgg_dir, args.n * 2)  # over-fetch; some skipped by control check
    print(f"[data] loaded {len(pairs)} image+mask pairs from {args.lgg_dir}")

    # Build the usable set once: real image + tumour mask + area-matched control mask.
    real_arrs, tum_masks, ctrl_masks, lesion_areas = [], [], [], []
    skipped = 0
    for rgb, tum in pairs:
        brain = brain_mask(rgb)
        ctrl, ok = mirrored_control(tum, brain, tum)
        if not ok:
            skipped += 1
            continue
        real_arrs.append(rgb)
        tum_masks.append(tum)
        ctrl_masks.append(ctrl)
        lesion_areas.append(int(tum.sum()))
        if len(real_arrs) >= args.n:
            break
    N = len(real_arrs)
    print(f"[control] usable={N}  skipped(bad mirror)={skipped}  "
          f"mean matched area={np.mean(lesion_areas):.0f}px (lesion == control by construction)")
    if N < 30:
        print("WARNING: too few usable pairs for a stable estimate.")

    # Real features once (reused across all perturbation modes).
    metric = M3EntropyMetric(device=args.device, single_layer=12, seed=args.seed)
    L = metric.num_layers
    rf = metric._extract_raw_features(to_m3(real_arrs, args.device), layers_to_keep={L})[L - 1]
    real_if = inception_feats(to_fid01(real_arrs, args.device), args.device)

    report = {"config": {"n_used": N, "skipped": skipped, "seed": args.seed,
                         "n_perm": args.n_perm, "matched_area_px": float(np.mean(lesion_areas)),
                         "m3_layer": L, "inception_baseline": "KID (unbiased Inception-MMD)"},
              "modes": {}}

    for mode in modes:
        lesion_arrs = [perturb(r, t, mode) for r, t in zip(real_arrs, tum_masks)]
        control_arrs = [perturb(r, c, mode) for r, c in zip(real_arrs, ctrl_masks)]

        lf = metric._extract_raw_features(to_m3(lesion_arrs, args.device), layers_to_keep={L})[L - 1]
        cf = metric._extract_raw_features(to_m3(control_arrs, args.device), layers_to_keep={L})[L - 1]
        m3_les = metric._mmd2(rf.to(args.device), lf.to(args.device)).item()
        m3_ctl = metric._mmd2(rf.to(args.device), cf.to(args.device)).item()
        p_les, z_les, _ = metric._permutation_test_fidelity(rf, lf, n_permutations=args.n_perm, seed=args.seed)
        p_ctl, z_ctl, _ = metric._permutation_test_fidelity(rf, cf, n_permutations=args.n_perm, seed=args.seed)

        les_if = inception_feats(to_fid01(lesion_arrs, args.device), args.device)
        ctl_if = inception_feats(to_fid01(control_arrs, args.device), args.device)
        kid_les = kid_mmd2(real_if, les_if)
        kid_ctl = kid_mmd2(real_if, ctl_if)

        r_m3 = m3_les / max(m3_ctl, 1e-12)
        r_kid = kid_les / max(kid_ctl, 1e-12)
        report["modes"][mode] = {
            "M3_L12": {"delta_lesion": m3_les, "delta_control": m3_ctl, "specificity_ratio": r_m3,
                       "z_lesion": z_les, "z_control": z_ctl, "p_lesion": p_les, "p_control": p_ctl},
            "KID_inception": {"delta_lesion": kid_les, "delta_control": kid_ctl, "specificity_ratio": r_kid},
            "m3_more_specific_than_kid": bool(r_m3 > r_kid),
        }
        print(f"[{mode:5s}] M3 ratio={r_m3:.3f} (les {m3_les:.4f}/ctl {m3_ctl:.4f})  "
              f"KID ratio={r_kid:.3f} (les {kid_les:.5f}/ctl {kid_ctl:.5f})  "
              f"=> M3 {'MORE' if r_m3 > r_kid else 'NOT more'} specific")

        # example triptych for this mode
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(9, 3))
            ax[0].imshow(real_arrs[0]); ax[0].set_title("real"); ax[0].axis("off")
            ax[1].imshow(lesion_arrs[0]); ax[1].set_title(f"lesion {mode}"); ax[1].axis("off")
            ax[2].imshow(control_arrs[0]); ax[2].set_title(f"control {mode}"); ax[2].axis("off")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, f"example_{mode}.png"), dpi=110)
            plt.close(fig)
        except Exception as e:
            print("plot skipped:", e)

    out_json = os.path.join(args.out, "lesion_specificity.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {time.time()-t0:.1f}s -> {out_json}")


if __name__ == "__main__":
    main()
