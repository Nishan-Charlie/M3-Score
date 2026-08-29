"""
Pathology Masking Experiment
==============================
Clinical-relevance test for the paper (Section 3.x).

Hypothesis
----------
FID and CMMD are insensitive to pathology masking because InceptionV3/CLIP
features do not encode fine-grained brain anatomy. M3 with RadioDino-s16 L12
features should detect the distributional shift when tumor regions are erased.
The per-patch heatmap should localise the deviation to the erased region.

Protocol
--------
Three conditions, each comparing against the clean real distribution:
  (a) real vs generated           — baseline
  (b) real vs masked-real         — pathology hidden, same modality
  (c) real vs masked-generated    — double degradation

If BraTS segmentation masks are available (--mask_dir):
  Use the actual tumor segmentation mask (union of all label channels).
Fallback (no --mask_dir):
  Simulate with a centre square mask (--mask_size, default 64 px).

Localisation check (when masks available):
  Build patch memory bank on clean real images.
  Score each masked-real image with M3PatchScorer.
  Compute IoU(top-K% heatmap, ground-truth mask).

Expected results
----------------
  M3:  b >> a  (RadioDino features encode brain anatomy — masking is visible)
  FID: b ~= a  (InceptionV3 cannot see the difference)
  CMMD:b ~= a  (CLIP features too semantic)
  Patch heatmap IoU > chance

Output
------
  pathology_masking_report.json
  pathology_masking_scores.png
  heatmap_examples/heatmap_NNN.png   (up to 5 examples, if masks available)
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
# Image / mask loading helpers
# ---------------------------------------------------------------------------

def _glob_images(directory: str, recursive: bool = True) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, "**" if recursive else "", ext),
                               recursive=recursive))
    return sorted(set(paths))


def _load_image_mask_pairs(
    image_dir: str,
    mask_dir:  str,
    min_mask_frac: float,
    n_max: int,
) -> tuple[list[str], list[str]]:
    img_paths, msk_paths = [], []
    for ip in _glob_images(image_dir):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = next(
            (c for c in [
                os.path.join(mask_dir, f"{stem}_mask.png"),
                os.path.join(mask_dir, f"{stem}.png"),
                os.path.join(mask_dir, f"{stem}_seg.png"),
            ] if os.path.isfile(c)),
            None
        )
        if mp is None:
            continue
        frac = (np.array(Image.open(mp).convert("L")) > 0).mean()
        if frac < min_mask_frac:
            continue
        img_paths.append(ip); msk_paths.append(mp)
        if len(img_paths) >= n_max:
            break
    return img_paths, msk_paths


def _apply_mask(img_t: torch.Tensor, mask_t: torch.Tensor,
                fill: str = "zero") -> torch.Tensor:
    out = img_t.clone()
    if fill == "mean":
        out = out * (1 - mask_t) + img_t.mean() * mask_t
    else:
        out = out * (1 - mask_t)
    return out


def _centre_mask(imgs: torch.Tensor, mask_size: int) -> torch.Tensor:
    B, C, H, W = imgs.shape
    sh, sw = H // 2 - mask_size // 2, W // 2 - mask_size // 2
    out = imgs.clone()
    out[:, :, sh:sh + mask_size, sw:sw + mask_size] = 0.0
    return out


# ---------------------------------------------------------------------------
# FID helper (scipy, eps-regularised)
# ---------------------------------------------------------------------------

def _build_inception(device: str):
    inc = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inc.fc = torch.nn.Identity()
    return inc.eval().to(device)


def _inc_feats(t: torch.Tensor, inc, device: str, bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(t), bs):
        b = t[i:i+bs]
        b = F.interpolate(b, (299, 299), mode="bilinear", align_corners=False)
        mn = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        st = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        b  = (b - mn) / st
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


# ---------------------------------------------------------------------------
# CMMD helper
# ---------------------------------------------------------------------------

def _cmmd_score(fr: np.ndarray, fg: np.ndarray) -> float:
    from evaluation.cmmd_metric import CMMDMetric
    return float(CMMDMetric.gaussian_mmd2_unbiased(fr, fg))


# ---------------------------------------------------------------------------
# Heatmap IoU
# ---------------------------------------------------------------------------

def _heatmap_iou(heatmap: np.ndarray, mask: np.ndarray,
                 top_k: float = 0.20) -> float:
    thresh = np.percentile(heatmap, 100 * (1 - top_k))
    pred   = heatmap >= thresh
    inter  = (pred & mask).sum()
    union  = (pred | mask).sum()
    return float(inter / (union + 1e-8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pathology_masking(
    real_dir:      str,
    gen_dir:       str,
    output_dir:    str           = "./pathology_masking",
    mask_dir:      Optional[str] = None,
    n_images:      int           = 200,
    mask_size:     int           = 64,
    mask_fill:     str           = "zero",
    min_mask_frac: float         = 0.01,
    top_k_frac:    float         = 0.20,
    device:        Optional[str] = None,
    seed:          int           = 42,
    backbone_id:   str           = "Snarcy/RadioDino-s16",
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    # ── Load images ───────────────────────────────────────────────────────
    use_real_masks = mask_dir is not None and os.path.isdir(mask_dir)
    have_heatmaps  = False

    if use_real_masks:
        img_paths, msk_paths = _load_image_mask_pairs(
            real_dir, mask_dir, min_mask_frac, n_images
        )
        print(f"Loaded {len(img_paths)} image-mask pairs")
        if not img_paths:
            print("No pairs found — falling back to centre mask")
            use_real_masks = False

    if use_real_masks:
        real_t = torch.stack([_TFM_224(Image.open(p).convert("RGB")) for p in img_paths])
        mask_t = torch.stack([
            torch.from_numpy(
                (np.array(Image.open(p).convert("L").resize((224,224))) > 0)
                .astype(np.float32)
            ).unsqueeze(0)
            for p in msk_paths
        ])
        masked_real_t = torch.stack([_apply_mask(real_t[i], mask_t[i], mask_fill)
                                      for i in range(len(real_t))])
        have_heatmaps = True
    else:
        all_real_paths = _glob_images(real_dir)[:n_images]
        real_t = torch.stack([_TFM_224(Image.open(p).convert("RGB"))
                               for p in all_real_paths])
        masked_real_t = _centre_mask(real_t, mask_size)
        mask_t = None
        print(f"Using centre mask ({mask_size}x{mask_size}) on {len(real_t)} images")

    gen_paths = _glob_images(gen_dir, recursive=True)[:len(real_t)]
    if not gen_paths:
        raise FileNotFoundError(f"No images found in gen_dir={gen_dir!r}")
    gen_t     = torch.stack([_TFM_224(Image.open(p).convert("RGB")) for p in gen_paths])
    masked_gen_t = (_centre_mask(gen_t, mask_size) if mask_t is None else
                    torch.stack([_apply_mask(gen_t[i], mask_t[i], mask_fill)
                                  for i in range(len(gen_t))]))

    N = min(len(real_t), len(gen_t))
    real_t = real_t[:N]; masked_real_t = masked_real_t[:N]
    gen_t  = gen_t[:N];  masked_gen_t  = masked_gen_t[:N]
    print(f"N = {N}")

    # ── M3 (RBF, L12) ────────────────────────────────────────────────────
    metric = M3V2Metric(device=device, backbone_id=backbone_id,
                        kernel="rbf", single_layer=12, seed=seed)

    def _m3(r, g):
        with torch.no_grad():
            return float(metric(r, g)["m3_score"])

    # ── InceptionV3 for FID ───────────────────────────────────────────────
    inc = _build_inception(device)
    f_real        = _inc_feats(real_t, inc, device)
    f_gen         = _inc_feats(gen_t,  inc, device)
    f_masked_real = _inc_feats(masked_real_t, inc, device)
    f_masked_gen  = _inc_feats(masked_gen_t,  inc, device)

    # ── CMMD (optional) ───────────────────────────────────────────────────
    try:
        from evaluation.cmmd_metric import CMMDMetric
        cmmd_model = CMMDMetric(device=device)
        _save = lambda t, tag: [
            (p := os.path.join(output_dir, f"_tmp_{tag}_{i}.png"),
             __import__("torchvision").transforms.functional.to_pil_image(
                 t[i].clamp(0,1)).save(p))[0]
            for i in range(len(t))
        ]
        os.makedirs(output_dir, exist_ok=True)
        cf_real        = cmmd_model.extract_features(
            img_paths if use_real_masks else _glob_images(real_dir)[:N])
        cf_gen         = cmmd_model.extract_features(gen_paths[:N])
        cf_masked_real = cmmd_model.extract_features(_save(masked_real_t, "mr"))
        cf_masked_gen  = cmmd_model.extract_features(_save(masked_gen_t,  "mg"))
        have_cmmd = True
    except Exception as e:
        print(f"CMMD skipped: {e}")
        have_cmmd = False

    # ── Three conditions ──────────────────────────────────────────────────
    print("\n(a) real vs gen (baseline)")
    a = {"m3": _m3(real_t, gen_t), "fid": _fid(f_real, f_gen),
         "cmmd": _cmmd_score(cf_real, cf_gen) if have_cmmd else None}
    print(f"    M3={a['m3']:.4f}  FID={a['fid']:.2f}  CMMD={a['cmmd']}")

    print("(b) real vs masked-real (pathology hidden)")
    b = {"m3": _m3(real_t, masked_real_t), "fid": _fid(f_real, f_masked_real),
         "cmmd": _cmmd_score(cf_real, cf_masked_real) if have_cmmd else None}
    print(f"    M3={b['m3']:.4f}  FID={b['fid']:.2f}  CMMD={b['cmmd']}")

    print("(c) real vs masked-gen (double degradation)")
    c = {"m3": _m3(real_t, masked_gen_t), "fid": _fid(f_real, f_masked_gen),
         "cmmd": _cmmd_score(cf_real, cf_masked_gen) if have_cmmd else None}
    print(f"    M3={c['m3']:.4f}  FID={c['fid']:.2f}  CMMD={c['cmmd']}")

    # Sensitivity: relative change from (a) to (b)
    sensitivity = {
        "m3":  round((b["m3"]  - a["m3"])  / (a["m3"]  + 1e-8), 4),
        "fid": round((b["fid"] - a["fid"]) / (a["fid"] + 1e-8), 4),
        "cmmd": round((b["cmmd"] - a["cmmd"]) / (a["cmmd"] + 1e-8), 4) if have_cmmd else None,
    }
    print(f"\nSensitivity to masking: M3={sensitivity['m3']:+.3f}  "
          f"FID={sensitivity['fid']:+.3f}  CMMD={sensitivity.get('cmmd')}")

    # ── Patch heatmaps (if masks available) ──────────────────────────────
    iou_list: list[float] = []
    if have_heatmaps:
        print("\nBuilding patch memory bank ...")
        from evaluation.m3_patch_scorer import M3PatchScorer
        scorer = M3PatchScorer(device=device, backbone_id=backbone_id)
        scorer.fit(real_t)
        hmap_dir = os.path.join(output_dir, "heatmap_examples")
        os.makedirs(hmap_dir, exist_ok=True)

        for i in range(N):
            r = scorer.score(masked_real_t[i])
            gt = (mask_t[i, 0].numpy() > 0)
            iou = _heatmap_iou(r["heatmap"], gt, top_k=top_k_frac)
            iou_list.append(iou)

            if i < 5:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=110)
                axes[0].imshow(real_t[i].permute(1,2,0).numpy().clip(0,1), cmap="gray")
                axes[0].set_title("Real (clean)"); axes[0].axis("off")
                axes[1].imshow(masked_real_t[i].permute(1,2,0).numpy().clip(0,1), cmap="gray")
                axes[1].set_title("Masked (tumor removed)"); axes[1].axis("off")
                im = axes[2].imshow(r["heatmap"], cmap="hot")
                axes[2].contour(gt, colors=["cyan"], linewidths=1.5)
                axes[2].set_title(f"Deviation heatmap  IoU={iou:.3f}"); axes[2].axis("off")
                plt.colorbar(im, ax=axes[2])
                fig.tight_layout()
                fig.savefig(os.path.join(hmap_dir, f"heatmap_{i:03d}.png"), bbox_inches="tight")
                plt.close()

    mean_iou = float(np.mean(iou_list)) if iou_list else None

    # ── Summary bar chart ─────────────────────────────────────────────────
    n_metrics = 3 if have_cmmd else 2
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), dpi=120)
    if n_metrics == 1: axes = [axes]
    conds = ["(a) baseline\nreal vs gen",
             "(b) masked-real\n(pathology hidden)",
             "(c) masked-gen\n(double degrad.)"]
    colors = ["#1f77b4", "#e05c5c", "#ff7f0e"]

    for ax, (key, label, note) in zip(axes, [
        ("m3",   "M3-Score (RBF L12)", "higher = more different from real"),
        ("fid",  "FID",                "should barely change at (b)"),
    ] + ([("cmmd", "CMMD", "should barely change at (b)")] if have_cmmd else [])):
        vals = [a[key], b[key], c[key]]
        if any(v is None for v in vals):
            continue
        bars = ax.bar(conds, vals, color=colors, edgecolor="black", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f"{v:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{label}\n({note})", fontsize=10)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.3)

    iou_txt = f"  |  Heatmap IoU={mean_iou:.3f}" if mean_iou is not None else ""
    fig.suptitle(f"Pathology Masking Sensitivity{iou_txt}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "pathology_masking_scores.png")
    fig.savefig(plot_path, bbox_inches="tight"); plt.close()
    print(f"Plot saved: {plot_path}")

    report = {
        "n": N, "mask_fill": mask_fill, "use_real_masks": use_real_masks,
        "condition_a": a, "condition_b": b, "condition_c": c,
        "sensitivity_b_vs_a": sensitivity,
        "heatmap_iou_mean": mean_iou,
        "heatmap_iou_list": [round(v, 4) for v in iou_list],
        "plots": {"scores": plot_path},
    }
    rp = os.path.join(output_dir, "pathology_masking_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {rp}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",      required=True)
    p.add_argument("--gen_dir",       required=True)
    p.add_argument("--output_dir",    default="results/pathology_masking")
    p.add_argument("--mask_dir",      default=None)
    p.add_argument("--n_images",      type=int,   default=200)
    p.add_argument("--mask_size",     type=int,   default=64)
    p.add_argument("--mask_fill",     default="zero", choices=["zero", "mean"])
    p.add_argument("--min_mask_frac", type=float, default=0.01)
    p.add_argument("--device",        default=None)
    p.add_argument("--seed",          type=int,   default=42)
    a = p.parse_args()
    run_pathology_masking(**vars(a))
