"""
experiments/lgg_noise_sweep.py
==============================
Workstream B extension: harden the one defensible medical finding from
`lgg_lesion_specificity.py` into a publication-grade result.

Two additions over the base experiment:

  1. Noise-sigma SWEEP. The base run showed M3 is lesion-selective under *subtle* noise but
     not under *gross* erasure. If that story is real, the lesion/healthy specificity ratio
     should be high at small sigma and decay toward 1 as sigma grows (the low-level signal
     starts to dominate and swamps the semantic difference). We trace ratio vs sigma.

  2. TEXTURE-MATCHED control, alongside the mirrored control. The mirrored (contralateral)
     region matches area+shape but a tumour is more textured than healthy tissue, so the
     selectivity could be a texture-complexity artifact. The texture control translates the
     tumour mask to the healthy-brain location whose local gradient-energy best matches the
     tumour's — matching area AND texture. If M3 stays selective vs BOTH controls while the
     Inception-MMD (KID) baseline does not, the effect is backbone-attributable, not texture.

Baseline = KID (unbiased Inception-MMD): small-N valid and same estimator family as M3.

Output: results/lgg_noise_sweep/{sweep.json, sweep.png}. One process (no GPU orphans).

Usage:
    python experiments/lgg_noise_sweep.py --n 200 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.m3_score_v2 import M3EntropyMetric
from experiments.lgg_lesion_specificity import (
    load_pairs, brain_mask, mirrored_control, to_m3, to_fid01,
    inception_feats, kid_mmd2,
)


# ---------------------------------------------------------------------------
# Texture-matched control
# ---------------------------------------------------------------------------

def gradient_energy(gray: np.ndarray) -> np.ndarray:
    from scipy.ndimage import sobel
    gx = sobel(gray.astype(float), axis=1)
    gy = sobel(gray.astype(float), axis=0)
    return np.hypot(gx, gy)


def texture_matched_control(tumour, brain, rgb, step: int = 12):
    """Translate the tumour mask to the healthy-brain location whose gradient energy best
    matches the tumour's. Returns (ctrl_mask, ok). Preserves shape/area; matches texture."""
    H, W = tumour.shape
    gmap = gradient_energy(rgb.max(axis=2))
    ys, xs = np.where(tumour)
    if len(ys) == 0:
        return tumour.copy(), False
    tum_energy = float(gmap[tumour].sum())
    tum_area = int(tumour.sum())

    best, best_diff = None, None
    for dy in range(-H + 1, H, step):
        for dx in range(-W + 1, W, step):
            if dy == 0 and dx == 0:
                continue
            y2, x2 = ys + dy, xs + dx
            valid = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
            if valid.mean() < 0.98:
                continue
            shifted = np.zeros_like(tumour)
            shifted[y2[valid], x2[valid]] = True
            a = int(shifted.sum())
            if a < 0.9 * tum_area:
                continue
            if (shifted & brain).sum() / a < 0.9:
                continue
            if (shifted & tumour).sum() / a > 0.05:
                continue
            diff = abs(float(gmap[shifted].sum()) - tum_energy)
            if best is None or diff < best_diff:
                best, best_diff = shifted, diff
    return (best, True) if best is not None else (tumour.copy(), False)


# ---------------------------------------------------------------------------
# Perturbation at a given sigma (region-local additive Gaussian noise)
# ---------------------------------------------------------------------------

def add_noise(rgb: np.ndarray, region: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rgb.astype(np.float32)
    noise = rng.normal(0, sigma, rgb.shape).astype(np.float32)
    out[region] = np.clip(out[region] + noise[region], 0, 255)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Bootstrap CI on the specificity ratio, using pre-extracted features (cheap)
# ---------------------------------------------------------------------------

def ratio_ci(mmd_fn, real_f, les_f, ctl_f, floor, n_boot=200, seed=0):
    """Bootstrap the ratio MMD(real,les)/MMD(real,ctl) by resampling images.

    Returns dict with raw deltas and a `measurable` flag: if either delta is below `floor`
    the MMD has underflowed to ~0 (perturbation too subtle to move the distribution) and the
    ratio is a 0/0 artifact — flagged so downstream plotting/summary can exclude it.
    """
    rng = np.random.default_rng(seed)
    nr, nl, nc = real_f.shape[0], les_f.shape[0], ctl_f.shape[0]
    d_les = mmd_fn(real_f, les_f)
    d_ctl = mmd_fn(real_f, ctl_f)
    point = d_les / max(d_ctl, 1e-12)
    measurable = (d_les >= floor) and (d_ctl >= floor)
    vals = []
    for _ in range(n_boot):
        ri, li, ci = rng.integers(0, nr, nr), rng.integers(0, nl, nl), rng.integers(0, nc, nc)
        bl = mmd_fn(real_f[ri], les_f[li])
        bc = mmd_fn(real_f[ri], ctl_f[ci])
        vals.append(bl / max(bc, 1e-12))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"ratio": float(point), "lo": float(lo), "hi": float(hi),
            "d_lesion": float(d_les), "d_control": float(d_ctl), "measurable": bool(measurable)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgg_dir", default=os.path.join(ROOT, "data_mri", "lgg_with_masks"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lgg_noise_sweep"))
    ap.add_argument("--sigmas", type=float, nargs="+", default=[10, 20, 30, 45, 60, 90])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    pairs = load_pairs(args.lgg_dir, args.n * 3)
    print(f"[data] loaded {len(pairs)} pairs")

    # Build usable set with BOTH controls (skip if either control is invalid → fair contrast).
    real_arrs, tum_masks, mir_masks, tex_masks = [], [], [], []
    skip_mir = skip_tex = 0
    for rgb, tum in pairs:
        brain = brain_mask(rgb)
        mir, ok_m = mirrored_control(tum, brain, tum)
        if not ok_m:
            skip_mir += 1
            continue
        tex, ok_t = texture_matched_control(tum, brain, rgb)
        if not ok_t:
            skip_tex += 1
            continue
        real_arrs.append(rgb); tum_masks.append(tum)
        mir_masks.append(mir); tex_masks.append(tex)
        if len(real_arrs) >= args.n:
            break
    N = len(real_arrs)
    tum_e = np.mean([gradient_energy(r.max(2))[m].sum() for r, m in zip(real_arrs, tum_masks)])
    mir_e = np.mean([gradient_energy(r.max(2))[m].sum() for r, m in zip(real_arrs, mir_masks)])
    tex_e = np.mean([gradient_energy(r.max(2))[m].sum() for r, m in zip(real_arrs, tex_masks)])
    print(f"[control] usable={N}  skip(mirror)={skip_mir} skip(texture)={skip_tex}")
    print(f"[texture] mean gradient-energy  tumour={tum_e:.0f}  mirror={mir_e:.0f}  "
          f"texture-matched={tex_e:.0f}  (texture control closes the gap: "
          f"{100*abs(tex_e-tum_e)/tum_e:.0f}% vs mirror {100*abs(mir_e-tum_e)/tum_e:.0f}%)")

    metric = M3EntropyMetric(device=args.device, single_layer=12, seed=args.seed)
    L = metric.num_layers

    def m3_feats(arrs):
        return metric._extract_raw_features(to_m3(arrs, args.device), layers_to_keep={L})[L - 1].cpu().numpy()

    def m3_mmd(a, b):
        return metric._mmd2(torch.from_numpy(a).to(args.device),
                            torch.from_numpy(b).to(args.device)).item()

    real_m3 = m3_feats(real_arrs)
    real_kid = inception_feats(to_fid01(real_arrs, args.device), args.device)

    results = {"config": {"n_used": N, "sigmas": args.sigmas, "seed": args.seed,
                          "gradient_energy": {"tumour": float(tum_e), "mirror": float(mir_e),
                                              "texture_matched": float(tex_e)}},
               "sweep": []}

    for sig in args.sigmas:
        les = [add_noise(r, m, sig, args.seed + i) for i, (r, m) in enumerate(zip(real_arrs, tum_masks))]
        mir = [add_noise(r, m, sig, args.seed + i) for i, (r, m) in enumerate(zip(real_arrs, mir_masks))]
        tex = [add_noise(r, m, sig, args.seed + i) for i, (r, m) in enumerate(zip(real_arrs, tex_masks))]

        les_m3, mir_m3, tex_m3 = m3_feats(les), m3_feats(mir), m3_feats(tex)
        les_k = inception_feats(to_fid01(les, args.device), args.device)
        mir_k = inception_feats(to_fid01(mir, args.device), args.device)
        tex_k = inception_feats(to_fid01(tex, args.device), args.device)

        # Detection floors: M3 MMD^2 lives ~1e-3..1e-2; KID ~1e-2..1e-1. Below floor => underflow.
        row = {"sigma": sig}
        row["M3_mirror"] = ratio_ci(m3_mmd, real_m3, les_m3, mir_m3, 5e-4, args.n_boot, args.seed)
        row["M3_texture"] = ratio_ci(m3_mmd, real_m3, les_m3, tex_m3, 5e-4, args.n_boot, args.seed)
        row["KID_mirror"] = ratio_ci(kid_mmd2, real_kid, les_k, mir_k, 1e-3, args.n_boot, args.seed)
        row["KID_texture"] = ratio_ci(kid_mmd2, real_kid, les_k, tex_k, 1e-3, args.n_boot, args.seed)
        results["sweep"].append(row)
        def _f(r):
            return f"{r['ratio']:.2f}" if r["measurable"] else "n/a"
        print(f"[sig={sig:5.0f}] M3(mir)={_f(row['M3_mirror'])} M3(tex)={_f(row['M3_texture'])} "
              f"| KID(mir)={_f(row['KID_mirror'])} KID(tex)={_f(row['KID_texture'])}")

    with open(os.path.join(args.out, "sweep.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Figure ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sg = [r["sigma"] for r in results["sweep"]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        styles = {"M3_mirror": ("#c0392b", "-", "M3 (mirror control)"),
                  "M3_texture": ("#e67e22", "--", "M3 (texture-matched control)"),
                  "KID_mirror": ("#2c3e50", "-", "KID/Inception (mirror)"),
                  "KID_texture": ("#7f8c8d", "--", "KID/Inception (texture)")}
        for key, (c, ls, lab) in styles.items():
            # Plot only measurable points (exclude MMD-underflow 0/0 artifacts).
            sig_m = [r["sigma"] for r in results["sweep"] if r[key]["measurable"]]
            pts = [r[key]["ratio"] for r in results["sweep"] if r[key]["measurable"]]
            lo = [r[key]["lo"] for r in results["sweep"] if r[key]["measurable"]]
            hi = [r[key]["hi"] for r in results["sweep"] if r[key]["measurable"]]
            if not sig_m:
                continue
            ax.plot(sig_m, pts, ls, color=c, label=lab, linewidth=2, marker="o", markersize=4)
            ax.fill_between(sig_m, lo, hi, color=c, alpha=0.12)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_ylim(0.8, None)
        ax.set_xlabel("noise sigma (region-local additive Gaussian)")
        ax.set_ylabel("lesion / healthy specificity ratio")
        ax.set_title("LGG lesion-selectivity vs perturbation strength\n(real expert masks, N=%d)" % N)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "sweep.png"), dpi=130)
        print(f"[fig] -> {os.path.join(args.out, 'sweep.png')}")
    except Exception as e:
        print("plot skipped:", e)

    print(f"[done] {time.time()-t0:.1f}s -> {os.path.join(args.out, 'sweep.json')}")


if __name__ == "__main__":
    main()
