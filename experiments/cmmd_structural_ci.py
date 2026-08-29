"""
experiments/cmmd_structural_ci.py
=================================
Statistical support for the paper's central claim (reviewer 3.3): CMMD suffers a *structural
ordering failure* -- it cannot separate a weak same-modality generator (WDM-3D MRI) from a full
cross-modality shift (LIDC CT), while the M3 fidelity axis does. We add bootstrap 95% CIs on both
CMMD values and on their paired difference Delta, and the same for M3, so the claim rests on
CIs, not point estimates.

Structural-failure test: if the two CMMD CIs overlap (equivalently, the Delta CI includes ~0),
CMMD cannot statistically distinguish the two regimes. M3 should show a Delta CI excluding 0.

Usage: python experiments/cmmd_structural_ci.py --n 400 --n_boot 500 --device cuda
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def paths_in(d, n):
    ps = []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        ps.extend(glob.glob(os.path.join(d, "**", e), recursive=True))
    ps = [p for p in sorted(ps) if "_segmask_" not in os.path.basename(p)]
    return ps[:n]


def boot_mmd(feat_a, feat_b, mmd_fn, n_boot, seed):
    """Bootstrap point + 95% CI of mmd(a,b) by resampling both sets with replacement."""
    rng = np.random.default_rng(seed)
    na, nb = len(feat_a), len(feat_b)
    point = mmd_fn(feat_a, feat_b)
    vals = []
    for _ in range(n_boot):
        vals.append(mmd_fn(feat_a[rng.integers(0, na, na)], feat_b[rng.integers(0, nb, nb)]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), np.array(vals)


def paired_delta_ci(feat_real, feat_x, feat_y, mmd_fn, n_boot, seed):
    """Bootstrap CI of Delta = mmd(real,y) - mmd(real,x) using a shared real resample."""
    rng = np.random.default_rng(seed)
    nr, nx, ny = len(feat_real), len(feat_x), len(feat_y)
    point = mmd_fn(feat_real, feat_y) - mmd_fn(feat_real, feat_x)
    d = []
    for _ in range(n_boot):
        ri = rng.integers(0, nr, nr)
        r = feat_real[ri]
        d.append(mmd_fn(r, feat_y[rng.integers(0, ny, ny)]) - mmd_fn(r, feat_x[rng.integers(0, nx, nx)]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cmmd_structural"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    real_d = os.path.join(ROOT, "data_mri", "brats_axial_multislice")
    ddpm_d = os.path.join(ROOT, "output", "generated_500_standard")     # same-domain DDPM
    mri_d  = os.path.join(ROOT, "output", "generated_wdm3d", "brats")   # weak same-modality
    ct_d   = os.path.join(ROOT, "output", "generated_wdm3d", "lidc")    # cross-modality CT
    real_p = paths_in(real_d, args.n)
    ddpm_p, mri_p, ct_p = paths_in(ddpm_d, args.n), paths_in(mri_d, args.n), paths_in(ct_d, args.n)
    print(f"[data] real={len(real_p)} DDPM={len(ddpm_p)} wdm3d-MRI={len(mri_p)} LIDC-CT={len(ct_p)}")

    report = {"config": {"n": args.n, "n_boot": args.n_boot, "seed": args.seed}}

    # ---------- CMMD (CLIP ViT-L/14) ----------
    from evaluation.cmmd_metric import CMMDMetric
    cm = CMMDMetric(device=args.device)
    fr = cm.extract_features(real_p, "real")
    fd = cm.extract_features(ddpm_p, "ddpm")
    fm = cm.extract_features(mri_p, "wdm3d-mri")
    fc = cm.extract_features(ct_p, "lidc-ct")
    mmd = lambda a, b: CMMDMetric.gaussian_mmd2_unbiased(a, b)
    c_ddpm = boot_mmd(fr, fd, mmd, args.n_boot, args.seed + 3)
    c_mri = boot_mmd(fr, fm, mmd, args.n_boot, args.seed)
    c_ct  = boot_mmd(fr, fc, mmd, args.n_boot, args.seed + 1)
    c_delta = paired_delta_ci(fr, fm, fc, mmd, args.n_boot, args.seed + 2)
    overlap = not (c_mri[2] < c_ct[1] or c_ct[2] < c_mri[1])
    report["CMMD"] = {
        "real_vs_ddpm":      {"cmmd": c_ddpm[0], "ci95": [c_ddpm[1], c_ddpm[2]]},
        "real_vs_wdm3d_mri": {"cmmd": c_mri[0], "ci95": [c_mri[1], c_mri[2]]},
        "real_vs_lidc_ct":   {"cmmd": c_ct[0],  "ci95": [c_ct[1],  c_ct[2]]},
        "delta_ct_minus_mri": {"delta": c_delta[0], "ci95": [c_delta[1], c_delta[2]],
                               "includes_zero": bool(c_delta[1] <= 0 <= c_delta[2])},
        "rank_inversion_ct_below_mri": bool(c_ct[0] < c_mri[0]),
        "ci_overlap": bool(overlap),
    }
    print(f"[CMMD] DDPM={c_ddpm[0]:.4f} CI[{c_ddpm[1]:.4f},{c_ddpm[2]:.4f}] | "
          f"MRI={c_mri[0]:.4f} CI[{c_mri[1]:.4f},{c_mri[2]:.4f}] | "
          f"CT={c_ct[0]:.4f} CI[{c_ct[1]:.4f},{c_ct[2]:.4f}] | "
          f"Delta(CT-MRI)={c_delta[0]:+.4f} CI[{c_delta[1]:+.4f},{c_delta[2]:+.4f}] "
          f"invert={report['CMMD']['rank_inversion_ct_below_mri']}")
    del cm, fr, fd, fm, fc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- M3 (RadioDINO-s16 L12) ----------
    from evaluation.m3_score_v2 import M3EntropyMetric
    t01 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    load = lambda ps: torch.stack([t01(Image.open(p).convert("RGB")) for p in ps])
    m = M3EntropyMetric(device=args.device, single_layer=12, seed=args.seed)
    L = m.num_layers
    ef = lambda ps: m._extract_raw_features(load(ps).to(args.device), layers_to_keep={L})[L - 1].cpu().numpy()
    ed, em, ec = ef(ddpm_p), ef(mri_p), ef(ct_p)
    er = ef(real_p)
    m3mmd = lambda a, b: m._mmd2(torch.from_numpy(a).to(args.device), torch.from_numpy(b).to(args.device)).item()
    x_ddpm = boot_mmd(er, ed, m3mmd, args.n_boot, args.seed + 3)
    x_mri = boot_mmd(er, em, m3mmd, args.n_boot, args.seed)
    x_ct  = boot_mmd(er, ec, m3mmd, args.n_boot, args.seed + 1)
    x_delta = paired_delta_ci(er, em, ec, m3mmd, args.n_boot, args.seed + 2)
    m_overlap = not (x_mri[2] < x_ct[1] or x_ct[2] < x_mri[1])
    report["M3"] = {
        "real_vs_ddpm":      {"mmd2": x_ddpm[0], "ci95": [x_ddpm[1], x_ddpm[2]]},
        "real_vs_wdm3d_mri": {"mmd2": x_mri[0], "ci95": [x_mri[1], x_mri[2]]},
        "real_vs_lidc_ct":   {"mmd2": x_ct[0],  "ci95": [x_ct[1],  x_ct[2]]},
        "delta_ct_minus_mri": {"delta": x_delta[0], "ci95": [x_delta[1], x_delta[2]],
                               "includes_zero": bool(x_delta[1] <= 0 <= x_delta[2])},
        "correct_order_ct_above_mri": bool(x_ct[0] > x_mri[0]),
        "ci_overlap": bool(m_overlap),
    }
    print(f"[M3]   DDPM={x_ddpm[0]:.4f} CI[{x_ddpm[1]:.4f},{x_ddpm[2]:.4f}] | "
          f"MRI={x_mri[0]:.4f} CI[{x_mri[1]:.4f},{x_mri[2]:.4f}] | "
          f"CT={x_ct[0]:.4f} CI[{x_ct[1]:.4f},{x_ct[2]:.4f}] | "
          f"Delta(CT-MRI)={x_delta[0]:+.4f} CI[{x_delta[1]:+.4f},{x_delta[2]:+.4f}]")

    with open(os.path.join(args.out, "structural_ci.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] -> {os.path.join(args.out, 'structural_ci.json')}")


if __name__ == "__main__":
    main()
