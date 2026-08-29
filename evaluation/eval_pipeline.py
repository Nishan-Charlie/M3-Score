"""
evaluate.py — Unified MRI Diffusion Evaluation CLI
====================================================
Runs any combination of evaluation metrics against a real/<->generated image pair,
writes a consolidated `evaluation_report.json`, and saves all plots to --output_dir.

Metrics available
-----------------
  fid          — Fréchet Inception Distance       (torchmetrics)
  kid          — Kernel Inception Distance         (torchmetrics)
  ssim         — SSIM / PSNR nearest-neighbour     (pytorch_msssim)
  alpha        — α-precision, β-recall, Authenticity, Improved P&R
  tsne         — t-SNE real-vs-gen scatter plot
  downstream   — CNN downstream classification
  anomaly      — Isolation Forest + Autoencoder anomaly detection
  extended     — MS-SSIM, LPIPS, IS, Coverage, Density
  all          — All of the above

Usage
-----
    python evaluate.py \\
        --real_dir  /path/to/real \\
        --gen_dir   /path/to/generated \\
        --output_dir /path/to/results \\
        --metrics all \\
        --num_images 1000 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Ensure project root is on sys.path so metrics/ package is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _safe_run(fn, section_name: str, fallback: dict | None = None) -> dict:
    """Run fn(), catching exceptions gracefully and returning fallback."""
    try:
        return fn()
    except Exception as exc:
        print(f"  [WARN] {section_name} failed: {exc}")
        return fallback or {"error": str(exc)}


# ---------------------------------------------------------------------------
# FID / KID / SSIM / PSNR  (from existing generate.py logic, refactored)
# ---------------------------------------------------------------------------

def _fid_kid_ssim(args, use_tqdm) -> dict:
    """Re-use torchmetrics FID/KID and pytorch_msssim SSIM/PSNR."""
    import glob
    import torch
    import torch.nn.functional as F
    from PIL import Image

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from pytorch_msssim import ssim as pt_ssim
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    from tqdm.auto import tqdm

    device = args.device

    def load_rgb(directory, n=None):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        paths = sorted(paths)[:n]
        imgs = []
        for p in tqdm(paths, desc=f"Load RGB {os.path.basename(directory)}", disable=not use_tqdm):
            arr = np.array(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.uint8)
            imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
        return torch.stack(imgs)

    def load_gray(directory, n=None):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        paths = sorted(paths)[:n]
        imgs = []
        for p in tqdm(paths, desc=f"Load gray {os.path.basename(directory)}", disable=not use_tqdm):
            arr = np.array(Image.open(p).convert("L").resize((256, 256)), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).unsqueeze(0))
        return torch.stack(imgs)

    N = args.num_images
    real_rgb = load_rgb(args.real_dir, N).to(device)
    gen_rgb  = load_rgb(args.gen_dir,  N).to(device)

    # FID
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.update(real_rgb, real=True)
    fid.update(gen_rgb, real=False)
    fid_score = float(fid.compute().item())

    # KID
    kid_ss = min(50, len(gen_rgb))
    kid = KernelInceptionDistance(subset_size=kid_ss).to(device)
    kid.update(real_rgb, real=True)
    kid.update(gen_rgb, real=False)
    kid_mean, kid_std = kid.compute()

    # SSIM / PSNR
    real_gray = load_gray(args.real_dir, N).to(device)
    gen_gray  = load_gray(args.gen_dir,  N).to(device)

    def pt_psnr(a, b, dr):
        mse = torch.mean((a - b) ** 2)
        if mse == 0:
            return 100.0
        return (20 * torch.log10(torch.tensor(dr, device=a.device))
                - 10 * torch.log10(mse)).item()

    num_real = len(real_gray)
    comp = min(50, num_real)
    ssim_vals, psnr_vals = [], []
    for i in tqdm(range(len(gen_gray)), desc="SSIM/PSNR", disable=not use_tqdm):
        g = gen_gray[i:i+1]
        sub_idx = torch.randperm(num_real)[:comp]
        best_s, best_p = -1.0, 0.0
        for j in range(comp):
            r = real_gray[sub_idx[j]:sub_idx[j]+1]
            dr = max(float(r.max() - r.min()), 1.0)
            s = pt_ssim(g, r, data_range=dr, size_average=True).item()
            if s > best_s:
                best_s = s
                best_p = pt_psnr(g, r, dr)
        ssim_vals.append(best_s)
        psnr_vals.append(best_p)

    return {
        "fid": round(fid_score, 4),
        "kid_mean": round(float(kid_mean.item()), 6),
        "kid_std":  round(float(kid_std.item()),  6),
        "ssim":     round(float(np.mean(ssim_vals)), 6),
        "psnr":     round(float(np.mean(psnr_vals)), 4),
    }


# ---------------------------------------------------------------------------
# M3-Score
# ---------------------------------------------------------------------------

def _m3_score(args, use_tqdm) -> dict:
    """Evaluates the MS-MMD (M3-Score) Medical Metric."""
    import torch
    import glob
    from torchvision import transforms
    from tqdm.auto import tqdm
    from PIL import Image
    from evaluation.m3_score_v2 import M3V2Metric

    def load_images_for_m3(directory, n):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        paths = sorted(paths)[:n] if n else paths
        imgs = []

        # M3 Metric preprocesses internally — load raw [0,1] tensors
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        for p in tqdm(paths, desc=f"Load M3 {os.path.basename(directory)}", disable=not use_tqdm):
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                pass

        if not imgs:
            return torch.empty(0)
        return torch.stack(imgs)

    device = args.device
    m3 = M3V2Metric(device=device)

    real_images = load_images_for_m3(args.real_dir, n=args.num_images).to(device)
    gen_images = load_images_for_m3(args.gen_dir, n=args.num_images).to(device)

    if len(real_images) == 0 or len(gen_images) == 0:
         return {"error": "Not enough images to compute M3-Score."}

    with torch.no_grad():
        m3.prune_layers_via_cka(real_images[:20])
        results = m3(real_images, gen_images)

    return results


# ---------------------------------------------------------------------------
# Summary radar chart
# ---------------------------------------------------------------------------

def _plot_radar(metrics: dict, output_dir: str):
    """Radar chart of normalised scalar metrics (best-effort)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect scalar metrics we want to show (normalised where needed)
    candidates = {
        "α-Precision":       ("alpha_precision",   1.0, False),
        "β-Recall":          ("beta_recall",        1.0, False),
        "Authenticity":      ("authenticity",       1.0, False),
        "Imp. Precision":    ("improved_precision", 1.0, False),
        "Imp. Recall":       ("improved_recall",    1.0, False),
        "SSIM":              ("ssim",               1.0, False),
        "MS-SSIM":           ("ms_ssim",            1.0, False),
        "Coverage":          ("coverage",           1.0, False),
        "Density":           ("density",            1.0, False),
        "1-LPIPS":           ("lpips",              1.0, True),    # invert: lower=better
        "Rad-FID":           ("rad_fid",            50.0, True),   # scale: 50.0, invert: lower=better
    }

    labels, values = [], []
    flat = {}
    # Flatten nested metric dict
    for v in metrics.values():
        if isinstance(v, dict):
            flat.update(v)
    flat.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})

    for label, (key, scale, invert) in candidates.items():
        val = flat.get(key)
        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
            nv = val / scale
            if invert:
                nv = 1.0 - nv
            nv = float(np.clip(nv, 0.0, 1.0))
            labels.append(label)
            values.append(nv)

    if len(labels) < 3:
        return  # Not enough metrics for radar

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals   = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True}, dpi=120)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.fill(angles, vals, color="#1565C0", alpha=0.20)
    ax.plot(angles, vals, color="#1565C0", linewidth=2)
    ax.scatter(angles[:-1], vals[:-1], color="#c62828", s=60, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="#222222")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#666666", fontsize=7)
    ax.spines["polar"].set_color("#cccccc")
    ax.set_title("Evaluation Metrics — Radar", color="#222222", fontsize=13, pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_radar.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[evaluate] Radar chart → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified MRI diffusion model evaluation toolkit.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--real_dir",    required=True,  help="Directory of real MRI images")
    parser.add_argument("--gen_dir",     required=True,  help="Directory of generated images")
    parser.add_argument("--output_dir",  default="./evaluation_output", help="Where to save all results")
    parser.add_argument("--metrics",     nargs="+", default=["all"],
                        choices=["all", "fid", "kid", "ssim", "alpha", "tsne",
                                 "downstream", "anomaly", "extended", "m3", "rad_fid",
                                 "coverage", "novelty", "statistical", "conditional"],
                        help="Which metrics to compute. 'all' runs everything.")
    parser.add_argument("--rad_fid_checkpoint", default="output/radiodino_segmentation/backbone_final.pth",
                        help="Path to fine-tuned RaDDINO backbone (.pth)")
    parser.add_argument("--num_images",  type=int,  default=None,  help="Limit images per split (None=all)")
    parser.add_argument("--k",           type=int,  default=5,     help="k for k-NN manifold (α/β/coverage)")
    parser.add_argument("--batch_size",  type=int,  default=32,    help="Batch size for feature extraction")
    parser.add_argument("--device",      default="cpu",            help="Torch device (e.g. cuda:0, cpu)")
    parser.add_argument("--no_tqdm",     action="store_true",      help="Disable progress bars")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_n_iter",     type=int,   default=1000)
    parser.add_argument("--downstream_mode", choices=["augment", "only_gen"], default="augment")
    parser.add_argument("--downstream_epochs", type=int, default=10)
    parser.add_argument("--anomaly_method",    choices=["iforest", "autoencoder", "both"], default="both")
    parser.add_argument("--anomaly_top_n",     type=int, default=10)
    parser.add_argument("--ae_epochs",         type=int, default=15)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--backbone_id",       default="Snarcy/RadioDino-s16",
                        help="timm backbone for coverage/statistical/conditional metrics")
    parser.add_argument("--n_perm",            type=int, default=500,
                        help="Permutations for statistical rigor p-value")
    parser.add_argument("--n_boot",            type=int, default=200,
                        help="Bootstrap resamples for confidence intervals")
    parser.add_argument("--stratify_by",       default="intensity_quartile",
                        choices=["intensity_quartile", "slice_position", "kmeans"],
                        help="Stratification strategy for conditional MMD")
    parser.add_argument("--n_strata",          type=int, default=4,
                        help="Number of strata for conditional MMD")

    args = parser.parse_args()
    use_tqdm = not args.no_tqdm
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve "all" shorthand
    all_metrics = {"fid", "kid", "ssim", "alpha", "tsne", "downstream", "anomaly", "extended", "m3", "rad_fid"}
    requested = all_metrics if "all" in args.metrics else set(args.metrics)

    report: dict = {
        "config": {
            "real_dir":   args.real_dir,
            "gen_dir":    args.gen_dir,
            "metrics":    sorted(requested),
            "num_images": args.num_images,
            "device":     args.device,
        }
    }

    t0 = time.time()

    # ---- FID / KID / SSIM / PSNR ------------------------------------------
    needs_base = requested & {"fid", "kid", "ssim"}
    if needs_base:
        _section("FID / KID / SSIM / PSNR")
        report["base_metrics"] = _safe_run(
            lambda: _fid_kid_ssim(args, use_tqdm), "FID/KID/SSIM"
        )

    # ---- α-precision / β-recall / Authenticity / Improved P&R -------------
    if "alpha" in requested:
        _section("α-Precision / β-Recall / Authenticity / Improved P&R")
        from metrics.alpha_precision_recall import compute_manifold_metrics
        report["manifold_metrics"] = _safe_run(
            lambda: compute_manifold_metrics(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                num_images=args.num_images,
                k=args.k,
                batch_size=args.batch_size,
                device=args.device,
                use_tqdm=use_tqdm,
            ),
            "Manifold metrics",
        )

    # ---- t-SNE Visualisation -----------------------------------------------
    if "tsne" in requested:
        _section("t-SNE Visualisation")
        from metrics.tsne_visualizer import plot_tsne
        def _tsne():
            path = plot_tsne(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=args.output_dir,
                num_images=args.num_images,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_n_iter,
                batch_size=args.batch_size,
                device=args.device,
                use_tqdm=use_tqdm,
                random_state=args.seed,
            )
            return {"tsne_plot": path}
        report["tsne"] = _safe_run(_tsne, "t-SNE")

    # ---- Downstream classification -----------------------------------------
    if "downstream" in requested:
        _section("Downstream Classification")
        from metrics.downstream_classifier import evaluate_downstream
        report["downstream"] = _safe_run(
            lambda: evaluate_downstream(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "downstream"),
                mode=args.downstream_mode,
                num_images=args.num_images,
                epochs=args.downstream_epochs,
                batch_size=args.batch_size,
                device=args.device,
                use_tqdm=use_tqdm,
                seed=args.seed,
            ),
            "Downstream classifier",
        )

    # ---- Anomaly detection -------------------------------------------------
    if "anomaly" in requested:
        _section("Anomaly Detection")
        from metrics.anomaly_detection import run_anomaly_detection
        report["anomaly"] = _safe_run(
            lambda: run_anomaly_detection(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "anomaly"),
                method=args.anomaly_method,
                num_images=args.num_images,
                batch_size=args.batch_size,
                ae_epochs=args.ae_epochs,
                top_n=args.anomaly_top_n,
                device=args.device,
                use_tqdm=use_tqdm,
                seed=args.seed,
            ),
            "Anomaly detection",
        )

    # ---- Extended metrics --------------------------------------------------
    if "extended" in requested:
        _section("Extended Metrics (MS-SSIM / LPIPS / IS / Coverage / Density)")
        from metrics.extended_tests import compute_extended_metrics
        report["extended_metrics"] = _safe_run(
            lambda: compute_extended_metrics(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "extended"),
                num_images=args.num_images,
                k=args.k,
                batch_size=args.batch_size,
                device=args.device,
                use_tqdm=use_tqdm,
            ),
            "Extended metrics",
        )

    # ---- M3-Score ----------------------------------------------------------
    if "m3" in requested:
        _section("M3-Score Metric")
        report["m3_score"] = _safe_run(
            lambda: _m3_score(args, use_tqdm), "M3-Score"
        )

    # ---- Coverage & Novelty (Axes 2 & 3) -----------------------------------
    if "coverage" in requested or "novelty" in requested or "all" in requested:
        _section("Coverage & Novelty (Axes 2 & 3)")
        from evaluation.coverage_novelty import compute_coverage_novelty
        from experiments._shared_utils import load_pils_recursive
        report["coverage_novelty"] = _safe_run(
            lambda: compute_coverage_novelty(
                real_imgs=load_pils_recursive(args.real_dir, n=args.num_images),
                gen_imgs =load_pils_recursive(args.gen_dir,  n=args.num_images),
                backbone_id=args.backbone_id,
                device=args.device,
                k_neighbors=5,
                pct_memorize=5.0,
                batch_size=args.batch_size,
            ),
            "Coverage & Novelty",
        )

    # ---- Statistical Rigor (p-value, effect size, CI) ----------------------
    if "statistical" in requested or "all" in requested:
        _section("Statistical Rigor (permutation p-value + bootstrap CI)")
        from evaluation.statistical_rigor import statistical_m3
        from experiments._shared_utils import load_pils_recursive
        report["statistical_rigor"] = _safe_run(
            lambda: statistical_m3(
                real_imgs=load_pils_recursive(args.real_dir, n=args.num_images),
                gen_imgs =load_pils_recursive(args.gen_dir,  n=args.num_images),
                backbone_id=args.backbone_id,
                device=args.device,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                ci_level=0.95,
                batch_size=args.batch_size,
                single_layer=12,
            ),
            "Statistical Rigor",
        )

    # ---- Conditional MMD (per-stratum worst-case) --------------------------
    if "conditional" in requested or "all" in requested:
        _section("Conditional MMD (per-stratum worst-case)")
        from evaluation.conditional_mmd import run_conditional_mmd
        report["conditional_mmd"] = _safe_run(
            lambda: run_conditional_mmd(
                real_dir    = args.real_dir,
                gen_dir     = args.gen_dir,
                stratify_by = args.stratify_by,
                n_strata    = args.n_strata,
                n_images    = args.num_images,
                backbone_id = getattr(args, "backbone_id", "Snarcy/RadioDino-s16"),
                device      = args.device,
                output_dir  = os.path.join(args.output_dir, "conditional_mmd"),
            ),
            "Conditional MMD",
        )

    # ---- Rad-FID (RaDDINO-FID) ---------------------------------------------
    if "rad_fid" in requested:
        _section("Rad-FID (RaDDINO-based Fréchet Distance)")
        from metrics.rad_fid import compute_rad_fid
        report["rad_fid"] = _safe_run(
            lambda: compute_rad_fid(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                checkpoint_path=args.rad_fid_checkpoint,
                num_images=args.num_images or 1000,
                batch_size=args.batch_size,
                device=args.device,
                use_tqdm=use_tqdm,
            ),
            "Rad-FID",
        )

    # ---- Radar chart -------------------------------------------------------
    _plot_radar(report, args.output_dir)

    # ---- Save consolidated report ------------------------------------------
    report["elapsed_seconds"] = round(time.time() - t0, 2)
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    _section("Evaluation Complete")
    print(f"  Total time : {report['elapsed_seconds']:.1f}s")
    print(f"  Report     : {report_path}")
    print(f"  Output dir : {args.output_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
