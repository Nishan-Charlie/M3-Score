"""
experiments/frd_comparison.py
==============================
Experiment: FRD vs M3-Score comparison (Section 3.12).

Computes FRD (Frechet Radiomic Distance) alongside M3-Score, FID, and KID
on the same real/generated image sets so all four metrics can be compared in
the same table.

Dependencies:
    pip install pytorch-fid
    pip install frd-score
    pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master

If any dependency is missing the corresponding metric returns NaN and a
warning is printed; the other metrics still run.

Outputs:
    <output_dir>/frd_comparison.json   -- raw scores for all metrics
    <output_dir>/frd_comparison.tex    -- LaTeX table (requires booktabs)
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_images_from_dir(
    directory: str,
    num_images: Optional[int],
) -> torch.Tensor:
    """
    Load images from a directory (recursively) as a CPU float tensor of
    shape (N, 3, 224, 224) in [0, 1].

    Grayscale images are expanded to three channels.
    Multi-page TIFF files: PIL loads only the first frame.

    Returns a CPU tensor; device transfer is handled inside each metric.
    """
    from torchvision import transforms
    from torchvision.datasets.folder import default_loader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))

    if num_images is not None:
        paths = paths[:num_images]

    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")

    imgs = torch.stack([transform(default_loader(p)) for p in paths])
    return imgs   # CPU tensor, shape (N, 3, 224, 224)


# ---------------------------------------------------------------------------
# InceptionV3 feature extraction (shared for FID and KID)
# ---------------------------------------------------------------------------

def _load_uint8_rgb(directory: str, n: Optional[int] = None) -> torch.Tensor:
    """uint8 RGB tensors at 256x256, matching evaluation/eval_pipeline.load_rgb.

    torchmetrics' FID/KID expect uint8 input and handle the 299x299 resize and
    the FID-specific Inception normalisation internally. Keeping this identical
    to eval_pipeline is what makes the FID reported here comparable with the
    rest of the paper.
    """
    from PIL import Image
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    paths: list = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(paths)[:n]
    imgs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.uint8)
        imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
    return torch.stack(imgs)


def _extract_inception_features(
    imgs: torch.Tensor,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Extract 2048-d InceptionV3 pool3 features using torchvision.
    """
    from torchvision import models, transforms
    import torch.nn as nn

    try:
        # Load InceptionV3 with pretrained weights
        net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        net.fc = nn.Identity()
        net.eval().to(device)

        # Standard Inception preprocessing (Resize and ToTensor already done in _load_images)
        # But we need to handle the normalization specifically if not done.
        # Images are in [0, 1]. Inception expects [-1, 1] usually or specific normalization.
        # We'll use the same as ood_detection for consistency.
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 32):
                batch = imgs[i:i + 32].to(device)
                # Normalize
                batch = torch.stack([norm(b) for b in batch])
                out = net(batch)
                feats.append(out.cpu())
        
        return torch.cat(feats, dim=0)
    except Exception as exc:
        print(f"  [WARN] Inception feature extraction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def _compute_fid_from_features(
    f_r: np.ndarray,
    f_g: np.ndarray,
) -> float:
    """
    Frechet Inception Distance from pre-extracted feature arrays.

    Applies a numerical stability offset (eps=1e-6) to the covariance
    matrices before the matrix square root, matching the standard
    pytorch-fid implementation and preventing incorrect results when
    covariance matrices are ill-conditioned at small sample sizes.
    """
    from scipy.linalg import sqrtm

    mu_r, sig_r = f_r.mean(0), np.cov(f_r, rowvar=False)
    mu_g, sig_g = f_g.mean(0), np.cov(f_g, rowvar=False)
    diff = mu_r - mu_g

    eps = 1e-6
    offset = np.eye(sig_r.shape[0]) * eps
    covmean = sqrtm((sig_r + offset) @ (sig_g + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))


# ---------------------------------------------------------------------------
# KID
# ---------------------------------------------------------------------------

def _compute_kid_from_features(
    f_r: torch.Tensor,
    f_g: torch.Tensor,
) -> float:
    """
    Kernel Inception Distance (polynomial kernel MMD^2) from pre-extracted
    feature tensors.

    Uses the same degree-3 polynomial kernel as M3V2Metric for consistency,
    with gamma = 1/D where D is the feature dimension.
    """
    def poly_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 / a.shape[1]
        return (gamma * torch.mm(a, b.t()) + 1.0) ** 3

    K_XX = poly_kernel(f_r, f_r)
    K_YY = poly_kernel(f_g, f_g)
    K_XY = poly_kernel(f_r, f_g)

    m = K_XX.shape[0]
    n = K_YY.shape[0]
    mmd2 = (
        (K_XX.sum() - torch.trace(K_XX)) / (m * (m - 1))
        + (K_YY.sum() - torch.trace(K_YY)) / (n * (n - 1))
        - 2.0 * K_XY.mean()
    ).clamp(min=0.0)
    return float(mmd2)


# ---------------------------------------------------------------------------
# FRD
# ---------------------------------------------------------------------------

def _compute_frd(
    real_dir: str,
    gen_dir: str,
    num_images: Optional[int] = None,
    num_workers: int = 4,
) -> float:
    """
    Frechet Radiomic Distance using the official frd-score package.

    If num_images is set, we create temporary directories with copies to
    the first num_images to ensure FRD is computed on the same subset as
    other metrics.

    num_workers caps frd-score's multiprocessing pool. Its own default is
    cpu_count() - 2, which on a 20-core machine spawns 18 processes that each
    re-import numpy/scipy/pyradiomics; with limited free RAM this exhausts the
    Windows paging file and dies with

        ImportError: DLL load failed while importing bit_generator:
        The paging file is too small for this operation to complete.

    Radiomic extraction is I/O- and CPU-bound per image, so a small pool costs
    little wall-clock time and keeps the run within memory.
    """
    import tempfile
    import shutil
    try:
        from frd_score import frd
    except ImportError:
        return float("nan")

    tmp_real = None
    tmp_gen = None
    
    try:
        if num_images is not None:
            # Create temp directories
            tmp_real = tempfile.mkdtemp(prefix="frd_real_")
            tmp_gen = tempfile.mkdtemp(prefix="frd_gen_")
            
            # Helper to copy subset
            def _copy_subset(src, dst, n):
                paths = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
                    paths.extend(glob.glob(os.path.join(src, "**", ext), recursive=True))
                paths = sorted(set(paths))[:n]
                for i, p in enumerate(paths):
                    shutil.copy2(p, os.path.join(dst, f"img_{i:06d}{os.path.splitext(p)[1]}"))
            
            _copy_subset(real_dir, tmp_real, num_images)
            _copy_subset(gen_dir,  tmp_gen,  num_images)
            
            # Pass list of paths to compute_frd
            score = frd.compute_frd([tmp_real, tmp_gen], num_workers=num_workers)
        else:
            score = frd.compute_frd([real_dir, gen_dir], num_workers=num_workers)
            
        return float(score)
    except Exception as exc:
        print(f"  [WARN] FRD computation failed: {exc}")
        return float("nan")
    finally:
        # Cleanup
        if tmp_real and os.path.exists(tmp_real): shutil.rmtree(tmp_real)
        if tmp_gen and os.path.exists(tmp_gen):   shutil.rmtree(tmp_gen)


# ── M3-Score ───────────────────────────────────────────────────────────────

def _compute_m3(
    real: torch.Tensor,
    gen: torch.Tensor,
    device: str,
    cka_threshold: float = 0.95,
    num_sub_batches: int = 5,
) -> dict:
    """M3-Score: CKA layer selection on real[:20], then full evaluation."""
    from evaluation.m3_score_v2 import M3V2Metric

    metric = M3V2Metric(
        device=device,
        cka_threshold=cka_threshold,
        num_sub_batches=num_sub_batches,
    )
    prune_n = min(20, len(real))
    metric.prune_layers_via_cka(real[:prune_n])
    return metric(real, gen)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def _write_latex_table(results: dict, path: str) -> None:
    """
    Write a two-column LaTeX table (Score, Time) comparing all four metrics.

    Requires \\usepackage{booktabs} and \\usepackage{amsmath} in the
    enclosing document preamble (both present in the paper preamble).

    NaN values are rendered as plain text N/A rather than inside $...$
    to avoid the en-dash ligature that -- produces in math mode.
    """
    rows = [
        ("FRD",         results.get("frd",    float("nan")), results.get("frd_time_s",    0)),
        ("FID",         results.get("fid",    float("nan")), results.get("fid_time_s",    0)),
        ("KID",         results.get("kid",    float("nan")), results.get("kid_time_s",    0)),
        ("M3-Score", results.get("m3_v2",  float("nan")), results.get("m3_v2_time_s", 0)),
    ]

    def fmt_score(v: float) -> str:
        if isinstance(v, float) and np.isnan(v):
            return r"\multicolumn{1}{c}{N/A}"
        return f"${v:.6f}$"

    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{amsmath}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Distributional metrics evaluated on the same image set. "
        r"FRD uses standardised radiomic features~\cite{konz2025frd}; "
        r"FID and KID use InceptionV3 features~\cite{heusel2017gans,binkowski2018demystifying}; "
        r"M3-Score uses CKA-selected RadioDino embeddings with stability weighting.}",
        r"\label{tab:frd_comparison}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Score & Time\,(s) \\",
        r"\midrule",
    ]

    for name, score, t in rows:
        lines.append(f"{name} & {fmt_score(score)} & {t:.1f} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_frd_comparison(
    real_dir:      str,
    gen_dir:       str,
    output_dir:    str,
    num_images:    Optional[int] = None,
    device:        str   = "cuda",
    use_tqdm:      bool  = True,
    cka_threshold: float = 0.95,
    seed:          int   = 42,
) -> dict:
    """
    Run FRD, FID, KID, and M3-Score on the same image sets.

    InceptionV3 features are extracted once and reused for both FID and KID.

    FRD reads from disk directly; image-count control is applied at the
    load stage, not via the frd-score API, for version compatibility.

    Args:
        real_dir:      Directory of real images (searched recursively).
        gen_dir:       Directory of generated images (searched recursively).
        output_dir:    Destination for JSON and LaTeX outputs.
        num_images:    Maximum images per set (None = all found).
        device:        Torch device string.
        use_tqdm:      Unused; retained for orchestrator compatibility.
        cka_threshold: CKA redundancy threshold for M3 layer pruning.
        seed:          Random seed.

    Returns:
        dict with keys frd, fid, kid, m3_v2, per-field timing, m3_v2_details.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Loading images  real={real_dir}  gen={gen_dir}")
    real = _load_images_from_dir(real_dir, num_images)
    gen  = _load_images_from_dir(gen_dir,  num_images)
    print(f"  real: {real.shape}   gen: {gen.shape}")

    results: dict = {}

    # ── FRD (now respects num_images via temp symlinks) ────────────────────
    print("\n  Computing FRD ...")
    t = time.time()
    results["frd"] = _compute_frd(real_dir, gen_dir, num_images)
    results["frd_time_s"] = round(time.time() - t, 2)
    print(f"  FRD = {results['frd']}  ({results['frd_time_s']} s)")

    # ── FID and KID via torchmetrics ───────────────────────────────────────
    # NOTE: this file previously computed FID from torchvision's ImageNet
    # InceptionV3 with ImageNet mean/std normalisation. That is NOT standard
    # FID (which uses the FID-specific pt_inception-2015-12-05 weights at
    # 299x299) and it returned 284.5 where every other experiment in this
    # repository reports 68.8 for the identical comparison. Ten other modules
    # use torchmetrics; this one now does too, so the numbers are comparable.
    print("\n  Computing FID and KID (torchmetrics, uint8 RGB @256) ...")
    t = time.time()
    real_u8 = _load_uint8_rgb(real_dir, num_images)
    gen_u8  = _load_uint8_rgb(gen_dir,  num_images)
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        _fid = FrechetInceptionDistance(feature=2048).to(device)
        _fid.update(real_u8.to(device), real=True)
        _fid.update(gen_u8.to(device),  real=False)
        results["fid"] = float(_fid.compute().item())
        inception_available = True
    except Exception as exc:
        print(f"  [WARN] torchmetrics FID failed: {exc}")
        results["fid"] = float("nan")
        inception_available = False
    results["fid_time_s"] = round(time.time() - t, 2)
    print(f"  FID = {results['fid']}  ({results['fid_time_s']} s)")

    # ── KID ────────────────────────────────────────────────────────────────
    print("\n  Computing KID ...")
    t = time.time()
    if inception_available:
        _sub = min(50, len(gen_u8))
        _kid = KernelInceptionDistance(subset_size=_sub).to(device)
        _kid.update(real_u8.to(device), real=True)
        _kid.update(gen_u8.to(device),  real=False)
        results["kid"] = float(_kid.compute()[0].item())
    else:
        results["kid"] = float("nan")
    results["kid_time_s"] = round(time.time() - t, 2)
    print(f"  KID = {results['kid']}  ({results['kid_time_s']} s)")

    # ── M3-Score ────────────────────────────────────────────────────────────
    print("\n  Computing M3-Score ...")
    t = time.time()
    m3_results = _compute_m3(real, gen, device, cka_threshold)
    results["m3_v2"] = m3_results["m3_v2_final_score"]
    results["m3_v2_details"] = m3_results
    results["m3_v2_time_s"] = round(time.time() - t, 2)
    print(f"  M3-Score = {results['m3_v2']:.6f}  ({results['m3_v2_time_s']} s)")

    # ── Save outputs ───────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "frd_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\n  JSON saved: {json_path}")

    tex_path = os.path.join(output_dir, "frd_comparison.tex")
    _write_latex_table(results, tex_path)
    print(f"  LaTeX saved: {tex_path}")

    return results