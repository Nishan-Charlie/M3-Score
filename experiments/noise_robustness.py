"""
Robustness Experiment -- Section 3.x
=======================================
Tests how each metric responds to two classes of known image perturbation:

  Perturbation A -- Gaussian noise
      Sigma in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5} added to real images.
      Validates monotonicity and sensitivity of each metric to additive noise.

  Perturbation B -- Gaussian blur
      Radius in {0, 1, 2, 3, 4} pixels applied via PIL GaussianBlur.
      Validates that each metric responds to low-frequency structural
      distortion independently of pixel-level noise.

For each perturbation level and type, all metrics are computed between the
clean reference set and the perturbed set.  Spearman correlation between
metric value and perturbation level quantifies monotonic sensitivity.

Metrics tested: M3-Score V2, FID, KID, SSIM, PSNR, MS-SSIM, LPIPS, Alpha-Precision, Beta-Recall.

Model objects (RadioDino, InceptionV3) are instantiated once and reused
across all perturbation levels to avoid repeated loading from disk.

Absorbs progressive_degradation.py (blur sweep) into a single unified
robustness experiment. FRD is excluded (pyradiomics LoG filter issues
with BraTS image sizes).
"""

from __future__ import annotations

import json
import os
import sys
import glob
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LAYERS_CACHE = os.path.join(_PROJECT_ROOT, "canonical_layers.json")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from scipy import stats as sp_stats

from evaluation.m3_score_v2 import M3V2Metric


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_paths(directory: str, n: Optional[int] = None) -> list[str]:
    paths: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))
    if n is not None:
        paths = paths[:n]
    return paths


def _load_images_float(paths: list[str], size: tuple = (256, 256)) -> torch.Tensor:
    """
    Load images as (N, 1, H, W) float32 tensors in [0, 1].
    Size 256x256 is used for SSIM/PSNR/MS-SSIM/LPIPS; MS-SSIM requires
    at least 161px per side for four downsampling levels (256 satisfies this).
    """
    imgs = []
    for p in paths:
        arr = np.array(
            Image.open(p).convert("L").resize(size), dtype=np.float32
        ) / 255.0
        imgs.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.stack(imgs)


def _load_images_rgb_uint8(paths: list[str],
                            size: tuple = (256, 256)) -> torch.Tensor:
    """Load images as (N, 3, H, W) uint8 tensors for FID/KID."""
    imgs = []
    for p in paths:
        arr = np.array(
            Image.open(p).convert("RGB").resize(size), dtype=np.uint8
        )
        imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
    return torch.stack(imgs)


def _load_images_for_m3(paths: list[str]) -> torch.Tensor:
    """Load images as (N, 3, 224, 224) float32 tensors in [0, 1] for M3V2Metric."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return torch.stack([
        transform(Image.open(p).convert("RGB")) for p in paths
    ])


# ---------------------------------------------------------------------------
# Noise injection and disk I/O
# ---------------------------------------------------------------------------

def _add_gaussian_noise(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to a [0, 1] tensor and clamp to [0, 1]."""
    return torch.clamp(img_tensor + torch.randn_like(img_tensor) * sigma, 0.0, 1.0)


def _add_gaussian_blur(img_tensor: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Apply Gaussian blur to a (N, C, H, W) float tensor in [0, 1].

    Blur is applied per-image via PIL GaussianBlur, which matches the
    blur applied in progressive_degradation.py and ensures consistency
    with the disk-saved versions used by FID.

    Args:
        img_tensor: (N, C, H, W) float tensor in [0, 1].
        radius:     GaussianBlur radius in pixels (0 = no blur).

    Returns:
        Blurred (N, C, H, W) float tensor in [0, 1].
    """
    from PIL import ImageFilter
    if radius == 0:
        return img_tensor.clone()

    blurred = []
    for i in range(img_tensor.shape[0]):
        arr = (img_tensor[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        C = arr.shape[-1]
        if C == 1:
            # Single-channel: squeeze to (H, W), blur as greyscale, re-expand
            pil = Image.fromarray(arr.squeeze(-1), mode="L").filter(
                ImageFilter.GaussianBlur(radius=radius)
            )
            out = torch.from_numpy(
                np.array(pil, dtype=np.float32) / 255.0
            ).unsqueeze(0)          # (1, H, W)
        else:
            # RGB: blur all three channels jointly
            pil = Image.fromarray(arr, mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=radius)
            )
            out = torch.from_numpy(
                np.array(pil, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)      # (3, H, W)
        blurred.append(out)
    return torch.stack(blurred)


def _save_perturbed_images(tensor_float: torch.Tensor, output_dir: str) -> None:
    """
    Save perturbed grayscale images to disk as PNG files.

    Accepts either (N, 1, H, W) or (N, 3, H, W) tensors. The first channel
    is used for single-channel saving (mode='L'), compatible with
    FID/KID loaders that call
    .convert('RGB') on load.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(tensor_float.shape[0]):
        arr = (tensor_float[i, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(
            os.path.join(output_dir, f"perturbed_{i:04d}.png")
        )


# ---------------------------------------------------------------------------
# Per-metric computation helpers
# (model objects passed in to avoid repeated loading)
# ---------------------------------------------------------------------------

def _compute_ssim_psnr(
    clean_float: torch.Tensor,
    noisy_float: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute mean SSIM and PSNR over paired images.

    PSNR = -10 * log10(MSE) for images in [0, 1] range
    (equivalent to 20*log10(1.0) - 10*log10(MSE) since log10(1) = 0).
    """
    from pytorch_msssim import ssim as pt_ssim

    num = min(len(clean_float), len(noisy_float))
    ssim_vals: list[float] = []
    psnr_vals: list[float] = []

    for i in range(num):
        c   = clean_float[i:i + 1]
        noi = noisy_float[i:i + 1]
        ssim_vals.append(
            pt_ssim(c, noi, data_range=1.0, size_average=True).item()
        )
        mse = torch.mean((c - noi) ** 2)
        if mse == 0:
            psnr_vals.append(100.0)
        else:
            psnr_vals.append(-10.0 * torch.log10(mse).item())

    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


def _compute_msssim_lpips(
    clean_float: torch.Tensor,
    noisy_float: torch.Tensor,
    device: str,
    lpips_fn,
) -> dict:
    """
    Compute MS-SSIM and LPIPS.

    lpips_fn is a pre-instantiated lpips.LPIPS model passed in to avoid
    repeated model loading across sigma levels.
    """
    results: dict = {}
    num = min(len(clean_float), len(noisy_float))

    try:
        from pytorch_msssim import ms_ssim
        ms_vals: list[float] = []
        batch = 8
        for s in range(0, num, batch):
            c   = clean_float[s:s + batch].to(device)
            noi = noisy_float[s:s + batch].to(device)
            if c.shape[0] > 0:
                ms_vals.append(
                    ms_ssim(c, noi, data_range=1.0, size_average=True).item()
                )
        results["ms_ssim"] = float(np.mean(ms_vals)) if ms_vals else float("nan")
    except Exception:
        results["ms_ssim"] = float("nan")

    try:
        lpips_vals: list[float] = []
        batch = 16
        for s in range(0, num, batch):
            c   = clean_float[s:s + batch].repeat(1, 3, 1, 1).to(device) * 2 - 1
            noi = noisy_float[s:s + batch].repeat(1, 3, 1, 1).to(device) * 2 - 1
            with torch.no_grad():
                lpips_vals.append(lpips_fn(c, noi).mean().item())
        results["lpips"] = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    except Exception:
        results["lpips"] = float("nan")

    return results


def _compute_fid_kid_from_features(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
) -> tuple[float, float]:
    """
    Compute FID and polynomial-kernel KID from pre-extracted 2048-d feature arrays.

    Uses scipy sqrtm with eps=1e-6 regularisation so the result is finite even
    when n < feature_dim (rank-deficient covariance).  Torchmetrics' FID calls
    torch.linalg.eigh on the unregularised product and returns nan in that case.

    KID uses a degree-3 polynomial MMD matching frd_comparison.py.
    """
    from scipy.linalg import sqrtm as scipy_sqrtm

    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sig_g = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)
    eps    = 1e-6
    offset = np.eye(sig_r.shape[0]) * eps
    covmean = scipy_sqrtm((sig_r + offset) @ (sig_g + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    fid  = float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))

    r   = torch.from_numpy(real_feats)
    g   = torch.from_numpy(gen_feats)
    gam = 1.0 / r.shape[1]
    def _poly(a, b):
        return (gam * torch.mm(a, b.t()) + 1.0) ** 3
    K_rr, K_gg, K_rg = _poly(r, r), _poly(g, g), _poly(r, g)
    m, n_g = K_rr.shape[0], K_gg.shape[0]
    kid = float((
        (K_rr.sum() - torch.trace(K_rr)) / (m   * (m   - 1))
        + (K_gg.sum() - torch.trace(K_gg)) / (n_g * (n_g - 1))
        - 2.0 * K_rg.mean()
    ).clamp(min=0.0))

    return fid, kid




def _compute_m3_score(
    real_t:  torch.Tensor,
    noisy_t: torch.Tensor,
    metric:  M3V2Metric,
) -> float:
    """
    Compute M3-Score V2 using a pre-initialised and CKA-pruned metric object.
    """
    with torch.no_grad():
        results = metric(real_t, noisy_t)
    return float(results.get("m3_v2_final_score", float("nan")))


def _extract_inception_features_from_tensor(
    rgb_uint8:  torch.Tensor,
    extractor,
    device:     str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract 2048-d InceptionV3 features from a (N, 3, H, W) uint8 tensor.
    Images are resized to 299×299 and ImageNet-normalised before forwarding.
    """
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    all_feats: list[np.ndarray] = []
    for s in range(0, len(rgb_uint8), batch_size):
        batch = rgb_uint8[s : s + batch_size].float().to(device) / 255.0
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        with torch.no_grad():
            feats = extractor(batch)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def _compute_alpha_precision_beta_recall(
    clean_feats: np.ndarray,
    pert_dir:    str,
    n:           int,
    extractor,
    k:           int = 5,
) -> tuple[float, float]:
    """
    Compute alpha-precision (fidelity) and beta-recall (diversity).

    Uses pre-extracted InceptionV3 features for the real set (clean_feats)
    and extracts features on-the-fly from perturbed disk images.
    The extractor is shared across sweep levels to avoid redundant loading.
    Returns (nan, nan) when there are too few samples for reliable k-NN.
    """
    from metrics.alpha_precision_recall import (
        _extract_features as _alpha_extract,
        _build_knn,
        alpha_precision,
        beta_recall,
    )
    pert_paths = sorted(glob.glob(os.path.join(pert_dir, "*.png")))[:n]
    if len(pert_paths) < k + 1 or len(clean_feats) < k + 1:
        return float("nan"), float("nan")

    pert_feats = _alpha_extract(
        pert_paths, extractor, batch_size=32, use_tqdm=False, label="Pert"
    )

    real_radii, real_knn = _build_knn(clean_feats, k)
    gen_radii,  gen_knn  = _build_knn(pert_feats,  k)

    prec = alpha_precision(clean_feats, pert_feats, real_radii, real_knn)
    rec  = beta_recall(clean_feats, pert_feats, gen_radii, gen_knn)
    return prec, rec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_noise_robustness(
    real_dir:     str,
    output_dir:   str            = "./noise_robustness_output",
    num_images:   Optional[int]  = 200,
    noise_levels: Optional[list] = None,
    blur_levels:  Optional[list] = None,
    device:       str            = "cpu",
    use_tqdm:     bool           = True,
    seed:         int            = 42,
    backbone_id:  str            = "Snarcy/RadioDino-s16",
) -> dict:
    """
    Run the robustness experiment across Gaussian noise and Gaussian blur.

    Args:
        real_dir:     Directory of real images.
        output_dir:   Destination for plots and JSON report.
        num_images:   Number of images to load (default 200).
        noise_levels: List of Gaussian noise sigma values.
        blur_levels:  List of Gaussian blur radii in pixels.
        device:       Torch device string.
        use_tqdm:     Whether to show progress bars.
        seed:         Random seed.

    Returns:
        dict with noise and blur metric_values, Spearman correlations, and
        plot paths, compatible with master_report["noise_robustness"] in
        run_experiments.py.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    if blur_levels is None:
        blur_levels = [0, 1, 2, 3, 4]

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    real_paths = _load_paths(real_dir, num_images)
    if not real_paths:
        raise FileNotFoundError(
            f"No images found in {real_dir}. "
            "Check the directory path and supported extensions."
        )
    n = len(real_paths)
    print(f"[Robustness] Using {n} real images")
    print(f"             Noise levels: {noise_levels}")
    print(f"             Blur radii:   {blur_levels}")

    # ── Load clean images once ────────────────────────────────────────────────
    clean_float   = _load_images_float(real_paths)
    clean_rgb     = _load_images_rgb_uint8(real_paths).to(device)
    # Load M3 images as a capped subset to avoid holding
    # 2 x 500 x (3,224,224) float32 = 600 MB in RAM simultaneously.
    # 200 images gives stable Spearman rho for the noise sweep.
    _m3_n = min(len(real_paths), 200)
    clean_m3      = _load_images_for_m3(real_paths[:_m3_n])



    # ── Instantiate models once ───────────────────────────────────────────────
    print("[Robustness] Initialising models ...")
    m3_metric = M3V2Metric(device=device, backbone_id=backbone_id)
    m3_metric.prune_layers_via_cka(clean_m3[:20], cache_path=_LAYERS_CACHE)
    print(f"  M3 active layers: {m3_metric.active_layers}")

    # FID and KID are now computed from shared InceptionV3 features using
    # scipy sqrtm with eps regularisation (see _compute_fid_kid_from_features).
    # torchmetrics is no longer used for FID/KID.

    lpips_fn = None
    try:
        import warnings
        import lpips as lpips_lib
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
            lpips_fn.eval()
    except ImportError:
        print("  [WARN] lpips not installed; LPIPS skipped.")

    # ── InceptionV3 for Precision/Recall (instantiated once, clean feats cached) ─
    prec_recall_extractor  = None
    clean_inception_feats  = None
    try:
        from metrics.alpha_precision_recall import InceptionFeatureExtractor
        prec_recall_extractor = InceptionFeatureExtractor(device=device)
        prec_recall_extractor.eval()
        print("  Extracting clean InceptionV3 features for Precision/Recall ...")
        clean_inception_feats = _extract_inception_features_from_tensor(
            clean_rgb, prec_recall_extractor, device
        )
        print(f"  Clean InceptionV3 features shape: {clean_inception_feats.shape}")
    except Exception as e:
        print(f"  [WARN] Precision/Recall extractor init failed: {e}")

    metric_names = ["m3", "fid", "kid", "ssim", "psnr", "ms_ssim", "lpips", "precision", "recall"]

    def _run_sweep(
        levels:     list,
        level_key:  str,
        perturb_fn,
        sweep_label: str,
    ) -> list[dict]:
        """Generic inner loop shared by noise and blur sweeps."""
        sweep_results: list[dict] = []
        for level in tqdm(levels, desc=sweep_label, disable=not use_tqdm):
            print(f"\n[{sweep_label}] {level_key} = {level}")
            row: dict = {level_key: float(level)}

            pert_float = perturb_fn(clean_float, level)
            pert_m3    = perturb_fn(clean_m3,    level)  # capped at 200

            pert_dir = os.path.join(output_dir, f"pert_{sweep_label}_{level_key}{level}")
            _pert_existing = len(glob.glob(os.path.join(pert_dir, "*.png")))
            if _pert_existing == len(pert_float):
                print(f"  [INFO] Perturbed images reused from previous run ({_pert_existing} files)")
            else:
                _save_perturbed_images(pert_float, pert_dir)

            try:
                row["m3"] = round(_compute_m3_score(clean_m3, pert_m3, m3_metric), 6)
            except Exception as e:
                row["m3"] = float("nan"); print(f"  M3 failed: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()



            # Extract perturbed Inception features once; shared for FID/KID and P/R
            pert_inception_feats = None
            if prec_recall_extractor is not None and clean_inception_feats is not None:
                try:
                    from metrics.alpha_precision_recall import _extract_features as _alpha_extract
                    pert_paths = sorted(glob.glob(os.path.join(pert_dir, "*.png")))[:n]
                    if len(pert_paths) >= 2:
                        pert_inception_feats = _alpha_extract(
                            pert_paths, prec_recall_extractor,
                            batch_size=32, use_tqdm=False, label="Pert"
                        )
                except Exception as e:
                    print(f"  Inception feature extraction failed: {e}")

            if clean_inception_feats is not None and pert_inception_feats is not None:
                try:
                    fid_val, kid_val = _compute_fid_kid_from_features(
                        clean_inception_feats, pert_inception_feats
                    )
                    row["fid"] = round(fid_val, 4)
                    row["kid"] = round(kid_val, 6)
                except Exception as e:
                    row["fid"] = row["kid"] = float("nan")
                    print(f"  FID/KID failed: {e}")
            else:
                row["fid"] = row["kid"] = float("nan")

            try:
                ssim_val, psnr_val = _compute_ssim_psnr(clean_float, pert_float)
                row["ssim"] = round(ssim_val, 6)
                row["psnr"] = round(psnr_val, 4)
            except Exception as e:
                row["ssim"] = row["psnr"] = float("nan")
                print(f"  SSIM/PSNR failed: {e}")

            try:
                ext = _compute_msssim_lpips(clean_float, pert_float, device, lpips_fn)
                row["ms_ssim"] = round(ext.get("ms_ssim", float("nan")), 6)
                row["lpips"]   = round(ext.get("lpips",   float("nan")), 6)
            except Exception as e:
                row["ms_ssim"] = row["lpips"] = float("nan")
                print(f"  MS-SSIM/LPIPS failed: {e}")

            # Alpha-Precision / Beta-Recall (reuse perturbed features from above)
            if clean_inception_feats is not None and pert_inception_feats is not None:
                try:
                    from metrics.alpha_precision_recall import (
                        _build_knn, alpha_precision, beta_recall,
                    )
                    real_radii, real_knn = _build_knn(clean_inception_feats, 5)
                    gen_radii,  gen_knn  = _build_knn(pert_inception_feats,  5)
                    prec = alpha_precision(
                        clean_inception_feats, pert_inception_feats, real_radii, real_knn
                    )
                    rec = beta_recall(
                        clean_inception_feats, pert_inception_feats, gen_radii, gen_knn
                    )
                    row["precision"] = round(float(prec), 4)
                    row["recall"]    = round(float(rec),  4)
                except Exception as e:
                    row["precision"] = row["recall"] = float("nan")
                    print(f"  Precision/Recall failed: {e}")
            else:
                row["precision"] = row["recall"] = float("nan")

            sweep_results.append(row)
            print(f"  {row}")

        # Perturbed directories are kept for reuse on subsequent runs.
        # To force a fresh save (e.g. after changing seed or num_images),
        # manually delete the pert_* directories in the output folder.

        return sweep_results

    # ── Noise sweep ───────────────────────────────────────────────────────────
    noise_results = _run_sweep(
        levels      = noise_levels,
        level_key   = "sigma",
        perturb_fn  = lambda imgs, s: (
            imgs.clone() if s == 0.0 else _add_gaussian_noise(imgs, s)
        ),
        sweep_label = "noise",
    )

    # ── Blur sweep ────────────────────────────────────────────────────────────
    blur_results = _run_sweep(
        levels      = blur_levels,
        level_key   = "radius",
        perturb_fn  = _add_gaussian_blur,
        sweep_label = "blur",
    )

    # ── Spearman correlations ─────────────────────────────────────────────────
    def _spearman_table(sweep: list[dict], level_key: str) -> dict:
        """
        Spearman correlation between perturbation level and each metric,
        with Fisher-z 95% CI and monotonicity verdict.

        Metrics that should INCREASE with perturbation (worse = higher):
            m3, fid, kid, lpips
        Metrics that should DECREASE with perturbation (worse = lower):
            ssim, psnr, ms_ssim

        Monotone = rho has the expected sign AND |rho| >= 0.7.
        """
        level_vals = np.array([r[level_key] for r in sweep])
        # Expected direction: +1 = should increase, -1 = should decrease
        expected_direction = {
            "m3": 1, "fid": 1, "kid": 1, "lpips": 1,
            "ssim": -1, "psnr": -1, "ms_ssim": -1,
            # Additional metrics
            "precision": -1, "recall": -1,
        }
        corr: dict = {}
        for m in metric_names:
            vals  = np.array([r.get(m, float("nan")) for r in sweep])
            valid = ~np.isnan(vals)
            n_valid = int(valid.sum())
            if n_valid < 3:
                continue
            r_val, p_val = sp_stats.spearmanr(level_vals[valid], vals[valid])
            # Fisher-z 95% CI
            if abs(r_val) >= 1.0:
                ci_lo = ci_hi = float(r_val)
            else:
                z  = np.arctanh(r_val)
                se = 1.0 / np.sqrt(max(n_valid - 3, 1))
                ci_lo = float(np.tanh(z - 1.96 * se))
                ci_hi = float(np.tanh(z + 1.96 * se))
            exp_dir = expected_direction.get(m, 1)
            monotone = bool((float(r_val) * exp_dir > 0) and (float(abs(r_val)) >= 0.7))
            corr[m] = {
                "spearman_r":  round(float(r_val), 4),
                "spearman_p":  float(p_val),
                "ci_95":       [round(ci_lo, 4), round(ci_hi, 4)],
                "sample_size": n_valid,
                "monotone":    monotone,
                "expected_direction": "increase" if exp_dir == 1 else "decrease",
            }
        return corr

    noise_corr = _spearman_table(noise_results, "sigma")
    blur_corr  = _spearman_table(blur_results,  "radius")

    # ── Plots ─────────────────────────────────────────────────────────────────
    colors = {
        "m3":        "#ef5350",
        "fid":       "#4fc3f7",
        "kid":       "#66bb6a",
        "ssim":      "#ffe082",
        "psnr":      "#ce93d8",
        "ms_ssim":   "#ff8a65",
        "lpips":     "#80cbc4",
        "precision": "#ab47bc",
        "recall":    "#26c6da",

    }

    def _dark_ax(ax):
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333")
        ax.grid(True, alpha=0.4)

    def _make_sweep_plots(
        sweep:      list[dict],
        level_key:  str,
        corr:       dict,
        prefix:     str,
        xlabel:     str,
    ) -> str:
        levels = np.array([r[level_key] for r in sweep])
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)
        fig.patch.set_facecolor("white")

        ax = axes[0, 0]; _dark_ax(ax)
        for m in ["m3", "fid", "kid"]:
            vals = [r.get(m, float("nan")) for r in sweep]
            ax.plot(levels, vals, "o-", color=colors[m], label=m.upper(),
                    linewidth=2, markersize=6)
        ax.set_xlabel(xlabel, color="#222222", fontsize=11)
        ax.set_ylabel("Metric value", color="#222222", fontsize=11)
        ax.set_title("Distributional metrics", color="#222222", fontsize=13)
        ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc")

        ax = axes[0, 1]; _dark_ax(ax)
        for m in ["ssim", "psnr", "ms_ssim", "lpips"]:
            vals = [r.get(m, float("nan")) for r in sweep]
            ax.plot(levels, vals, "o-", color=colors[m], label=m.upper(),
                    linewidth=2, markersize=6)
        ax.set_xlabel(xlabel, color="#222222", fontsize=11)
        ax.set_ylabel("Metric value", color="#222222", fontsize=11)
        ax.set_title("Perceptual metrics", color="#222222", fontsize=13)
        ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc")

        ax = axes[1, 0]; _dark_ax(ax)
        for m in metric_names:
            vals = np.array([r.get(m, float("nan")) for r in sweep])
            valid = ~np.isnan(vals)
            if valid.sum() >= 2:
                vmin, vmax = vals[valid].min(), vals[valid].max()
                normed = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
                ax.plot(levels[valid], normed[valid], "o-", color=colors[m],
                        label=m.upper(), linewidth=2, markersize=5)
        ax.set_xlabel(xlabel, color="#222222", fontsize=11)
        ax.set_ylabel("Normalised metric (0 to 1)", color="#222222", fontsize=11)
        ax.set_title("All metrics normalised", color="#222222", fontsize=13)
        ax.legend(fontsize=8, facecolor="white", edgecolor="#cccccc", ncol=2)

        ax = axes[1, 1]; ax.set_facecolor("white"); ax.tick_params(colors="#333333")
        corr_ms = list(corr.keys())
        corr_vs = [abs(corr[m]["spearman_r"]) for m in corr_ms]
        bars = ax.barh(corr_ms, corr_vs,
                       color=[colors.get(m, "#888") for m in corr_ms],
                       edgecolor="#cccccc", height=0.6)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, color="#222222", padding=4)
        ax.set_xlabel("|Spearman rho|", color="#222222", fontsize=11)
        ax.set_title("Metric sensitivity", color="#222222", fontsize=13)
        ax.set_xlim(0, 1.1)

        plt.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_robustness.png")
        plt.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    noise_plot = _make_sweep_plots(
        noise_results, "sigma", noise_corr, "noise", "Noise level sigma"
    )
    blur_plot = _make_sweep_plots(
        blur_results, "radius", blur_corr, "blur", "Blur radius (pixels)"
    )



    # ── Save report ───────────────────────────────────────────────────────────
    results = {
        "noise": {
            "levels":       noise_levels,
            "metric_values": noise_results,
            "spearman":     noise_corr,
            "plot":         noise_plot,
        },
        "blur": {
            "levels":       blur_levels,
            "metric_values": blur_results,
            "spearman":     blur_corr,
            "plot":         blur_plot,
        },
    }
    report_path = os.path.join(output_dir, "robustness_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"[Robustness] Report saved: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Robustness experiment: Gaussian noise and Gaussian blur"
    )
    parser.add_argument("--real_dir",    required=True)
    parser.add_argument("--output_dir",  default="./noise_robustness_output")
    parser.add_argument("--num_images",  type=int, default=200)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--no_tqdm",     action="store_true")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument(
        "--noise_levels", nargs="+", type=float,
        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        help="Gaussian noise sigma values",
    )
    parser.add_argument(
        "--blur_levels", nargs="+", type=int,
        default=[0, 1, 2, 3, 4],
        help="Gaussian blur radii in pixels",
    )
    args = parser.parse_args()
    run_noise_robustness(
        real_dir     = args.real_dir,
        output_dir   = args.output_dir,
        num_images   = args.num_images,
        noise_levels = args.noise_levels,
        blur_levels  = args.blur_levels,
        device       = args.device,
        use_tqdm     = not args.no_tqdm,
        seed         = args.seed,
    )