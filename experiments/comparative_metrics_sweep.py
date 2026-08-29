import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import glob
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.m3_score_v2 import M3V2Metric
from experiments.fid_infinity import run_fid_infinity

def load_images_simple(directory, n=300, recursive=True):
    """Load up to *n* images from *directory* as (N,3,H,W) uint8 + float tensors."""
    if recursive:
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        paths = sorted(set(paths))[:n]
    else:
        paths = sorted(glob.glob(os.path.join(directory, "*.png")))[:n]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imgs_rgb = []
    imgs_raw = []
    for p in tqdm(paths, desc=f"Loading {os.path.basename(directory)}", leave=False):
        img = Image.open(p).convert("RGB")
        img_t = transform(img)               # raw [0,1]
        imgs_rgb.append((img_t * 255).byte())
        imgs_raw.append(img_t)
    if not imgs_rgb:
        raise FileNotFoundError(f"No images found in {directory}")
    return torch.stack(imgs_rgb), torch.stack(imgs_raw)

def apply_center_mask(imgs_raw, mask_size=64):
    """Zero-out the centre of raw [0,1] tensors."""
    masked = imgs_raw.clone()
    H, W = masked.shape[-2:]
    start_h = H // 2 - mask_size // 2
    start_w = W // 2 - mask_size // 2
    masked[:, :, start_h:start_h + mask_size, start_w:start_w + mask_size] = 0.0
    return masked

def inject_hallucination(imgs_raw, blob_size=32):
    """Inject a bright blob into raw [0,1] tensors."""
    hallucinated = imgs_raw.clone()
    H, W = hallucinated.shape[-2:]
    start_h = H // 2 - blob_size // 2
    start_w = W // 2 - blob_size // 2
    hallucinated[:, :, start_h:start_h + blob_size, start_w:start_w + blob_size] = 1.0
    return hallucinated.clamp(0.0, 1.0)


def run_comparative_sweep(
    real_dir: str,
    gen_dir: str,
    output_dir: str = "./comparative_output",
    n: int = 300,
    device=None,
    seed: int = 42,
) -> dict:
    """
    Comparative sensitivity sweep: tests all metrics across 4 conditions —
      1. Baseline (Real vs Real)
      2. Pathology Masking (centre-cropped anomaly)
      3. Hallucination injection (bright blob)
      4. Generated Model output

    Metrics compared: M3-Score, FID, KID, SSIM, PSNR, FID-Infinity.

    Both FID and KID are computed on every condition.

    Returns a dict compatible with master_report["comparative_sweep"].
    """
    import time
    import json

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print("--- Loading Datasets ---")
    real_rgb, real_raw = load_images_simple(real_dir, n=n, recursive=True)
    gen_rgb,  gen_raw  = load_images_simple(gen_dir,  n=n, recursive=False)

    # Create Perturbations (in raw [0,1] space)
    masked_raw = apply_center_mask(real_raw)
    masked_rgb = (masked_raw.clamp(0, 1) * 255).byte()

    halluc_raw = inject_hallucination(real_raw)
    halluc_rgb = (halluc_raw.clamp(0, 1) * 255).byte()

    # Initialise metrics once
    m3_metric = M3V2Metric(device=device)
    m3_metric.prune_layers_via_cka(real_raw[:20])

    fid_obj = FrechetInceptionDistance(feature=2048).to(device)
    kid_obj = KernelInceptionDistance(subset_size=min(50, n)).to(device)

    results_rows = []

    def evaluate_all(target_rgb, target_raw, label):
        print(f"\nEvaluating: {label}")
        row = {"test": label}

        with torch.no_grad():
            # M3-Score
            t0 = time.time()
            m3_res = m3_metric(real_raw, target_raw)
            row["m3_score"] = round(float(m3_res["m3_v2_final_score"]), 6)
            row["m3_score_time_s"] = round(time.time() - t0, 2)
            row["m3_sub_scores"] = {
                k: round(float(v), 6)
                for k, v in m3_res.items()
                if k != "m3_v2_final_score" and isinstance(v, (int, float, torch.Tensor))
            }

            # FID
            t0 = time.time()
            fid_obj.reset()
            fid_obj.update(real_rgb.to(device), real=True)
            fid_obj.update(target_rgb.to(device), real=False)
            row["fid"] = round(float(fid_obj.compute().item()), 6)
            row["fid_time_s"] = round(time.time() - t0, 2)

            # KID
            t0 = time.time()
            kid_obj.reset()
            kid_obj.update(real_rgb.to(device), real=True)
            kid_obj.update(target_rgb.to(device), real=False)
            kid_m, kid_std = kid_obj.compute()
            row["kid"] = round(float(kid_m.item()), 6)
            row["kid_std"] = round(float(kid_std.item()), 6)
            row["kid_time_s"] = round(time.time() - t0, 2)

            # SSIM
            ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            row["ssim"] = round(float(ssim_fn(
                target_rgb.float().to(device) / 255.0,
                real_rgb.float().to(device) / 255.0,
            ).item()), 6)

            # PSNR
            psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
            row["psnr"] = round(float(psnr_fn(
                target_rgb.float().to(device) / 255.0,
                real_rgb.float().to(device) / 255.0,
            ).item()), 4)

            # FID-Infinity (Generated Model only)
            if label == "Generated Model":
                print("    Running FID-Infinity...")
                t0 = time.time()
                fid_inf_res = run_fid_infinity(
                    real_dir=real_dir,
                    gen_dir=gen_dir,
                    output_dir=output_dir,
                    device=device,
                    sample_sizes=[50, 100, 200, min(300, n)],
                    num_repeats=3,
                    use_tqdm=False,
                )
                row["fid_infinity"] = fid_inf_res.get("fid_infinity", float("nan"))
                row["fid_infinity_time_s"] = round(time.time() - t0, 2)
            else:
                row["fid_infinity"] = float("nan")
                row["fid_infinity_time_s"] = 0.0

        return row

    # 4 test conditions
    results_rows.append(evaluate_all(real_rgb,    real_raw,    "Baseline (Real vs Real)"))
    results_rows.append(evaluate_all(masked_rgb,  masked_raw,  "Pathology Masking (Anomaly)"))
    results_rows.append(evaluate_all(halluc_rgb,  halluc_raw,  "Hallucination (Anomaly)"))
    results_rows.append(evaluate_all(gen_rgb,     gen_raw,     "Generated Model"))

    # Sensitivity ratios vs Baseline
    baseline = results_rows[0]
    sensitivity = {}
    for test_row in results_rows[1:]:
        label = test_row["test"]
        ratios = {}
        for col in ["m3_score", "fid", "kid", "ssim", "psnr"]:
            b_val = baseline.get(col, float("nan"))
            t_val = test_row.get(col, float("nan"))
            if col in ("ssim", "psnr"):
                # These decrease under degradation → ratio = baseline/test
                ratios[col] = round((b_val + 1e-9) / (t_val + 1e-9), 4)
            else:
                # These increase under degradation → ratio = test/baseline
                ratios[col] = round((t_val + 1e-9) / (b_val + 1e-9), 4)
        sensitivity[label] = ratios

    print("\n--- Comparative Sensitivity Ratios (Perturbed / Baseline) ---")
    for label, ratios in sensitivity.items():
        print(f"  {label}: {ratios}")

    out = {"conditions": results_rows, "sensitivity_ratios": sensitivity}
    json_path = os.path.join(output_dir, "comparative_sweep_report.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=4, default=str)
    print(f"\n  Saved: {json_path}")
    return out


if __name__ == "__main__":
    run_experiment()
