"""
run_multidataset.py
====================
Runs the core M3-Score evaluation experiments across three datasets/modalities:

  Dataset A — BraTS (Brain MRI)
    Real:      data_mri/brats_axial_multislice/
    Generated: output/generated_500_standard/   (standard DDPM, 1000 steps, no best-of-N)

  Dataset B — Retinal Fundus
    Real:      data_mri/medmnist/retinamnist/all/  (RetinaMNIST 64×64)
    Generated: output/generated_retinal/           (HF DDPM, GS-23 model, 128×128→gray)

  Dataset C — Chest X-Ray
    Real:      data_mri/medmnist/pneumoniamnist/all/  (PneumoniaMNIST 64×64)
    Generated: output/generated_cxr_corrupted/        (Gaussian-corrupted real CXR at σ=0.3)
               (No pretrained CXR generator available; corruption proxy tests metric
               sensitivity on a second modality without requiring a trained generator)

Experiments run per dataset:
  1. Core M3 score
  2. Statistical rigor (permutation p-value + bootstrap CI)
  3. Noise robustness monotonicity (Spearman rho under Gaussian noise)
  4. Conditional MMD (intensity quartile stratification)
  5. Coverage & Novelty

Results saved to:  results/multidataset_run/

Usage:
    python run_multidataset.py --device cuda --seed 42
    python run_multidataset.py --datasets brats retinal --device cuda
    python run_multidataset.py --datasets brats --skip_generation  # if already generated
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

DATASETS = {
    "brats": {
        "name": "BraTS Brain MRI",
        "modality": "brain_mri",
        "real_dir": str(ROOT / "data_mri" / "brats_axial_multislice"),
        "gen_dir":  str(ROOT / "output" / "generated_500_standard"),
        "gen_script": "brats_standard",
    },
    "retinal": {
        "name": "Retinal Fundus (RetinaMNIST + HF DDPM)",
        "modality": "retinal_fundus",
        "real_dir": str(ROOT / "data_mri" / "medmnist" / "retinamnist" / "all"),
        "gen_dir":  str(ROOT / "output" / "generated_retinal"),
        "gen_script": "retinal_hf",
    },
    "cxr": {
        "name": "Chest X-Ray (PneumoniaMNIST + corrupted proxy)",
        "modality": "chest_xray",
        "real_dir": str(ROOT / "data_mri" / "medmnist" / "pneumoniamnist" / "all"),
        "gen_dir":  str(ROOT / "output" / "generated_cxr_corrupted"),
        "gen_script": "cxr_corrupted",
    },
}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def ensure_brats_standard(gen_dir: str, checkpoint_dir: str, device: str, seed: int):
    """Generate 500 BraTS images with standard sampling (no best-of-N)."""
    if Path(gen_dir).exists() and len(list(Path(gen_dir).glob("*.png"))) >= 500:
        print(f"  [SKIP] {gen_dir} already has ≥500 images.")
        return
    Path(gen_dir).mkdir(parents=True, exist_ok=True)
    from generate import generate_images
    generate_images(
        checkpoint_dir=checkpoint_dir,
        output_dir=gen_dir,
        num_images=500,
        batch_size=16,
        device=device,
        reward_type="none",
        best_of_n=1,
    )


def ensure_retinal_generated(gen_dir: str, device: str, seed: int):
    """Run the HF retinal DDPM to generate 500 images."""
    if Path(gen_dir).exists() and len(list(Path(gen_dir).glob("*.png"))) >= 500:
        print(f"  [SKIP] {gen_dir} already has ≥500 images.")
        return
    sys.path.insert(0, str(ROOT))
    from tools.generate_retinal import generate_retinal
    generate_retinal(
        num_images=500,
        num_inference_steps=1000,
        output_dir=gen_dir,
        batch_size=8,
        device=device,
        seed=seed,
    )


def ensure_cxr_corrupted(real_dir: str, gen_dir: str, sigma: float = 0.30, seed: int = 42):
    """
    Create corrupted chest X-ray images as a generated proxy.
    Applies Gaussian noise at σ=0.30 to real CXR images.
    This tests metric sensitivity on a second modality without a trained generator.
    """
    if Path(gen_dir).exists() and len(list(Path(gen_dir).glob("*.png"))) >= 500:
        print(f"  [SKIP] {gen_dir} already has ≥500 images.")
        return
    Path(gen_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    real_files = sorted(Path(real_dir).glob("*.png"))[:500]
    if not real_files:
        raise FileNotFoundError(f"No PNG files found in {real_dir}. "
                                "Run: python tools/download_medmnist.py "
                                "--datasets pneumoniamnist --also_merge")

    print(f"  Creating {len(real_files)} corrupted CXR images (σ={sigma}) ...")
    for i, fpath in enumerate(real_files):
        img = np.array(Image.open(fpath).convert("L")).astype(np.float32) / 255.0
        noisy = img + rng.normal(0, sigma, img.shape)
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(Path(gen_dir) / f"cxr_corrupted_{i:04d}.png")
    print(f"  Saved to {gen_dir}")


def ensure_medmnist_real(dataset_name: str, real_dir: str, size: int = 64):
    """Download and flatten a MedMNIST dataset to real_dir/all/ if not already done."""
    if Path(real_dir).exists() and len(list(Path(real_dir).glob("*.png"))) > 0:
        print(f"  [SKIP] {real_dir} already exists with images.")
        return
    print(f"  Downloading {dataset_name} at {size}×{size} ...")
    # Call download_medmnist via subprocess to avoid import side-effects
    import subprocess
    cmd = [
        sys.executable, str(ROOT / "tools" / "download_medmnist.py"),
        "--datasets", dataset_name,
        "--size", str(size),
        "--output_dir", str(ROOT / "data_mri" / "medmnist"),
        "--also_merge",
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Core evaluation per dataset
# ---------------------------------------------------------------------------

def run_experiments_for_dataset(
    ds_key: str,
    ds_info: dict,
    output_root: Path,
    device: str,
    n_images: int,
    n_perm: int,
    n_boot: int,
    backbone_id: str,
):
    real_dir = ds_info["real_dir"]
    gen_dir  = ds_info["gen_dir"]
    ds_out   = output_root / ds_key
    ds_out.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": ds_key,
        "name": ds_info["name"],
        "modality": ds_info["modality"],
        "real_dir": real_dir,
        "gen_dir": gen_dir,
    }

    from experiments._shared_utils import load_pils_recursive

    print(f"\n  Loading images (N={n_images}) ...")
    real_imgs = load_pils_recursive(real_dir, n_images)
    gen_imgs  = load_pils_recursive(gen_dir,  n_images)
    print(f"  Real: {len(real_imgs)}  Gen: {len(gen_imgs)}")

    if len(real_imgs) < 50 or len(gen_imgs) < 50:
        results["error"] = f"Too few images: real={len(real_imgs)}, gen={len(gen_imgs)}"
        return results

    # ---- 1. Core M3 --------------------------------------------------------
    print("  [1/5] Core M3 score ...")
    try:
        from evaluation.m3_score_v2 import M3EntropyMetric
        import torch
        from torchvision import transforms
        tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        real_tensors = torch.stack([tfm(img) for img in real_imgs])
        gen_tensors  = torch.stack([tfm(img) for img in gen_imgs])
        metric = M3EntropyMetric(device=device, backbone_id=backbone_id, kernel="rbf")
        res = metric(real_tensors, gen_tensors)
        results["core_m3"] = float(res.get("m3_score", res.get("m3_v2_final_score", 0)))
        print(f"    M3 = {results['core_m3']:.4f}")
    except Exception as e:
        results["core_m3"] = {"error": str(e)}
        traceback.print_exc()

    # ---- 2. Statistical Rigor ----------------------------------------------
    print(f"  [2/5] Statistical rigor (n_perm={n_perm}, n_boot={n_boot}) ...")
    try:
        from evaluation.statistical_rigor import statistical_m3
        sr = statistical_m3(
            real_imgs=real_imgs, gen_imgs=gen_imgs,
            backbone_id=backbone_id, device=device,
            n_perm=n_perm, n_boot=n_boot,
        )
        results["statistical_rigor"] = {
            "m3_score":    float(sr["m3_score"]),
            "p_value":     float(sr["p_value"]),
            "effect_size": float(sr["effect_size"]),
            "ci_low":      float(sr["ci_low"]),
            "ci_high":     float(sr["ci_high"]),
        }
        print(f"    Z={sr['effect_size']:.1f}  p={sr['p_value']:.4f}  "
              f"CI=[{sr['ci_low']:.4f}, {sr['ci_high']:.4f}]")
    except Exception as e:
        results["statistical_rigor"] = {"error": str(e)}
        traceback.print_exc()

    # ---- 3. Noise Robustness (monotonicity) --------------------------------
    print("  [3/5] Noise robustness (Spearman tau) ...")
    try:
        from experiments.noise_robustness import run_noise_robustness
        nr = run_noise_robustness(
            real_dir=real_dir,
            output_dir=str(ds_out / "noise_robustness"),
            device=device, num_images=min(n_images, 200),
            backbone_id=backbone_id,
        )
        results["noise_robustness"] = {
            "m3_spearman":  nr.get("noise", {}).get("spearman", {}).get("m3", {}).get("spearman_r"),
            "fid_spearman": nr.get("noise", {}).get("spearman", {}).get("fid", {}).get("spearman_r"),
        }
        print(f"    M3 rho={results['noise_robustness']['m3_spearman']}  "
              f"FID rho={results['noise_robustness']['fid_spearman']}")
    except Exception as e:
        results["noise_robustness"] = {"error": str(e)}
        traceback.print_exc()

    # ---- 4. Conditional MMD ------------------------------------------------
    print("  [4/5] Conditional MMD (intensity_quartile) ...")
    try:
        from evaluation.conditional_mmd import run_conditional_mmd
        cm = run_conditional_mmd(
            real_dir=real_dir, gen_dir=gen_dir,
            stratify_by="intensity_quartile", n_strata=4,
            n_images=n_images, backbone_id=backbone_id, device=device,
            output_dir=str(ds_out / "conditional_mmd"),
        )
        results["conditional_mmd"] = {
            "mean_m3":      float(cm["mean_m3"]),
            "worst_m3":     float(cm["worst_m3"]),
            "worst_stratum": cm["worst_stratum"],
            "std_m3":       float(cm["std_m3"]),
        }
        print(f"    mean={cm['mean_m3']:.4f}  worst={cm['worst_m3']:.4f} ({cm['worst_stratum']})")
    except Exception as e:
        results["conditional_mmd"] = {"error": str(e)}
        traceback.print_exc()

    # ---- 5. Coverage & Novelty ---------------------------------------------
    print("  [5/5] Coverage & Novelty ...")
    try:
        from evaluation.coverage_novelty import compute_coverage_novelty
        cn = compute_coverage_novelty(
            real_imgs=real_imgs, gen_imgs=gen_imgs,
            backbone_id=backbone_id, device=device,
        )
        results["coverage_novelty"] = {
            "coverage":         float(cn["coverage"]),
            "novelty":          float(cn["novelty"]),
            "memorization_rate": float(cn["memorization_rate"]),
        }
        print(f"    Coverage={cn['coverage']:.4f}  "
              f"Novelty={cn['novelty']:.4f}  "
              f"Memorization={cn['memorization_rate']:.4f}")
    except Exception as e:
        results["coverage_novelty"] = {"error": str(e)}
        traceback.print_exc()

    # Save per-dataset results
    out_path = ds_out / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved → {out_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["brats", "retinal", "cxr"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--checkpoint_dir",
                        default="output/output_unet/checkpoints/best",
                        help="BraTS DDPM checkpoint (for standard generation)")
    parser.add_argument("--output_dir", default="results/multidataset_run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_images", type=int, default=500)
    parser.add_argument("--n_perm", type=int, default=500)
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip image generation; assume all generated dirs exist")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MULTI-DATASET M3-SCORE EVALUATION")
    print("=" * 60)
    print(f"  Datasets:   {args.datasets}")
    print(f"  N images:   {args.n_images}")
    print(f"  Backbone:   {args.backbone_id}")
    print(f"  Device:     {args.device}")
    print(f"  Output:     {output_root.resolve()}")
    print()

    all_results = {}

    for ds_key in args.datasets:
        ds_info = DATASETS[ds_key]
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_info['name']}")
        print(f"{'='*60}")

        # --- Ensure data exists ---
        if not args.skip_generation:
            print("  [SETUP] Ensuring data directories exist ...")
            try:
                if ds_key == "brats":
                    ensure_brats_standard(
                        ds_info["gen_dir"], args.checkpoint_dir, args.device, args.seed
                    )
                elif ds_key == "retinal":
                    ensure_medmnist_real("retinamnist", ds_info["real_dir"], size=64)
                    ensure_retinal_generated(ds_info["gen_dir"], args.device, args.seed)
                elif ds_key == "cxr":
                    ensure_medmnist_real("pneumoniamnist", ds_info["real_dir"], size=64)
                    ensure_cxr_corrupted(ds_info["real_dir"], ds_info["gen_dir"],
                                         sigma=0.30, seed=args.seed)
            except Exception as e:
                print(f"  [SETUP ERROR] {e}")
                traceback.print_exc()
                all_results[ds_key] = {"error": f"Setup failed: {e}"}
                continue

        # --- Check dirs exist ---
        if not Path(ds_info["real_dir"]).exists():
            print(f"  [SKIP] real_dir not found: {ds_info['real_dir']}")
            print("         Run with --skip_generation=False or provide the data manually.")
            all_results[ds_key] = {"error": "real_dir missing"}
            continue
        if not Path(ds_info["gen_dir"]).exists():
            print(f"  [SKIP] gen_dir not found: {ds_info['gen_dir']}")
            all_results[ds_key] = {"error": "gen_dir missing"}
            continue

        # --- Run experiments ---
        try:
            result = run_experiments_for_dataset(
                ds_key=ds_key,
                ds_info=ds_info,
                output_root=output_root,
                device=args.device,
                n_images=args.n_images,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                backbone_id=args.backbone_id,
            )
            all_results[ds_key] = result
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            all_results[ds_key] = {"error": str(e)}

    # --- Save combined results ---
    combined_path = output_root / "all_datasets_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Print summary ---
    print(f"\n\n{'='*60}")
    print("  MULTI-DATASET SUMMARY")
    print(f"{'='*60}")
    headers = ["Dataset", "M3", "Z", "p-val", "CI", "Cov", "Nov", "CondMMD-worst"]
    print(f"{'Dataset':12s} {'M3':7s} {'Z':7s} {'p':6s} {'95% CI':16s} {'Cov':5s} {'Nov':5s} {'CondMMD-worst':14s}")
    print("-" * 80)
    for ds_key, r in all_results.items():
        if "error" in r and "core_m3" not in r:
            print(f"{ds_key:12s} FAILED: {r['error']}")
            continue
        m3  = r.get("core_m3", "N/A")
        sr  = r.get("statistical_rigor", {})
        cn  = r.get("coverage_novelty", {})
        cm  = r.get("conditional_mmd", {})
        z   = sr.get("effect_size", "N/A")
        p   = sr.get("p_value", "N/A")
        cil = sr.get("ci_low", "N/A")
        cih = sr.get("ci_high", "N/A")
        cov = cn.get("coverage", "N/A")
        nov = cn.get("novelty", "N/A")
        worst = cm.get("worst_m3", "N/A")
        ci_str = f"[{cil:.3f},{cih:.3f}]" if isinstance(cil, float) else "N/A"
        print(f"{ds_key:12s} {m3:<7.4f} {z:<7.1f} {p:<6.4f} {ci_str:16s} "
              f"{cov:<5.3f} {nov:<5.3f} {worst:<14.4f}"
              if isinstance(m3, float) else f"{ds_key:12s} {m3}")

    print(f"\nFull results → {combined_path}")


if __name__ == "__main__":
    main()
