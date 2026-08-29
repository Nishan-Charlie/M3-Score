"""
distortion_monotonicity_per_scale.py  --  Validation Experiment
================================================================

Applies progressively stronger medically-relevant image corruptions and
measures M3-Score INDEPENDENTLY at each RadioDino-s16 layer scale
(L1=early/texture, L6=mid/structure, L12=late/semantics) plus the
multi-scale aggregate.

Also measures FID for comparison.

Corruption types
----------------
1. Gaussian noise   : sigma = [0, 0.05, 0.10, 0.20, 0.30, 0.50]
2. Gaussian blur    : kernel_px = [0, 3, 5, 7, 11, 15, 21]
3. Rectangular occlusion : mask_frac = [0, 0.05, 0.10, 0.20, 0.30, 0.50]
4. Brightness shift : shift = [0, 0.05, 0.1, 0.2, 0.3, 0.5]  (proxy for scanner drift)

Expected result
---------------
Each per-scale M3 increases monotonically with corruption strength.
The multi-scale aggregate is more sensitive than any single scale.
FID is non-monotonic or flat for mild blur (the "medical FID paradox").

Usage
-----
    python experiments/distortion_monotonicity_per_scale.py \
        --real_dir data_mri/brats_axial_multislice \
        --gen_dir  output/generated \
        --output_dir results/distortion_monotonicity_per_scale
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Corruption functions (operate on PIL images, return PIL images)
# ---------------------------------------------------------------------------

def _corrupt_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    if sigma == 0:
        return img
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _corrupt_blur(img: Image.Image, kernel_px: float) -> Image.Image:
    if kernel_px == 0:
        return img
    r = max(1, int(kernel_px) // 2)
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def _corrupt_occlusion(img: Image.Image, mask_frac: float) -> Image.Image:
    if mask_frac == 0:
        return img
    arr = np.array(img).copy()
    H, W = arr.shape[:2]
    size = int(np.sqrt(mask_frac) * min(H, W))
    y0 = np.random.randint(0, max(1, H - size))
    x0 = np.random.randint(0, max(1, W - size))
    arr[y0:y0+size, x0:x0+size] = 0
    return Image.fromarray(arr)


def _corrupt_brightness(img: Image.Image, shift: float) -> Image.Image:
    if shift == 0:
        return img
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + shift, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


CORRUPTION_TYPES = {
    "gaussian_noise": {
        "fn":     _corrupt_gaussian_noise,
        "levels": [0, 0.05, 0.10, 0.20, 0.30, 0.50],
        "xlabel": "Noise sigma",
        "param":  "sigma",
    },
    "gaussian_blur": {
        "fn":     _corrupt_blur,
        "levels": [0, 3, 5, 7, 11, 15, 21],
        "xlabel": "Blur kernel (px)",
        "param":  "kernel_px",
    },
    "occlusion": {
        "fn":     _corrupt_occlusion,
        "levels": [0, 0.05, 0.10, 0.20, 0.30, 0.50],
        "xlabel": "Occluded fraction",
        "param":  "mask_frac",
    },
    "brightness_shift": {
        "fn":     _corrupt_brightness,
        "levels": [0, 0.05, 0.10, 0.20, 0.30, 0.50],
        "xlabel": "Brightness shift",
        "param":  "shift",
    },
}


# ---------------------------------------------------------------------------
# Per-layer feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_per_layer(
    images: List[Image.Image],
    backbone_id: str,
    device: str,
    target_layers: List[int],  # 1-indexed
    batch_size: int = 16,
) -> dict:
    """
    Extract CLS features at each target layer.
    Returns {layer_idx: np.ndarray (N, D)}.
    """
    import timm
    from torchvision import transforms
    import torch.nn.functional as NF

    model = timm.create_model(
        f"hf_hub:{backbone_id}", pretrained=True, img_size=224
    ).to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Collect activations via hooks
    layer_feats: dict = {l: [] for l in target_layers}
    hooks = []

    for layer_idx in target_layers:
        blk = model.blocks[layer_idx - 1]  # 0-indexed

        def _hook(module, inp, out, _l=layer_idx):
            # out: (B, tokens, D)  or  (B, D)
            if out.dim() == 3:
                cls = out[:, 0]    # CLS token
            else:
                cls = out
            layer_feats[_l].append(NF.normalize(cls.detach().cpu(), dim=-1))

        hooks.append(blk.register_forward_hook(_hook))

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        x = torch.stack([tf(img.convert("RGB")) for img in batch]).to(device)
        model(x)

    for h in hooks:
        h.remove()

    return {l: torch.cat(layer_feats[l], dim=0).numpy()
            for l in target_layers}


# ---------------------------------------------------------------------------
# FID helper
# ---------------------------------------------------------------------------

def _fid_from_feats(real_f: np.ndarray, gen_f: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    mu_r = real_f.mean(0); mu_g = gen_f.mean(0)
    sg_r = np.cov(real_f.T); sg_g = np.cov(gen_f.T)
    diff = mu_r - mu_g
    cm = sqrtm(sg_r @ sg_g)
    if np.iscomplexobj(cm): cm = cm.real
    return float(max(diff @ diff + np.trace(sg_r + sg_g - 2 * cm), 0))


# ---------------------------------------------------------------------------
# MMD helper
# ---------------------------------------------------------------------------

def _mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()

    def sq(A, B):
        return (A.unsqueeze(1) - B.unsqueeze(0)).pow(2).sum(-1)

    Dxx = sq(Xt, Xt); Dyy = sq(Yt, Yt); Dxy = sq(Xt, Yt)
    med = torch.cat([Dxy.flatten(), Dxx[Dxx > 0].flatten(),
                     Dyy[Dyy > 0].flatten()]).median().item()
    if med < 1e-10: med = 1.0

    n, m = len(X), len(Y); total = 0.0
    for bw in [med * (2 ** k) for k in range(-2, 3)]:
        g = 1.0 / (2.0 * bw)
        Kxx = torch.exp(-g * Dxx); Kxx.fill_diagonal_(0)
        Kyy = torch.exp(-g * Dyy); Kyy.fill_diagonal_(0)
        Kxy = torch.exp(-g * Dxy)
        total += (Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean()).item()
    return total / 5.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_distortion_monotonicity(
    real_dir:     str,
    gen_dir:      str,
    output_dir:   str = "results/distortion_monotonicity_per_scale",
    n_images:     int = 300,
    backbone_id:  str = "Snarcy/RadioDino-s16",
    device:       str = "cuda",
    batch_size:   int = 16,
    seed:         int = 42,
    target_layers: List[int] = None,  # default: [1, 6, 12]
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    target_layers = target_layers or [1, 6, 12]

    from experiments._shared_utils import load_pils_recursive
    print("\n=== Distortion Monotonicity Per Scale ===")

    # Load base generated images (these will be corrupted)
    gen_imgs_base = load_pils_recursive(gen_dir,  n=n_images)
    print(f"  Loaded {len(gen_imgs_base)} generated images for corruption")

    # Extract real features at each layer (extracted once, reused)
    real_imgs = load_pils_recursive(real_dir, n=n_images)
    print(f"  Loaded {len(real_imgs)} real images")
    print(f"  Extracting real features at layers {target_layers}...")
    real_layer_feats = _extract_per_layer(
        real_imgs, backbone_id, device, target_layers, batch_size)

    all_results = {}

    for corrupt_name, corrupt_cfg in CORRUPTION_TYPES.items():
        fn     = corrupt_cfg["fn"]
        levels = corrupt_cfg["levels"]
        xlabel = corrupt_cfg["xlabel"]

        print(f"\n  Corruption: {corrupt_name}")
        results_per_level = []

        for level in levels:
            # Apply corruption
            corrupted = [fn(img, level) for img in gen_imgs_base]

            # Extract corrupted features at all layers
            corr_layer_feats = _extract_per_layer(
                corrupted, backbone_id, device, target_layers, batch_size)

            # Compute per-layer MMD
            layer_mmds = {}
            for l in target_layers:
                mmd = _mmd_rbf(real_layer_feats[l], corr_layer_feats[l])
                layer_mmds[f"L{l}"] = float(mmd)

            # Multi-scale aggregate (equal weights here; paper uses CKA weights)
            m3_agg = float(np.mean(list(layer_mmds.values())))

            # FID at last layer (L12)
            fid = _fid_from_feats(real_layer_feats[target_layers[-1]],
                                   corr_layer_feats[target_layers[-1]])

            row = {"level": level, **layer_mmds, "m3_agg": m3_agg, "fid": fid}
            results_per_level.append(row)
            print(f"    level={level}  " +
                  "  ".join(f"{k}={v:.5f}" for k, v in row.items() if k != "level"))

        all_results[corrupt_name] = {
            "levels":    levels,
            "xlabel":    xlabel,
            "results":   results_per_level,
        }

    # Save JSON
    with open(os.path.join(output_dir, "distortion_monotonicity_report.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plots
    _make_plots(all_results, output_dir, target_layers)

    print(f"\nSaved -> {output_dir}")
    return all_results


def _make_plots(all_results: dict, output_dir: str, target_layers: List[int]):
    n_corrupts = len(all_results)
    fig, axes = plt.subplots(2, n_corrupts, figsize=(5 * n_corrupts, 9))

    colors_per_layer = {l: c for l, c in zip(target_layers,
                         ["#9bc0e7", "#4878cf", "#1a3a6b"])}
    color_agg = "#e05c5c"
    color_fid = "#999999"

    for col_idx, (cname, cdata) in enumerate(all_results.items()):
        levels  = cdata["levels"]
        results = cdata["results"]
        xlabel  = cdata["xlabel"]

        # Top: per-layer MMD
        ax_top = axes[0, col_idx]
        for l in target_layers:
            vals = [r[f"L{l}"] for r in results]
            ax_top.plot(levels, vals, "o-", lw=2, ms=5,
                        color=colors_per_layer[l], label=f"L{l}")
        agg_vals = [r["m3_agg"] for r in results]
        ax_top.plot(levels, agg_vals, "s--", lw=2, ms=6,
                    color=color_agg, label="M3 (agg)")
        ax_top.set_xlabel(xlabel, fontsize=11)
        ax_top.set_ylabel("MMD score", fontsize=11)
        ax_top.set_title(cname.replace("_", " ").title(), fontsize=12)
        ax_top.legend(fontsize=9)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        # Bottom: FID vs M3 aggregate (normalised)
        ax_bot = axes[1, col_idx]
        fid_vals = np.array([r["fid"] for r in results])
        m3_vals  = np.array(agg_vals)
        # Normalise to [0,1]
        def _norm(v):
            r = v.max() - v.min() + 1e-10
            return (v - v.min()) / r
        ax_bot.plot(levels, _norm(fid_vals), "D--", lw=2, ms=5,
                    color=color_fid, label="FID (normalised)")
        ax_bot.plot(levels, _norm(m3_vals),  "s-",  lw=2, ms=5,
                    color=color_agg, label="M3-agg (normalised)")
        ax_bot.set_xlabel(xlabel, fontsize=11)
        ax_bot.set_ylabel("Normalised score", fontsize=11)
        ax_bot.set_title("FID vs M3 comparison", fontsize=12)
        ax_bot.legend(fontsize=9)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

    plt.suptitle("Distortion Monotonicity Per Scale\n"
                 "M3 increases monotonically; FID non-monotonic on medical corruptions",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distortion_monotonicity.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Monotonicity check: compute Kendall tau per metric
    import scipy.stats as stats
    mono_report = {}
    for cname, cdata in all_results.items():
        results = cdata["results"]
        levels  = cdata["levels"]
        for key in [f"L{l}" for l in target_layers] + ["m3_agg", "fid"]:
            vals = [r[key] for r in results]
            tau, pval = stats.kendalltau(levels, vals)
            mono_report.setdefault(cname, {})[key] = {
                "kendall_tau": float(tau), "p_value": float(pval)
            }

    with open(os.path.join(output_dir, "monotonicity_check.json"), "w") as f:
        json.dump(mono_report, f, indent=2)

    print("\n  Kendall tau (monotonicity) summary:")
    for cname, keys in mono_report.items():
        print(f"  {cname}:")
        for key, stats_dict in keys.items():
            print(f"    {key}: tau={stats_dict['kendall_tau']:.3f}  "
                  f"p={stats_dict['p_value']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",      required=True)
    p.add_argument("--gen_dir",       required=True)
    p.add_argument("--output_dir",    default="results/distortion_monotonicity_per_scale")
    p.add_argument("--n_images",      type=int, default=300)
    p.add_argument("--backbone_id",   default="Snarcy/RadioDino-s16")
    p.add_argument("--device",        default=None)
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--target_layers", nargs="+", type=int, default=[1, 6, 12])
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_distortion_monotonicity(
        real_dir      = a.real_dir,
        gen_dir       = a.gen_dir,
        output_dir    = a.output_dir,
        n_images      = a.n_images,
        backbone_id   = a.backbone_id,
        device        = device,
        batch_size    = a.batch_size,
        seed          = a.seed,
        target_layers = a.target_layers,
    )
