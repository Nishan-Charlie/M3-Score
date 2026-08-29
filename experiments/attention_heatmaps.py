"""
RadioDino Spatial Attention Heatmaps — Section 3.10 (Interpretability)
======================================================================
Computes POPULATION-LEVEL attention shift maps between the real and
generated image distributions.

Because this is unconditional image generation, there is no pairing between
real and generated images.  All comparisons are distributional:

    shift_map[l] = mean_gen_attn[l] − mean_real_attn[l]

where mean_*_attn[l] is the CLS-to-patch self-attention averaged across all
images in the respective population for layer l.

This mirrors how M3-Score itself works: comparing distributions, not pairs.

For each layer l in {L1, L3, L12}:
    attn[l] = mean over heads of attention_weight[CLS → patch_i]
    reshaped to (16, 16) for ViT-B/14 with 224×224 input, patch_size=14
    upsampled to (224, 224) via bilinear interpolation

Usage:
    python experiments/attention_heatmaps.py \
        --real_dir   data_mri/brats_axial_multislice \
        --gen_dir    output/generated_500_best \
        --output_dir results/experiments_output_v5/attention_heatmaps \
        --device     cpu \
        --n_real     200 \
        --n_gen      200
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


_BACKBONE_ID = "Snarcy/RadioDino-s16"
_PATCH_SIZE  = 16   # ViT-S/16 patch size (rad-dino was ViT-B/14 → 14)


def _load_backbone_with_attention(device: str, backbone_id: str = _BACKBONE_ID):
    """Load timm RadioDino-s16 with fused attention disabled so hooks can capture attn weights."""
    import timm
    model = timm.create_model(f"hf_hub:{backbone_id}", pretrained=True)
    # Disable fused (flash) attention so the explicit softmax path is used, enabling hooks
    for blk in model.blocks:
        if hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = False
    return model.to(device).eval()


class _AttnHook:
    """Forward hook that captures attention weights from timm's Attention.attn_drop."""
    def __init__(self):
        self.weights: torch.Tensor | None = None
    def __call__(self, module, inp, out):
        # inp[0] is the attention weight tensor (B, heads, N, N) before dropout
        if inp:
            self.weights = inp[0].detach().cpu()


def _image_paths(directory: str, n: int | None = None) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif"):
        paths.extend(_glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(set(paths))
    if n is not None:
        paths = paths[:n]
    return paths


def _image_to_display(path: str, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0


@torch.no_grad()
def _extract_attention_maps(
    model,
    img_tensor: torch.Tensor,
    layer_indices: list[int],
    patch_size: int = _PATCH_SIZE,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """
    Returns per-layer (H_patch, W_patch) mean-head CLS attention map, normalised [0,1].
    Uses forward hooks on timm's attn_drop to capture attention weights.
    """
    img_tensor = img_tensor.to(device)
    h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    H_p, W_p = h // patch_size, w // patch_size

    hooks: dict[int, _AttnHook] = {}
    handles = []
    for li in layer_indices:
        if li < len(model.blocks):
            hook = _AttnHook()
            hooks[li] = hook
            handles.append(model.blocks[li].attn.attn_drop.register_forward_hook(hook))

    model(img_tensor)

    for h_ in handles:
        h_.remove()

    result: dict[int, np.ndarray] = {}
    for li in layer_indices:
        if li not in hooks or hooks[li].weights is None:
            continue
        attn = hooks[li].weights             # (1, heads, tokens, tokens)
        cls_attn = attn[0, :, 0, 1:].mean(0)  # (H_p*W_p,)
        cls_attn = cls_attn.reshape(H_p, W_p).numpy()
        a_min, a_max = cls_attn.min(), cls_attn.max()
        if a_max > a_min:
            cls_attn = (cls_attn - a_min) / (a_max - a_min)
        result[li] = cls_attn.astype(np.float32)
    return result


def _upsample_map(patch_map: np.ndarray, size: int = 224) -> np.ndarray:
    t = torch.from_numpy(patch_map).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


@torch.no_grad()
def _population_attention(
    model,
    paths: list[str],
    layer_indices: list[int],
    transform,
    device: str,
    label: str = "",
    patch_size: int = _PATCH_SIZE,
) -> dict[int, np.ndarray]:
    """
    Compute the mean attention map across an entire image population.
    Returns dict: layer_index → (H_patch, W_patch) mean attention, NOT normalised.
    """
    accum: dict[int, list[np.ndarray]] = {li: [] for li in layer_indices}
    H_p = W_p = 224 // patch_size

    for path in tqdm(paths, desc=f"Attention [{label}]", leave=False):
        try:
            img = Image.open(path).convert("RGB")
            t   = transform(img).unsqueeze(0).to(device)
            maps = _extract_attention_maps(model, t, layer_indices,
                                           patch_size=patch_size, device=device)
            for li, raw in maps.items():
                accum[li].append(raw)
        except Exception as e:
            print(f"  [WARN] Skipped {path}: {e}")

    return {li: np.mean(accum[li], axis=0).astype(np.float32)
            for li in layer_indices if accum[li]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_heatmaps(
    real_dir:    str,
    gen_dir:     str,
    output_dir:  str,
    device:      str = "cpu",
    n_real:      int = 200,
    n_gen:       int = 200,
    layer_labels: list[str] | None = None,
    backbone_id:  str = _BACKBONE_ID,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # L1, L6, L12 → 0-indexed 0, 5, 11 (RadioDino-s16 has 12 blocks)
    layer_indices = [0, 5, 11]
    if layer_labels is None:
        layer_labels = ["Early (L1)", "Mid (L6)", "Deep (L12)"]

    real_paths = _image_paths(real_dir, n_real)
    gen_paths  = _image_paths(gen_dir,  n_gen)

    if not real_paths or not gen_paths:
        raise FileNotFoundError(f"No images found in {real_dir} or {gen_dir}")

    print(f"[Heatmaps] Real: {len(real_paths)}  Generated: {len(gen_paths)}")
    print(f"[Heatmaps] Loading {backbone_id} (timm, hooks) ...")
    model     = _load_backbone_with_attention(device, backbone_id)
    transform = _get_transform(224)

    print("[Heatmaps] Computing population-level mean attention (real distribution) ...")
    real_attn = _population_attention(model, real_paths, layer_indices, transform, device, "real")

    print("[Heatmaps] Computing population-level mean attention (generated distribution) ...")
    gen_attn  = _population_attention(model, gen_paths,  layer_indices, transform, device, "gen")

    # ── Figure 1: Population-level shift map ───────────────────────────────────
    # 3 rows (layers) × 3 cols (real mean | gen mean | signed diff)
    n_layers = len(layer_indices)
    fig, axes = plt.subplots(n_layers, 3, figsize=(12, 4 * n_layers), dpi=150)
    fig.patch.set_facecolor("white")
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Real Brain MRI",
        "Generated Brain MRI",
        "Attention Shift",
    ]

    for row, (li, lbl) in enumerate(zip(layer_indices, layer_labels)):
        r_raw  = real_attn[li]
        g_raw  = gen_attn[li]
        diff   = g_raw - r_raw   # signed: red = gen attends more, blue = real attends more

        r_up   = _upsample_map(r_raw)
        g_up   = _upsample_map(g_raw)
        d_up   = _upsample_map(diff)

        # Common scale for real/gen panels
        vmin_rg = min(r_up.min(), g_up.min())
        vmax_rg = max(r_up.max(), g_up.max())
        abs_max = max(abs(d_up.min()), abs(d_up.max()))

        for col, (data, cmap, vmin, vmax) in enumerate(zip(
            [r_up, g_up, d_up],
            ["hot", "hot", "RdBu_r"],
            [vmin_rg, vmin_rg, -abs_max],
            [vmax_rg, vmax_rg,  abs_max],
        )):
            ax = axes[row, col]
            ax.set_facecolor("white")
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, color="#222222")
            if col == 0:
                ax.set_ylabel(lbl, fontsize=11, color="#222222")
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        "RadioDino Spatial Attention: Real vs. Generated Brain MRI",
        fontsize=13, fontweight="bold", color="#222222", y=1.02,
    )
    plt.tight_layout()
    shift_path = os.path.join(output_dir, "attention_population_shift.png")
    plt.savefig(shift_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Heatmaps] Saved: {shift_path}")

    # ── Figure 2: Typical examples from each distribution (no pairing) ────────
    # Show 3 real and 3 generated images side-by-side with their L3 attention maps.
    # Caption will make clear these are independent samples, not matched pairs.
    import random
    random.seed(42)
    sample_real = random.sample(real_paths, min(3, len(real_paths)))
    sample_gen  = random.sample(gen_paths,  min(3, len(gen_paths)))
    mid_li = layer_indices[1]   # L6

    fig2, axes2 = plt.subplots(2, 6, figsize=(24, 8), dpi=150)
    fig2.patch.set_facecolor("white")
    top_titles = ["Real MRI", "L6 Attention",
                  "Real MRI", "L6 Attention",
                  "Real MRI", "L6 Attention"]
    bot_titles = ["Generated MRI", "L6 Attention",
                  "Generated MRI", "L6 Attention",
                  "Generated MRI", "L6 Attention"]

    for col_pair, (r_path, g_path) in enumerate(zip(sample_real, sample_gen)):
        c = col_pair * 2
        r_img = _image_to_display(r_path)
        g_img = _image_to_display(g_path)

        r_t   = transform(Image.open(r_path).convert("RGB")).unsqueeze(0)
        g_t   = transform(Image.open(g_path).convert("RGB")).unsqueeze(0)
        r_maps = _extract_attention_maps(model, r_t, [mid_li], device=device)
        g_maps = _extract_attention_maps(model, g_t, [mid_li], device=device)

        r_attn_up = _upsample_map(r_maps[mid_li])
        g_attn_up = _upsample_map(g_maps[mid_li])

        for ax, img_data, cmap, title in [
            (axes2[0, c],     r_img,     "gray", top_titles[c]),
            (axes2[0, c + 1], r_attn_up, "hot",  top_titles[c + 1]),
            (axes2[1, c],     g_img,     "gray", bot_titles[c]),
            (axes2[1, c + 1], g_attn_up, "hot",  bot_titles[c + 1]),
        ]:
            ax.set_facecolor("white")
            ax.imshow(img_data, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=9, color="#222222")
            ax.axis("off")

    axes2[0, 0].set_ylabel("Real MRI", fontsize=11, color="#222222")
    axes2[1, 0].set_ylabel("Generated MRI", fontsize=11, color="#222222")
    plt.suptitle(
        "Mid-Layer Spatial Attention: Real and Generated Brain MRI",
        fontsize=13, fontweight="bold", color="#222222",
    )
    plt.tight_layout()
    examples_path = os.path.join(output_dir, "attention_distribution_examples.png")
    plt.savefig(examples_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Heatmaps] Saved: {examples_path}")

    print(f"\n[Heatmaps] Done. Output: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="results/experiments_output_v5/attention_heatmaps")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--n_real",     type=int, default=200)
    parser.add_argument("--n_gen",      type=int, default=200)
    args = parser.parse_args()

    generate_heatmaps(
        real_dir   = args.real_dir,
        gen_dir    = args.gen_dir,
        output_dir = args.output_dir,
        device     = args.device,
        n_real     = args.n_real,
        n_gen      = args.n_gen,
    )
