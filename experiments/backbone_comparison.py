"""
Backbone Comparison Experiment
===============================
Six ViT / ResNet backbones × two MMD kernels:

  Backbones
  ---------
  1. microsoft/rad-dino        -- RAD-DINO     (ViT-B/14,  transformers)
  2. Snarcy/RadioDino-s16      -- RadioDINO    (ViT-S/16,  timm)
  3. facebook/dinov2-base      -- DINOv2       (ViT-B/14,  transformers)
  4. flaviagiammarino/pubmed-clip-vit-base-patch32
                               -- PubMedCLIP  (ViT-B/32,  transformers CLIP)
  5. microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
                               -- BiomedCLIP  (ViT-B/16,  open_clip)
  6. RadImageNet-ResNet50      -- RadImageNet  (ResNet-50, timm / torchvision)

  MMD kernels
  -----------
  poly  : k(u,v) = (u·v + 1)³   (M3-v2 default)
  rbf   : k(u,v) = mean_σ exp(-||u-v||²/2σ²)
          five-bandwidth, σ ∈ {σ_med/4, σ_med/2, σ_med, 2σ_med, 4σ_med}
          median-heuristic bandwidth on joint (real ∪ gen) sample

  Per backbone × kernel the script computes:
  - M3 score (equal-weighted MMD across active layers)
  - OOD AUC  (Gaussian density on real; score real vs gen)
  - Discriminability = score_rg / score_rr
  - Active layers chosen by greedy CKA pruning (τ=0.80)
"""

from __future__ import annotations

import json, os, glob
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from torchvision import transforms

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_EPS_L2  = 1e-8
_EPS_CKA = 1e-8
_BATCH   = 16   # images per GPU batch during feature extraction

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_images(directory: str, n: int = 500) -> torch.Tensor:
    """Return (N, 3, 224, 224) float32 normalised tensor."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(paths)[:n]
    if not paths:
        raise FileNotFoundError(f"No images in {directory}")
    return torch.stack([
        _TRANSFORM(Image.open(p).convert("RGB"))
        for p in tqdm(paths, desc="Loading", leave=False)
    ])


def _unnorm_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a normalised (N,3,H,W) tensor back to PIL list."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = (tensor.cpu() * std + mean).clamp(0, 1)
    pils = []
    for t in imgs:
        arr = (t.numpy().transpose(1, 2, 0) * 255).astype("uint8")
        pils.append(Image.fromarray(arr))
    return pils


# ===========================================================================
# Backbone extractors
# ===========================================================================

class _TransformersViTExtractor:
    """CLS-token features from every layer — HuggingFace AutoModel.

    Passes pixel_values directly; does NOT require AutoImageProcessor so it
    works offline as long as the model weights are cached.  Inputs are already
    normalised by _TRANSFORM (ImageNet mean/std, 224×224).
    """

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModel
        print(f"  Loading {model_id} (transformers) ...")
        self.model = AutoModel.from_pretrained(
            model_id, output_hidden_states=True, output_attentions=False
        ).to(device)
        self.model.eval()
        self.device     = device
        self.num_layers = len(self.model.encoder.layer)
        self.embed_dim  = self.model.config.hidden_size

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        n = imgs.shape[0]
        acc = [[] for _ in range(self.num_layers)]
        for s in range(0, n, _BATCH):
            pv  = imgs[s:s+_BATCH].to(self.device)   # already (B,3,224,224)
            out = self.model(pixel_values=pv, output_hidden_states=True)
            hs  = out.hidden_states[1:]
            for i in range(self.num_layers):
                acc[i].append(hs[i][:, 0, :].cpu())
            del out, pv, hs
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return [torch.cat(a) for a in acc]


class _TimmViTExtractor:
    """CLS-token features via manual timm ViT block iteration."""

    def __init__(self, model_id: str, device: str):
        import timm as _timm
        hf_id = f"hf_hub:{model_id}" if not model_id.startswith("hf_hub:") else model_id
        print(f"  Loading {model_id} (timm) ...")
        self.model     = _timm.create_model(hf_id, pretrained=True)
        self.model.eval().to(device)
        self.device    = device
        self.num_layers = len(self.model.blocks)
        self.embed_dim  = self.model.embed_dim

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        n = imgs.shape[0]
        acc = [[] for _ in range(self.num_layers)]
        for s in range(0, n, _BATCH):
            batch = imgs[s:s+_BATCH].to(self.device)
            x = self.model.patch_embed(batch)
            if hasattr(self.model, '_pos_embed'):
                x = self.model._pos_embed(x)
            else:
                cls = self.model.cls_token.expand(x.shape[0], -1, -1)
                x   = torch.cat([cls, x], dim=1) + self.model.pos_embed
            if hasattr(self.model, 'patch_drop'): x = self.model.patch_drop(x)
            if hasattr(self.model, 'norm_pre'):   x = self.model.norm_pre(x)
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
                acc[i].append(x[:, 0, :].cpu())
            del x, batch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return [torch.cat(a) for a in acc]


class _CLIPViTExtractor:
    """CLS-token features from every layer — HuggingFace CLIPVisionModel.

    Does NOT require CLIPProcessor / preprocessor_config.json — passes
    pixel_values directly from the pre-normalised _TRANSFORM tensors.
    """

    def __init__(self, model_id: str, device: str):
        from transformers import CLIPModel
        print(f"  Loading {model_id} (transformers CLIP) ...")
        full = CLIPModel.from_pretrained(model_id)
        self.model       = full.vision_model.to(device)
        self.model.eval()
        self.device      = device
        self.num_layers  = self.model.config.num_hidden_layers
        self.embed_dim   = self.model.config.hidden_size
        del full

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        n = imgs.shape[0]
        acc = [[] for _ in range(self.num_layers)]
        for s in range(0, n, _BATCH):
            pv  = imgs[s:s+_BATCH].to(self.device)   # (B,3,224,224) pre-normalised
            out = self.model(pixel_values=pv, output_hidden_states=True)
            hs  = out.hidden_states[1:]   # skip patch-embed
            for i in range(self.num_layers):
                acc[i].append(hs[i][:, 0, :].cpu())
            del out, pv, hs
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return [torch.cat(a) for a in acc]


class _OpenCLIPViTExtractor:
    """CLS-token features via manual open_clip ViT block iteration.

    Handles two open_clip visual-encoder layouts:
      - Standard open_clip ViT  : v.transformer.resblocks  (LND layout)
      - TimmModel wrapper        : v.trunk.blocks           (BSN layout)
        e.g. BiomedCLIP uses TimmModel wrapping a timm VisionTransformer
    """

    def __init__(self, model_id: str, device: str):
        import open_clip as _oc
        hf_id = f"hf-hub:{model_id}" if not model_id.startswith("hf-hub:") else model_id
        print(f"  Loading {model_id} (open_clip) ...")
        model, _, preprocess = _oc.create_model_and_transforms(hf_id)
        model.eval().to(device)
        self.model      = model
        self.preprocess = preprocess
        self.device     = device
        v = model.visual
        # Detect layout
        if hasattr(v, "trunk") and hasattr(v.trunk, "blocks"):
            # TimmModel wrapper (e.g. BiomedCLIP)
            self._layout    = "timm"
            self.blocks     = v.trunk.blocks
            self.num_layers = len(self.blocks)
            self.embed_dim  = v.trunk.num_features
        else:
            # Standard open_clip ViT
            self._layout    = "openclip"
            self.blocks     = v.transformer.resblocks
            self.num_layers = len(self.blocks)
            self.embed_dim  = v.transformer.width

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        n   = imgs.shape[0]
        acc = [[] for _ in range(self.num_layers)]
        v   = self.model.visual

        for s in range(0, n, _BATCH):
            pils  = _unnorm_to_pil(imgs[s:s+_BATCH])
            batch = torch.stack([self.preprocess(p) for p in pils]).to(self.device)

            if self._layout == "timm":
                # TimmModel / timm VisionTransformer — BSN layout (batch, seq, dim)
                trunk = v.trunk
                x = trunk.patch_embed(batch)
                if hasattr(trunk, "_pos_embed"):
                    x = trunk._pos_embed(x)
                else:
                    cls = trunk.cls_token.expand(x.shape[0], -1, -1)
                    x   = torch.cat([cls, x], dim=1) + trunk.pos_embed
                if hasattr(trunk, "norm_pre"):
                    x = trunk.norm_pre(x)
                for i, blk in enumerate(self.blocks):
                    x = blk(x)
                    acc[i].append(x[:, 0, :].cpu())   # CLS token (batch dim 0)
            else:
                # Standard open_clip ViT — LND layout (seq, batch, dim)
                x   = v.conv1(batch)
                x   = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                cls = v.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
                x   = torch.cat([cls, x], dim=1) + v.positional_embedding
                x   = v.ln_pre(x).permute(1, 0, 2)   # NLD -> LND
                for i, blk in enumerate(self.blocks):
                    x = blk(x)
                    acc[i].append(x[0, :, :].cpu())   # CLS token (seq dim 0)

            del x, batch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return [torch.cat(a) for a in acc]


class _ResNetExtractor:
    """
    Spatial feature extractor for ResNet-style models (RadImageNet).

    Extracts global-average-pooled features after each of the 4 residual
    stage groups, giving 4 comparison layers of increasing semantic depth.

    Loading priority:
      1. timm hf_hub:<model_id>   (community uploads)
      2. torchvision resnet50(pretrained=False) + custom weight path
      3. torchvision resnet50(pretrained=True)  as ImageNet fallback
    """

    _RADNET_IDS = [
        "Tonic/radImageNet-resnet50",
        "BMEII/RadImageNet-ResNet50",
        "hannaheargle/radImageNet",
    ]

    def __init__(self, model_id: str, device: str):
        import timm as _timm
        import torchvision.models as _tv

        print(f"  Loading {model_id} (ResNet) ...")
        loaded = False

        # 1. Try timm hub upload
        for hf_id in self._RADNET_IDS:
            try:
                self._model = _timm.create_model(f"hf_hub:{hf_id}", pretrained=True)
                self._model.eval().to(device)
                print(f"    Loaded from timm: {hf_id}")
                loaded = True
                break
            except Exception:
                pass

        # 2. Fallback: ImageNet ResNet50 with a warning
        if not loaded:
            print(f"    [WARN] RadImageNet weights not found on HuggingFace.")
            print(f"           Falling back to ImageNet-pretrained ResNet50.")
            print(f"           For true RadImageNet features download weights from")
            print(f"           https://github.com/BMEII-AI/RadImageNet")
            self._model = _tv.resnet50(weights=_tv.ResNet50_Weights.IMAGENET1K_V2)
            self._model.eval().to(device)

        self.device    = device
        self.num_layers = 4   # 4 residual stage outputs
        self.embed_dim  = 2048

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        import torch.nn.functional as F
        n = imgs.shape[0]
        acc = [[] for _ in range(self.num_layers)]
        m = self._model

        # Identify stage modules (works for timm & torchvision resnet50)
        if hasattr(m, 'layer1'):
            stages = [m.layer1, m.layer2, m.layer3, m.layer4]
            def stem(x):
                x = m.conv1(x); x = m.bn1(x); x = m.act1(x) if hasattr(m,'act1') else m.relu(x)
                return m.maxpool(x)
        else:
            # timm resnet: stages via m.layer1..4
            stages = [getattr(m, f'layer{i}') for i in range(1, 5)]
            def stem(x):
                return m.maxpool(m.act1(m.bn1(m.conv1(x))))

        for s in range(0, n, _BATCH):
            x = imgs[s:s+_BATCH].to(self.device)
            x = stem(x)
            for i, stage in enumerate(stages):
                x = stage(x)
                pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
                acc[i].append(pooled.cpu())
            del x
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return [torch.cat(a) for a in acc]


# ===========================================================================
# MMD kernels
# ===========================================================================

def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=_EPS_L2)


def _mmd2_poly(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased MMD² — degree-3 polynomial kernel on unit-sphere features."""
    x, y = _l2_norm(x), _l2_norm(y)
    Kxx  = (x @ x.t() + 1.0) ** 3
    Kyy  = (y @ y.t() + 1.0) ** 3
    Kxy  = (x @ y.t() + 1.0) ** 3
    m, n = x.shape[0], y.shape[0]
    return float(((Kxx.sum()-Kxx.trace())/(m*(m-1)) +
                  (Kyy.sum()-Kyy.trace())/(n*(n-1)) - 2*Kxy.mean()).clamp(min=0))


def _mmd2_rbf(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Unbiased MMD² — five-bandwidth Gaussian RBF kernel.

    Bandwidth schedule: σ² ∈ {σ²_med/4, σ²_med/2, σ²_med, 2σ²_med, 4σ²_med}
    where σ²_med = median pairwise squared distance on joint (real ∪ gen) sample.
    """
    x, y = _l2_norm(x), _l2_norm(y)

    def _sq(a, b):
        return ((a*a).sum(1,keepdim=True) + (b*b).sum(1) - 2*(a@b.t())).clamp(min=0)

    Dxx = _sq(x, x); Dyy = _sq(y, y); Dxy = _sq(x, y)

    joint  = torch.cat([x, y])
    D_joint = _sq(joint, joint)
    mask   = torch.triu(torch.ones(D_joint.shape[0], D_joint.shape[0], dtype=torch.bool), diagonal=1)
    s2_med = float(D_joint[mask].median().clamp(min=1e-10))

    m, n  = x.shape[0], y.shape[0]
    total = 0.0
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        g = 1.0 / (2.0 * s2_med * scale)
        Kxx = torch.exp(-g * Dxx); Kyy = torch.exp(-g * Dyy); Kxy = torch.exp(-g * Dxy)
        total += float(((Kxx.sum()-Kxx.trace())/(m*(m-1)) +
                        (Kyy.sum()-Kyy.trace())/(n*(n-1)) - 2*Kxy.mean()).clamp(min=0))
    return total / 5.0


# ===========================================================================
# Shared analysis helpers
# ===========================================================================

def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx, Ky = X @ X.t(), Y @ Y.t()
    return float((Kx*Ky).sum() / ((Kx.norm("fro")**2 * Ky.norm("fro")**2).sqrt() + _EPS_CKA))


def _select_layers(feats: list[torch.Tensor], tau: float = 0.80) -> list[int]:
    L = len(feats)
    sel = [L]
    for i in range(L-1, 0, -1):
        if all(_linear_cka(feats[i-1], feats[s-1]) < tau for s in sel):
            sel.append(i)
    return sorted(sel)


def _ood_auc(real_f: torch.Tensor, gen_f: torch.Tensor) -> float:
    r, g  = real_f.numpy().astype(np.float32), gen_f.numpy().astype(np.float32)
    mu, var = r.mean(0), r.var(0) + 1e-6
    def ll(x): return -0.5 * ((x-mu)**2/var).sum(1)
    labels = np.concatenate([np.ones(len(r)), np.zeros(len(g))])
    scores = np.concatenate([ll(r), ll(g)])
    try:    return float(roc_auc_score(labels, scores))
    except: return float("nan")


# ===========================================================================
# Per-backbone evaluation
# ===========================================================================

def _evaluate_backbone(extractor, real_imgs: torch.Tensor,
                        gen_imgs: torch.Tensor, cka_tau: float = 0.80) -> dict:
    print("  Extracting real features ...")
    real_f = extractor.extract(real_imgs)
    print("  Extracting gen  features ...")
    gen_f  = extractor.extract(gen_imgs)

    print("  CKA layer selection ...")
    active = _select_layers(real_f, tau=cka_tau)
    print(f"  Active layers: {active}")

    # Subset to active layers
    rf_act = [real_f[l-1] for l in active]
    gf_act = [gen_f[l-1]  for l in active]

    # -- Both kernels ----------------------------------------------------
    poly_mmds, rbf_mmds = {}, {}
    for lk, rf, gf in zip(active, rf_act, gf_act):
        poly_mmds[f"L{lk}"] = _mmd2_poly(rf, gf)
        rbf_mmds[f"L{lk}"]  = _mmd2_rbf(rf, gf)

    m3_poly = float(np.mean(list(poly_mmds.values())))
    m3_rbf  = float(np.mean(list(rbf_mmds.values())))

    # Discriminability (score_rg / score_rr) for both kernels
    half = len(real_imgs) // 2
    rr_poly = float(np.mean([_mmd2_poly(real_f[l-1][:half], real_f[l-1][half:]) for l in active]))
    rr_rbf  = float(np.mean([_mmd2_rbf (real_f[l-1][:half], real_f[l-1][half:]) for l in active]))
    disc_poly = m3_poly / (rr_poly + 1e-10)
    disc_rbf  = m3_rbf  / (rr_rbf  + 1e-10)

    # -- OOD AUC (per active layer + mean) ----------------------------
    ooc_per_layer = {f"L{l}": _ood_auc(real_f[l-1], gen_f[l-1]) for l in active}
    weighted_auc  = float(np.nanmean(list(ooc_per_layer.values())))

    return {
        "num_layers":     extractor.num_layers,
        "embed_dim":      extractor.embed_dim,
        "active_layers":  active,
        "cka_tau":        cka_tau,
        # Polynomial kernel
        "m3_poly":        round(m3_poly, 6),
        "disc_poly":      round(disc_poly, 4),
        "rr_poly":        round(rr_poly, 6),
        "layer_mmd_poly": {k: round(v, 6) for k, v in poly_mmds.items()},
        # RBF kernel
        "m3_rbf":         round(m3_rbf, 6),
        "disc_rbf":       round(disc_rbf, 4),
        "rr_rbf":         round(rr_rbf, 6),
        "layer_mmd_rbf":  {k: round(v, 6) for k, v in rbf_mmds.items()},
        # Shared
        "ooc_auc_per_layer": {k: round(v, 4) for k, v in ooc_per_layer.items()},
        "weighted_auc":   round(weighted_auc, 4),
    }


# ===========================================================================
# Plotting
# ===========================================================================

def _plot(results: dict, output_dir: str) -> dict:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    bnames = list(results.keys())
    short  = {
        "microsoft/rad-dino":   "RAD-DINO\n(B/14)",
        "Snarcy/RadioDino-s16": "RadioDINO\n(S/16)",
        "facebook/dinov2-base": "DINOv2\n(B/14)",
        "flaviagiammarino/pubmed-clip-vit-base-patch32": "PubMedCLIP\n(B/32)",
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": "BiomedCLIP\n(B/16)",
        "RadImageNet-ResNet50": "RadImageNet\n(R50)",
    }
    labels = [short.get(b, b.split("/")[-1]) for b in bnames]
    colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4"]

    metrics = [
        ("M3 (poly ↓)", "m3_poly"),
        ("M3 (RBF  ↓)", "m3_rbf"),
        ("Disc poly (↑)", "disc_poly"),
        ("Disc RBF  (↑)", "disc_rbf"),
        ("OOD AUC   (↑)", "weighted_auc"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Backbone × Kernel Comparison  (RAD-DINO / RadioDINO / DINOv2 / PubMedCLIP / BiomedCLIP / RadImageNet)",
                 fontsize=11, fontweight="bold")

    for ax, (title, key) in zip(axes.flat, metrics):
        vals = [results[b].get(key, float("nan")) if "error" not in results[b] else float("nan")
                for b in bnames]
        bars = ax.bar(labels, vals, color=colors[:len(labels)], edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_title(title, fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    p1 = os.path.join(output_dir, "backbone_comparison.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()

    # Per-layer MMD heatmap (poly and RBF)
    valid = [(b, r) for b, r in results.items() if "error" not in r]
    if valid:
        fig2, axes2 = plt.subplots(2, len(valid), figsize=(5*len(valid), 7))
        if len(valid) == 1: axes2 = [[axes2[0]], [axes2[1]]]
        for col, (bname, bres) in enumerate(valid):
            for row, kern in enumerate(("poly", "rbf")):
                ax  = axes2[row][col]
                mmd = bres.get(f"layer_mmd_{kern}", {})
                if mmd:
                    ax.barh(list(mmd.keys()), list(mmd.values()),
                            color=colors[col], alpha=0.85)
                ax.set_title(f"{short.get(bname, bname.split('/')[-1])}\n({kern})",
                             fontsize=8, fontweight="bold")
                ax.set_xlabel("MMD²", fontsize=7)
                ax.invert_yaxis()
                ax.spines[["top","right"]].set_visible(False)
                ax.tick_params(labelsize=7)
        plt.suptitle("Per-layer MMD²: polynomial vs RBF", fontsize=11)
        plt.tight_layout()
        p2 = os.path.join(output_dir, "backbone_layer_mmd.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    else:
        p2 = None

    return {"comparison_bar": p1, "layer_mmd": p2}


# ===========================================================================
# Backbone registry
# ===========================================================================

_BACKBONES: list[tuple[str, str]] = [
    ("microsoft/rad-dino",                                              "transformers"),
    ("Snarcy/RadioDino-s16",                                            "timm"),
    ("facebook/dinov2-base",                                            "transformers"),
    ("flaviagiammarino/pubmed-clip-vit-base-patch32",                   "clip_transformers"),
    ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",       "open_clip"),
    ("RadImageNet-ResNet50",                                             "resnet"),
]


def _build_extractor(model_id: str, backend: str, device: str):
    if backend == "transformers":
        return _TransformersViTExtractor(model_id, device)
    if backend == "timm":
        return _TimmViTExtractor(model_id, device)
    if backend == "clip_transformers":
        return _CLIPViTExtractor(model_id, device)
    if backend == "open_clip":
        return _OpenCLIPViTExtractor(model_id, device)
    if backend == "resnet":
        return _ResNetExtractor(model_id, device)
    raise ValueError(f"Unknown backend: {backend}")


# ===========================================================================
# Main entry point
# ===========================================================================

def run_backbone_comparison(
    real_dir:   str,
    gen_dir:    str,
    output_dir: str = "./backbone_comparison_output",
    n_images:   int = 500,
    device:     str = "cuda",
    cka_tau:    float = 0.80,
    backbones:  Optional[list] = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if backbones is None:
        backbones = _BACKBONES

    print(f"\n{'='*60}")
    print("  Backbone × Kernel Comparison")
    print(f"  Backbones: {len(backbones)}")
    print(f"  Kernels  : polynomial  +  multi-bandwidth RBF")
    print(f"  N images : {n_images}   CKA tau : {cka_tau}")
    print(f"{'='*60}\n")

    print("Loading images ...")
    real_imgs = _load_images(real_dir, n=n_images)
    gen_imgs  = _load_images(gen_dir,  n=n_images)
    print(f"  {len(real_imgs)} real  /  {len(gen_imgs)} generated\n")

    backbone_results: dict = {}

    for model_id, backend in backbones:
        print(f"\n{'-'*55}")
        print(f"  {model_id}  [{backend}]")
        print(f"{'-'*55}")
        try:
            ext = _build_extractor(model_id, backend, device)
            res = _evaluate_backbone(ext, real_imgs, gen_imgs, cka_tau=cka_tau)
            backbone_results[model_id] = res
            print(f"  M3-poly={res['m3_poly']:.6f}  M3-rbf={res['m3_rbf']:.6f}"
                  f"  disc-poly={res['disc_poly']:.3f}  disc-rbf={res['disc_rbf']:.3f}"
                  f"  OOD-AUC={res['weighted_auc']:.4f}")
            # unload to free VRAM
            del ext
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            backbone_results[model_id] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Summary table ----------------------------------------------------
    print(f"\n{'='*90}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*90}")
    hdr = f"{'Backbone':<44} {'Layers':>6} {'M3-poly':>10} {'M3-rbf':>10} {'Disc-poly':>10} {'Disc-rbf':>9} {'OOD-AUC':>8}"
    print(hdr); print("-"*len(hdr))
    rows = []
    for bid, res in backbone_results.items():
        if "error" in res:
            print(f"  {bid:<42}  FAILED: {res['error'][:40]}")
            continue
        row = {"backbone": bid, **{k: res[k] for k in
               ("active_layers","m3_poly","m3_rbf","disc_poly","disc_rbf","weighted_auc")}}
        rows.append(row)
        print(f"  {bid:<42}  {str(res['active_layers']):>6}  "
              f"{res['m3_poly']:>10.6f}  {res['m3_rbf']:>10.6f}  "
              f"{res['disc_poly']:>10.4f}  {res['disc_rbf']:>9.4f}  {res['weighted_auc']:>8.4f}")

    plots = _plot(backbone_results, output_dir)

    out = {
        "config": {"real_dir": real_dir, "gen_dir": gen_dir,
                   "n_images": n_images, "cka_tau": cka_tau,
                   "backbones": [{"id": b, "backend": t} for b, t in backbones],
                   "kernels": ["polynomial_deg3", "gaussian_rbf_5band"]},
        "per_backbone": backbone_results,
        "comparison":   rows,
        "plots":        plots,
    }
    report = os.path.join(output_dir, "backbone_comparison_report.json")
    with open(report, "w") as f:
        json.dump(out, f, indent=4, default=str)
    print(f"\nReport: {report}")
    return out


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",   required=True)
    p.add_argument("--gen_dir",    required=True)
    p.add_argument("--output_dir", default="./backbone_comparison_output")
    p.add_argument("--n_images",   type=int,   default=500)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--cka_tau",    type=float, default=0.80)
    args = p.parse_args()
    run_backbone_comparison(**vars(args))
