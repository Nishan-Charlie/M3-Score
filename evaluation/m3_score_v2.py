"""
M3-Score V3  --  Multi-scale Medical Manifold Metric
=====================================================
Weighting scheme:

    w_l  ∝  semanticity_l  *  stability_l  *  uniqueness_l

where:

    semanticity_l  = exp(-entropy_l / T)
                     high weight → low-entropy CLS attention (focused layers)

    stability_l    = min(mean_sub_l / (std_sub_l + eps),  SNR_CAP)
                     SNR formula from the methodology, capped at 1e4 to
                     prevent zero-variance layers from dominating completely.
                     (1/(std+eps) is NOT used -- it is unbounded and collapses
                     all weight onto one layer when std≈0.)

    uniqueness_l   = max(1 - mean_CKA_to_others_l,  1e-4)
                     downweights layers whose representation is already
                     captured by another retained layer.

Backward-compatible output keys
---------------------------------
The forward() dict contains both:
    "m3_score"           -- new canonical key
    "m3_v2_final_score"  -- alias for backward compat with all experiment files

Public API
----------
    M3V2Metric = M3EntropyMetric        (alias used by all experiment imports)
    .prune_layers_via_cka(ref_imgs)
    .forward(real_imgs, gen_imgs)       -- returns result dict
    ._extract_raw_features(imgs, ...)   -- used by interpretability / OOD experiments
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Optional, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNR_CAP = 1e4      # maximum stability value (prevents zero-std dominance)
_EPS_L2  = 1e-8     # L2 normalisation epsilon
_EPS_SNR = 1e-6     # SNR denominator epsilon
_EPS_ATT = 1e-8     # attention normalisation epsilon

# ImageNet normalization (used for all backends by default)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# RAD-DINO uses its own normalization stats (grayscale-centric)
_RADDINO_MEAN = torch.tensor([0.5307, 0.5307, 0.5307])
_RADDINO_STD  = torch.tensor([0.2583, 0.2583, 0.2583])

# Max-entropy dummy value for non-attention backends
# ≈ log(200); exp(-H_max / T) ≈ 0 → semanticity near-zero → weights
# driven purely by stability × uniqueness (honest for non-DINO models)
_DUMMY_ENTROPY = 5.3

# ---------------------------------------------------------------------------
# A-priori three-axis layer assignments
# ---------------------------------------------------------------------------
# ALL THREE VALUES ARE FIXED BEFORE SEEING ANY TEST-SET RESULTS.
# They are chosen by structural position alone. Changing them retroactively
# to maximise a reported score is the exact violation this rule prevents.
#
# FIDELITY_LAYER = 12
#   The final transformer block collapses all hierarchical representations
#   into the highest-level semantic feature. Using the last layer for
#   distribution distance is the standard choice, analogous to FID using
#   the InceptionV3 penultimate activation. For a 12-block ViT-S/16, L12
#   is the deepest representation available.
#
# MEMORIZATION_LAYER = 9  (75 % depth)
#   At 75 % depth features encode strong mid-level structure (edges,
#   textures, local anatomy) but have not yet collapsed to the final
#   semantic pooling. Nearest-neighbour distances here are sensitive to
#   individual image variation, so near-duplicate memorisation that L12
#   would obscure is detectable at L9.
#
# COVERAGE_LAYER = 4  (33 % depth)
#   Early-to-mid blocks encode spatial structure and low-to-mid frequency
#   content. Manifold spread measured at this depth reflects geometric /
#   structural diversity rather than semantic category counts, making it
#   appropriate for coverage and mode-drop detection.
#
FIDELITY_LAYER     = 12   # a-priori: deepest semantic layer
MEMORIZATION_LAYER = 9    # a-priori: 75 % depth, instance-level features
COVERAGE_LAYER     = 4    # a-priori: 33 % depth, structural / geometric features


# ---------------------------------------------------------------------------
# Main metric class
# ---------------------------------------------------------------------------

class M3EntropyMetric(nn.Module):
    """
    M3-Score V3: entropy-aware, stability-aware, uniqueness-aware weighting.

    Args:
        device:               Torch device string.
        backbone_id:          HuggingFace model ID for the encoder.
        cka_threshold:        Redundancy threshold tau for CKA pruning (default 0.95).
        num_sub_batches:      K sub-batches for stability estimation (default 5).
        entropy_temperature:  Temperature T in exp(-H/T) (default 1.0).
        kernel:               MMD kernel — "polynomial" (default) or "rbf".
                              "rbf" uses multi-bandwidth Gaussian RBF with
                              median-heuristic base bandwidth.
    """

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend(backbone_id: str) -> str:
        """Classify backbone_id into a loading strategy."""
        ml = backbone_id.lower()
        if "biomedclip" in ml or "pubmedbert_256" in ml:
            return "open_clip_timm"
        if "pubmed-clip" in ml or "pubmed_clip" in ml:
            return "clip_transformers"
        if "radiodino-s16" in ml or "radiodino_s16" in ml:
            return "timm_vit"
        if "resnet" in ml or "rad-imagenet" in ml:
            return "resnet"
        # Default: transformers AutoModel (RAD-DINO, DINOv2, …)
        return "transformers_vit"

    def __init__(
        self,
        device:               str            = "cuda",
        backbone_id:          str            = "Snarcy/RadioDino-s16",
        cka_threshold:        float          = 0.8,
        num_sub_batches:      int            = 5,
        entropy_temperature:  float          = 1.0,
        kernel:               str            = "rbf",
        backend:              str            = "auto",
        seed:                 int            = 42,
        single_layer:         Optional[int]  = 12,
    ):
        # NOTE (validated 2026-07-26, experiments/debug_metric_ablation.py):
        #   * `single_layer=12` is the shipping default. Single-layer L12 unbiased
        #     multi-bandwidth RBF MMD^2 matches or beats the legacy
        #     entropy*stability*uniqueness weighting on both decisiveness (permutation Z)
        #     and stability (bootstrap CV). The weighting path (single_layer=None +
        #     prune_layers_via_cka) is retained only for post-hoc ablation.
        #   * The entropy / "semanticity" axis is INERT for the default timm_vit backbone
        #     (RadioDino-s16): it exposes no attention entropy, so every layer receives the
        #     _DUMMY_ENTROPY constant and semanticity is uniform. Do not describe the default
        #     configuration as "entropy-aware" — that only applies to transformers_vit
        #     backbones (rad-dino, dinov2) run in the multi-layer path.
        super().__init__()

        self.device              = device
        self.backbone_id         = backbone_id
        self.cka_threshold       = cka_threshold
        self.num_sub_batches     = num_sub_batches
        self.entropy_temperature = entropy_temperature
        self.kernel              = kernel        # "polynomial" | "rbf"
        self.seed                = seed          # fixes sub-batch stability sampling → deterministic score
        self.single_layer        = single_layer  # None → CKA multi-scale; int → use that layer only
        self._backend = self._detect_backend(backbone_id) if backend == "auto" else backend

        # Normalization stats: RAD-DINO uses its own; everything else uses ImageNet
        if "rad-dino" in backbone_id.lower():
            self._norm_mean = _RADDINO_MEAN
            self._norm_std  = _RADDINO_STD
        else:
            self._norm_mean = _IMAGENET_MEAN
            self._norm_std  = _IMAGENET_STD

        print(f"Loading backbone: {backbone_id} (backend={self._backend})")

        if self._backend == "transformers_vit":
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(
                backbone_id,
                output_hidden_states=True,
                output_attentions=True,
            ).to(device)
            self.backbone.eval()
            self.num_layers = len(self.backbone.encoder.layer)

        elif self._backend == "clip_transformers":
            from transformers import CLIPModel
            _full = CLIPModel.from_pretrained(backbone_id)
            self.backbone = _full.vision_model.to(device)
            self.backbone.eval()
            self.num_layers = self.backbone.config.num_hidden_layers

        elif self._backend == "timm_vit":
            import timm
            self.backbone = timm.create_model(
                f"hf_hub:{backbone_id}", pretrained=True, img_size=224
            ).to(device)
            self.backbone.eval()
            self.num_layers = len(self.backbone.blocks)

        elif self._backend == "open_clip_timm":
            import open_clip
            _model, _, _ = open_clip.create_model_and_transforms(
                f"hf-hub:{backbone_id}"
            )
            _v = _model.visual
            if hasattr(_v, "trunk") and hasattr(_v.trunk, "blocks"):
                self._oc_trunk = _v.trunk.to(device)
                self.num_layers = len(self._oc_trunk.blocks)
            else:
                self._oc_trunk = _v.to(device)
                self.num_layers = len(_v.transformer.resblocks)
            self._oc_trunk.eval()
            self.backbone = self._oc_trunk  # alias

        elif self._backend == "resnet":
            import timm
            try:
                self.backbone = timm.create_model(backbone_id, pretrained=True).to(device)
            except Exception:
                from torchvision import models as _tv
                self.backbone = _tv.resnet50(weights=_tv.ResNet50_Weights.DEFAULT).to(device)
            self.backbone.eval()
            self.num_layers = 4  # 4 ResNet stages

        else:
            raise ValueError(f"Unknown backend: {self._backend}")

        self.active_layers: list[int] = list(range(1, self.num_layers + 1))

        # Single-layer mode: pin active_layers now; prune_layers_via_cka becomes a no-op
        if single_layer is not None:
            assert 1 <= single_layer <= self.num_layers, (
                f"single_layer={single_layer} out of range [1, {self.num_layers}]"
            )
            self.active_layers = [single_layer]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _to_pixel_values(self, img_tensor) -> torch.Tensor:
        """(N,3,H,W) float [0,1] → model-normalised tensor on self.device."""
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.stack(list(img_tensor))
        x = img_tensor.float()
        mean = self._norm_mean.to(x.device).view(1, 3, 1, 1)
        std  = self._norm_std.to(x.device).view(1, 3, 1, 1)
        return ((x - mean) / std).to(self.device)

    def _preprocess(self, img_tensor):
        """Backward-compatible wrapper used by legacy callers."""
        return {"pixel_values": self._to_pixel_values(img_tensor)}

    # ------------------------------------------------------------------
    # L2 normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise(x: torch.Tensor) -> torch.Tensor:
        """Project each row onto the unit sphere."""
        return x / x.norm(dim=1, keepdim=True).clamp(min=_EPS_L2)

    # ------------------------------------------------------------------
    # Linear CKA
    # ------------------------------------------------------------------

    def _linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Linear CKA between two (N, D) embedding matrices.
        Uses gram-matrix form (N x N) for numerical stability.
        """
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        Kx = X @ X.t()
        Ky = Y @ Y.t()
        hsic   = (Kx * Ky).sum()
        norm_x = torch.norm(Kx, p="fro").pow(2)
        norm_y = torch.norm(Ky, p="fro").pow(2)
        return (hsic / ((norm_x * norm_y).sqrt() + _EPS_L2)).item()

    # ------------------------------------------------------------------
    # Attention entropy
    # ------------------------------------------------------------------

    @staticmethod
    def _attention_entropy(attn_weights: torch.Tensor) -> float:
        """
        Mean Shannon entropy of the CLS→patch attention distribution.

        Args:
            attn_weights: (B, num_heads, N_seq, N_seq)

        Returns:
            Scalar entropy averaged over the batch.
        """
        avg_attn  = attn_weights.mean(dim=1)          # (B, N_seq, N_seq)
        cls_attn  = avg_attn[:, 0, 1:]                # (B, N_seq-1)
        cls_attn  = cls_attn / (cls_attn.sum(dim=1, keepdim=True) + _EPS_ATT)
        entropy   = -(cls_attn * torch.log(cls_attn + _EPS_ATT)).sum(dim=1)
        return entropy.mean().item()

    # ------------------------------------------------------------------
    # Feature extraction  (used by forward AND by experiment helpers)
    # ------------------------------------------------------------------

    def _extract_cls_features(
        self,
        img_tensor,
        layers_to_keep: Optional[Set[int]] = None,
    ) -> list[torch.Tensor]:
        """
        Extract CLS token features for CKA layer selection.

        Dispatches on self._backend so all six backbone families are supported.
        ResNet uses global-average-pooled stage features instead of CLS tokens.

        Returns list of length num_layers; (N, D) tensor per active layer.
        """
        if layers_to_keep is None:
            layers_to_keep = set(range(1, self.num_layers + 1))

        n = img_tensor.shape[0] if isinstance(img_tensor, torch.Tensor) else len(img_tensor)
        layer_feats = [[] for _ in range(self.num_layers)]

        with torch.no_grad():

            # ── transformers ViT (RAD-DINO, DINOv2, …) ───────────────────────
            if self._backend in ("transformers_vit", "clip_transformers"):
                BS = 16
                for start in range(0, n, BS):
                    pv  = self._to_pixel_values(img_tensor[start:start + BS])
                    out = self.backbone(
                        pixel_values=pv,
                        output_hidden_states=True,
                        output_attentions=False,
                    )
                    hs = out.hidden_states[1:]
                    for i in range(self.num_layers):
                        if (i + 1) not in layers_to_keep:
                            continue
                        layer_feats[i].append(hs[i][:, 0, :].cpu())
                    del hs, out, pv
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ── timm ViT (RadioDINO-s16, …) ───────────────────────────────────
            elif self._backend == "timm_vit":
                model = self.backbone
                BS = 16
                for start in range(0, n, BS):
                    x = self._to_pixel_values(img_tensor[start:start + BS])
                    x = model.patch_embed(x)
                    if hasattr(model, "_pos_embed"):
                        x = model._pos_embed(x)
                    else:
                        cls = model.cls_token.expand(x.shape[0], -1, -1)
                        x   = torch.cat([cls, x], dim=1) + model.pos_embed
                    if hasattr(model, "norm_pre"):
                        x = model.norm_pre(x)
                    for i, blk in enumerate(model.blocks):
                        x = blk(x)
                        if (i + 1) not in layers_to_keep:
                            continue
                        layer_feats[i].append(x[:, 0, :].cpu())
                    del x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ── open_clip TimmModel (BiomedCLIP, …) ───────────────────────────
            elif self._backend == "open_clip_timm":
                trunk = self._oc_trunk
                BS = 16
                for start in range(0, n, BS):
                    x = self._to_pixel_values(img_tensor[start:start + BS])
                    x = trunk.patch_embed(x)
                    if hasattr(trunk, "_pos_embed"):
                        x = trunk._pos_embed(x)
                    else:
                        cls = trunk.cls_token.expand(x.shape[0], -1, -1)
                        x   = torch.cat([cls, x], dim=1) + trunk.pos_embed
                    if hasattr(trunk, "norm_pre"):
                        x = trunk.norm_pre(x)
                    for i, blk in enumerate(trunk.blocks):
                        x = blk(x)
                        if (i + 1) not in layers_to_keep:
                            continue
                        layer_feats[i].append(x[:, 0, :].cpu())
                    del x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ── ResNet (stage-level global-avg-pool) ──────────────────────────
            elif self._backend == "resnet":
                model = self.backbone
                stages = [model.layer1, model.layer2, model.layer3, model.layer4]
                BS = 16
                for start in range(0, n, BS):
                    x = self._to_pixel_values(img_tensor[start:start + BS])
                    x = model.conv1(x)
                    x = model.bn1(x)
                    x = model.act1(x)
                    x = model.maxpool(x)
                    for i, stage in enumerate(stages):
                        x = stage(x)
                        if (i + 1) not in layers_to_keep:
                            continue
                        layer_feats[i].append(x.mean(dim=[2, 3]).cpu())
                    del x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return [
            torch.cat(layer_feats[i], dim=0) if layer_feats[i] else torch.empty(0)
            for i in range(self.num_layers)
        ]

    def _extract_features_and_entropy(
        self,
        img_tensor,
        layers_to_keep: Optional[Set[int]] = None,
    ) -> tuple[list[torch.Tensor], list[float]]:
        """
        Extract pooled features and per-layer entropy for weighting.

        transformers_vit: attention-weighted pooling + CLS attention entropy
                          (full semanticity weighting as per Section 2.2).
        All other backends: CLS / GAP token + _DUMMY_ENTROPY (max-entropy
                            fallback so semanticity ≈ 0, weights reduce to
                            stability × uniqueness — honest for non-DINO models).

        Returns:
            final_feats:   list of length num_layers; (N, D) feature tensors.
            final_entropy: list of length num_layers; scalar entropy per layer.
        """
        if layers_to_keep is None:
            layers_to_keep = set(range(1, self.num_layers + 1))

        # ── Non-DINO backends: CLS/GAP features + uniform max entropy ─────────
        if self._backend != "transformers_vit":
            feats = self._extract_cls_features(img_tensor, layers_to_keep)
            entropy = [
                _DUMMY_ENTROPY if (i + 1) in layers_to_keep else 0.0
                for i in range(self.num_layers)
            ]
            return feats, entropy

        # ── transformers_vit: full attention-weighted pooling + entropy ────────
        n = (
            img_tensor.shape[0]
            if isinstance(img_tensor, torch.Tensor)
            else len(img_tensor)
        )

        layer_feats   = [[] for _ in range(self.num_layers)]
        layer_entropy = [[] for _ in range(self.num_layers)]

        with torch.no_grad():
            for start in range(0, n, 1):   # batch_size=1 to avoid OOM
                pv      = self._to_pixel_values(img_tensor[start:start + 1])
                outputs = self.backbone(
                    pixel_values=pv,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                hidden_states = outputs.hidden_states[1:]
                attentions    = outputs.attentions

                for i in range(self.num_layers):
                    if (i + 1) not in layers_to_keep:
                        continue

                    h            = hidden_states[i]        # (1, N_seq, D)
                    patch_tokens = h[:, 1:, :]             # (1, N_seq-1, D)

                    avg_attn = attentions[i].mean(dim=1)   # (1, N_seq, N_seq)
                    cls_attn = avg_attn[:, 0, 1:]          # (1, N_seq-1)
                    cls_attn = cls_attn / (
                        cls_attn.sum(dim=1, keepdim=True) + _EPS_ATT
                    )
                    feat = (patch_tokens * cls_attn.unsqueeze(-1)).sum(dim=1)
                    layer_entropy[i].append(
                        self._attention_entropy(attentions[i])
                    )
                    layer_feats[i].append(feat.cpu())

                del attentions, hidden_states, outputs, pv
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        final_feats   = []
        final_entropy = []
        for i in range(self.num_layers):
            if layer_feats[i]:
                final_feats.append(torch.cat(layer_feats[i], dim=0))
                final_entropy.append(float(np.mean(layer_entropy[i])))
            else:
                final_feats.append(torch.empty(0))
                final_entropy.append(0.0)

        return final_feats, final_entropy

    def _extract_raw_features(
        self,
        img_tensor,
        use_attention: bool = True,
        layers_to_keep: Optional[Set[int]] = None,
    ) -> list[torch.Tensor]:
        """
        Public alias for experiment compatibility.

        Used by:
          - interpretability.py
          - ood_detection.py  (_compute_m3_distances)
          - metrics_interpretability.py

        Returns:
            list of length num_layers; each element is (N, D) float32 tensor
            (empty tensor for skipped layers).
        """
        feats, _ = self._extract_features_and_entropy(
            img_tensor,
            layers_to_keep=layers_to_keep,
        )
        return feats

    # ------------------------------------------------------------------
    # Polynomial MMD² (unbiased)
    # ------------------------------------------------------------------

    def _compute_mmd2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Unbiased MMD² with degree-3 polynomial kernel on unit-sphere features.

            k(u, v) = (u^T v + 1)^3,  gamma = 1

        Features are L2-normalised inside this function so the caller
        may pass raw or pre-normalised tensors.
        """
        x = self._l2_normalise(x)
        y = self._l2_normalise(y)

        Kxx = (x @ x.t() + 1.0) ** 3
        Kyy = (y @ y.t() + 1.0) ** 3
        Kxy = (x @ y.t() + 1.0) ** 3

        m = Kxx.shape[0]
        n = Kyy.shape[0]

        T_rr = (Kxx.sum() - torch.trace(Kxx)) / (m * (m - 1))
        T_gg = (Kyy.sum() - torch.trace(Kyy)) / (n * (n - 1))
        T_rg = Kxy.mean()

        return (T_rr + T_gg - 2.0 * T_rg).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Multi-bandwidth Gaussian RBF MMD² (unbiased)
    # ------------------------------------------------------------------

    def _compute_mmd2_rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Unbiased MMD² with multi-bandwidth Gaussian RBF kernel.

            k(u, v) = mean_σ [ exp(-||u-v||² / (2σ²)) ]

        Bandwidth schedule: {σ_med/4, σ_med/2, σ_med, 2·σ_med, 4·σ_med}
        where σ_med² is the median pairwise squared distance on the joint
        (real ∪ gen) sample — the standard median heuristic.

        Features are L2-normalised before distance computation.
        """
        x = self._l2_normalise(x)
        y = self._l2_normalise(y)

        def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (
                (a * a).sum(1, keepdim=True)
                + (b * b).sum(1)
                - 2.0 * (a @ b.t())
            ).clamp(min=0.0)

        Dxx = _sq_dist(x, x)
        Dyy = _sq_dist(y, y)
        Dxy = _sq_dist(x, y)

        # Median heuristic on joint sample
        joint = torch.cat([x, y], dim=0)
        D_joint = _sq_dist(joint, joint)
        mask = torch.triu(torch.ones(D_joint.shape[0], D_joint.shape[0],
                                      dtype=torch.bool, device=D_joint.device),
                          diagonal=1)
        sigma2_med = float(D_joint[mask].median().clamp(min=1e-10))

        # Five-bandwidth schedule
        bandwidths = [sigma2_med * s for s in (0.25, 0.5, 1.0, 2.0, 4.0)]

        m, n = x.shape[0], y.shape[0]
        mmd2_sum = torch.zeros(1, device=x.device)

        for sigma2 in bandwidths:
            gamma = 1.0 / (2.0 * sigma2)
            Kxx = torch.exp(-gamma * Dxx)
            Kyy = torch.exp(-gamma * Dyy)
            Kxy = torch.exp(-gamma * Dxy)

            T_rr = (Kxx.sum() - Kxx.trace()) / (m * (m - 1))
            T_gg = (Kyy.sum() - Kyy.trace()) / (n * (n - 1))
            T_rg = Kxy.mean()
            mmd2_sum = mmd2_sum + (T_rr + T_gg - 2.0 * T_rg)

        return (mmd2_sum / len(bandwidths)).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Kernel dispatch
    # ------------------------------------------------------------------

    def _mmd2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Route to polynomial or RBF kernel based on self.kernel."""
        if self.kernel == "rbf":
            return self._compute_mmd2_rbf(x, y)
        return self._compute_mmd2(x, y)

    # ------------------------------------------------------------------
    # CKA pruning
    # ------------------------------------------------------------------

    def prune_layers_via_cka(
        self,
        reference_images,
        cache_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Run greedy backward CKA pruning on `reference_images`.
        Sets self.active_layers in place.

        cache_path: if given and the file exists, active_layers are loaded
        from the cache (skipping recomputation). If the file does not exist,
        CKA is run and the result is saved to cache_path. Point all
        experiments at the same file (e.g. <project_root>/canonical_layers.json)
        to guarantee identical layer selection across every run.

        seed: optional integer seed that is applied *before* any feature
        extraction so that stochastic operations (dropout during eval, etc.)
        are reproducible. Falls back to self.seed when None.

        Stability rule: pass at least 200 representative reference images so
        that the greedy CKA selection is not sensitive to the specific draw.
        The cache stores the ref-image count so mismatches can be detected.

        Algorithm (from Section 2.3 of the methodology):
          - Layer L is always retained.
          - Iterate from L-1 down to 1; add layer i if its CKA similarity
            to every already-retained layer is below cka_threshold.
        """
        # In single-layer mode CKA pruning is irrelevant — active_layers is already pinned.
        if self.single_layer is not None:
            return

        import json, os as _os

        effective_seed = seed if seed is not None else self.seed

        if cache_path is not None and _os.path.isfile(cache_path):
            with open(cache_path) as _f:
                cached = json.load(_f)
            cached_tau = cached.get("cka_threshold")
            cached_n   = cached.get("ref_n", 0)
            n_ref = (
                reference_images.shape[0]
                if isinstance(reference_images, torch.Tensor)
                else len(reference_images)
            )
            if cached_tau == self.cka_threshold and cached_n == n_ref:
                self.active_layers = cached["active_layers"]
                print(f"CKA layers loaded from cache (tau={self.cka_threshold}, "
                      f"n_ref={n_ref}): {self.active_layers}")
                return
            print(f"Cache miss (tau {cached_tau}→{self.cka_threshold}, "
                  f"n_ref {cached_n}→{n_ref}) — recomputing.")

        # Seed before feature extraction for reproducibility
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)

        print("Running CKA pruning...")
        # Use CLS token features for CKA (not attention-weighted pooling).
        # CLS features have stable inter-layer similarity, ensuring tau=0.95
        # correctly prunes redundant layers. Attention-weighted features vary
        # too much between adjacent layers, causing all layers to be retained.
        feats = self._extract_cls_features(
            reference_images,
            layers_to_keep=set(range(1, self.num_layers + 1)),
        )

        selected = [self.num_layers]
        for i in range(self.num_layers - 1, 0, -1):
            redundant = False
            for s in selected:
                if self._linear_cka(feats[i - 1], feats[s - 1]) > self.cka_threshold:
                    redundant = True
                    break
            if not redundant:
                selected.append(i)

        self.active_layers = sorted(selected)
        print(f"Selected layers: {self.active_layers}")

        if cache_path is not None:
            n_ref = (
                reference_images.shape[0]
                if isinstance(reference_images, torch.Tensor)
                else len(reference_images)
            )
            _os.makedirs(_os.path.dirname(_os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w") as _f:
                json.dump({
                    "active_layers":  self.active_layers,
                    "cka_threshold":  self.cka_threshold,
                    "num_layers":     self.num_layers,
                    "ref_n":          n_ref,
                    "seed":           effective_seed,
                }, _f, indent=2)
            print(f"CKA layer selection cached: {cache_path}")

        # Release CUDA allocator caches so the main forward pass starts
        # with maximum available VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Three-axis helper methods (a-priori layer assignments)
    # ------------------------------------------------------------------

    def _knn_distances(
        self,
        query_feats: torch.Tensor,
        ref_feats: torch.Tensor,
        k: int = 1,
    ) -> torch.Tensor:
        """Return the k-th nearest-neighbour distance from each query to ref.

        Uses L2 distance in the feature space. Returns a (Nq,) tensor.
        """
        q = query_feats.to(self.device).float()
        r = ref_feats.to(self.device).float()
        # pairwise squared distances: (Nq, Nr)
        dists_sq = torch.cdist(q, r, p=2).pow(2)
        topk = dists_sq.topk(k, largest=False, dim=1).values
        return topk[:, -1].sqrt()  # k-th NN distance

    def _knn_distances_within(
        self,
        feats: torch.Tensor,
        k: int = 1,
    ) -> torch.Tensor:
        """k-th NN distance within a set, excluding self (diagonal = inf)."""
        f = feats.to(self.device).float()
        dists_sq = torch.cdist(f, f, p=2).pow(2)
        dists_sq.fill_diagonal_(float("inf"))
        topk = dists_sq.topk(k, largest=False, dim=1).values
        return topk[:, -1].sqrt()

    def _manifold_fraction(
        self,
        query_feats: torch.Tensor,
        ref_feats: torch.Tensor,
        ref_radii: torch.Tensor,
    ) -> float:
        """Fraction of query points that fall inside any ref k-NN ball.

        A query point is inside if its distance to the nearest ref point is
        less than or equal to that ref point's k-NN radius.
        """
        q = query_feats.to(self.device).float()
        r = ref_feats.to(self.device).float()
        rr = ref_radii.to(self.device)
        dists = torch.cdist(q, r, p=2)                # (Nq, Nr)
        min_dist, nn_idx = dists.min(dim=1)            # (Nq,)
        radii_at_nn = rr[nn_idx]                       # (Nq,)
        inside = (min_dist <= radii_at_nn).float().mean().item()
        return float(inside)

    def _compute_precision_recall(
        self,
        real_feats: torch.Tensor,
        gen_feats: torch.Tensor,
        k: int = 5,
    ) -> tuple:
        """kNN precision and recall (Kynkaanniemi et al. 2019).

        Precision: fraction of gen points inside the real manifold.
        Recall:    fraction of real points covered by the gen manifold.
        k=5 is the standard value from the original paper.
        Returns (precision, recall) as plain floats.
        """
        real_radii = self._knn_distances_within(real_feats, k=k)
        gen_radii  = self._knn_distances_within(gen_feats,  k=k)
        precision  = self._manifold_fraction(gen_feats,  real_feats, real_radii)
        recall     = self._manifold_fraction(real_feats, gen_feats,  gen_radii)
        return precision, recall

    def _compute_memorization(
        self,
        real_feats: torch.Tensor,
        gen_feats: torch.Tensor,
        k: int = 1,
    ) -> tuple:
        """Nearest-neighbour memorization detector at the mid-depth layer.

        Compares each generated image's distance to its nearest real image
        against the within-real 1-NN threshold.  If a gen point is closer
        to a real point than that real point's nearest real neighbour, it is
        a candidate near-duplicate.

        Returns:
            mean_nn_dist      -- mean gen-to-real 1-NN distance
            memorization_rate -- fraction of gen points flagged as near-duplicates
        """
        gen_to_real  = self._knn_distances(gen_feats, real_feats, k=k)
        within_real  = self._knn_distances_within(real_feats, k=k)
        threshold    = within_real.median().item()
        mean_nn_dist = gen_to_real.mean().item()
        mem_rate     = (gen_to_real < threshold).float().mean().item()
        return float(mean_nn_dist), float(mem_rate)

    def _permutation_test_fidelity(
        self,
        real_feats: torch.Tensor,
        gen_feats: torch.Tensor,
        n_permutations: int = 200,
        seed: int = 0,
    ) -> tuple:
        """One-sided permutation test for H0: MMD2(real, gen) = 0.

        Pools real and gen, repeatedly shuffles the labels, computes MMD2
        on each shuffle.  The p-value is the fraction of null MMD2 values
        that meet or exceed the observed MMD2.

        Returns (p_value, z_score, null_mean).
        """
        rf  = real_feats.to(self.device).float()
        gf  = gen_feats.to(self.device).float()
        nr  = rf.shape[0]
        ng  = gf.shape[0]
        obs = self._compute_mmd2_rbf(rf, gf).item()

        pooled = torch.cat([rf, gf], dim=0)
        rng    = torch.Generator().manual_seed(seed)
        null_vals = []
        for _ in range(n_permutations):
            perm = torch.randperm(nr + ng, generator=rng)
            r2   = pooled[perm[:nr]]
            g2   = pooled[perm[nr:]]
            null_vals.append(self._compute_mmd2_rbf(r2, g2).item())

        null_arr  = np.array(null_vals)
        p_value   = float((null_arr >= obs).mean())
        null_mean = float(null_arr.mean())
        null_std  = float(null_arr.std())
        z_score   = float((obs - null_mean) / max(null_std, 1e-10))
        return p_value, z_score, null_mean

    def _bootstrap_ci_fidelity(
        self,
        real_feats: torch.Tensor,
        gen_feats: torch.Tensor,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> tuple:
        """Bootstrap 95 % CI on the fidelity MMD2 estimate.

        Resamples real and gen independently with replacement.
        Returns (ci_low, ci_high).
        """
        rf  = real_feats.to(self.device).float()
        gf  = gen_feats.to(self.device).float()
        nr  = rf.shape[0]
        ng  = gf.shape[0]
        rng = torch.Generator().manual_seed(seed)
        boot_vals = []
        for _ in range(n_bootstrap):
            ri  = torch.randint(0, nr, (nr,), generator=rng)
            gi  = torch.randint(0, ng, (ng,), generator=rng)
            boot_vals.append(self._compute_mmd2_rbf(rf[ri], gf[gi]).item())

        arr     = np.array(boot_vals)
        ci_low  = float(np.percentile(arr, 100 * alpha / 2))
        ci_high = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return ci_low, ci_high

    def compute_axes(
        self,
        real_images,
        gen_images,
        n_permutations: int = 200,
        n_bootstrap: int = 200,
        knn_k: int = 5,
        seed: int = 0,
    ) -> dict:
        """Three-axis M3 evaluation using a-priori layer assignments.

        Extracts features at FIDELITY_LAYER (12), MEMORIZATION_LAYER (9),
        and COVERAGE_LAYER (4) in a single forward pass through RadioDINO-s16.

        Returns a dict with keys:
            fidelity/mmd2          -- MMD2 at L12
            fidelity/p_value       -- permutation test p-value (H0: MMD2=0)
            fidelity/z_score       -- effect-size Z relative to null
            fidelity/null_mean     -- mean null MMD2
            fidelity/ci_low        -- 95 % bootstrap CI lower bound
            fidelity/ci_high       -- 95 % bootstrap CI upper bound
            memorization/mean_nn_dist  -- mean gen-to-real L9 distance
            memorization/rate          -- fraction of gen flagged as near-dup
            coverage/precision         -- kNN precision at L4
            coverage/recall            -- kNN recall at L4
            -- backward-compat aliases --
            m3_score               -- fidelity/mmd2  (same as canonical forward())
            m3_v2_final_score      -- alias of m3_score
        """
        target_layers = {FIDELITY_LAYER, MEMORIZATION_LAYER, COVERAGE_LAYER}

        print("compute_axes: extracting features at layers "
              f"{sorted(target_layers)}...")

        real_cls, _ = self._extract_features_and_entropy(
            real_images, layers_to_keep=target_layers
        )
        gen_cls, _ = self._extract_features_and_entropy(
            gen_images, layers_to_keep=target_layers
        )

        rf12 = real_cls[FIDELITY_LAYER - 1].to(self.device).float()
        gf12 = gen_cls[FIDELITY_LAYER - 1].to(self.device).float()
        rf9  = real_cls[MEMORIZATION_LAYER - 1].to(self.device).float()
        gf9  = gen_cls[MEMORIZATION_LAYER - 1].to(self.device).float()
        rf4  = real_cls[COVERAGE_LAYER - 1].to(self.device).float()
        gf4  = gen_cls[COVERAGE_LAYER - 1].to(self.device).float()

        # Fidelity axis
        print("compute_axes: fidelity MMD2...")
        fid_mmd2 = self._compute_mmd2_rbf(rf12, gf12).item()
        print("compute_axes: permutation test...")
        p_val, z_sc, null_mean = self._permutation_test_fidelity(
            rf12, gf12, n_permutations=n_permutations, seed=seed
        )
        print("compute_axes: bootstrap CI...")
        ci_low, ci_high = self._bootstrap_ci_fidelity(
            rf12, gf12, n_bootstrap=n_bootstrap, seed=seed
        )

        # Memorization axis
        print("compute_axes: memorization...")
        mean_nn, mem_rate = self._compute_memorization(rf9, gf9, k=1)

        # Coverage axis
        print("compute_axes: precision-recall...")
        precision, recall = self._compute_precision_recall(rf4, gf4, k=knn_k)

        return {
            "fidelity/mmd2":             float(fid_mmd2),
            "fidelity/p_value":          float(p_val),
            "fidelity/z_score":          float(z_sc),
            "fidelity/null_mean":        float(null_mean),
            "fidelity/ci_low":           float(ci_low),
            "fidelity/ci_high":          float(ci_high),
            "memorization/mean_nn_dist": float(mean_nn),
            "memorization/rate":         float(mem_rate),
            "coverage/precision":        float(precision),
            "coverage/recall":           float(recall),
            # backward-compat aliases
            "m3_score":                  float(fid_mmd2),
            "m3_v2_final_score":         float(fid_mmd2),
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        real_images,
        gen_images,
    ) -> dict:
        """
        Compute M3-Score between real and generated image sets.

        Args:
            real_images: (N, 3, H, W) float32 tensor in [0, 1].
            gen_images:  (N, 3, H, W) float32 tensor in [0, 1].

        Returns dict with keys:
            m3_score            -- final weighted score (canonical)
            m3_v2_final_score   -- alias for backward compatibility
            active_layers
            layer_distances     -- {L4: float, ...}
            layer_weights       -- {L4: float, ...}  (normalised)
            semanticity         -- {L4: float, ...}
            stability           -- {L4: float, ...}  (capped SNR)
            uniqueness          -- {L4: float, ...}
        """
        active_set = set(self.active_layers)
        M = len(active_set)

        print(f"Extracting features from {M} layers...")

        real_feats, entropy_vals = self._extract_features_and_entropy(
            real_images, layers_to_keep=active_set
        )
        gen_feats, _ = self._extract_features_and_entropy(
            gen_images, layers_to_keep=active_set
        )

        # Reindex to active layers only
        real_feats   = [real_feats[l - 1]   for l in self.active_layers]
        gen_feats    = [gen_feats[l - 1]    for l in self.active_layers]
        entropy_vals = [entropy_vals[l - 1] for l in self.active_layers]

        num_samples = (
            real_images.shape[0]
            if isinstance(real_images, torch.Tensor)
            else len(real_images)
        )

        # Use the smaller of real/gen sizes for sub-batch indexing so we never
        # index gf out of bounds when len(gen) < len(real).
        num_gen = (
            gen_images.shape[0]
            if isinstance(gen_images, torch.Tensor)
            else len(gen_images)
        )
        num_sub_samples = min(num_samples, num_gen)
        # Fix: guard n_sub so sub_dists is never empty (num_sub_samples < 4)
        n_sub = max(1, min(self.num_sub_batches, num_sub_samples // 4))
        sub_batch_size = max(2, num_sub_samples // n_sub)

        # Determinism: route every sub-batch draw through a single seeded
        # CPU generator. Without this the stability/SNR estimate (and hence
        # the layer weights and final score) varies run-to-run on identical
        # inputs. One generator per forward() call → the whole score is a
        # deterministic function of (real, gen, self.seed).
        sub_gen = torch.Generator().manual_seed(self.seed)

        # ── Uniqueness: mean CKA to all other retained layers ──────────────
        redundancy_scores = []
        for i in range(M):
            cka_sum = 0.0
            n_pairs = 0
            for j in range(M):
                if i == j:
                    continue
                cka_sum += self._linear_cka(real_feats[i], real_feats[j])
                n_pairs += 1
            redundancy_scores.append(cka_sum / max(n_pairs, 1))

        # ── Per-layer scores and weight components ──────────────────────────
        layer_scores     = []
        semantic_arr     = []
        stability_arr    = []
        uniqueness_arr   = []

        for i in tqdm(range(M), desc="Computing metrics"):
            rf = real_feats[i].to(self.device)
            gf = gen_feats[i].to(self.device)

            # Full-set MMD²
            dist = self._mmd2(rf, gf).item()

            # Sub-batch MMD² for stability estimation
            sub_dists = []
            for _ in range(n_sub):
                idx_r = torch.randperm(rf.shape[0], generator=sub_gen)[:sub_batch_size].to(rf.device)
                idx_g = torch.randperm(gf.shape[0], generator=sub_gen)[:sub_batch_size].to(gf.device)
                sub_dists.append(
                    self._mmd2(rf[idx_r], gf[idx_g]).item()
                )

            mean_sub = float(np.mean(sub_dists))
            std_sub  = float(np.std(sub_dists))

            # ── Semanticity  = exp(-entropy / T) ─────────────────────────
            semanticity = float(np.exp(
                -entropy_vals[i] / max(self.entropy_temperature, 1e-8)
            ))

            # ── Stability  = capped SNR  (Fix: replaces unbounded 1/std) ─
            # 1/(std+eps) is unbounded: if std=0, stability=1e6 and one
            # layer monopolises the weight completely. Using the SNR formula
            # (mean/std) with the same cap used in the old weighting ensures
            # layers with zero variance receive high but bounded weight.
            stability = float(
                min(mean_sub / (std_sub + _EPS_SNR), _SNR_CAP)
            )

            # ── Uniqueness  = 1 - mean_CKA_to_others ─────────────────────
            uniqueness = float(max(1.0 - redundancy_scores[i], 1e-4))

            layer_scores.append(dist)
            semantic_arr.append(semanticity)
            stability_arr.append(stability)
            uniqueness_arr.append(uniqueness)

        # ── Final weights  w_l ∝ semanticity * stability * uniqueness ──────
        sem   = np.array(semantic_arr)
        stab  = np.array(stability_arr)
        uniq  = np.array(uniqueness_arr)

        raw_weights = sem * stab * uniq
        weight_sum  = raw_weights.sum() + 1e-8
        weights     = raw_weights / weight_sum

        final_score = float(np.sum(np.array(layer_scores) * weights))

        # ── Output dict ─────────────────────────────────────────────────────
        result = {
            # canonical key
            "m3_score":           final_score,
            # backward-compatible alias (used by all experiment files)
            "m3_v2_final_score":  final_score,
            "active_layers":      self.active_layers,
            "layer_distances": {
                f"L{l}": float(v)
                for l, v in zip(self.active_layers, layer_scores)
            },
            "layer_weights": {
                f"L{l}": float(v)
                for l, v in zip(self.active_layers, weights)
            },
            "semanticity": {
                f"L{l}": float(v)
                for l, v in zip(self.active_layers, sem)
            },
            "stability": {
                f"L{l}": float(v)
                for l, v in zip(self.active_layers, stab)
            },
            "uniqueness": {
                f"L{l}": float(v)
                for l, v in zip(self.active_layers, uniq)
            },
        }

        # Optional three-axis keys — populated when both L9 (memorization)
        # and L4 (coverage) are in the active set, so forward() callers
        # that already load those layers get the extra diagnostics for free.
        active_set_fwd = set(self.active_layers)
        all_feats_fwd  = {l: rf for l, rf in zip(self.active_layers, real_feats)}
        all_gen_fwd    = {l: gf for l, gf in zip(self.active_layers, gen_feats)}

        if MEMORIZATION_LAYER in active_set_fwd:
            rf9 = all_feats_fwd[MEMORIZATION_LAYER].to(self.device).float()
            gf9 = all_gen_fwd[MEMORIZATION_LAYER].to(self.device).float()
            mn, mr = self._compute_memorization(rf9, gf9)
            result["memorization/mean_nn_dist"] = mn
            result["memorization/rate"]         = mr

        if COVERAGE_LAYER in active_set_fwd:
            rf4 = all_feats_fwd[COVERAGE_LAYER].to(self.device).float()
            gf4 = all_gen_fwd[COVERAGE_LAYER].to(self.device).float()
            prec, rec = self._compute_precision_recall(rf4, gf4)
            result["coverage/precision"] = prec
            result["coverage/recall"]    = rec

        return result


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

M3V2Metric = M3EntropyMetric
"""Alias used by all experiment files:
    from evaluation.m3_score_v2 import M3V2Metric
"""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = M3EntropyMetric(device=device)

    dummy_real = torch.rand(50, 3, 224, 224)
    dummy_gen  = torch.rand(50, 3, 224, 224)

    metric.prune_layers_via_cka(dummy_real[:20])

    results = metric(dummy_real, dummy_gen)

    print("\n=== RESULTS ===")
    print(f"m3_score:          {results['m3_score']:.6f}")
    print(f"m3_v2_final_score: {results['m3_v2_final_score']:.6f}")
    print(f"Active layers:     {results['active_layers']}")
    print()
    print(f"{'Layer':<8} {'MMD²':>10} {'Weight':>10} "
          f"{'Semantic':>10} {'Stability':>12} {'Uniqueness':>12}")
    print("-" * 66)
    for l in results["active_layers"]:
        k = f"L{l}"
        print(
            f"{k:<8} "
            f"{results['layer_distances'][k]:>10.4f} "
            f"{results['layer_weights'][k]:>10.4f} "
            f"{results['semanticity'][k]:>10.4f} "
            f"{results['stability'][k]:>12.2f} "
            f"{results['uniqueness'][k]:>12.4f}"
        )

    # Verify backward compat
    assert results["m3_score"] == results["m3_v2_final_score"]
    assert "_extract_raw_features" in dir(metric)
    print("\nBackward compatibility: OK")

    # Verify determinism: identical inputs must yield an identical score.
    results_repeat = metric(dummy_real, dummy_gen)
    assert results["m3_score"] == results_repeat["m3_score"], (
        "Non-deterministic score: sub-batch sampling is not seeded."
    )
    assert results["layer_weights"] == results_repeat["layer_weights"]
    print("Determinism: OK")