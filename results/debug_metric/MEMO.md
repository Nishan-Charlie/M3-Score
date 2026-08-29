# Workstream A — Metric debug: results & decision

Date: 2026-07-25 | Script: `experiments/debug_metric_ablation.py`
Data: BraTS real vs `generated_500_standard` (unfiltered), + OOD (wdm3d brain, retinal).
Two runs: (N=300, seed=42) and (N=250, seed=7). Backbone: Snarcy/RadioDino-s16 (timm ViT-S/16).

## Headline decisions

1. **Drop the entropy/"semanticity" factor — it is inert for the default backbone.**
   Measured entropy = **5.300 on every layer** = the `_DUMMY_ENTROPY` fallback. RadioDino-s16
   loads via the `timm_vit` backend, which returns no attention entropy, so
   `semanticity = exp(-5.3/T)` is an identical constant across all 12 layers. The
   "entropy-aware / semanticity" axis contributes nothing for the shipping default. The
   paper must not claim entropy-awareness for RadioDino-s16.

2. **Drop the multi-layer weighting — it does not beat a single layer.**
   Shared-null permutation Z (higher = more decisive) and bootstrap CV (lower = more stable):

   | Config | Z (N=300,s42) | CV% | Z (N=250,s7) | CV% |
   |---|---:|---:|---:|---:|
   | single L12 (a-priori) | 302.2 | 3.30 | 288.4 | 3.34 |
   | best single (L7/L9) | **318.9** | **2.92** | **305.6** | **2.93** |
   | uniform over 12 | 287.6 | 3.42 | 226.1 | 3.77 |
   | sem×stab×uniq (shipping scheme) | 299.2 | 3.20 | 233.3 | 3.63 |

   The weighted scheme is **worse than single-layer L12 on both decisiveness and stability**,
   in both runs, and markedly worse at N=250. Mechanism: the *uniqueness* factor penalizes the
   most-discriminative middle layers (L5–L9 are mutually similar → low uniqueness), misallocating
   weight toward weaker early layers.

3. **Middle-late layers (L7–L9) discriminate best; L12 is a defensible a-priori runner-up.**
   Per-layer Z rises from L1 (≈106) to a plateau at L7–L9 (≈314–319), then eases to L12 (≈302).
   Recommendation: keep **L12 as the a-priori "deepest-semantic" fidelity layer** (preserves the
   paper's "never tuned on test" integrity; only ~5% below the empirical best) **and report
   honestly** that L7–L9 are empirically stronger. Do not silently switch to the argmax layer —
   that would be the exact test-set tuning the paper claims to avoid.

4. **Demote the OOD-AUC claim.** On every OOD set available locally, **both M3 and InceptionV3
   reach AUC ≈ 1.000** (retinal and wdm3d), with identical Spearman (ρ≈0.866). OOD separation is
   trivially easy here (consistent with Coverage=0) and does **not** differentiate M3 from FID.
   The paper's 0.974-vs-0.778 gap is specific to the BraTS-vs-LIDC-CT comparison (data not present
   locally); the AUC-vs-Spearman contradiction does **not** reproduce on these sets.

## What this means for the code (low-risk — the default is already right)

`m3_score_v2.py` already defaults to `single_layer=12`, so the **shipping default already computes
a single-layer L12 MMD²** — the weighting/multi-axis path only runs when `single_layer=None`. So no
risky metric surgery is required. Recommended cleanups (deferred until approved):
- Document that `semanticity` is a constant for non-attention (timm/resnet) backbones; either gate
  the entropy factor to attention backbones or drop it entirely.
- Mark the multi-axis / weighted path as **exploratory** in docstrings; keep it for backward compat.

## What this means for the paper (Workstream C)

- Retitle away from "Multi-Axis / A-Priori Layers / entropy-aware." The defensible object is a
  **single-layer (L12) unbiased multi-bandwidth RBF MMD² on a radiology backbone, reported with a
  permutation p-value and bootstrap CI.**
- Move OOD to a small backbone-validation note; stop leading with it.
- Keep statistical rigour + (later) the conditional/diagnostic and LGG specificity as the contribution.

Artifacts: `results/debug_metric/ablation_report.json`, `results/debug_metric_seed7/ablation_report.json`.
