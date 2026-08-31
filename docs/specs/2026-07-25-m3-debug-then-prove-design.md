# M3-Score — "Debug-then-Prove" design spec

Date: 2026-07-25
Status: approved (Directions 1 + 3 + Debug). Workstream A first.
Author: Nishankar S. (with Claude)

## Problem (independent analysis)

1. **Headline novelty is not novel.** The fidelity axis is an unbiased multi-bandwidth RBF
   MMD²; the permutation p-value is the *original* MMD kernel two-sample test (Gretton 2012).
   KID is already an unbiased MMD on Inception features, so a permutation p-value/CI is
   available to any FID/KID user. "Native significance testing" cannot be the methodological
   contribution.
2. **The metric's own ablations undercut its title.** EXPERIMENT_FINDINGS shows single-layer
   L12 ≥ multi-layer; the coverage axis is degenerate (Coverage=0) and unstable; the
   entropy×stability×uniqueness weighting is ad hoc and unvalidated.
3. **The core medical claim is missing/reversed.** Noise/blur/step monotonicity is matched by
   FID. The lesion-specificity claim (the only thing a medical backbone should buy) was the
   reversed "pathology masking" result.
4. **Unresolved contradiction.** OOD AUC favors M3 (0.974 vs 0.778) yet InceptionV3 has higher
   per-sample Spearman with the real/gen label (0.397 vs 0.271).

## Scope decision

Pursue **Direction 1 (honesty/fixes) + Direction 3 (LGG real-mask medical evidence) + Debug
the metric**, sequenced A → B → C. Not pursuing Direction 2 (diagnostic-MMD as a new method)
as the spine.

## Feasibility (verified 2026-07-25)

- GPU: RTX 4070 Laptop (8 GB), torch 2.6.0+cu124, `timm` 1.0.16 present.
- No trained DDPM checkpoint locally → step-ladder NOT runnable; use existing generated sets.
- `output/generated_500_standard` (500) present → unfiltered-set fix ready.
- `data_mri/lgg_download/Brain-MRI-LGG-Segmentation.zip` present → real expert LGG masks
  available by extraction (no network needed).
- Metric contract: feed **raw [0,1]** 224×224 tensors (Resize+ToTensor only); the metric
  normalizes internally (`_preprocess`). Do NOT pre-normalize.

---

## Workstream A — Debug & simplify the metric (this spec's focus)

**Goal:** empirically settle the metric design and remove indefensible complexity, producing a
"metric-final" recommendation before any paper work.

**Sets (raw [0,1], 224², N≈250 each unless GPU-limited):**
- real = `data_mri/brats_axial_multislice`
- gen  = `output/generated_500_standard` (unfiltered)
- ood_near = `output/generated_wdm3d/brats`
- ood_far  = `output/generated_retinal`

**Procedure (single script, reuse `M3EntropyMetric` API, `single_layer=None` to expose all 12 layers):**
- Extract per-layer features once per set via `_extract_raw_features(x, layers_to_keep={1..12})`.
- **A1 Layer discriminability:** per layer L∈{1..12}, compute MMD²(real,gen) and its permutation
  Z via `_permutation_test_fidelity`, plus bootstrap CV via repeated `_bootstrap_ci_fidelity`/
  resampling. Also MMD²(real, ood_near/far). Report which single layer maximizes Z at lowest CV.
- **A2 Weighting ablation:** compare, for real-vs-gen, the decisiveness (Z) and stability (CV) of:
  (i) single-layer L12, (ii) uniform mean over active layers, (iii) sem×stab×uniq weighting
  (the current scheme, via `forward()`). Question: does (iii) beat (i)?
- **A3 OOD-vs-Spearman:** on real-vs-ood_far, compute ROC-AUC and per-sample Spearman(label,
  distance-to-real-centroid) for M3 (L12 features) and for InceptionV3 (FID features), to
  reproduce and explain the AUC-vs-Spearman gap (rank vs value).

**Deliverables:**
- `results/debug_metric/ablation_report.json` (all numbers)
- `results/debug_metric/MEMO.md` (decision: keep/drop weighting; single-layer vs multi-axis;
  OOD framing) — feeds the metric-final and the paper reframe.

**Decision rule:** if L12-only Z ≥ weighted Z at ≤ CV, recommend shipping single-layer L12
unbiased multi-bandwidth RBF MMD² + permutation p + bootstrap CI, and scope multi-axis as
exploratory. Change `m3_score_v2.py` defaults only if the evidence supports it; keep the
`M3V2Metric`/`m3_v2_final_score` back-compat aliases untouched.

**Suitability gate:** a smoke test (N=16) must confirm the backbone loads and feature
extraction fits in 8 GB before the full run.

## Workstreams B and C (later)

- **B (LGG real masks):** extract zip → lesion-vs-matched-healthy perturbation test (ΔM3 vs
  ΔFID) + `M3PatchScorer` heatmap→IoU. Honest scoping: metric-sensitivity test on real images
  (no LGG generator), report nulls if they occur.
- **C (honesty pass):** re-run core/statistical-rigor/conditional-MMD on the standard set; fix
  TSTR exact small-n permutation p; retitle/reframe `paper.tex`; sync numbers to
  EXPERIMENT_FINDINGS.

## Novelty positioning (honest)

Not novel: MMD + permutation test; medical-backbone Fréchet/MMD (FRD, CMMD). Defensible:
LGG lesion-**specificity** evidence, CMMD cross-modality structural-failure finding, stratified
MMD as a diagnostic. Target framing: rigorous domain-evaluation contribution, not method-novelty.

Note: repo is not a git repository, so this spec is not committed.
