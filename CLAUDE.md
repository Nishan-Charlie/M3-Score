# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

**MRI-Diffuser / M3-Score** is a research framework for unconditional 2D MRI image
generation with DDPMs, whose real deliverable is the **M3-Score** — a medical
generative-model evaluation metric built on a frozen medical ViT backbone (default
`Snarcy/RadioDino-s16`). The metric computes an entropy-/stability-/uniqueness-weighted,
multi-bandwidth RBF MMD² between real and generated feature distributions.

Most of the active work is **evaluation/experiments and paper writing**, not model
training. The generator (a fine-tuned DDPM) exists mainly to produce samples the metric
is validated on.

> ⚠️ **The headline numbers in `paper/paper.tex` and older docs are stale.**
> Newer runs in `docs/EXPERIMENT_FINDINGS.md` supersede them. Do **not** repeat the paper's
> original claims as fact. See [Contested Claims](#contested-claims-read-before-quoting-numbers)
> and `docs/next_note.md` before quoting any metric value. (`README.md` was rewritten
> 2026-08-29 against the corrected values and is safe to quote.)

## Environment Setup

```bash
pip install -r requirements.txt
```
`timm` and `open_clip_torch` are now **pinned** in `requirements.txt` — the default
RadioDino-s16 backbone will not load without them. Per-experiment extras (pyradiomics,
frd-score, clean-fid, medigan, medmnist, gradio, …) are listed commented-out at the
bottom of that file; install them individually when a specific experiment needs one.

## Repository Map (what actually exists)

```
train.py                     DDPM training entry point (argparse)
generate.py                  Sampling + optional Best-of-N reward rejection
evaluation/
  m3_score_v2.py             THE metric. class M3EntropyMetric (alias M3V2Metric).
                             Header says "M3-Score V3" — same file, current version.
  eval_pipeline.py           Multi-metric CLI: fid | kid | ssim | alpha | m3
  run_experiments.py         Orchestrator that writes results/master.json (~12 curated experiments)
  cmmd_metric.py             CLIP-based CMMD baseline (Jayasumana 2024)
  frd_wrapper.py             Fréchet RadDino Distance baseline
  conditional_mmd.py, coverage_novelty.py, statistical_rigor.py, m3_patch_scorer.py
  finetune_raddino_{classification,segmentation}.py   backbone finetuning
experiments/                 44 validation scripts (NOT 21). See "Experiments" below.
  _shared_utils.py           get_m3_transform(), get_fid_transform(), image loaders
metrics/                     Supplementary: alpha_precision_recall, anomaly_detection,
                             downstream_classifier, rad_fid, tsne_visualizer, extended_tests
models/unet.py               get_model(model_type) → unet | attention_unet | dit
data/dataloader.py           MONAI CacheDataset, [-1,1] scaling, PILReader (see Gotchas)
utils/                       trainer.py, rl_trainer.py, finetune_rl.py, reward_system.py
tools/                       data prep + WDM-3D + medigan + 3D visualization utilities,
                             plus one-off helpers: collage.py, regen_noise_plots.py,
                             _verify_determinism.py, check_cka.py
scripts/                     shell runners (*.sh)
notebooks/                   00–04 exploration/dashboard notebooks (built by _build_notebooks.py)
external/wdm-3d/             Vendored WDM-3D (3D wavelet diffusion) baseline generator
pretrained/wdm3d/            WDM-3D checkpoints (brats_unet_128_1200k.pt, lidc-idri_...)
config/global.json           medigan model registry (3rd-party GAN zoo metadata) — NOT app config
paper/                       paper.tex, paper.pdf, paper.bbl, refs.bib, Images/ (paper-local
                             figures; \includegraphics paths resolve inside paper/)
figures/                     Figure staging area, copied into paper/Images/. Written by
                             experiments/regen_ood_roc.py. (Was top-level Images/.)
results/                     Experiment outputs (*_report.json + PNGs), versioned per run
  master.json                OUTPUT of evaluation/run_experiments.py (~385KB) — not input
docs/
  EXPERIMENT_FINDINGS.md     Ground-truth log of latest experiment results (authoritative)
  METHODOLOGY_AND_DIRECTION.md  Honest self-assessment + action plan for the paper
  next_note.md               Research state + what to do next
  specs/                     Design specs for completed workstreams
README.md / LICENSE / CITATION.cff
```

> **Repo cleaned + `git init`-ed 2026-08-29.** ~719 MB of regenerable cache (84 `pert_*`
> dirs, 2 `calibration_sanity/` dumps, 400 `_tmp_*` PNGs, `__pycache__`, stray logs,
> an aborted `pretrained/wdm3d_new/` download) was moved to `_quarantine_2026-08-29/`.
> Delete that folder once you're satisfied nothing is missing. All `*_report.json`,
> `master_results.json` and figures were retained.

Top-level orchestrators (each different — pick deliberately):

| Script | Purpose |
|---|---|
| `run_all_experiments.py` | **Primary** master runner; EXP 1–12, publication-styled plots. `--run_id --seed --out_dir` |
| `run_experiments_13_19.py` | Adds the tri-axial framework EXP 13–19; merges into `master_results.json` |
| `run_all_backbones.py` | Runs `run_all_experiments.py` for all 6 backbones into per-backbone dirs |
| `run_multidataset.py` | Cross-modality: BraTS + RetinaMNIST + PneumoniaMNIST |
| `evaluation/run_experiments.py` | Separate curated orchestrator → writes `results/master.json` |

## Common Commands

### Train
```bash
python train.py --data_dir data_mri/brats_axial_multislice \
    --output_dir output/output_unet --model_type unet \
    --batch_size 16 --img_size 256 --lr 1e-5 --epochs 50 \
    --mixed_precision fp16 --device cuda:0
```
`--model_type`: `unet | attention_unet | dit`. `--mixed_precision`: `fp16 | no`.

### Generate
```bash
python generate.py --checkpoint_dir output/output_unet/checkpoints/best \
    --num_images 500 --output_dir output/generated \
    --calculate_metrics --data_dir data_mri/brats_axial_multislice \
    --reward_type trajectory_efficiency --best_of_n 4
```
`--reward_type`: `none` (default) | `trajectory_efficiency` (no external model) |
`deep_cosine_diversity` (loads VGG-16). Best-of-N only active when `reward_type != none`.
`--num_inference_steps` lower (50/20/10) for the quality-ladder experiment.

### Multi-metric evaluation
```bash
python evaluation/eval_pipeline.py --real_dir data_mri/brats_axial_multislice \
    --gen_dir output/generated --output_dir results/evaluation \
    --metrics fid kid ssim m3 alpha
```

### Full experiment suites
```bash
python run_all_experiments.py --run_id 5 --seed 42 --device cuda:0        # EXP 1–12
python run_experiments_13_19.py --run_id 5 --seed 42 --device cuda        # EXP 13–19
python run_all_backbones.py --device cuda:0 --backbones rad-dino dinov2    # subset
python evaluation/run_experiments.py --real_dir ... --gen_dir ... --experiments all
```

### RL fine-tuning (DDPO)
```bash
python utils/finetune_rl.py --checkpoint_dir output/output_unet/checkpoints/best \
    --reward_type deep_cosine_diversity --rl_epochs 50 --batch_size 4 --lr 1e-6
```

## M3-Score Implementation (`evaluation/m3_score_v2.py`)

`M3EntropyMetric(device, backbone_id="Snarcy/RadioDino-s16", cka_threshold=0.8,
kernel="rbf", single_layer=12, seed=42)`.

- **Weighting:** `w_l ∝ semanticity_l · stability_l · uniqueness_l`, where
  `semanticity=exp(-H_l/T)`, `stability=min(mean/std, 1e4)` (SNR, capped),
  `uniqueness=max(1-mean_CKA_to_others, 1e-4)`.
- **MMD:** unbiased multi-bandwidth RBF (median heuristic on real∪gen) — default `kernel="rbf"`.
- **Output dict:** both `m3_score` and `m3_v2_final_score` keys (identical value; use `m3_score`).
- **6 backbone families**, auto-detected from `backbone_id` in `_detect_backend`:
  `transformers_vit` (rad-dino, dinov2), `timm_vit` (RadioDino-s16), `clip_transformers`
  (pubmed-clip), `open_clip_timm` (biomedclip), `resnet` (resnet50/rad-imagenet).
- **`single_layer=12` is the DEFAULT** → the metric runs in single-layer mode and
  `prune_layers_via_cka()` is a **no-op**. To get the CKA-selected multi-scale behavior,
  construct with `single_layer=None` and call `prune_layers_via_cka(ref_imgs)` first.
  This directly supersedes the old "[1, 4, 12]" three-layer story (see Contested Claims).

Backbone registry (`run_all_backbones.py`): `rad-dino`=microsoft/rad-dino,
`dinov2`=facebook/dinov2-base, `radiodino-s16`=Snarcy/RadioDino-s16,
`pubmedclip`=flaviagiammarino/pubmed-clip-vit-base-patch32,
`biomedclip`=microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, `resnet50`.

## Model Architectures (`models/unet.py`)

`get_model(model_type, in_channels=1, out_channels=1, ...)`; first/last conv auto-adapted
to 1-channel grayscale via `_adapt_channels`:

| `model_type` | Pretrained checkpoint | Scheduler source |
|---|---|---|
| `unet` | `google/ddpm-celebahq-256` | itself |
| `attention_unet` | `benetraco/brain_ddpm_256` | itself |
| `dit` | `facebook/DiT-XL-2-256` | `google/ddpm-celebahq-256` (standard DDPM) |

## Data (`data/dataloader.py`)

`get_mri_2d_dataloader(data_dir, batch_size, spatial_size=(256,256))`: MONAI `CacheDataset`
(`cache_rate=1.0`), recursive JPG/PNG/TIF, transform `LoadImaged(reader="PILReader") →
EnsureChannelFirst → Resized → ScaleIntensityd(-1,1) → ToTensor`. Output range **[-1, 1]**.

Datasets under `data_mri/` (all gitignored):
`brats_axial_multislice/` (38,781 BraTS axial PNGs, the main real set), `brats_with_proxymasks/`,
`lgg_download/`, `lgg_with_masks/`, `medmnist/{retinamnist,pneumoniamnist}/`.
Generated sets live under `output/`: `generated_500_standard`, `generated_500_best`,
`generated_retinal`, `generated_cxr_corrupted`, `generated_wdm3d/brats`, etc.

## Experiments (`experiments/` — ~40 scripts)

Each validates a paper claim; shared helpers in `experiments/_shared_utils.py`. The
`evaluation/run_experiments.py` orchestrator runs a curated subset keyed as:
`interpretability, ood, noise, efficiency, fid_infinity, weight_ablation, permutation_test,
hallucination, orthogonality, frd_comparison, metric_interpretability, comparative_sweep`.

Notable scripts: `noise_robustness.py`, `ood_detection.py` / `ood_detection_comparison.py`,
`hallucination_detection.py`, `pathology_masking.py`, `interpretability.py`,
`feature_orthogonality.py`, `weight_ablation.py`, `permutation_test.py`,
`cka_layer_similarity.py`, `backbone_comparison.py`, `comparative_metrics_sweep.py`,
`tstr_utility.py`, `distortion_monotonicity_per_scale.py`, `normality_violation.py`,
`sample_size_consistency.py`, `{generate,evaluate}_medigan_samples.py`.

Results are versioned under `results/` (`experiments_output_v5/` is latest) plus purpose-named
dirs (`ood_detection_comparison/`, `pathology_masking/`, `cka_analysis/`,
`radiodino-s16_run5/`, `multidataset_run/`, ...). Each holds a `*_report.json` + PNGs.

**`pert_*/` dirs are a cache, not results.** `noise_robustness.py` reuses perturbed images
across runs and reloads them if the file count matches; delete a `pert_*` dir to force a
fresh save (e.g. after changing seed or `num_images`). They are gitignored and were purged
in the 2026-08-29 cleanup — the first re-run of a noise sweep will regenerate them.

## WDM-3D Baseline

`external/wdm-3d/` is a vendored 3D wavelet-diffusion generator used as a second generative
baseline (its axial slices go to `output/generated_wdm3d/brats`). Weights in `pretrained/wdm3d/`.
Driver utilities: `tools/download_wdm3d.py`, `tools/generate_wdm3d.py`.

## Reward & RL (`utils/`)

- `reward_system.py`: `TrajectoryEfficiencyReward` (pixel variance + step penalty, no model),
  `DeepCosineDiversityReward` (VGG-16, needs GPU); factory `build_reward_fn`. Used by
  `generate.py` Best-of-N rejection — no weight updates.
- `rl_trainer.py` / `finetune_rl.py`: DDPO policy gradient with KL regularization (DPOK-style);
  needs a pre-trained checkpoint; saves to a separate dir.

## Gotchas

- **Stale paper numbers.** `docs/EXPERIMENT_FINDINGS.md` is authoritative over `paper/paper.tex`.
  See [Contested Claims](#contested-claims-read-before-quoting-numbers). `README.md` was
  rewritten against the corrected values on 2026-08-29.
- **`single_layer=12` default** means CKA multi-scale pruning is off unless you pass
  `single_layer=None`. (`canonical_layers.json` no longer exists — it was an empty 0-byte
  stub and is not referenced by any code.)
- **Rotation fix.** `generate.py` applies a 90° CCW rotation fix by default (`--no_rotation_fix`
  disables it). This corrects for models trained *before* the dataloader switched to `PILReader`
  (MONAI's default ITKReader loads PNG/JPG transposed as (W,H)). Retrain-clean models don't need it.
- **`config/global.json` is medigan's registry**, not project config; **`results/master.json`
  is an output file**, not input.
- **Feature extraction OOM:** use `batch_size=1` in `M3EntropyMetric` — attention-weighted
  pooling is memory-heavy.
- **Git history starts 2026-08-29.** The repo was untracked before that; the initial commit
  is the post-cleanup state, so `git log` will not explain anything older. Use the `docs/*.md`
  logs for pre-2026-08-29 context.
- **Figure paths differ by location.** `paper/paper.tex` uses `Images/...`, which resolves to
  `paper/Images/`. The repo-root staging dir is `figures/` (renamed from `Images/`). Keep them
  in sync by copying `figures/.` into `paper/Images/`.
- **BraTS prep:** `tools/prepare_brats_slices.py` extracts 2D axial slices from `.nii.gz` volumes.

## Contested Claims (read before quoting numbers)

Per `docs/EXPERIMENT_FINDINGS.md` ("Claims That Must Be Updated"), newer runs (run5, N=500) give:

| Old claim (paper/README) | Corrected value |
|---|---|
| OOD AUC M3 = 0.863 | **0.9743** (FID ≈ 0.778) |
| Active layers [1, 4, 12] | **Single layer [12]** (higher discriminability than multi-layer) |
| CV = 4.70% (M3), 44% (FID) | **M3 ≈ 1.93%**, **FID ≈ 1.89%** at N=500 (both stable) |
| 4.23× pathology-masking sensitivity over FID | **Reversed** — FID/CMMD are *more* artifact-sensitive; M3 is more *specific* to semantic shift, not more sensitive to masking |
| Z = 486 | **Z ≈ 446** (500 perms) |

New positive findings: TSTR utility correlation ρ=1.000 for M3 (vs 0.800 FID); zero memorization
in the DDPM; conditional-MMD reveals slice-intensity bias. The current defensible framing lives in
`docs/METHODOLOGY_AND_DIRECTION.md` Part IV.
