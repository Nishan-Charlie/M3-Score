# M3-Score

**A Multi-Axis Medical-MMD Metric with A-Priori Layers and Native Significance Testing
for Generative Radiology Image Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/get-started/locally/)
[![MONAI](https://img.shields.io/badge/MONAI-0.9+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation and full experimental record for **M3-Score**, an evaluation
metric for generative medical imaging models. M3-Score computes a multi-bandwidth RBF
MMD² between real and generated feature distributions drawn from a **frozen medical ViT
backbone** (default `Snarcy/RadioDino-s16`), and — unlike FID — ships a **native
hypothesis test**: a permutation p-value and a bootstrap confidence interval.

This is a **metric paper**. The DDPM generator and the vendored WDM-3D baseline exist to
produce the samples the metric is validated on; they are not the contribution.

---

## Results

All numbers below are from `run5` at N=500 with seed 42. **`docs/EXPERIMENT_FINDINGS.md`
is the authoritative source** — it is a dated experiment log and supersedes any figure
quoted elsewhere, including in earlier drafts of this README.

| | FID (Inception) | CMMD (CLIP) | **M3 (ours)** |
|---|:---:|:---:|:---:|
| OOD detection AUC | 0.7781 | — | **0.9743** |
| Stability, CV @ N=500 | 1.89% | — | 1.93% |
| Stability, CV @ N=50 | — | — | 11.3% |
| TSTR utility correlation (ρ) | 0.800 (p=0.200) | — | **1.000** (n=4) |
| Native significance test | ✗ | ✗ | **✓** permutation p + bootstrap CI |

**Where M3 wins.** Effect size Z ≈ 446 over 500 permutations — FID is a point estimate and
structurally cannot produce this. M3 also perfectly predicts downstream task utility across
four generators (ρ=1.000) where FID does not, and stratified conditional MMD exposes
per-stratum generator failure (worst on high-intensity Q4 slices, MMD 0.251 vs 0.208 mean)
that any aggregate scalar hides.

**Lesion specificity.** Against **real LGG expert masks**, under subtle noise M3 is
lesion-selective at **1.4–1.8×** versus Inception-MMD's ~1.05×, robust to two area-matched
controls (mirror + texture-matched) and a σ-sweep.

**Where M3 does not win — stated plainly.**
- FID is **not** unstable at N=500 (CV 1.89% vs M3's 1.93%); at N<500 FID has *lower*
  variance. M3's advantage is the statistical test, not lower variance. Use N≥500.
- M3 is **less** artifact-sensitive than FID/CMMD. Under a 64px centre mask FID rises
  +101% and CMMD +192% while M3 *falls* −54%. The correct framing is that M3 is more
  **specific** to semantic distributional shift, not more **sensitive** to masking.
- The metric runs **single-layer (L12)** by default. CKA ablation showed L12 alone reaches
  higher discriminability (9.54) than two-layer (9.42) or three-layer configurations.
- CMMD shows a significant **rank inversion**: it ranks LIDC-CT closer to BraTS-MRI (0.699)
  than a weak MRI generator (0.738), Δ=−0.039 [−0.059, −0.021]. M3 and FID order correctly.

Additional validated findings: **zero training-image memorization** in the DDPM; RadioDino-s16
is the best of six backbones on OOD-AUC and discriminability.

---

## Install

```bash
conda create -n mri-diffuser python=3.10 && conda activate mri-diffuser
pip install -r requirements.txt
```

`timm` and `open_clip_torch` are **required** — the default RadioDino-s16 backbone will not
load without them. Optional per-experiment extras are listed, commented, at the bottom of
`requirements.txt`.

---

## Quick start

### Score two image directories

```python
from evaluation.m3_score_v2 import M3EntropyMetric

metric = M3EntropyMetric(device="cuda", backbone_id="Snarcy/RadioDino-s16", seed=42)
result = metric.compute(real_images, generated_images)   # tensors, [-1, 1]
print(result["m3_score"])
```

`single_layer=12` is the **default**, which puts the metric in single-layer mode and makes
`prune_layers_via_cka()` a no-op. For the CKA-selected multi-scale path, construct with
`single_layer=None` and call `prune_layers_via_cka(ref_imgs)` first. On OOM during feature
extraction, set `batch_size=1` — attention-weighted pooling is memory-heavy.

### Multi-metric evaluation

```bash
python evaluation/eval_pipeline.py \
    --real_dir data_mri/brats_axial_multislice --gen_dir output/generated \
    --output_dir results/evaluation --metrics fid kid ssim m3 alpha
```

### Reproduce the main result set

```bash
python run_all_experiments.py   --run_id 5 --seed 42 --device cuda:0   # EXP 1–12
python run_experiments_13_19.py --run_id 5 --seed 42 --device cuda     # EXP 13–19
# → results/radiodino-s16_run5/ + master_results.json
```

### Train / generate

```bash
python train.py --data_dir data_mri/brats_axial_multislice \
    --output_dir output/output_unet --model_type unet \
    --batch_size 16 --img_size 256 --lr 1e-5 --epochs 50 --mixed_precision fp16

python generate.py --checkpoint_dir output/output_unet/checkpoints/best \
    --num_images 500 --output_dir output/generated
```

`--model_type`: `unet` | `attention_unet` | `dit`. Note that `generate.py` applies a 90° CCW
rotation fix by default (`--no_rotation_fix` to disable); it corrects models trained before
the dataloader switched to MONAI's `PILReader`.

---

## Repository layout

```
evaluation/            The metric and its baselines
  m3_score_v2.py         M3EntropyMetric (alias M3V2Metric) — THE metric
  eval_pipeline.py       Multi-metric CLI: fid | kid | ssim | alpha | m3
  cmmd_metric.py         CLIP-based CMMD baseline (Jayasumana 2024)
  frd_wrapper.py         Fréchet RadDino Distance baseline
  conditional_mmd.py     Intensity-stratified MMD diagnostic
  statistical_rigor.py   Permutation test + bootstrap CI
  run_experiments.py     Curated orchestrator → results/master.json
experiments/           44 validation scripts, one per paper claim
  _shared_utils.py       Shared transforms and image loaders
metrics/               Supplementary metrics (alpha-precision/recall, rad-FID, t-SNE, …)
models/unet.py         get_model(): unet | attention_unet | dit
data/dataloader.py     MONAI CacheDataset, output range [-1, 1]
utils/                 DDPM trainer, DDPO RL fine-tuning, reward system
tools/                 Data prep, WDM-3D drivers, plotting, 3D visualization
scripts/               Shell runners
notebooks/             00–04 exploration + dashboard (built by _build_notebooks.py)
external/wdm-3d/       Vendored WDM-3D 3D wavelet-diffusion baseline
paper/                 paper.tex, paper.pdf, refs.bib, Images/ (paper-local figures)
figures/               Figure staging area, copied into paper/Images/
results/               Experiment outputs (*_report.json + PNGs), versioned per run
docs/                  See below
config/global.json     medigan registry metadata — NOT project configuration
```

**Top-level orchestrators** — each is different, pick deliberately:

| Script | Purpose |
|---|---|
| `run_all_experiments.py` | Primary master runner; EXP 1–12 + publication plots |
| `run_experiments_13_19.py` | Tri-axial framework EXP 13–19; merges into `master_results.json` |
| `run_all_backbones.py` | Runs the above across all 6 backbones into per-backbone dirs |
| `run_multidataset.py` | Cross-modality: BraTS + RetinaMNIST + PneumoniaMNIST |
| `evaluation/run_experiments.py` | Separate curated orchestrator → `results/master.json` |

---

## Documentation

| File | What it is |
|---|---|
| `docs/EXPERIMENT_FINDINGS.md` | **Authoritative.** Dated log of every experiment result. |
| `docs/METHODOLOGY_AND_DIRECTION.md` | Honest self-assessment; current defensible framing in Part IV. |
| `docs/next_note.md` | Research state and what to do next. |
| `docs/specs/` | Design specs for completed workstreams. |
| `CLAUDE.md` | Codebase guide for AI coding assistants. |

Precedence for any number: `docs/EXPERIMENT_FINDINGS.md` → `docs/METHODOLOGY_AND_DIRECTION.md`
→ `paper/paper.tex`.

---

## Data

Datasets live under `data_mri/` and are gitignored. The main real set is
`brats_axial_multislice/` (38,781 BraTS axial PNGs), prepared from `.nii.gz` volumes with
`tools/prepare_brats_slices.py`. MedMNIST subsets via `tools/download_medmnist.py`;
WDM-3D checkpoints via `tools/download_wdm3d.py`.

---

## Citation

```bibtex
@article{nishankar2026m3score,
  title   = {M3-Score: A Multi-Axis Medical-MMD Metric with A-Priori Layers and
             Native Significance Testing for Generative Radiology Image Models},
  author  = {Sathiyamohan, Nishankar},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Vendored WDM-3D, medigan registry metadata, and the
Hugging Face backbones carry their own licenses; see the LICENSE file for details.
