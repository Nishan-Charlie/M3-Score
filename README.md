# M3-Score

## Fidelity, memorization, and coverage as separate axes for evaluating generative radiology image models

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project page](https://img.shields.io/badge/Project%20page-academic-6f42c1.svg)](docs/index.html)

M3-Score is a medical-image evaluation framework for generative radiology models. It does not
collapse quality into a single number. Instead, it reports three interpretable axes in a
frozen radiology-specific feature space:

- **Fidelity:** distributional agreement between real and generated images.
- **Memorization:** proximity of generated images to real-image neighbors.
- **Coverage:** how much of the real-image manifold is represented by generated images.

The framework uses frozen RadioDINO-s16 features and reads the axes from transformer blocks
fixed before evaluation: fidelity at L12, memorization at L9, and coverage at L4. The fidelity
axis also provides a native permutation test and bootstrap confidence interval.

![M3-Score overview](paper_release/Images/three_axis_severity.png)

**Read more:** [HTML project page](docs/index.html) · [academic project notes](docs/PROJECT.md) · ArXiv link coming soon ·
[publication figures](paper_release/Images/) · [release scope](RELEASE.md)

> This is a pre-publication research repository. It does not claim acceptance or submission to
> MICAI. The repository is organized as a publication-grade companion release for the paper.

## Calculation methodology

[View the M3-Score calculation methodology diagram (PDF)](docs/assets/M3Score-Calculation.pdf)

The diagram summarizes the feature extraction, fixed-layer selection, and separate fidelity, memorization, and coverage calculations used by M3-Score.

## Why three axes?

A generator can produce individually plausible images while missing large portions of the real
distribution. A single fidelity score cannot distinguish that failure from a well-covered
generator. M3-Score keeps the failure modes separate and makes each axis testable against the
perturbation it is intended to detect.

| Axis | Feature depth | What it measures | Direction |
| --- | ---: | --- | :---: |
| Fidelity | L12 | Unbiased multi-bandwidth RBF MMD^2 between real and generated features | lower |
| Memorization | L9 | Generated images that are unusually close to real neighbors | lower |
| Coverage | L4 | Real images with a generated neighbor inside their real-set k-NN radius | higher |

## Main findings from the paper

The reported experiments use subject-diverse BraTS references and generated radiology images.
The complete numerical record is retained in [`docs/EXPERIMENT_FINDINGS.md`](docs/EXPERIMENT_FINDINGS.md)
and [`results/`](results/).

- A DDPM obtains fidelity MMD^2 = **0.073** with a bootstrap 95% CI of **[0.071, 0.080]**, yet
  covers only **38%** of the real manifold.
- Conventional k-NN manifold recall increases as generated diversity is removed at all twelve
  encoder blocks (Spearman rho = **+0.70 to +1.00**), while real-radius coverage decreases
  monotonically (rho = **-1.00**).
- RadioDINO separates real brain MRI from in-domain DDPM output at ROC-AUC **0.819**, compared
  with **0.555** for InceptionV3/FID features and **0.582** for CLIP/CMMD features.
- Across a twenty-fold change in evaluation size, M3 changes by **1.05x** and FID by **2.52x**
  on the same data.
- With expert glioma masks and area-matched controls, M3 is **1.4-1.8x** more responsive to
  tumor-region perturbations than matched healthy-region perturbations; the Inception-MMD
  comparison is approximately **1.05x**.

The paper also reports important limitations: FID is more repeatable than M3 for fixed sample
sizes below 500; the axis depths are fixed by representational position rather than optimized on
test outcomes; and the permutation test is descriptive because its weak null is rejected by all
generators in the study.

## Visual summary

### Real and generated radiology samples

![Real and generated samples](paper_release/Images/samples/real_vs_generated.png)

### Coverage and mode-drop validation

![Coverage validation](paper_release/Images/coverage_axis_validation.png)

### Domain-specific feature-space comparison

![OOD comparison](paper_release/Images/OOD/ood_roc_auc.png)

### Lesion-specific perturbation analysis

![Lesion specificity](paper_release/Images/lesion_specificity/lgg_noise_sweep.png)

## Quick start

### Install

```bash
conda create -n m3score python=3.10
conda activate m3score
pip install -r requirements.txt
```

The default backbone is `Snarcy/RadioDino-s16`. Dataset files, checkpoints, and generated
training outputs are intentionally excluded from this repository.

### Score two image directories

```python
from evaluation.m3_score_v2 import M3EntropyMetric

metric = M3EntropyMetric(
    device="cuda",
    backbone_id="Snarcy/RadioDino-s16",
    seed=42,
)
result = metric.compute(real_images, generated_images)
print(result["m3_score"])
```

The default evaluation uses the fixed L12 fidelity layer. See the implementation and the
academic project page for the memorization and coverage protocols.

### Run the multi-metric pipeline

```bash
python evaluation/eval_pipeline.py \
  --real_dir /path/to/real_images \
  --gen_dir /path/to/generated_images \
  --output_dir results/evaluation \
  --metrics fid kid ssim m3 alpha
```

## Repository map

```text
evaluation/       M3-Score and baseline evaluation implementations
metrics/           Supplementary image-quality metrics
models/            Generator model definitions
data/              Dataset loading utilities
figures/           Publication figures and image assets
results/           Numeric reports, plots, and retained experiment logs
tools/             Audits, result tables, and analysis helpers
external/wdm-3d/  Vendored WDM-3D baseline
paper_release/    Supporting publication figures and retained logs
docs/              Academic project page and experiment records
```

## Reproducibility and data policy

This release contains code, figures, result reports, and logs needed to inspect the paper’s
claims. Medical datasets and model weights are not redistributed. Obtain them from their
original sources, verify their licenses and access conditions, and keep local copies in ignored
directories. See [`RELEASE.md`](RELEASE.md) for the publication boundary.

## Publish the project page

GitHub Pages can publish this site directly from the `docs/` folder:

1. Open the repository on GitHub and go to **Settings -> Pages**.
2. Select **Deploy from a branch**.
3. Choose the `main` branch and the `/docs` folder.
4. Save the setting and open the generated Pages URL.

The entry point is [`docs/index.html`](docs/index.html). After enabling Pages, the live site will be available at `https://nishan-charlie.github.io/M3-Score/`. The page includes a self-contained copy
of selected figures under [`docs/assets/`](docs/assets/); the ArXiv link will be added later.
## Citation

```bibtex
@article{nishankar2026m3score,
  title   = {M3-Score: Fidelity, Memorization and Coverage as Separate Axes for Evaluating
             Generative Radiology Image Models},
  author  = {Sathiyamohan, Nishankar},
  year    = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE). Vendored WDM-3D and pretrained backbones retain their own
licenses; consult the relevant upstream projects before redistribution.