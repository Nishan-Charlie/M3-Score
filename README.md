# M3-Score

## Fidelity, memorization, and coverage as separate axes for evaluating generative radiology image models

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Nishan-Charlie/M3-Score/blob/main/LICENSE)
[![Project page](https://img.shields.io/badge/Project%20page-academic-6f42c1.svg)](https://nishan-charlie.github.io/M3-Score/)

M3-Score is a medical-image evaluation framework for generative radiology models. It does not
collapse quality into a single number. It reports three interpretable axes in a frozen
radiology-specific feature space:

- **Fidelity:** distributional agreement between real and generated images.
- **Memorization:** proximity of generated images to real-image neighbors.
- **Coverage:** how much of the real-image manifold is represented by generated images.

![M3-Score overview](https://raw.githubusercontent.com/Nishan-Charlie/M3-Score/main/paper_release/Images/three_axis_severity.png)

**Read more:** [project page](https://nishan-charlie.github.io/M3-Score/) ·
[academic project notes](https://github.com/Nishan-Charlie/M3-Score/blob/main/docs/PROJECT.md) ·
ArXiv link coming soon ·
[calculation diagram (PDF)](https://github.com/Nishan-Charlie/M3-Score/blob/main/docs/assets/M3Score-Calculation.pdf)

> This is a pre-publication research project. It does not claim acceptance or submission to
> any venue.

## Install

```bash
pip install m3-score            # 2D images: PNG, JPG, TIFF, BMP
pip install "m3-score[all]"     # adds NIfTI and DICOM
```

Python 3.10 or later and PyTorch 2.0 or later. The RadioDINO-s16 encoder (about 90 MB,
CC-BY-4.0) is downloaded from the Hugging Face Hub the first time you score something, and
cached. To install the development version instead:
`pip install "git+https://github.com/Nishan-Charlie/M3-Score"`.

## Score two folders

```bash
m3score path/to/real path/to/generated
```

```text
M3-Score  protocol m3-v1  (m3-score 0.1.0)
  real       2,000 images, 412 subjects
  generated  2,000 images

  axis          value    better   detail
  fidelity      0.0731   lower    95% CI [0.0689, 0.0774]; real-vs-real floor 0.0015 +/- 0.0004 (by subject)
  memorization  0.012    lower    mean nearest-real distance 14.2
  coverage      0.380    higher   k = 5

  noise level: 0.0023 (floor + 2 sd, from halves of the real set); fidelity is above it
  permutation test: z = 347.2, p = 0.00498 (descriptive, not a model-selection tool)
```

The numbers above illustrate the layout; they are not a published result.

| Option | Meaning |
| --- | --- |
| `--n N` | Images scored per set, drawn at random. Default: all, up to 5,000. |
| `--seed S` | Seed for sampling and statistics. Default 0. |
| `--device auto\|cpu\|cuda` | Default `auto`. |
| `--include GLOB` | Only read real files whose path inside the folder matches, e.g. `"*_flair.nii.gz"`. |
| `--group-pattern REGEX` | First capture group names the subject of each real file, e.g. `"(BraTS2021_\d+)"`. |
| `--slice-band LO HI` | Fraction of each volume's depth to keep. Default `0.355 0.555`. |
| `--window LO HI` | Intensity window for volumes, e.g. `-1000 400` for CT. |
| `--no-stats` | Skip the confidence interval, permutation test and real-vs-real floor. |
| `--json FILE` | Also write the result as JSON. |

Statistics for 5,000 images per set take about a minute on a CPU; extracting the features
takes longer. Use `--n 1000` for a quick estimate, or a GPU.

## Python

```python
import m3score

result = m3score.score("path/to/real", "path/to/generated", n=2000)
print(result.fidelity, result.memorization, result.coverage)
print(result)             # the table shown above
result.to_dict()          # JSON-serialisable
```

Tensors and arrays work too: shape `(N, H, W)`, `(N, 1, H, W)` or `(N, 3, H, W)`, either
`uint8` or floating point in `[0, 1]`. Values outside `[0, 1]` are rejected, so a generator
that outputs `[-1, 1]` needs `(x + 1) / 2` first.

```python
result = m3score.score(real_tensor, generated_tensor, device="cuda")
```

## In a training loop

```python
from m3score import M3Score

metric = M3Score(device="cuda")
for batch in real_loader:            # a random subset of the real set, about 5,000 images
    metric.update(batch, real=True)

for epoch in range(epochs):
    ...
    for batch in sample_batches():
        metric.update(batch, real=False)
    result = metric.compute()
    metric.reset(keep_real=True)     # the real features are reused next epoch
```

`M3Score` holds per-image features, about 4.6 kB per image, and computes when asked. It
uses the method names of `torchmetrics` without depending on it. It does not synchronise
across processes.

It keeps every image you pass, and the statistics need memory that grows with the square of
the total. Give each set a random subset of about 5,000 images, which is what `score` draws
by default; past 20,000 images in total it stops with an explanation instead of running out
of memory. Pass `groups=` to `update` when you know each image's subject.

## NIfTI and DICOM

```bash
pip install "m3-score[all]"
m3score BraTS2021/ generated/ --include "*_flair.nii.gz" --group-pattern "(BraTS2021_\d+)"
```

- **NIfTI.** Each volume is reoriented to one display convention (anterior up,
  radiological), so stored orientation does not matter. The central axial band is kept
  (slices 55 to 85 of a 155-slice BraTS volume, scaled to other depths) and each volume is
  scaled by the 0.5 to 99.5 percentile of its non-zero voxels. One file is one subject, so
  a folder that holds several files per subject needs `--include` and `--group-pattern`;
  the result says so when it sees that layout.
- **DICOM.** A series of positioned slices is a volume: its slices are ordered by position
  and the same band is kept along the direction they were acquired in. Slices are not
  resampled to another plane, so a sagittal series is scored as sagittal slices. Single
  images such as chest radiographs are scored as they are. The stored window is used when
  present (for a volume, the median over its slices); `MONOCHROME1` is inverted. The subject
  is `PatientID`, so a dataset anonymised to one shared ID counts as one subject unless you
  pass `--group-pattern`. Compressed files need a pydicom decoder plugin such as `pylibjpeg`.
- **CT.** Pass an explicit window, for example `--window -1000 400`.

A dataset folder that holds several modalities and segmentations per subject must be
filtered with `--include`, or they will be mixed.

Files that cannot be read, are blank, or claim an implausible size are left out and named in
the warnings. The image counts in the result are what was actually scored.

`--include` and `--group-pattern` describe the real folder and must match it. Generated
files are usually named differently, so each is applied to the generated folder only if it
matches something there; when `--include` is not applied, the result says so.

If your generator was trained on slices you exported yourself, score it against a real
folder exported the same way. A different slice selection, orientation or intensity scaling
between the two sets shows up as distance that has nothing to do with the generator.

## Build a valid reference set

How the real set is assembled can matter more than which generator is scored. In this
project, taking the first 500 BraTS slices in filename order selected 17 subjects from one
contributing site and gave a fidelity of 0.150; a random 500 from the same pool, covering
413 subjects, gave 0.077
([details](https://github.com/Nishan-Charlie/M3-Score/blob/main/docs/REFERENCE_COHORT_FINDING.md)).

The package is built around that finding:

- A subset is always a seeded random draw, never the head of a sorted folder.
- When subjects are known, the draw is balanced across them. NIfTI files and DICOM
  `PatientID` provide subjects automatically; for image folders pass `--group-pattern`.
- The result reports distinct subjects next to the image count, warns below 50 subjects or
  500 images, and says so whenever the real set's subjects are unknown, because then none of
  this can be checked. That is the normal case for a plain folder of PNGs, so pass
  `--group-pattern` when the file names carry a subject identifier.
- It warns when real images are exact duplicates of each other, which distorts memorization
  and coverage.

Report both counts with any M3-Score you publish, and keep generator training data and the
reference set subject-disjoint.

## Reading the result

- **Fidelity** is an unbiased MMD² and has no natural scale. Read it against the
  **real-vs-real floor**, the same statistic between random halves of your real set (mean
  and spread over ten halvings), and the **noise level** printed under the table, which is
  the floor plus two standard deviations. A fidelity at or below the noise level cannot be
  told apart from real data at this sample size; the further above it, the clearer the gap.
  - The floor is conservative. Each half is half the size of the real set, so it reads
    roughly twice what a full-size real-vs-real comparison would.
  - With known subjects the halves are subject-disjoint, and the floor includes genuine
    variation between subjects. With unknown subjects the split is by image: its mean is
    zero by construction and only the spread is informative.
- The **permutation test** rejects for every generator tested in the paper. Treat it as
  descriptive, not as a way to choose between models.
- **Memorization** is the fraction of generated images closer to a real image than real
  images typically are to each other. Identical copies give 1.0.
- **Coverage** is the fraction of real images with a generated image inside their own
  5-nearest-neighbour radius. It cannot rise when generated samples are removed.
- Scores are comparable only under the same protocol. Results from a custom backbone or
  custom layers are labelled `custom`.

### The `m3-v1` protocol

| | |
| --- | --- |
| Encoder | `Snarcy/RadioDino-s16` (ViT-S/16), pinned Hub revision `773d28d` |
| Input | grey repeated to RGB, antialiased bilinear resize to 224x224, ImageNet normalisation |
| Feature | CLS token after a transformer block |
| Fidelity | block 12, unbiased MMD² with five RBF bandwidths around the pooled median |
| Memorization | block 9, nearest-real distance against the median real-to-real distance |
| Coverage | block 4, real-radius coverage with k = 5 |

### Differences from the paper's research code

The package reproduces the research code's fidelity, memorization and coverage values
exactly on identical features (`tests/test_parity.py`). It differs deliberately in seven places:

- **Coverage axis.** `compute_axes()` in the research code returns k-NN precision and recall
  at block 4. The package returns real-radius coverage, the estimator the paper validates.
- **The real-vs-real floor and noise level** are new.
- **Confidence interval.** The research code's bootstrap counted duplicated images as
  distinct pairs, which shifts the interval upward. The package leaves those pairs out. The
  paper's interval [0.071, 0.080] was computed with the earlier procedure.
- **Permutation p-value** is `(1 + b) / (1 + B)` and so is never exactly 0.
- **`m3score.stats.precision_recall`** uses the standard any-ball definition of
  Kynkäänniemi et al. (2019); the research code tested only the nearest reference ball.
- **Inputs outside `[0, 1]`** are rejected.
- **Resizing keeps floating-point precision.** The research pipeline resized each 8-bit
  image with Pillow, which rounds the result back to whole grey levels. The package uses
  the same bilinear filter without that rounding. On two test pairs this moved fidelity by
  0.06% and 1.2% and left memorization and coverage unchanged, so scoring a PNG folder
  lands within about 1% of the research pipeline, not exactly on it.

## Scope and limitations

- The encoder was trained on radiology images. M3-Score is not validated for natural
  images, histopathology or other domains.
- FID is more repeatable than M3 at fixed sample sizes below 500.
- The axis depths are fixed by representational position, not optimised on test outcomes.
- Images are scored as 2D slices; there is no 3D feature space.

## Main findings from the paper

The reported experiments use subject-diverse BraTS references and generated radiology images.
The complete numerical record is in
[`docs/EXPERIMENT_FINDINGS.md`](https://github.com/Nishan-Charlie/M3-Score/blob/main/docs/EXPERIMENT_FINDINGS.md)
and [`results/`](https://github.com/Nishan-Charlie/M3-Score/tree/main/results).

- A DDPM obtains fidelity MMD² = **0.073** with a bootstrap 95% CI of **[0.071, 0.080]**, yet
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

### Real and generated radiology samples

![Real and generated samples](https://raw.githubusercontent.com/Nishan-Charlie/M3-Score/main/paper_release/Images/samples/real_vs_generated.png)

### Coverage and mode-drop validation

![Coverage validation](https://raw.githubusercontent.com/Nishan-Charlie/M3-Score/main/paper_release/Images/coverage_axis_validation.png)

### Domain-specific feature-space comparison

![OOD comparison](https://raw.githubusercontent.com/Nishan-Charlie/M3-Score/main/paper_release/Images/OOD/ood_roc_auc.png)

### Lesion-specific perturbation analysis

![Lesion specificity](https://raw.githubusercontent.com/Nishan-Charlie/M3-Score/main/paper_release/Images/lesion_specificity/lgg_noise_sweep.png)

## Reproducing the paper

The experiments were run with the research implementation in
[`evaluation/m3_score_v2.py`](https://github.com/Nishan-Charlie/M3-Score/blob/main/evaluation/m3_score_v2.py),
which is kept unchanged.

```bash
git clone https://github.com/Nishan-Charlie/M3-Score
cd M3-Score
pip install -r requirements.txt
python evaluation/eval_pipeline.py \
  --real_dir /path/to/real_images \
  --gen_dir /path/to/generated_images \
  --output_dir results/evaluation \
  --metrics fid kid ssim m3 alpha
```

With `--num_images N` that pipeline takes the first N files in sorted order, the selection
the reference-cohort finding above warns against. Point it at a folder that already holds
the sample you want.

## Repository map

```text
src/m3score/      The installable package
tests/            Package tests, including parity with the research code
evaluation/       Research implementation of M3-Score and baseline metrics
metrics/          Supplementary image-quality metrics
models/           Generator model definitions
data/             Dataset loading utilities
tools/            Audits, result tables, and analysis helpers
figures/          Publication figures and image assets
results/          Numeric reports, plots, and retained experiment logs
external/wdm-3d/  Vendored WDM-3D baseline
paper_release/    Supporting publication figures and retained logs
docs/             Academic project page and experiment records
```

## Reproducibility and data policy

This repository contains code, figures, result reports, and logs needed to inspect the paper's
claims. Medical datasets and model weights are not redistributed. Obtain them from their
original sources, verify their licenses and access conditions, and keep local copies in ignored
directories. See
[`RELEASE.md`](https://github.com/Nishan-Charlie/M3-Score/blob/main/RELEASE.md) for the
publication boundary.

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

MIT. See [`LICENSE`](https://github.com/Nishan-Charlie/M3-Score/blob/main/LICENSE). The
RadioDINO-s16 weights are downloaded from their publisher under CC-BY-4.0 and are not
redistributed here. Vendored WDM-3D retains its own license.
