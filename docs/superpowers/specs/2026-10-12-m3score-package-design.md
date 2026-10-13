# M3-Score as an installable package (`m3-score` 0.1.0): design

**Date:** 2026-10-12
**Status:** design approved in conversation; awaiting review of this written spec
**Next step after approval:** implementation plan

## 1. Goal

Someone who has never cloned this repository can run

```bash
pip install m3-score
m3score path/to/real path/to/generated
```

and get the three M3 axes for their own images, or call the same computation from
Python, including from inside a training loop, on 2D images, NIfTI volumes or DICOM.

### Success criteria

1. `pip install` of the built wheel into a clean environment gives a working `m3score`
   command and `import m3score`, with no repository checkout on the path.
2. On identical feature tensors the package returns the same fidelity, memorization and
   coverage values as the research code in `evaluation/m3_score_v2.py`, to 1e-6.
3. One-shot scoring, streaming `update()/compute()`, and the CLI give the same result
   for the same inputs and seed.
4. PNG/JPG/TIFF folders, torch tensors, numpy arrays, NIfTI volumes and DICOM files are
   all accepted.
5. No code path selects images by taking the head of a sorted file list.
6. The wheel and sdist build, pass `twine check`, and CI is green on Linux and Windows.

### Explicitly not a goal of this work

Uploading to PyPI. Claiming a name there is permanent and needs the maintainer's
account. This work ends with a verified wheel and a release workflow; section 12 lists
the maintainer's one-time steps.

## 2. Decisions taken

| Question | Decision |
| --- | --- |
| Usage modes in 0.1 | CLI on folders, Python API, streaming `update()/compute()`, NIfTI/DICOM volumes |
| Structure | New focused package in `src/m3score/`. `evaluation/m3_score_v2.py` is not modified |
| Default volume-to-slice rule | Central axial band matching the paper's slices, per-volume percentile scaling |
| Real-vs-real floor | Included, reported beside fidelity |
| Bootstrap interval | Corrected in the package; differs from the research code (section 4) |
| `stats.precision_recall` | Standard any-ball definition; differs from the research code (section 4) |
| `CITATION.cff` | Updated to the current title and repository URL |

## 3. Background that constrains the design

Read from the repository on 2026-10-11:

- The metric lives in one 1,258-line file importable only with the repository root on
  `sys.path`. There is no `pyproject.toml`.
- The README quick start calls `metric.compute(real, gen)`. That method does not exist.
- `compute_axes()` returns k-NN precision and recall on the coverage axis. The
  real-radius coverage the paper argues for is implemented as `_compute_coverage` and
  is not called by anything.
- `docs/REFERENCE_COHORT_FINDING.md`: selecting the first 500 files in sorted order
  chose 17 subjects from one site and gave M3 = 0.150 against 0.077 for a random draw.
  Its protocol (section 8) requires sampling by subject, reporting distinct subjects,
  and contextualising against real-vs-real distance.
- The script that sliced BraTS for the paper is not in the repository. Retained
  evidence: 38,781 slices / 1,251 subjects = 31 per subject, filenames running
  consecutively from `slice055`, 240x240, anterior at the top of the image.
- Default backbone `Snarcy/RadioDino-s16`: timm ViT-S/16, 12 blocks, 384-d, CC-BY-4.0,
  not gated, Hub revision `773d28d873455153495d39179fc42ea816aa1212`.
- `m3-score` and `m3score` were unclaimed on PyPI on 2026-10-11.
- The real BraTS reference set is not on the development machine. Generated sets are
  (`output/generated_500_best`, `generated_retinal`, `generated_cxr_corrupted`, 500
  PNGs each).

## 4. The `m3-v1` protocol

"M3-Score" must mean one computation so that published values are comparable. Every
constant below lives in `src/m3score/protocol.py` and every result carries the tag
`m3-v1`.

| Constant | Value |
| --- | --- |
| Backbone | `Snarcy/RadioDino-s16` via timm, `img_size=224`, pinned to revision `773d28d873455153495d39179fc42ea816aa1212` |
| Input | float32 in `[0, 1]`, 3 channels (grayscale repeated) |
| Resize | to 224x224, bilinear, antialiased, in torch (`F.interpolate`); no crop; skipped when already 224x224 |
| Normalisation | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |
| Feature | CLS token after transformer block L (after `patch_embed`, position embedding, `norm_pre`; before the final `norm`) |
| Layers | fidelity 12, memorization 9, coverage 4 |

All input types go through the same resize, so a folder and the same images passed as
a tensor score identically.

### Fidelity (block 12)

Unbiased MMD² with a mixture of five Gaussian RBF kernels on L2-normalised features.
Bandwidths are `s * m` for `s` in `(0.25, 0.5, 1, 2, 4)`, where `m` is the median of
the pairwise squared distances over the pooled real and generated sample (upper
triangle, `torch.median`, floored at 1e-10). The five estimates are averaged and the
result clamped at 0.

Statistics, all seeded and all skipped when `stats=False`:

- **Bootstrap 95% CI.** 200 replicates, real and generated resampled independently
  with replacement, percentiles 2.5 and 97.5, lower bound clamped at 0. The kernel is
  the one used for the point estimate (bandwidths are not re-estimated per replicate),
  and within-set pairs that are two copies of the same original image are left out of
  the within-set averages. This keeps the bootstrap distribution centred on the point
  estimate.
- **Permutation test.** 200 label permutations of the pooled sample. Reports the
  z-score against the null and `p = (1 + b) / (1 + B)`.
- **Real-vs-real floor.** 10 random splits of the real set into two halves. When
  subjects are known the split is by subject, so the halves are subject-disjoint;
  otherwise by image. Each split gives one unclamped MMD². Reported as
  `fidelity_floor = max(mean, 0)` and `fidelity_floor_std` (sample standard deviation).

### Memorization (block 9)

On raw features. For each generated image, the Euclidean distance to its nearest real
image. Threshold: the median (`torch.median`) of each real image's distance to its
nearest other real image. `memorization` is the fraction of generated images below the
threshold. `memorization_nn_dist` is the mean generated-to-real nearest distance.

### Coverage (block 4)

On raw features, k = 5. Each real image's radius is its distance to its 5th nearest
other real image. `coverage` is the fraction of real images with at least one
generated image within their radius (Naeem et al. 2020).

### Differences from `compute_axes()` in the research code

1. The coverage axis returns real-radius coverage. k-NN precision and recall are not in
   the default result.
2. `m3score.stats.precision_recall` is offered as a side utility and uses the standard
   definition (Kynkäänniemi et al. 2019): a point counts when it lies inside the ball
   of *any* reference point. The research code's `_manifold_fraction` tests only the
   ball of the *nearest* reference point and returns lower values (on a synthetic
   check, precision 0.84 against 0.97 and recall 0.36 against 0.53).
3. The permutation p-value is `(1 + b) / (1 + B)`. The research code reports `b / B`,
   which can be exactly 0.
4. Inputs outside `[0, 1]` raise `ValueError`. The research code scores them.
5. The real-vs-real floor is new.
6. The bootstrap interval is corrected. The research code recomputes the statistic on
   each resample as if its points were distinct; duplicated images then count as
   off-diagonal pairs at the kernel's maximum, which shifts the interval upward. On a
   synthetic check the point estimate was 0.0050 and the research interval was
   [0.0067, 0.0100]. The fidelity value itself is unaffected. Intervals from the
   package therefore differ from the paper's [0.071, 0.080], and the README says so.
7. Resizing keeps floating-point precision. The research pipeline resizes each 8-bit
   image with Pillow, which rounds the result back to whole grey levels; the package's
   resize is numerically the same bilinear filter without that rounding (a Pillow
   resize in float mode gives the identical fidelity to six digits). On two local
   pairs this moved fidelity by -0.06% and -1.19% and left memorization and coverage
   unchanged. Scoring a PNG folder therefore lands within about 1% of the research
   pipeline, not on it. Not measured on real BraTS data.

### Behaviour preserved on purpose, for parity with the paper

- The MMD² estimate is clamped at 0.
- `torch.median` returns the lower of the two middle values for an even count.

### Other backbones

`M3Score(backbone=..., layers=(f, m, c))` accepts a timm model name or an `nn.Module`
exposing `patch_embed`, `blocks` and a CLS token. `layers` is then mandatory: depths
are not mapped automatically, because the reference-cohort protocol says to audit
read-out depth per encoder. Passing either `backbone` or `layers` makes the result
carry `protocol = "custom"` and a warning; only the defaults produce `m3-v1`.
Backbone families that are not timm ViTs are out of scope.

## 5. Package layout

```text
pyproject.toml
src/m3score/
  __init__.py     public names: M3Score, M3Result, score, M3ScoreWarning, __version__
  protocol.py     frozen constants and PROTOCOL_VERSION
  backbone.py     load the ViT; extract CLS features at requested blocks, batched
  stats.py        pure functions on feature tensors
  metric.py       M3Score, M3Result, score()
  cli.py          argparse entry point
  py.typed
  io/
    __init__.py   open_source(), ImageSource
    arrays.py     torch tensors and numpy arrays; value and shape validation
    images.py     2D image files
    nifti.py      NIfTI volumes (needs nibabel)
    dicom.py      DICOM files and series (needs pydicom)
    sampling.py   seeded random and subject-balanced index sampling
tests/
.github/workflows/ci.yml
.github/workflows/publish.yml
```

Each module has one job and can be tested alone: `stats.py` knows nothing about images
or models, `backbone.py` nothing about files, `io/` nothing about the metric.

### Names and dependencies

- Distribution `m3-score`, import `m3score`, command `m3score`, version `0.1.0`.
- Build backend hatchling. Python `>=3.10`. Licence MIT.
- Core: `torch>=2.0`, `timm>=1.0`, `numpy>=1.23`, `pillow>=9.1`, `tqdm>=4.60`.
- Extras: `nifti` = `nibabel>=4.0`; `dicom` = `pydicom>=2.4`; `all` = both;
  `dev` = `all` plus `pytest`, `ruff`, `build`, `twine`.
- `requirements.txt` is unchanged and remains the environment for the paper's
  experiments.

## 6. Python API

```python
import m3score

# one-shot: folders, tensors or arrays
r = m3score.score("real/", "gen/", n=2000, seed=0)
print(r.fidelity, r.memorization, r.coverage)
print(r)                       # formatted table
r.to_dict()                    # JSON-serialisable

# streaming
m = m3score.M3Score(device="cuda")
for x in real_loader:
    m.update(x, real=True)
for x in gen_loader:
    m.update(x, real=False)
r = m.compute()
m.reset(keep_real=True)        # next epoch reuses the real features
```

### `M3Score`

```python
M3Score(device=None, *, backbone=None, layers=None, batch_size=32, seed=0, progress=False)
  .update(images, *, real: bool, groups=None) -> None
  .compute(*, stats=True) -> M3Result
  .reset(*, keep_real=False) -> None
  .score(real, gen, *, n=None, stats=True, include=None, group_pattern=None,
         slice_band=None, window=None) -> M3Result
```

- A plain class, not an `nn.Module` and not a `torchmetrics.Metric`. It uses the same
  method names so it fits the same call sites without the dependency.
- `device=None` selects CUDA when available, otherwise CPU.
- The backbone loads on first use, so construction and `--help` do not download.
- `update` extracts the three feature layers for the batch and stores them on the CPU
  (about 4.6 kB per image). `groups` is an optional sequence of subject identifiers,
  one per image.
- `score` resets the metric, then runs both sources through `update` and returns
  `compute()`. Module-level `m3score.score(...)` builds a temporary `M3Score` and takes
  the constructor's keyword arguments as well.
- One `seed` drives sampling, bootstrap, permutation and floor. The generated set is
  sampled with `seed + 1` so that two identically structured folders are not sampled
  in lockstep.

### `M3Result`

A frozen dataclass.

| Field | Type | Meaning |
| --- | --- | --- |
| `fidelity` | float | MMD² at block 12; lower is better |
| `fidelity_ci` | (float, float) or None | bootstrap 95% interval |
| `fidelity_p`, `fidelity_z` | float or None | permutation test |
| `fidelity_floor`, `fidelity_floor_std` | float or None | real-vs-real floor |
| `memorization` | float | near-duplicate rate at block 9; lower is better |
| `memorization_nn_dist` | float | mean generated-to-real nearest distance |
| `coverage` | float | real-radius coverage at block 4; higher is better |
| `n_real`, `n_gen` | int | images scored |
| `n_real_subjects`, `n_gen_subjects` | int or None | distinct subjects when known |
| `protocol` | str | `"m3-v1"` or `"custom"` |
| `backbone`, `layers` | str, (int, int, int) | what was actually used |
| `warnings` | tuple of str | every warning raised while scoring |

There is no single `m3_score` scalar. The README's position is that quality is not
collapsed into one number.

### Array inputs

- Shape `(N, H, W)`, `(N, 1, H, W)` or `(N, 3, H, W)`, as a torch tensor or numpy array.
- `uint8` is divided by 255. Floating point must be finite and inside `[0, 1]`
  (tolerance 1e-6); otherwise `ValueError` reporting the observed range and noting that
  `[-1, 1]` outputs need `(x + 1) / 2`. Other dtypes are rejected with a message.

## 7. Files, volumes and sampling

`io.open_source(source, ...)` returns an `ImageSource`: a length, a `load(i)` returning
one `(C, H, W)` float tensor in `[0, 1]`, and `groups` (one subject identifier per
item, or `None`). A folder is scanned, the index is sampled, and only the sampled items
are decoded, in batches. Decoding is single-process.

### Folder scanning

Recursive. Each file is classified by type:

- **2D images:** `.png .jpg .jpeg .tif .tiff .bmp`. 8-bit modes go through
  `convert("RGB")` and are divided by 255, as in the research code. 16-bit modes are
  divided by 65535.
- **NIfTI:** `.nii`, `.nii.gz`.
- **DICOM:** `.dcm`, `.dicom`, `.ima`, or no extension with `DICM` at byte 128.

`include` (CLI `--include GLOB`) restricts the scan to relative paths matching a glob.
It is needed for datasets such as BraTS where one subject folder holds several
modalities and a segmentation: `--include "*_flair.nii.gz"`.

### Subjects

| Source | Subject identifier |
| --- | --- |
| NIfTI | the file |
| DICOM | `PatientID`, else `StudyInstanceUID`, else the parent directory |
| 2D images | unknown |
| any, with `group_pattern` | the first capture group of the regex applied to the relative path |

`group_pattern` (CLI `--group-pattern`) overrides the default for every source type,
for example `"(BraTS2021_\d+)"`.

### Sampling

- `n=None` uses every item up to 5,000 per set. An explicit `n` larger than the set
  uses the whole set.
- When a subset is drawn it is always seeded and random.
- Subjects known: shuffle the subjects, shuffle the items within each subject, then
  take one item per subject in rotation until `n` is reached. The sample is balanced
  across subjects and maximally subject-diverse.
- Subjects unknown: uniform without replacement.

### NIfTI

1. Load with nibabel and reorient to the closest canonical (RAS+) axes, so stored
   orientation does not matter. Only 3D volumes; a 4D file raises an error naming it.
2. Intensity, per volume: with `window=(lo, hi)`, clip to the window and scale to
   `[0, 1]`. Otherwise take the 0.5 and 99.5 percentiles of the non-zero voxels, clip
   and scale. An all-zero or constant volume is skipped with a warning.
3. Slices: along the inferior-superior axis, indices `floor(lo * D + 0.5)` up to but
   excluding `floor(hi * D + 0.5)`, with the default band `(55/155, 86/155)`. For
   D = 155 this is slices 55 to 85, 31 slices.
4. Display orientation: anterior at the top, patient right on the image left
   (radiological). For BraTS-native files this equals the common `volume[:, :, k].T`
   export.

Up to two decoded volumes are cached, and sampled indices are processed in file order
so each volume is read once.

### DICOM

Headers are read first to group files by `SeriesInstanceUID`.

- A series with two or more instances carrying `ImagePositionPatient` and
  `ImageOrientationPatient` is a **volume**: slices are sorted along the slice normal
  (falling back to `InstanceNumber`), and the same band rule and per-volume intensity
  rule apply. A multi-frame instance is a volume in stored frame order.
- Any other instance is a **2D image**, scaled on its own.
- Pixel pipeline: rescale slope and intercept; then an explicit `window` if given,
  otherwise the stored `WindowCenter`/`WindowWidth` (first value), otherwise the
  percentile rule; `MONOCHROME1` is inverted.
- Pixel data is used in its stored display orientation.
- 8-bit RGB is supported. Other colour encodings are skipped with a warning.
- Compressed transfer syntaxes need a pydicom decoding plugin installed by the user;
  the error message says which.

### Recorded assumptions

- The band 55 to 85 is inferred from retained filenames, not read from the original
  slicing script.
- The left/right convention of the paper's PNGs cannot be determined from the images.
  Radiological display is assumed.
- Per-volume percentile scaling was chosen in review; the paper's original scaling is
  unknown.
- The README will state that a generator trained on slices exported by another
  pipeline should be scored against a real folder exported by that same pipeline.

## 8. Warnings and errors

Warnings are raised as `M3ScoreWarning` and also collected in `M3Result.warnings`:

- Fewer than 500 images in either set: estimates are less repeatable.
- Real set with known subjects and fewer than 50 of them: low subject diversity
  inflates distances.
- Real set sampled as a subset with subjects unknown: subject balance could not be
  enforced; suggests `group_pattern`.
- Custom protocol.
- Any skipped file or volume, with the reason.
- Floor computed by image because only one subject is known.

Errors:

- Fewer than 6 real images or fewer than 2 generated: `ValueError`.
- No readable images under a path: `FileNotFoundError` listing supported types and the
  `include` pattern in effect.
- NIfTI or DICOM input without the extra installed: `ImportError` giving the exact
  `pip install "m3-score[nifti]"` or `[dicom]` command.
- Out-of-range, non-finite or wrongly shaped arrays: `ValueError` as in section 6.

Library code does not print. Progress uses `tqdm` when `progress=True`; diagnostics go
to the `m3score` logger.

## 9. Command line

```text
m3score REAL GEN [--n N] [--seed S] [--device auto|cpu|cuda] [--batch-size B]
                 [--include GLOB] [--group-pattern REGEX]
                 [--slice-band LO HI] [--window LO HI]
                 [--no-stats] [--json FILE] [--quiet] [--version]
```

Output (values illustrative):

```text
M3-Score  protocol m3-v1  (m3-score 0.1.0)
  real       2,000 images, 412 subjects
  generated  2,000 images

  axis           value    better   detail
  fidelity       0.0731   lower    95% CI [0.0712, 0.0803]; real-vs-real floor 0.0015 +/- 0.0004
  memorization   0.012    lower    mean nearest-real distance 14.2
  coverage       0.380    higher   k = 5

  permutation test: z = 347.2, p = 0.005 (descriptive; rejects for every generator tested)
```

`--json FILE` writes `M3Result.to_dict()` plus the package version. `--quiet`
suppresses progress bars. Exit status is 0 on success, 2 for usage or input errors,
1 for anything else.

## 10. Performance envelope

Cost is quadratic in the number of images. At the default cap of 5,000 per set the
pooled distance matrix is 10,000 x 10,000 float32 (400 MB), within an 8 GB GPU.
The permutation null and the bootstrap are both computed from one pooled kernel matrix
held in float64. For the permutation test this is exact, because the pooled median,
and so the bandwidths, do not change under permutation; results match recomputation to
1e-5. Each replicate is then a matrix-vector product. Measured on CPU: statistics for
2,000 images per set take about a minute with the slower per-replicate recomputation,
so the pooled form is well inside that. Feature extraction dominates on CPU; the
README recommends `--n 1000` for a quick estimate.

## 11. Testing

All tests run offline using a randomly initialised 12-block timm ViT injected as the
backbone.

- **`stats`:** MMD² against a naive double-loop implementation; identical sets give 0;
  symmetry. Coverage and memorization on hand-built geometric cases. Property test:
  removing generated samples never increases coverage. Permutation via the pooled
  kernel equals recomputation. Floor splits are subject-disjoint when subjects are
  given.
- **Parity with the research code:** `evaluation/m3_score_v2.py` is imported unmodified
  and its `_compute_mmd2_rbf`, `_compute_memorization`, `_compute_coverage` and
  `_permutation_test_fidelity` are called on the same random features: values agree to
  1e-6 (z-score to relative 1e-3). Feature extraction agrees to 1e-5 on the same tiny
  ViT. Skipped when the research file is absent, as in an sdist. Precision/recall and
  the bootstrap are excluded because the package deliberately differs (section 4).
- **Bootstrap:** agrees with a slow reference that builds each resample explicitly and
  masks same-original pairs; the interval contains the point estimate in the synthetic
  case where the research procedure's interval does not.
- **Precision/recall:** a hand-built case where a point is outside its nearest
  reference ball but inside a farther one counts as inside.
- **`metric`:** streaming equals one-shot; independence from batch size; same seed
  gives the same result; `reset(keep_real=True)` keeps real features; input validation.
- **`io`:** synthetic 8-bit, 16-bit and RGB PNGs; NIfTI written in two different stored
  orientations yields identical slices; band arithmetic for D = 155 gives 55 to 85;
  DICOM CT series with rescale and window, a `MONOCHROME1` image, and slice ordering by
  position; missing-extra error messages.
- **Sampling:** with 100 subjects named in sorted order, a sample of 100 spans 100
  subjects, and no sample equals the head of the sorted list; determinism by seed.
- **CLI:** table and JSON output; exit codes.
- **Network-marked:** the real backbone loads at the pinned revision, and two fixed
  synthetic image sets reproduce stored golden values. Run locally and on demand.

Acceptance check run once by hand before release, recorded in the final report:

- Package against the research pipeline (PIL resize) end to end on the local
  500-image generated sets: fidelity within 2% relative, memorization and coverage
  within 0.02 absolute. A larger gap is reported to the maintainer, not tuned away.
  The fidelity tolerance was first set at 1%. The measured gaps were -0.06% and
  -1.19%, traced entirely to the research pipeline rounding resized images to 8 bits
  (section 4, item 7). The maintainer chose to keep the float resize and accept the
  gap, and the tolerance was set to 2% on that decision.
- The wheel installed into a clean virtual environment, `m3score` run from a directory
  outside the repository.

Not verifiable on the development machine: the paper's headline values (fidelity 0.073,
coverage 38%), because the BraTS reference set is not present.

## 12. CI, release and documentation

- **`ci.yml`:** ruff and pytest (`-m "not network"`) on Python 3.10 to 3.13, Ubuntu
  and Windows, CPU torch.
- **`publish.yml`:** on a published GitHub release: build sdist and wheel,
  `twine check`, upload with PyPI trusted publishing (OIDC, environment `pypi`, no
  stored token).
- **Maintainer's one-time steps**, written into `RELEASE.md`: create a PyPI account;
  add a pending trusted publisher for project `m3-score`, repository
  `Nishan-Charlie/M3-Score`, workflow `publish.yml`, environment `pypi`; publish a
  GitHub release tagged `v0.1.0`.
- **README:** install, CLI, Python, streaming, volumes, building a valid reference set,
  reading the output, reproducing the paper. The quick start is corrected. Image links
  become absolute so the PyPI page renders them.
- **`CITATION.cff`:** title updated to "M3-Score: Fidelity, Memorization and Coverage
  as Separate Axes for Evaluating Generative Radiology Image Models"; repository URL
  updated to `https://github.com/Nishan-Charlie/M3-Score`.
- **`docs/index.html`:** an install snippet.

## 13. Order of work

1. Core: `protocol`, `backbone`, `stats`, `metric`, array and 2D-image inputs,
   sampling, CLI, `pyproject.toml`. Installable and usable at the end of this phase.
2. Volumes: NIfTI, then DICOM.
3. CI, release workflow, README, `CITATION.cff`, landing page, the two acceptance
   checks.

## 14. Out of scope for 0.1

- Synchronising accumulated features across processes or GPUs.
- Saving and loading reference-feature files.
- Backbones that are not timm ViTs.
- The layer-weighting scheme and CKA pruning in the research code.
- Multi-process decoding.
- conda-forge.
