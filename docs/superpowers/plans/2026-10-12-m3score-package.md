# M3-Score Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `m3-score` 0.1.0: a pip-installable package with a `m3score REAL GEN` command and a Python API (one-shot and streaming) that computes the three M3 axes on 2D images, NIfTI volumes and DICOM.

**Architecture:** A new package in `src/m3score/` implements only the published three-axis path as small single-purpose modules: `protocol` (frozen constants), `stats` (pure functions on feature tensors), `backbone` (preprocessing and CLS extraction), `io/` (sources and sampling), `metric` (accumulate then compute), `cli`. The research file `evaluation/m3_score_v2.py` is never modified; parity tests import it and assert identical numbers.

**Tech Stack:** Python >= 3.10, PyTorch, timm, numpy, Pillow, tqdm; optional nibabel and pydicom; hatchling build backend; pytest; ruff; GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-12-m3score-package-design.md`

## Global Constraints

- **Do not modify `evaluation/m3_score_v2.py`** or anything else under `evaluation/`, `metrics/`, `models/`, `data/`, `utils/`, `tools/` (new files in `tools/` are allowed), `results/`, `figures/`, `paper_release/`.
- Distribution name `m3-score`, import name `m3score`, command `m3score`, version `0.1.0`.
- `requires-python = ">=3.10"`. Core dependencies exactly: `torch>=2.0`, `timm>=1.0`, `numpy>=1.23`, `pillow>=9.1`, `tqdm>=4.60`. Extras: `nifti` = `nibabel>=4.0`, `dicom` = `pydicom>=2.4`, `all`, `dev`.
- Protocol tag `m3-v1`. Backbone `Snarcy/RadioDino-s16` at revision `773d28d873455153495d39179fc42ea816aa1212`. Layers fidelity 12, memorization 9, coverage 4. Image size 224. ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- Library code never calls `print`. Progress goes through `tqdm` only when `progress=True`. `cli.py` is the only module that prints.
- Every user-facing string (CLI output, error and warning messages) is ASCII. Write `MMD^2`, not a superscript.
- No code path may select images by taking the head of a sorted list. Subsets are always a seeded random draw.
- `requirements.txt` is not changed.
- **Protect the local environment.** The development machine has `torch 2.6.0+cu124` pinned by hand. Install the package with `pip install -e . --no-deps`, and install `nibabel`, `pydicom`, `hatchling` individually. Never run a bare `pip install -e ".[dev]"` on this machine; that form is for CI only.
- **Do not upload to PyPI or TestPyPI.** Do not push. Do not create a GitHub release.
- **Lint hygiene.** The code in this plan was written to the 100-column limit but has not been machine-checked. Before every `ruff check src tests`, run `ruff format src tests` and `ruff check --fix src tests`. If E501 (line too long) remains, wrap the line or split the string; never raise the limit and never add `noqa` for it.
- **If a test in this plan fails for a reason in the test itself** (a fixture detail, a tolerance on a randomly initialised network, a library version difference), fix the test and say so in the task report. If it fails because the implementation disagrees with the spec, fix the implementation. Never weaken an assertion that pins a spec requirement, parity with the research code, or a Review Focus item.
- **Commits:** work on branch `feat/pip-package`. Run the `Commit` steps only if the maintainer approved committing at handoff; otherwise skip every `Commit` step and leave the changes in the working tree.
- Run every command from the repository root.

## Review Focus

Inputs and conditions the spec implies but does not spell out, most likely to bite a real user first. Each has a pinning test in the task named.

1. **`--group-pattern` fits the real filenames but not the generated ones** (generated files are usually `generated_0000.png`). Expected: the run succeeds, real subjects are counted, generated subjects are unknown. Pinned in Task 6 (`test_optional_group_pattern_that_matches_nothing_leaves_groups_unknown`) and Task 7 (`test_group_pattern_need_not_match_the_generated_set`).
2. **Images of different sizes**, across the two sets (240x240 real against 256x256 generated) and within one folder, including non-square. Expected: each image is resized on its own; nothing fails to stack. Pinned in Task 7 (`test_mixed_image_sizes_are_scored`).
3. **Degenerate comparisons a user will try as a sanity check:** the same folder as both real and generated, or a collapsed generator emitting blank images. Expected: finite values, fidelity 0 and coverage 1 for identical sets, no NaN, no division by zero. Pinned in Task 3 (`test_statistics_stay_finite_on_degenerate_features`) and Task 7 (`test_same_images_as_real_and_generated`).
4. **Environment failures on first run:** `--device cuda` on a CPU-only machine, or no route to the Hugging Face Hub. Expected: a one-line error naming the fix, not a traceback. Pinned in Task 4 (`test_backbone_download_failure_explains_itself`), Task 7 (`test_cuda_request_without_cuda_is_a_clear_error`) and Task 8 (`test_backbone_failure_exits_1_with_one_line`).
5. **Very unequal set sizes, or `--n` larger than one of the sets.** Expected: each set contributes what it has; no index error. Pinned in Task 7 (`test_unequal_set_sizes_and_oversized_n`).

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, dependencies, entry point, pytest and ruff configuration |
| `src/m3score/__init__.py` | Public names only |
| `src/m3score/_version.py` | `__version__` from installed metadata |
| `src/m3score/_errors.py` | `M3ScoreError`, `M3ScoreWarning` |
| `src/m3score/protocol.py` | Every frozen constant of `m3-v1` |
| `src/m3score/stats.py` | Pure functions on `(N, D)` feature tensors |
| `src/m3score/backbone.py` | `Backbone` (prepare, extract) and `load_backbone` |
| `src/m3score/io/base.py` | `ImageSource`, `ItemSource`, `SkippedItem` |
| `src/m3score/io/arrays.py` | In-memory tensors and arrays: validation, `ArraySource` |
| `src/m3score/io/sampling.py` | `sample_indices` |
| `src/m3score/io/images.py` | 2D image files |
| `src/m3score/io/volumes.py` | Slice band and intensity scaling shared by NIfTI and DICOM |
| `src/m3score/io/nifti.py` | NIfTI volumes |
| `src/m3score/io/dicom.py` | DICOM files and series |
| `src/m3score/io/__init__.py` | `open_source` and folder scanning |
| `src/m3score/metric.py` | `M3Score`, `M3Result`, `score` |
| `src/m3score/cli.py`, `__main__.py` | Command line |
| `tests/` | One test module per source module, plus `test_parity.py` and `test_golden.py` |
| `tools/check_package_parity.py` | End-to-end comparison with the research pipeline on two folders |
| `tools/make_golden.py` | Regenerates `tests/golden_m3_v1.json` |
| `.github/workflows/ci.yml`, `publish.yml` | CI and release |

Phases: Tasks 1-8 give an installable package for 2D images. Tasks 9-10 add volumes. Tasks 11-13 are verification, release tooling and documentation.

---

## Phase 1: core package

### Task 1: Package skeleton and protocol constants

**Files:**
- Create: `pyproject.toml`, `src/m3score/__init__.py`, `src/m3score/_version.py`, `src/m3score/_errors.py`, `src/m3score/protocol.py`, `src/m3score/py.typed`
- Modify: `.gitignore` (append)
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `m3score.__version__: str`; `m3score.M3ScoreError(RuntimeError)`; `m3score.M3ScoreWarning(UserWarning)`; module `m3score.protocol` with the constants listed in Step 3.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b feat/pip-package
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_protocol.py`:

```python
import m3score
from m3score import protocol


def test_version_is_exposed():
    assert m3score.__version__ == "0.1.0"


def test_protocol_constants_are_the_published_ones():
    assert protocol.PROTOCOL_VERSION == "m3-v1"
    assert protocol.BACKBONE_ID == "Snarcy/RadioDino-s16"
    assert protocol.BACKBONE_REVISION == "773d28d873455153495d39179fc42ea816aa1212"
    assert protocol.IMAGE_SIZE == 224
    assert protocol.LAYERS == (12, 9, 4)
    assert (protocol.FIDELITY_LAYER, protocol.MEMORIZATION_LAYER, protocol.COVERAGE_LAYER) == (12, 9, 4)
    assert protocol.RBF_BANDWIDTH_SCALES == (0.25, 0.5, 1.0, 2.0, 4.0)
    assert protocol.COVERAGE_K == 5
    assert protocol.MIN_REAL_IMAGES == 6


def test_errors_and_warnings_have_the_documented_bases():
    assert issubclass(m3score.M3ScoreWarning, UserWarning)
    assert issubclass(m3score.M3ScoreError, RuntimeError)
```

Run: `pytest tests/test_protocol.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'm3score'`.

- [ ] **Step 3: Write the package skeleton**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "m3-score"
version = "0.1.0"
description = "M3-Score: fidelity, memorization and coverage as separate axes for evaluating generative radiology image models"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Nishankar Sathiyamohan" }]
keywords = [
  "generative models",
  "evaluation metric",
  "medical imaging",
  "radiology",
  "maximum mean discrepancy",
  "diffusion models",
]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Science/Research",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Topic :: Scientific/Engineering :: Medical Science Apps.",
  "Typing :: Typed",
]
dependencies = [
  "torch>=2.0",
  "timm>=1.0",
  "numpy>=1.23",
  "pillow>=9.1",
  "tqdm>=4.60",
]

[project.optional-dependencies]
nifti = ["nibabel>=4.0"]
dicom = ["pydicom>=2.4"]
all = ["m3-score[nifti,dicom]"]
dev = ["m3-score[all]", "pytest>=7", "ruff>=0.4", "build>=1.0", "twine>=5"]

[project.urls]
Homepage = "https://nishan-charlie.github.io/M3-Score/"
Repository = "https://github.com/Nishan-Charlie/M3-Score"
Issues = "https://github.com/Nishan-Charlie/M3-Score/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/m3score"]

[tool.hatch.build.targets.sdist]
include = ["/src", "/tests", "/README.md", "/LICENSE", "/CITATION.cff", "/pyproject.toml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = ["network: needs the real backbone weights (Hugging Face Hub or a local cache)"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Create `src/m3score/_version.py`:

```python
"""The installed distribution's version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("m3-score")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
```

Create `src/m3score/_errors.py`:

```python
"""Exception and warning types raised by m3score."""


class M3ScoreError(RuntimeError):
    """Scoring could not proceed for a reason outside the caller's arguments.

    Examples: the backbone weights cannot be loaded, or pixel data cannot be decoded.
    """


class M3ScoreWarning(UserWarning):
    """Something about the inputs weakens the result without invalidating it."""
```

Create `src/m3score/protocol.py`:

```python
"""Frozen constants of the m3-v1 protocol.

Changing any value here changes what an M3-Score means, so it requires a new
PROTOCOL_VERSION.
"""

PROTOCOL_VERSION = "m3-v1"
CUSTOM_PROTOCOL = "custom"

# Encoder
BACKBONE_ID = "Snarcy/RadioDino-s16"
BACKBONE_REVISION = "773d28d873455153495d39179fc42ea816aa1212"
IMAGE_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Read-out depth of each axis, counted from 1
FIDELITY_LAYER = 12
MEMORIZATION_LAYER = 9
COVERAGE_LAYER = 4
LAYERS = (FIDELITY_LAYER, MEMORIZATION_LAYER, COVERAGE_LAYER)

# Statistics
RBF_BANDWIDTH_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
COVERAGE_K = 5
N_BOOTSTRAP = 200
N_PERMUTATIONS = 200
CI_ALPHA = 0.05
FLOOR_SPLITS = 10

# Volumes
SLICE_BAND = (55 / 155, 86 / 155)
INTENSITY_PERCENTILES = (0.5, 99.5)

# Sample sizes
MAX_IMAGES = 5000
MIN_IMAGES_WARN = 500
MIN_SUBJECTS_WARN = 50
MIN_REAL_IMAGES = COVERAGE_K + 1
MIN_GEN_IMAGES = 2
```

Create `src/m3score/__init__.py`:

```python
"""M3-Score: fidelity, memorization and coverage for generative radiology image models."""

from ._errors import M3ScoreError, M3ScoreWarning
from ._version import __version__
from .protocol import PROTOCOL_VERSION

__all__ = ["M3ScoreError", "M3ScoreWarning", "PROTOCOL_VERSION", "__version__"]
```

Create `src/m3score/py.typed` as an empty file.

Append to `.gitignore`:

```gitignore

# ─── Python packaging ───
dist/
build/
*.egg-info/
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 4: Install in editable mode without touching dependencies**

Run: `pip install -e . --no-deps`
Expected: ends with `Successfully installed m3-score-0.1.0`. pip downloads hatchling into an isolated build environment; the local torch is untouched.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_protocol.py -v`
Expected: 3 passed.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/m3score tests/test_protocol.py
git commit -m "feat: add m3score package skeleton and m3-v1 protocol constants"
```

---

### Task 2: Core statistics (MMD^2, memorization, coverage, precision/recall)

**Files:**
- Create: `src/m3score/stats.py`, `tests/_legacy.py`, `tests/test_stats.py`, `tests/test_parity.py`

**Interfaces:**
- Consumes: `m3score.protocol.RBF_BANDWIDTH_SCALES`, `COVERAGE_K`.
- Produces, all taking `(N, D)` float tensors:
  - `stats.mmd2_rbf(real, gen, *, clamp=True) -> float`
  - `stats.memorization(real, gen) -> tuple[float, float]` returning `(rate, mean_nearest_real_distance)`
  - `stats.coverage(real, gen, k=5) -> float`
  - `stats.precision_recall(real, gen, k=5) -> tuple[float, float]`
  - private helpers reused by Task 3: `stats._l2_normalise(x)`, `stats._sq_dist(a, b)`, `stats._median_bandwidth(d)`
- Test helper: `tests/_legacy.py::load_legacy_metric()` returns a research-code metric instance on the CPU with no backbone, or skips the test when the research file is absent.

- [ ] **Step 1: Write the failing tests**

Create `tests/_legacy.py`:

```python
"""Access to the frozen research implementation, for parity tests."""

import sys
from pathlib import Path

import pytest
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]


def load_legacy_module():
    """Import evaluation/m3_score_v2.py unmodified, or skip when it is not present."""
    if not (_ROOT / "evaluation" / "m3_score_v2.py").is_file():
        pytest.skip("research implementation not present (sdist or installed copy)")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from evaluation import m3_score_v2

    return m3_score_v2


def load_legacy_metric():
    """A research-code metric on the CPU, built without loading any backbone."""
    module = load_legacy_module()
    metric = module.M3EntropyMetric.__new__(module.M3EntropyMetric)
    nn.Module.__init__(metric)
    metric.device = "cpu"
    return metric
```

Create `tests/test_stats.py`:

```python
import numpy as np
import pytest
import torch

from m3score import stats


def _features(seed, n, d=16, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) + shift


def _naive_mmd2(x, y):
    """Float64 double-loop reference for the unbiased multi-bandwidth RBF MMD^2."""
    x = x.double().numpy()
    y = y.double().numpy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    joint = np.concatenate([x, y])
    d = ((joint[:, None, :] - joint[None, :, :]) ** 2).sum(-1)
    upper = np.sort(d[np.triu_indices(len(joint), k=1)])
    median = upper[(len(upper) - 1) // 2]  # lower median, as torch.median returns

    def k(a, b, sigma2):
        return np.exp(-((a - b) ** 2).sum() / (2.0 * sigma2))

    m, n = len(x), len(y)
    total = 0.0
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        s2 = median * scale
        rr = sum(k(x[i], x[j], s2) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
        gg = sum(k(y[i], y[j], s2) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
        rg = sum(k(x[i], y[j], s2) for i in range(m) for j in range(n)) / (m * n)
        total += rr + gg - 2.0 * rg
    return total / 5.0


def test_mmd_matches_a_naive_reference():
    real, gen = _features(0, 12), _features(1, 9, shift=0.4)
    assert stats.mmd2_rbf(real, gen) == pytest.approx(_naive_mmd2(real, gen), rel=1e-3, abs=1e-5)


def test_mmd_of_a_set_with_itself_is_zero_when_clamped_and_negative_when_not():
    x = _features(2, 30)
    assert stats.mmd2_rbf(x, x) == 0.0
    assert stats.mmd2_rbf(x, x, clamp=False) < 0.0


def test_mmd_is_symmetric():
    real, gen = _features(3, 20), _features(4, 25, shift=0.3)
    assert stats.mmd2_rbf(real, gen) == pytest.approx(stats.mmd2_rbf(gen, real), abs=1e-6)


def test_mmd_grows_with_distribution_shift():
    real = _features(5, 80)
    near, far = _features(6, 80, shift=0.1), _features(6, 80, shift=1.0)
    assert stats.mmd2_rbf(real, far) > stats.mmd2_rbf(real, near)


def test_mmd_needs_two_samples_per_set():
    with pytest.raises(ValueError, match="at least 2"):
        stats.mmd2_rbf(_features(0, 1), _features(1, 5))


def _line(*values):
    return torch.tensor([[float(v), 0.0] for v in values])


def test_coverage_on_a_hand_built_line():
    real = _line(*range(10))  # every real point has its nearest neighbour at distance 1
    gen = _line(0.5)
    # reals at 0 and 1 are within 0.5 <= 1 of the generated point; real at 2 is 1.5 away
    assert stats.coverage(real, gen, k=1) == pytest.approx(0.2)
    assert stats.coverage(real, real, k=1) == 1.0
    assert stats.coverage(real, _line(1000.0), k=1) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_coverage_never_rises_when_generated_samples_are_removed(seed):
    real, gen = _features(seed, 60, d=8), _features(seed + 100, 60, d=8, shift=0.2)
    values = [stats.coverage(real, gen[:n]) for n in (60, 40, 20, 5)]
    assert values == sorted(values, reverse=True)


def test_memorization_on_a_hand_built_line():
    real = _line(*range(10))  # median real-to-real nearest distance is 1
    gen = _line(0.2, 5.5, 100.0)  # nearest-real distances 0.2, 0.5, 91
    rate, nn_dist = stats.memorization(real, gen)
    assert rate == pytest.approx(2 / 3)
    assert nn_dist == pytest.approx((0.2 + 0.5 + 91.0) / 3, rel=1e-5)


def test_exact_copies_are_fully_memorized():
    real = _features(7, 40)
    rate, nn_dist = stats.memorization(real, real[:10].clone())
    assert rate == 1.0
    assert nn_dist == pytest.approx(0.0, abs=1e-3)


def test_precision_recall_uses_any_reference_ball():
    # k=1 radii: real 0 -> 0.1, real 0.1 -> 0.1, real 3 -> 2.9, real 8 -> 5.
    # The generated point at 1.0 is outside its nearest real ball (0.1, distance 0.9)
    # but inside the ball of the real point at 3 (distance 2.0 <= 2.9).
    real = _line(0.0, 0.1, 3.0, 8.0)
    gen = _line(1.0, 1.5)
    precision, recall = stats.precision_recall(real, gen, k=1)
    assert precision == 1.0
    assert recall == 0.0


def test_precision_recall_needs_more_than_k_points():
    with pytest.raises(ValueError, match="more than k"):
        stats.precision_recall(_features(0, 5), _features(1, 5), k=5)
```

Create `tests/test_parity.py`:

```python
"""The package must reproduce the research code's numbers on identical features."""

import pytest
import torch

from _legacy import load_legacy_metric
from m3score import stats


def _pair(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(60, 32, generator=g), torch.randn(45, 32, generator=g) * 1.1 + 0.25


@pytest.mark.parametrize("seed", range(4))
def test_fidelity_matches_research_code(seed):
    legacy = load_legacy_metric()
    real, gen = _pair(seed)
    expected = legacy._compute_mmd2_rbf(real, gen).item()
    assert stats.mmd2_rbf(real, gen) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_memorization_matches_research_code(seed):
    legacy = load_legacy_metric()
    real, gen = _pair(seed)
    expected_nn_dist, expected_rate = legacy._compute_memorization(real, gen)
    rate, nn_dist = stats.memorization(real, gen)
    assert rate == pytest.approx(expected_rate, abs=1e-6)
    assert nn_dist == pytest.approx(expected_nn_dist, abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_coverage_matches_research_code(seed):
    legacy = load_legacy_metric()
    real, gen = _pair(seed)
    assert stats.coverage(real, gen) == pytest.approx(legacy._compute_coverage(real, gen), abs=1e-6)
```

Run: `pytest tests/test_stats.py tests/test_parity.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'stats' from 'm3score'`.

- [ ] **Step 2: Implement `stats.py`**

Create `src/m3score/stats.py`:

```python
"""Statistics of the m3-v1 protocol, as pure functions on (N, D) feature tensors.

Nothing here knows about images or models. The fidelity, memorization and coverage
arithmetic mirrors the research implementation in ``evaluation/m3_score_v2.py``
operation for operation; ``tests/test_parity.py`` holds the two together.
"""

from __future__ import annotations

import torch

from . import protocol

_EPS_L2 = 1e-8


def _l2_normalise(x: torch.Tensor) -> torch.Tensor:
    """Project each row onto the unit sphere."""
    return x / x.norm(dim=1, keepdim=True).clamp(min=_EPS_L2)


def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances, (Na, Nb)."""
    return ((a * a).sum(1, keepdim=True) + (b * b).sum(1) - 2.0 * (a @ b.t())).clamp(min=0.0)


def _median_bandwidth(d: torch.Tensor) -> float:
    """Median of the upper triangle of a pooled squared-distance matrix, floored at 1e-10."""
    n = d.shape[0]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=d.device), diagonal=1)
    return float(d[upper].median().clamp(min=1e-10))


def mmd2_rbf(real: torch.Tensor, gen: torch.Tensor, *, clamp: bool = True) -> float:
    """Unbiased MMD^2 under a mixture of Gaussian RBF kernels, on L2-normalised features.

    Bandwidths are the pooled median squared distance times each of
    ``protocol.RBF_BANDWIDTH_SCALES``. With ``clamp`` the estimate is floored at 0.
    """
    x = _l2_normalise(real.float())
    y = _l2_normalise(gen.float())
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(f"MMD^2 needs at least 2 samples per set; got {m} and {n}")
    dxx, dyy, dxy = _sq_dist(x, x), _sq_dist(y, y), _sq_dist(x, y)
    joint = torch.cat([x, y], dim=0)
    sigma2 = _median_bandwidth(_sq_dist(joint, joint))

    total = torch.zeros(1, device=x.device)
    for scale in protocol.RBF_BANDWIDTH_SCALES:
        gamma = 1.0 / (2.0 * (sigma2 * scale))
        kxx = torch.exp(-gamma * dxx)
        kyy = torch.exp(-gamma * dyy)
        kxy = torch.exp(-gamma * dxy)
        t_rr = (kxx.sum() - kxx.trace()) / (m * (m - 1))
        t_gg = (kyy.sum() - kyy.trace()) / (n * (n - 1))
        total = total + (t_rr + t_gg - 2.0 * kxy.mean())
    value = total / len(protocol.RBF_BANDWIDTH_SCALES)
    return float(value.clamp(min=0.0) if clamp else value)


def _kth_nn_dist(query: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each query row to its k-th nearest row of ``ref``."""
    d2 = torch.cdist(query, ref, p=2).pow(2)
    return d2.topk(k, largest=False, dim=1).values[:, -1].sqrt()


def _kth_nn_dist_within(x: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each row to its k-th nearest other row of the same set."""
    d2 = torch.cdist(x, x, p=2).pow(2)
    d2.fill_diagonal_(float("inf"))
    return d2.topk(k, largest=False, dim=1).values[:, -1].sqrt()


def memorization(real: torch.Tensor, gen: torch.Tensor) -> tuple[float, float]:
    """Near-duplicate rate and mean nearest-real distance.

    A generated point is flagged when it is closer to a real point than the median
    real point is to its own nearest real neighbour.
    """
    real, gen = real.float(), gen.float()
    gen_to_real = _kth_nn_dist(gen, real, 1)
    threshold = _kth_nn_dist_within(real, 1).median().item()
    return float((gen_to_real < threshold).float().mean()), float(gen_to_real.mean())


def coverage(real: torch.Tensor, gen: torch.Tensor, k: int = protocol.COVERAGE_K) -> float:
    """Fraction of real points with a generated point inside their own k-NN radius.

    The radius is estimated within the real set (Naeem et al. 2020), so removing
    generated samples can only lower the value.
    """
    real, gen = real.float(), gen.float()
    radii = _kth_nn_dist_within(real, k)
    nearest_gen = torch.cdist(real, gen, p=2).min(dim=1).values
    return float((nearest_gen <= radii).float().mean())


def precision_recall(
    real: torch.Tensor, gen: torch.Tensor, k: int = protocol.COVERAGE_K
) -> tuple[float, float]:
    """k-NN precision and recall (Kynkaanniemi et al. 2019).

    A point counts when it lies inside the k-NN ball of *any* point of the other set.
    Not part of the M3 axes: recall rises as generated diversity is removed, which is
    why M3 reports ``coverage`` instead. The research code's ``_manifold_fraction``
    tests only the nearest reference point's ball and returns lower values.
    """
    real, gen = real.float(), gen.float()
    if real.shape[0] <= k or gen.shape[0] <= k:
        raise ValueError(
            f"precision/recall needs more than k={k} points per set; "
            f"got {real.shape[0]} and {gen.shape[0]}"
        )

    def inside_any(query: torch.Tensor, ref: torch.Tensor, radii: torch.Tensor) -> float:
        return float((torch.cdist(query, ref, p=2) <= radii[None, :]).any(dim=1).float().mean())

    precision = inside_any(gen, real, _kth_nn_dist_within(real, k))
    recall = inside_any(real, gen, _kth_nn_dist_within(gen, k))
    return precision, recall
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_stats.py tests/test_parity.py -v`
Expected: all pass, none skipped (the research file is present in the repository).

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/m3score/stats.py tests/_legacy.py tests/test_stats.py tests/test_parity.py
git commit -m "feat: add MMD, memorization, coverage and precision/recall statistics"
```

---

### Task 3: Fidelity statistics (bootstrap, permutation test, real-vs-real floor)

**Files:**
- Modify: `src/m3score/stats.py` (append), `tests/test_stats.py` (append), `tests/test_parity.py` (append)

**Interfaces:**
- Consumes: `stats.mmd2_rbf`, `stats._l2_normalise`, `stats._sq_dist`, `stats._median_bandwidth`; `protocol.N_BOOTSTRAP`, `N_PERMUTATIONS`, `CI_ALPHA`, `FLOOR_SPLITS`.
- Produces:
  - `stats.bootstrap_ci(real, gen, *, n_bootstrap=200, alpha=0.05, seed=0) -> tuple[float, float]`
  - `stats.permutation_test(real, gen, *, n_permutations=200, seed=0) -> tuple[float, float, float]` returning `(p_value, z_score, null_mean)`
  - `stats.real_floor(real, groups=None, *, n_splits=10, seed=0) -> tuple[float, float, bool] | None` returning `(mean clamped at 0, sample std, split_was_subject_disjoint)`, or `None` when no valid split exists. `groups` is a sequence of `str`, one per row, or `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stats.py`:

```python
def _slow_bootstrap_values(real, gen, n_bootstrap, seed):
    """Reference: build each resample explicitly and skip pairs sharing an original index."""
    nr, ng = real.shape[0], gen.shape[0]
    kernel = stats._pooled_kernel(real, gen).numpy()
    rng = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_bootstrap):
        ri = torch.randint(0, nr, (nr,), generator=rng).numpy()
        gi = torch.randint(0, ng, (ng,), generator=rng).numpy()
        rr = [kernel[a, b] for a in ri for b in ri if a != b]
        gg = [kernel[nr + a, nr + b] for a in gi for b in gi if a != b]
        rg = [kernel[a, nr + b] for a in ri for b in gi]
        out.append(sum(rr) / len(rr) + sum(gg) / len(gg) - 2.0 * sum(rg) / len(rg))
    return np.array(out)


def test_bootstrap_matches_a_slow_reference():
    real, gen = _features(1, 25), _features(2, 20, shift=0.3)
    values = _slow_bootstrap_values(real, gen, 8, seed=4)
    expected = (max(np.percentile(values, 2.5), 0.0), max(np.percentile(values, 97.5), 0.0))
    assert stats.bootstrap_ci(real, gen, n_bootstrap=8, seed=4) == pytest.approx(expected, abs=1e-9)


def test_bootstrap_interval_contains_the_point_estimate():
    g = torch.Generator().manual_seed(2)
    real = torch.randn(300, 64, generator=g)
    gen = torch.randn(300, 64, generator=g) + 0.15
    low, high = stats.bootstrap_ci(real, gen, seed=0)
    assert low <= stats.mmd2_rbf(real, gen) <= high


def test_bootstrap_interval_reaches_zero_for_one_distribution():
    g = torch.Generator().manual_seed(2)
    real = torch.randn(300, 64, generator=g)
    gen = torch.randn(300, 64, generator=g)
    low, _ = stats.bootstrap_ci(real, gen, seed=0)
    assert low == 0.0


def test_bootstrap_is_deterministic_per_seed():
    real, gen = _features(0, 50), _features(1, 50, shift=0.2)
    assert stats.bootstrap_ci(real, gen, seed=3) == stats.bootstrap_ci(real, gen, seed=3)
    assert stats.bootstrap_ci(real, gen, seed=3) != stats.bootstrap_ci(real, gen, seed=4)


def test_permutation_test_separates_shifted_from_identical_distributions():
    real = _features(0, 80, d=32)
    same = _features(1, 80, d=32)
    shifted = _features(1, 80, d=32, shift=0.5)
    p_same, _, _ = stats.permutation_test(real, same, n_permutations=100, seed=1)
    p_shift, z_shift, _ = stats.permutation_test(real, shifted, n_permutations=100, seed=1)
    assert p_same > 0.05
    assert p_shift == pytest.approx(1 / 101)
    assert z_shift > 5


def test_permutation_null_equals_recomputation():
    real, gen = _features(3, 30), _features(4, 25, shift=0.2)
    _, _, null_mean = stats.permutation_test(real, gen, n_permutations=20, seed=7)
    pooled = torch.cat([real, gen])
    rng = torch.Generator().manual_seed(7)
    slow = []
    for _ in range(20):
        perm = torch.randperm(55, generator=rng)
        slow.append(stats.mmd2_rbf(pooled[perm[:30]], pooled[perm[30:]]))
    assert null_mean == pytest.approx(float(np.mean(slow)), abs=1e-5)


def test_floor_is_small_for_one_distribution_and_deterministic():
    real = _features(0, 200, d=32)
    mean, std, disjoint = stats.real_floor(real, seed=0)
    assert not disjoint
    assert 0.0 <= mean < 0.01
    assert std >= 0.0
    assert stats.real_floor(real, seed=0) == (mean, std, disjoint)


def test_floor_splits_are_subject_disjoint(monkeypatch):
    # Column 0 carries the subject index so the two halves can be traced back.
    subjects = [f"s{i:02d}" for i in range(10) for _ in range(6)]
    real = _features(0, 60)
    real[:, 0] = torch.tensor([float(s[1:]) for s in subjects])
    seen = []
    original = stats.mmd2_rbf

    def spy(a, b, **kwargs):
        seen.append((set(a[:, 0].tolist()), set(b[:, 0].tolist())))
        return original(a, b, **kwargs)

    monkeypatch.setattr(stats, "mmd2_rbf", spy)
    _, _, disjoint = stats.real_floor(real, subjects, n_splits=5, seed=0)
    assert disjoint
    assert len(seen) == 5
    assert all(not (a & b) for a, b in seen)


def test_floor_falls_back_to_images_with_one_subject():
    result = stats.real_floor(_features(0, 40), ["only"] * 40, seed=0)
    assert result is not None and result[2] is False


def test_floor_is_none_without_enough_images():
    assert stats.real_floor(_features(0, 3)) is None


def test_statistics_stay_finite_on_degenerate_features():
    normal = _features(0, 40)
    constant = torch.ones(12, 16)
    for real, gen in ((normal, normal.clone()), (normal, constant), (constant, constant)):
        values = [
            stats.mmd2_rbf(real, gen),
            *stats.bootstrap_ci(real, gen, n_bootstrap=10),
            *stats.permutation_test(real, gen, n_permutations=10),
            *stats.memorization(real, gen),
            stats.coverage(real, gen),
        ]
        assert all(np.isfinite(v) for v in values), values
    assert stats.mmd2_rbf(normal, normal.clone()) == 0.0
    assert stats.coverage(normal, normal.clone()) == 1.0
```

Append to `tests/test_parity.py`:

```python
@pytest.mark.parametrize("seed", range(4))
def test_permutation_test_matches_research_code(seed):
    legacy = load_legacy_metric()
    real, gen = _pair(seed)
    research_p, research_z, research_null = legacy._permutation_test_fidelity(
        real, gen, n_permutations=50, seed=seed
    )
    p, z, null_mean = stats.permutation_test(real, gen, n_permutations=50, seed=seed)
    assert null_mean == pytest.approx(research_null, abs=1e-6)
    assert z == pytest.approx(research_z, rel=1e-3)
    # research p is b / B; the package reports (1 + b) / (1 + B)
    assert p == pytest.approx((1 + research_p * 50) / 51, abs=1e-9)


def test_research_bootstrap_excludes_the_point_estimate_and_the_package_does_not():
    """Documents the deliberate difference recorded in the spec, section 4, item 6."""
    legacy = load_legacy_metric()
    g = torch.Generator().manual_seed(2)
    real = torch.randn(300, 64, generator=g)
    gen = torch.randn(300, 64, generator=g) + 0.15
    point = stats.mmd2_rbf(real, gen)
    research_low, _ = legacy._bootstrap_ci_fidelity(real, gen, n_bootstrap=200, seed=0)
    low, high = stats.bootstrap_ci(real, gen, seed=0)
    assert research_low > point
    assert low <= point <= high
```

Run: `pytest tests/test_stats.py tests/test_parity.py -v`
Expected: the new tests FAIL with `AttributeError: module 'm3score.stats' has no attribute '_pooled_kernel'` (or `bootstrap_ci`, `permutation_test`, `real_floor`); the Task 2 tests still pass.

- [ ] **Step 2: Implement the statistics**

In `src/m3score/stats.py`, change the imports at the top to:

```python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from . import protocol
```

Append to `src/m3score/stats.py`:

```python
def _pooled_kernel(real: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
    """Bandwidth-averaged RBF kernel on the pooled sample, real rows first.

    Held in float64 so that sums over millions of entries keep their precision.
    Averaging the kernels equals averaging the per-bandwidth MMD^2 estimates.
    """
    pooled = _l2_normalise(torch.cat([real.float(), gen.float()], dim=0))
    d = _sq_dist(pooled, pooled)
    sigma2 = _median_bandwidth(d)
    dtype = torch.float32 if d.device.type == "mps" else torch.float64
    kernel = torch.zeros(d.shape, dtype=dtype, device=d.device)
    for scale in protocol.RBF_BANDWIDTH_SCALES:
        kernel += torch.exp(-d / (2.0 * (sigma2 * scale))).to(dtype)
    return kernel / len(protocol.RBF_BANDWIDTH_SCALES)


def bootstrap_ci(
    real: torch.Tensor,
    gen: torch.Tensor,
    *,
    n_bootstrap: int = protocol.N_BOOTSTRAP,
    alpha: float = protocol.CI_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the fidelity MMD^2.

    Real and generated sets are resampled independently with replacement. The kernel
    is the one used for the point estimate. A resample contains repeated images;
    within-set pairs that are two copies of the same original are left out of the
    within-set averages, which keeps the bootstrap distribution centred on the point
    estimate. (Counting them, as the research code does, shifts the interval upward.)
    """
    nr, ng = real.shape[0], gen.shape[0]
    kernel = _pooled_kernel(real, gen)
    k_rr, k_gg, k_rg = kernel[:nr, :nr], kernel[nr:, nr:], kernel[:nr, nr:]
    diag_r, diag_g = k_rr.diagonal(), k_gg.diagonal()

    rng = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(n_bootstrap):
        # multiplicity of each original image in this resample
        c_r = torch.bincount(torch.randint(0, nr, (nr,), generator=rng), minlength=nr).to(kernel)
        c_g = torch.bincount(torch.randint(0, ng, (ng,), generator=rng), minlength=ng).to(kernel)
        pairs_r = nr * nr - float((c_r * c_r).sum())
        pairs_g = ng * ng - float((c_g * c_g).sum())
        if pairs_r <= 0 or pairs_g <= 0:  # a resample made of one image repeated
            continue
        t_rr = (torch.dot(c_r, k_rr @ c_r) - torch.dot(c_r * c_r, diag_r)) / pairs_r
        t_gg = (torch.dot(c_g, k_gg @ c_g) - torch.dot(c_g * c_g, diag_g)) / pairs_g
        t_rg = torch.dot(c_r, k_rg @ c_g) / (nr * ng)
        values.append(float(t_rr + t_gg - 2.0 * t_rg))
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    low = max(float(np.percentile(arr, 100 * alpha / 2)), 0.0)
    high = max(float(np.percentile(arr, 100 * (1 - alpha / 2))), 0.0)
    return low, high


def permutation_test(
    real: torch.Tensor,
    gen: torch.Tensor,
    *,
    n_permutations: int = protocol.N_PERMUTATIONS,
    seed: int = 0,
) -> tuple[float, float, float]:
    """One-sided permutation test of H0: MMD^2(real, gen) = 0.

    Returns ``(p_value, z_score, null_mean)``. The pooled median, and so every
    bandwidth, is unchanged by relabelling, so the null is evaluated on one pooled
    kernel matrix: each permutation costs a single matrix-vector product.
    """
    nr, ng = real.shape[0], gen.shape[0]
    n = nr + ng
    observed = mmd2_rbf(real, gen)
    kernel = _pooled_kernel(real, gen)
    diag = kernel.diagonal().clone()
    total, diag_total = kernel.sum(), diag.sum()

    rng = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=rng)
        is_real = torch.zeros(n, dtype=kernel.dtype, device=kernel.device)
        is_real[perm[:nr].to(kernel.device)] = 1.0
        k_real = kernel @ is_real
        sum_rr = torch.dot(is_real, k_real)
        sum_rg = k_real.sum() - sum_rr
        sum_gg = total - sum_rr - 2.0 * sum_rg
        trace_r = torch.dot(is_real, diag)
        value = (
            (sum_rr - trace_r) / (nr * (nr - 1))
            + (sum_gg - (diag_total - trace_r)) / (ng * (ng - 1))
            - 2.0 * sum_rg / (nr * ng)
        )
        null.append(max(float(value), 0.0))
    arr = np.array(null)
    exceed = int((arr >= observed).sum())
    p_value = (1 + exceed) / (1 + n_permutations)
    z_score = float((observed - arr.mean()) / max(float(arr.std()), 1e-10))
    return float(p_value), z_score, float(arr.mean())


def real_floor(
    real: torch.Tensor,
    groups: Sequence[str] | None = None,
    *,
    n_splits: int = protocol.FLOOR_SPLITS,
    seed: int = 0,
) -> tuple[float, float, bool] | None:
    """Real-vs-real MMD^2 over random halvings of the real set.

    Gives the fidelity value a scale: this is what the metric reads when both sides
    are real. With two or more known subjects the halves are subject-disjoint.
    Returns ``(mean clamped at 0, sample std, subject_disjoint)``, or ``None`` when no
    split leaves two images on each side.
    """
    real = real.float()
    n = real.shape[0]
    subjects = sorted(set(groups)) if groups is not None else []
    by_subject = len(subjects) >= 2
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(n_splits):
        if by_subject:
            order = rng.permutation(len(subjects))
            half = {subjects[i] for i in order[: len(subjects) // 2]}
            in_a = np.fromiter((g in half for g in groups), dtype=bool, count=n)
        else:
            in_a = np.zeros(n, dtype=bool)
            in_a[rng.permutation(n)[: n // 2]] = True
        if in_a.sum() < 2 or (~in_a).sum() < 2:
            continue
        mask = torch.from_numpy(in_a).to(real.device)
        values.append(mmd2_rbf(real[mask], real[~mask], clamp=False))
    if not values:
        return None
    arr = np.array(values)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return max(float(arr.mean()), 0.0), std, by_subject
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_stats.py tests/test_parity.py -v`
Expected: all pass.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/m3score/stats.py tests/test_stats.py tests/test_parity.py
git commit -m "feat: add bootstrap interval, permutation test and real-vs-real floor"
```

---

### Task 4: Backbone (preprocessing and feature extraction)

**Files:**
- Create: `src/m3score/backbone.py`, `tests/conftest.py`, `tests/test_backbone.py`

**Interfaces:**
- Consumes: `protocol.BACKBONE_ID`, `BACKBONE_REVISION`, `IMAGE_SIZE`, `NORM_MEAN`, `NORM_STD`; `m3score.M3ScoreError`.
- Produces:
  - `backbone.Backbone(model: torch.nn.Module, name: str)` with attributes `name: str`, `image_size: tuple[int, int]`, `num_layers: int`
  - `Backbone.prepare(images) -> torch.Tensor`: `(N, 1|3, H, W)` float in `[0, 1]` to `(N, 3, S, S)`
  - `Backbone.extract(prepared, layers, device) -> dict[int, torch.Tensor]`: block number (from 1) to `(N, D)` float32 on the CPU
  - `backbone.load_backbone(spec=None, device="cpu") -> Backbone`
- Test fixtures in `tests/conftest.py`: `tiny_model` (session-scoped random 12-block ViT for 32x32 inputs, embedding size 48) and `tiny_backbone`.

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
import pytest
import timm
import torch

from m3score.backbone import Backbone


@pytest.fixture(scope="session")
def tiny_model():
    """A randomly initialised 12-block ViT for 32x32 inputs: fast, and needs no download."""
    torch.manual_seed(0)
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=32,
        patch_size=8,
        embed_dim=48,
        depth=12,
        num_heads=3,
    )
    return model.eval()


@pytest.fixture()
def tiny_backbone(tiny_model):
    return Backbone(tiny_model, "tiny-test-vit")
```

Create `tests/test_backbone.py`:

```python
import pytest
import timm
import torch
import torch.nn as nn

from _legacy import load_legacy_metric, load_legacy_module
from m3score import M3ScoreError, protocol
from m3score.backbone import Backbone, load_backbone

LAYERS = (12, 9, 4)


def test_extract_returns_cls_features_for_the_requested_blocks(tiny_backbone):
    prepared = tiny_backbone.prepare(torch.rand(5, 3, 32, 32))
    features = tiny_backbone.extract(prepared, LAYERS, "cpu")
    assert set(features) == {12, 9, 4}
    assert all(f.shape == (5, 48) and f.dtype == torch.float32 for f in features.values())
    assert not torch.equal(features[4], features[12])


def test_prepare_repeats_grey_and_resizes(tiny_backbone):
    prepared = tiny_backbone.prepare(torch.rand(2, 1, 50, 40))
    assert prepared.shape == (2, 3, 32, 32)
    assert torch.equal(prepared[:, 0], prepared[:, 1])
    assert float(prepared.min()) >= 0.0 and float(prepared.max()) <= 1.0


def test_prepare_leaves_native_size_untouched(tiny_backbone):
    images = torch.rand(2, 3, 32, 32)
    assert torch.equal(tiny_backbone.prepare(images), images)


def test_features_do_not_depend_on_how_images_are_batched(tiny_backbone):
    prepared = tiny_backbone.prepare(torch.rand(6, 3, 32, 32))
    whole = tiny_backbone.extract(prepared, LAYERS, "cpu")
    halves = [tiny_backbone.extract(prepared[i : i + 3], LAYERS, "cpu") for i in (0, 3)]
    for layer in LAYERS:
        assert torch.allclose(whole[layer], torch.cat([h[layer] for h in halves]), atol=1e-5)


def test_extraction_matches_research_code(tiny_model, tiny_backbone):
    module = load_legacy_module()
    legacy = load_legacy_metric()
    legacy.backbone = tiny_model
    legacy._backend = "timm_vit"
    legacy.num_layers = 12
    legacy._norm_mean = module._IMAGENET_MEAN
    legacy._norm_std = module._IMAGENET_STD
    images = torch.rand(20, 3, 32, 32)
    research = legacy._extract_cls_features(images, {4, 9, 12})
    package = tiny_backbone.extract(tiny_backbone.prepare(images), LAYERS, "cpu")
    for layer in LAYERS:
        assert torch.allclose(package[layer], research[layer - 1], atol=1e-5)


def test_rejects_a_model_that_is_not_a_vision_transformer():
    with pytest.raises(TypeError, match="vision transformer"):
        Backbone(nn.Linear(2, 2), "linear")


def test_load_backbone_accepts_a_module(tiny_model):
    backbone = load_backbone(tiny_model, "cpu")
    assert backbone.name == "VisionTransformer"
    assert backbone.num_layers == 12
    assert backbone.image_size == (32, 32)


def test_load_backbone_rejects_other_types():
    with pytest.raises(TypeError, match="backbone"):
        load_backbone(123, "cpu")


def test_default_backbone_is_the_pinned_revision(monkeypatch, tiny_model):
    calls = {}

    def fake_create(name, **kwargs):
        calls["name"], calls["kwargs"] = name, kwargs
        return tiny_model

    monkeypatch.setattr(timm, "create_model", fake_create)
    backbone = load_backbone(None, "cpu")
    pinned = f"{protocol.BACKBONE_ID}@{protocol.BACKBONE_REVISION}"
    assert calls["name"] == f"hf_hub:{pinned}"
    assert calls["kwargs"] == {"pretrained": True, "img_size": 224}
    assert backbone.name == pinned


def test_backbone_download_failure_explains_itself(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(timm, "create_model", refuse)
    with pytest.raises(M3ScoreError, match="Hugging Face") as info:
        load_backbone(None, "cpu")
    assert "HF_HUB_OFFLINE" in str(info.value)
    assert "connection refused" in str(info.value)
```

Run: `pytest tests/test_backbone.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'm3score.backbone'`.

- [ ] **Step 2: Implement `backbone.py`**

Create `src/m3score/backbone.py`:

```python
"""Loading the encoder and turning images into per-block CLS features."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from . import protocol
from ._errors import M3ScoreError

_REQUIRED = ("patch_embed", "_pos_embed", "norm_pre", "blocks")


class Backbone:
    """A timm vision transformer with the m3-v1 preprocessing in front of it."""

    def __init__(self, model: torch.nn.Module, name: str) -> None:
        missing = [attr for attr in _REQUIRED if not hasattr(model, attr)]
        if missing:
            raise TypeError(
                f"the backbone must be a timm vision transformer; "
                f"{type(model).__name__} has no {', '.join(missing)}"
            )
        self.model = model.eval()
        self.name = name
        size = model.patch_embed.img_size
        self.image_size = (int(size[0]), int(size[1]))
        self.num_layers = len(model.blocks)

    def prepare(self, images: torch.Tensor) -> torch.Tensor:
        """``(N, 1|3, H, W)`` float in [0, 1] to ``(N, 3, S, S)``.

        Grey is repeated to three channels, then one antialiased bilinear resize brings
        the image to the encoder's input size. This is the only resize in the package,
        so every input type is treated identically.
        """
        x = images.float()
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(
                x, size=self.image_size, mode="bilinear", antialias=True, align_corners=False
            )
        return x

    @torch.no_grad()
    def extract(
        self, prepared: torch.Tensor, layers: Iterable[int], device: torch.device | str
    ) -> dict[int, torch.Tensor]:
        """CLS token after each requested block (numbered from 1), as float32 on the CPU."""
        wanted = set(layers)
        last = max(wanted)
        model = self.model
        mean = torch.tensor(protocol.NORM_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(protocol.NORM_STD, device=device).view(1, 3, 1, 1)
        x = (prepared.to(device) - mean) / std
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        if hasattr(model, "patch_drop"):
            x = model.patch_drop(x)
        x = model.norm_pre(x)
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(model.blocks, start=1):
            x = block(x)
            if index in wanted:
                features[index] = x[:, 0, :].float().cpu()
            if index == last:
                break
        return features


def load_backbone(
    spec: str | torch.nn.Module | None = None, device: torch.device | str = "cpu"
) -> Backbone:
    """Build the encoder.

    ``None`` loads the m3-v1 encoder at its pinned Hub revision. A string is a timm
    model name loaded with pretrained weights. A module is used as given.
    """
    if isinstance(spec, torch.nn.Module):
        return Backbone(spec.to(device), type(spec).__name__)
    if spec is not None and not isinstance(spec, str):
        raise TypeError(
            f"backbone must be a timm model name or a torch.nn.Module, got {type(spec).__name__}"
        )

    import timm

    if isinstance(spec, str):
        return Backbone(timm.create_model(spec, pretrained=True, num_classes=0).to(device), spec)

    name = f"{protocol.BACKBONE_ID}@{protocol.BACKBONE_REVISION}"
    try:
        model = timm.create_model(f"hf_hub:{name}", pretrained=True, img_size=protocol.IMAGE_SIZE)
    except Exception as exc:
        raise M3ScoreError(
            f"could not load the M3-Score backbone {protocol.BACKBONE_ID} from the Hugging Face "
            f"Hub ({type(exc).__name__}: {exc}). It is downloaded once, about 90 MB, and cached. "
            f"Check access to huggingface.co, or copy a populated Hugging Face cache to this "
            f"machine and set HF_HUB_OFFLINE=1."
        ) from exc
    return Backbone(model.to(device), name)
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_backbone.py -v`
Expected: 10 passed.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/m3score/backbone.py tests/conftest.py tests/test_backbone.py
git commit -m "feat: add backbone loading, preprocessing and CLS feature extraction"
```

---

### Task 5: In-memory inputs and sampling

**Files:**
- Create: `src/m3score/io/__init__.py`, `src/m3score/io/base.py`, `src/m3score/io/arrays.py`, `src/m3score/io/sampling.py`, `tests/test_io_arrays.py`, `tests/test_io_sampling.py`

**Interfaces:**
- Consumes: `protocol.MAX_IMAGES`.
- Produces:
  - `io.base.SkippedItem(Exception)`: raised by an item's `load` when it cannot contribute an image; `str(exc)` is the reason, starting with the item's label.
  - `io.base.ImageSource`: base class with `groups: list[str] | None`, `warnings: list[str]`, `__len__()`, and `load(index) -> torch.Tensor` returning `(C, H, W)` float32 in `[0, 1]` with `C` in `{1, 3}`.
  - `io.base.ItemSource(items, groups, warnings=())`: an `ImageSource` over objects that each have a `load()` method.
  - `io.arrays.check_images(images) -> tuple[torch.Tensor, float]`: validates; returns the batch as `(N, C, H, W)` in its own dtype and the divisor mapping it to `[0, 1]`.
  - `io.arrays.to_float(tensor, divisor) -> torch.Tensor`
  - `io.arrays.ArraySource(images)`: an `ImageSource` over a tensor or array.
  - `io.sampling.sample_indices(n_total: int, n: int | None, groups: Sequence[str] | None, seed: int) -> list[int]`: sorted indices.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_io_arrays.py`:

```python
import numpy as np
import pytest
import torch

from m3score.io.arrays import ArraySource, check_images, to_float


def test_uint8_greyscale_stack_becomes_unit_range_with_a_channel():
    images = torch.tensor([[[0, 255], [51, 102]]], dtype=torch.uint8)
    tensor, divisor = check_images(images)
    out = to_float(tensor, divisor)
    assert out.shape == (1, 1, 2, 2)
    assert out.dtype == torch.float32
    assert out.flatten().tolist() == pytest.approx([0.0, 1.0, 0.2, 0.4])


def test_float_rgb_passes_through_unchanged():
    images = torch.rand(3, 3, 8, 8)
    tensor, divisor = check_images(images)
    assert divisor == 1.0
    assert torch.equal(to_float(tensor, divisor), images)


def test_numpy_arrays_are_accepted():
    tensor, divisor = check_images(np.zeros((2, 8, 8), dtype=np.uint8))
    assert tensor.shape == (2, 1, 8, 8)
    assert divisor == 255.0


def test_values_outside_unit_range_are_rejected_with_the_fix():
    with pytest.raises(ValueError, match=r"\(x \+ 1\) / 2"):
        check_images(torch.rand(2, 1, 8, 8) * 2 - 1)


def test_non_finite_values_are_rejected():
    images = torch.rand(2, 1, 8, 8)
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        check_images(images)


@pytest.mark.parametrize("shape", [(8, 8), (2, 2, 8, 8), (2, 8, 8, 3), (1, 2, 3, 8, 8)])
def test_wrong_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="shaped"):
        check_images(torch.zeros(shape))


def test_unsupported_dtypes_are_rejected():
    with pytest.raises(ValueError, match="dtype"):
        check_images(torch.zeros(2, 1, 8, 8, dtype=torch.int64))


def test_non_arrays_are_rejected():
    with pytest.raises(TypeError, match="torch.Tensor or numpy.ndarray"):
        check_images([[0.0]])


def test_empty_batches_are_rejected():
    with pytest.raises(ValueError, match="no images"):
        check_images(torch.zeros(0, 1, 8, 8))


def test_array_source_serves_single_images():
    source = ArraySource(np.full((4, 6, 5), 255, dtype=np.uint8))
    assert len(source) == 4
    assert source.groups is None
    assert source.warnings == []
    image = source.load(2)
    assert image.shape == (1, 6, 5)
    assert float(image.max()) == 1.0
```

Create `tests/test_io_sampling.py`:

```python
import pytest

from m3score import protocol
from m3score.io.sampling import sample_indices


def test_everything_is_used_when_n_covers_the_set():
    assert sample_indices(10, 10, None, 0) == list(range(10))
    assert sample_indices(10, 50, None, 0) == list(range(10))
    assert sample_indices(10, None, None, 0) == list(range(10))


def test_default_cap_applies_when_n_is_omitted():
    assert len(sample_indices(protocol.MAX_IMAGES + 10, None, None, 0)) == protocol.MAX_IMAGES


def test_a_subset_is_random_seeded_and_never_the_head_of_the_list():
    first = sample_indices(1000, 100, None, 0)
    assert len(set(first)) == 100
    assert first == sample_indices(1000, 100, None, 0)
    assert first != sample_indices(1000, 100, None, 1)
    assert first != list(range(100))
    assert max(first) > 500


def test_known_subjects_are_spread_across_the_sample():
    # 100 subjects x 5 items, listed subject by subject as a sorted folder would be.
    # Taking the head of this list would return 20 subjects; the finding in
    # docs/REFERENCE_COHORT_FINDING.md is that this is not a valid reference.
    groups = [f"subject{s:03d}" for s in range(100) for _ in range(5)]
    chosen = sample_indices(500, 100, groups, 0)
    assert len({groups[i] for i in chosen}) == 100


def test_known_subjects_are_balanced():
    groups = ["a"] * 50 + ["b"] * 50 + ["c"] * 50 + ["d"] * 2
    chosen = sample_indices(152, 40, groups, 0)
    counts = {g: sum(groups[i] == g for i in chosen) for g in "abcd"}
    assert sum(counts.values()) == 40
    assert counts["d"] == 2
    assert max(counts[g] for g in "abc") - min(counts[g] for g in "abc") <= 1


def test_grouped_sampling_is_seeded():
    groups = [f"s{i % 7}" for i in range(70)]
    assert sample_indices(70, 20, groups, 3) == sample_indices(70, 20, groups, 3)
    assert sample_indices(70, 20, groups, 3) != sample_indices(70, 20, groups, 4)


def test_n_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        sample_indices(10, 0, None, 0)


def test_groups_must_cover_every_item():
    with pytest.raises(ValueError, match="group labels"):
        sample_indices(10, 5, ["a"] * 9, 0)
```

Run: `pytest tests/test_io_arrays.py tests/test_io_sampling.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'm3score.io'`.

- [ ] **Step 2: Implement the modules**

Create `src/m3score/io/base.py`:

```python
"""What every image source looks like to the metric."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


class SkippedItem(Exception):
    """Raised by an item's ``load`` when it cannot contribute an image.

    The message is the reason, starting with the item's label.
    """


class ImageSource:
    """An indexable set of images.

    ``load(i)`` returns one image as ``(C, H, W)`` float32 in [0, 1] with C in {1, 3}.
    ``groups`` holds one subject identifier per item, or ``None`` when subjects are
    unknown. ``warnings`` lists anything the source left out and why.
    """

    groups: list[str] | None = None
    warnings: list[str]

    def __len__(self) -> int:
        raise NotImplementedError

    def load(self, index: int) -> torch.Tensor:
        raise NotImplementedError


class ItemSource(ImageSource):
    """A source over objects that each know how to ``load()`` themselves."""

    def __init__(
        self, items: Iterable, groups: Sequence[str] | None, warnings: Iterable[str] = ()
    ) -> None:
        self._items = list(items)
        self.groups = list(groups) if groups is not None else None
        self.warnings = list(warnings)

    def __len__(self) -> int:
        return len(self._items)

    def load(self, index: int) -> torch.Tensor:
        return self._items[index].load()
```

Create `src/m3score/io/arrays.py`:

```python
"""Images already in memory as a torch tensor or numpy array."""

from __future__ import annotations

import math

import numpy as np
import torch

from .base import ImageSource

_RANGE_TOLERANCE = 1e-6


def check_images(images) -> tuple[torch.Tensor, float]:
    """Validate a batch of images.

    Returns the batch as ``(N, C, H, W)`` in its own dtype, and the divisor that maps it
    to [0, 1]. Conversion to float is left to :func:`to_float` so that a large uint8
    array is never copied to float32 in one piece.
    """
    if isinstance(images, np.ndarray):
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(images))
        except TypeError as exc:
            raise ValueError(
                f"unsupported image dtype {images.dtype}; "
                f"pass uint8 in [0, 255] or floating point in [0, 1]"
            ) from exc
    elif isinstance(images, torch.Tensor):
        tensor = images.detach().cpu()
    else:
        raise TypeError(
            f"expected a torch.Tensor or numpy.ndarray of images, got {type(images).__name__}"
        )
    if tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] not in (1, 3):
        raise ValueError(
            f"expected images shaped (N, H, W), (N, 1, H, W) or (N, 3, H, W); "
            f"got {tuple(images.shape)}"
        )
    if tensor.shape[0] == 0:
        raise ValueError("no images given")
    if tensor.dtype == torch.uint8:
        return tensor, 255.0
    if not tensor.is_floating_point():
        raise ValueError(
            f"unsupported image dtype {tensor.dtype}; "
            f"pass uint8 in [0, 255] or floating point in [0, 1]"
        )
    low, high = float(tensor.min()), float(tensor.max())
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("images contain NaN or infinite values")
    if low < -_RANGE_TOLERANCE or high > 1.0 + _RANGE_TOLERANCE:
        raise ValueError(
            f"image values span [{low:.4g}, {high:.4g}] but must lie in [0, 1]; "
            f"if your generator outputs [-1, 1], pass (x + 1) / 2"
        )
    return tensor, 1.0


def to_float(tensor: torch.Tensor, divisor: float) -> torch.Tensor:
    """Float32 in [0, 1] from a tensor validated by :func:`check_images`."""
    return tensor.float().div(divisor).clamp_(0.0, 1.0)


class ArraySource(ImageSource):
    """A tensor or array of images held in memory. Subjects are unknown."""

    def __init__(self, images) -> None:
        self._tensor, self._divisor = check_images(images)
        self.groups = None
        self.warnings = []

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def load(self, index: int) -> torch.Tensor:
        return to_float(self._tensor[index], self._divisor)
```

Create `src/m3score/io/sampling.py`:

```python
"""Choosing which images of a source are scored.

A subset is always a seeded random draw. Taking the first N files of a sorted folder
selects a handful of subjects, which inflates every distance; see
docs/REFERENCE_COHORT_FINDING.md in the repository.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import protocol


def sample_indices(
    n_total: int, n: int | None, groups: Sequence[str] | None, seed: int
) -> list[int]:
    """Indices of the items to score, in ascending order.

    ``n=None`` means everything up to ``protocol.MAX_IMAGES``. With known ``groups``
    the draw takes one item per subject in rotation, subjects and items both shuffled,
    so the sample is balanced across subjects and spans as many as possible.
    """
    if n is None:
        n = protocol.MAX_IMAGES
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if groups is not None and len(groups) != n_total:
        raise ValueError(f"got {len(groups)} group labels for {n_total} items")
    if n >= n_total:
        return list(range(n_total))

    rng = np.random.default_rng(seed)
    if groups is None:
        return sorted(int(i) for i in rng.choice(n_total, size=n, replace=False))

    by_subject: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_subject.setdefault(group, []).append(index)
    subjects = sorted(by_subject)
    pools = [
        [int(i) for i in rng.permutation(by_subject[subjects[s]])]
        for s in rng.permutation(len(subjects))
    ]
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n:
        for pool in pools:
            if depth < len(pool):
                chosen.append(pool[depth])
                if len(chosen) == n:
                    break
        depth += 1
    return sorted(chosen)
```

Create `src/m3score/io/__init__.py`:

```python
"""Turning a path, tensor or array into an indexable set of images."""

from .arrays import ArraySource
from .base import ImageSource, ItemSource, SkippedItem
from .sampling import sample_indices

__all__ = ["ArraySource", "ImageSource", "ItemSource", "SkippedItem", "sample_indices"]
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_io_arrays.py tests/test_io_sampling.py -v`
Expected: all pass.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/m3score/io tests/test_io_arrays.py tests/test_io_sampling.py
git commit -m "feat: add in-memory image sources and subject-balanced sampling"
```

---

### Task 6: Image files and `open_source`

**Files:**
- Create: `src/m3score/io/images.py`, `tests/_images.py`, `tests/test_io_images.py`
- Modify: `src/m3score/io/__init__.py` (replace whole file)

**Interfaces:**
- Consumes: `io.base.ImageSource`, `ItemSource`, `SkippedItem`; `io.arrays.ArraySource`.
- Produces:
  - `io.images.IMAGE_SUFFIXES: frozenset[str]`
  - `io.images.ImageFile(path, label)` with `load() -> torch.Tensor`
  - `io.open_source(source, *, include=None, group_pattern=None, group_pattern_optional=False, slice_band=None, window=None) -> ImageSource`. `source` is a path, a tensor, an array, or an `ImageSource`. `slice_band` and `window` are accepted now and used by the volume readers added in Tasks 9 and 10.
- Test helpers in `tests/_images.py`: `write_png(path, array) -> Path`, `write_pngs(directory, count, *, size=(40, 40), seed=0, name="img_{:04d}.png") -> list[Path]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/_images.py`:

```python
"""Helpers that write small image files for tests."""

import numpy as np
from PIL import Image


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def write_pngs(directory, count, *, size=(40, 40), seed=0, name="img_{:04d}.png"):
    """Write ``count`` random 8-bit greyscale PNGs and return their paths."""
    rng = np.random.default_rng(seed)
    return [
        write_png(directory / name.format(i), rng.integers(0, 256, size=size, dtype=np.uint8))
        for i in range(count)
    ]
```

Create `tests/test_io_images.py`:

```python
import numpy as np
import pytest
import torch

from _images import write_png, write_pngs
from m3score.io import ArraySource, ImageSource, SkippedItem, open_source


def test_8bit_greyscale_png_loads_as_three_equal_channels(tmp_path):
    write_png(tmp_path / "a.png", np.array([[0, 255], [51, 102]], dtype=np.uint8))
    image = open_source(tmp_path).load(0)
    assert image.shape == (3, 2, 2)
    assert image.dtype == torch.float32
    assert image[0].flatten().tolist() == pytest.approx([0.0, 1.0, 0.2, 0.4])
    assert torch.equal(image[0], image[2])


def test_16bit_png_is_scaled_by_its_full_range(tmp_path):
    write_png(tmp_path / "a.png", np.array([[0, 65535], [13107, 32768]], dtype=np.uint16))
    image = open_source(tmp_path).load(0)
    assert image.shape == (1, 2, 2)
    assert image.flatten().tolist() == pytest.approx([0.0, 1.0, 0.2, 32768 / 65535], abs=1e-4)


def test_rgb_png_keeps_its_channels(tmp_path):
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    write_png(tmp_path / "a.png", pixels)
    image = open_source(tmp_path).load(0)
    assert float(image[0].min()) == 1.0
    assert float(image[1].max()) == 0.0


def test_folders_are_scanned_recursively_and_other_files_ignored(tmp_path):
    write_pngs(tmp_path / "b", 2)
    write_pngs(tmp_path / "a", 1)
    write_png(tmp_path / "UPPER.PNG", np.zeros((4, 4), dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")
    source = open_source(tmp_path)
    assert len(source) == 4
    assert source.groups is None
    assert source.warnings == []


def test_include_restricts_the_scan(tmp_path):
    write_pngs(tmp_path, 3, name="flair_{:04d}.png")
    write_pngs(tmp_path, 2, name="seg_{:04d}.png")
    assert len(open_source(tmp_path, include="flair_*")) == 3


def test_group_pattern_assigns_subjects(tmp_path):
    for subject in ("BraTS2021_00000", "BraTS2021_00002"):
        write_pngs(tmp_path, 3, name=subject + "_slice{:03d}.png")
    source = open_source(tmp_path, group_pattern=r"(BraTS2021_\d+)")
    assert source.groups == ["BraTS2021_00000"] * 3 + ["BraTS2021_00002"] * 3


def test_group_pattern_that_misses_a_file_is_an_error(tmp_path):
    write_pngs(tmp_path, 2, name="BraTS2021_00000_slice{:03d}.png")
    write_pngs(tmp_path, 1, name="other_{:04d}.png")
    with pytest.raises(ValueError, match="does not match"):
        open_source(tmp_path, group_pattern=r"(BraTS2021_\d+)")


def test_group_pattern_needs_a_capture_group(tmp_path):
    write_pngs(tmp_path, 1)
    with pytest.raises(ValueError, match="capture group"):
        open_source(tmp_path, group_pattern=r"img_\d+")


def test_invalid_group_pattern_is_a_value_error(tmp_path):
    write_pngs(tmp_path, 1)
    with pytest.raises(ValueError, match="not a valid regular expression"):
        open_source(tmp_path, group_pattern="(")


def test_optional_group_pattern_that_matches_nothing_leaves_groups_unknown(tmp_path):
    write_pngs(tmp_path, 3, name="generated_{:04d}.png")
    source = open_source(
        tmp_path, group_pattern=r"(BraTS2021_\d+)", group_pattern_optional=True
    )
    assert source.groups is None


def test_optional_group_pattern_still_rejects_a_partial_match(tmp_path):
    write_pngs(tmp_path, 2, name="BraTS2021_00000_slice{:03d}.png")
    write_pngs(tmp_path, 1, name="other_{:04d}.png")
    with pytest.raises(ValueError, match="does not match"):
        open_source(tmp_path, group_pattern=r"(BraTS2021_\d+)", group_pattern_optional=True)


def test_a_folder_without_images_lists_the_supported_types(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match=r"\.png"):
        open_source(tmp_path)


def test_a_missing_path_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file or directory"):
        open_source(tmp_path / "absent")


def test_an_unreadable_image_is_skipped_with_its_name(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"this is not a png")
    with pytest.raises(SkippedItem, match="broken.png"):
        open_source(tmp_path).load(0)


def test_a_single_file_is_a_source(tmp_path):
    path = write_pngs(tmp_path, 1)[0]
    assert len(open_source(path)) == 1


def test_tensors_arrays_and_sources_pass_through():
    assert isinstance(open_source(torch.zeros(2, 1, 4, 4)), ArraySource)
    assert isinstance(open_source(np.zeros((2, 4, 4), dtype=np.uint8)), ArraySource)
    source = ArraySource(torch.zeros(2, 1, 4, 4))
    assert open_source(source) is source
    assert isinstance(source, ImageSource)


def test_other_types_are_rejected():
    with pytest.raises(TypeError, match="path"):
        open_source(42)
```

Run: `pytest tests/test_io_images.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'open_source' from 'm3score.io'`.

- [ ] **Step 2: Implement image files and `open_source`**

Create `src/m3score/io/images.py`:

```python
"""2D image files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import SkippedItem

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I"})


class ImageFile:
    """One image file, decoded on demand."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label

    def load(self) -> torch.Tensor:
        try:
            with Image.open(self.path) as image:
                if image.mode in _SIXTEEN_BIT_MODES:
                    array = np.asarray(image, dtype=np.float32) / 65535.0
                    return torch.from_numpy(np.clip(array, 0.0, 1.0))[None]
                if image.mode == "F":
                    array = np.asarray(image, dtype=np.float32)
                    if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
                        raise SkippedItem(
                            f"{self.label}: floating-point image with values outside [0, 1]"
                        )
                    return torch.from_numpy(array.copy())[None]
                # 8-bit modes, as the research pipeline reads them
                array = np.array(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as exc:  # UnidentifiedImageError is an OSError
            raise SkippedItem(f"{self.label}: unreadable image ({exc})") from exc
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
```

Replace `src/m3score/io/__init__.py` with:

```python
"""Turning a path, tensor or array into an indexable set of images."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

import numpy as np
import torch

from .arrays import ArraySource
from .base import ImageSource, ItemSource, SkippedItem
from .images import IMAGE_SUFFIXES, ImageFile
from .sampling import sample_indices

__all__ = [
    "ArraySource",
    "ImageSource",
    "ItemSource",
    "SkippedItem",
    "open_source",
    "sample_indices",
]

_SUPPORTED = ", ".join(sorted(IMAGE_SUFFIXES))


def open_source(
    source,
    *,
    include: str | None = None,
    group_pattern: str | None = None,
    group_pattern_optional: bool = False,
    slice_band: tuple[float, float] | None = None,
    window: tuple[float, float] | None = None,
) -> ImageSource:
    """Make an :class:`ImageSource` from a path, a tensor, an array, or a source.

    ``include`` is a glob on each file's path relative to the folder.
    ``group_pattern`` is a regex whose first capture group names the subject of each
    file. It must match every file, unless ``group_pattern_optional`` is set, in which
    case a pattern that matches no file at all is ignored. ``slice_band`` and
    ``window`` apply to volumes.
    """
    if isinstance(source, ImageSource):
        return source
    if isinstance(source, (torch.Tensor, np.ndarray)):
        return ArraySource(source)
    if isinstance(source, (str, os.PathLike)):
        return _open_path(
            Path(source), include, group_pattern, group_pattern_optional, slice_band, window
        )
    raise TypeError(
        f"cannot read images from {type(source).__name__}; "
        f"pass a path, a torch.Tensor or a numpy.ndarray"
    )


def _list_files(root: Path, include: str | None) -> list[tuple[Path, str]]:
    """Files under ``root`` with their labels (posix paths relative to it), sorted by label."""
    if root.is_dir():
        found = [(p, p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file()]
    elif root.is_file():
        found = [(root, root.name)]
    else:
        raise FileNotFoundError(f"no such file or directory: {root}")
    if include is not None:
        found = [(p, label) for p, label in found if fnmatch.fnmatch(label, include)]
    # sort on the label string so the order is the same on every platform
    return sorted(found, key=lambda entry: entry[1])


def _open_path(
    root: Path,
    include: str | None,
    group_pattern: str | None,
    group_pattern_optional: bool,
    slice_band: tuple[float, float] | None,
    window: tuple[float, float] | None,
) -> ImageSource:
    items: list = []
    labels: list[str] = []
    groups: list[str | None] = []
    notes: list[str] = []
    for path, label in _list_files(root, include):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(ImageFile(path, label))
            labels.append(label)
            groups.append(None)
    if not items:
        matching = f" matching {include!r}" if include else ""
        raise FileNotFoundError(
            f"no readable images under {root}{matching}; supported: {_SUPPORTED}"
        )

    if group_pattern is not None:
        matched = _groups_from_pattern(labels, group_pattern, group_pattern_optional)
        if matched is not None:
            groups = list(matched)
    known = None if any(g is None for g in groups) else [str(g) for g in groups]
    return ItemSource(items, known, notes)


def _groups_from_pattern(labels: list[str], pattern: str, optional: bool) -> list[str] | None:
    """Subject of each label from the regex's first capture group."""
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"group pattern {pattern!r} is not a valid regular expression: {exc}"
        ) from exc
    if regex.groups < 1:
        raise ValueError(
            f"group pattern {pattern!r} needs one capture group naming the subject, "
            f"for example '(BraTS2021_\\d+)'"
        )
    matches = [regex.search(label) for label in labels]
    if optional and not any(matches):
        return None
    for label, match in zip(labels, matches):
        if match is None:
            raise ValueError(f"group pattern {pattern!r} does not match {label!r}")
    return [match.group(1) for match in matches]
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_io_images.py tests/test_io_arrays.py tests/test_io_sampling.py -v`
Expected: all pass.

Run: `ruff check src tests`
Expected: `All checks passed!` If a line is reported as too long (E501), wrap it; do not raise the limit.

- [ ] **Step 4: Commit**

```bash
git add src/m3score/io tests/_images.py tests/test_io_images.py
git commit -m "feat: read image folders with include filter and subject patterns"
```

---

### Task 7: The metric (`M3Score`, `M3Result`, `score`)

**Files:**
- Create: `src/m3score/metric.py`, `tests/test_metric.py`
- Modify: `src/m3score/__init__.py` (replace whole file), `tests/conftest.py` (append one fixture)

**Interfaces:**
- Consumes: `stats.mmd2_rbf`, `bootstrap_ci`, `permutation_test`, `real_floor`, `memorization`, `coverage`; `backbone.Backbone`, `load_backbone`; `io.open_source`, `sample_indices`, `SkippedItem`, `ImageSource`; `io.arrays.check_images`, `to_float`; `protocol` constants; `M3ScoreWarning`.
- Produces:
  - `M3Score(device=None, *, backbone=None, layers=None, batch_size=32, seed=0, progress=False)`
    - `.update(images, *, real: bool, groups=None) -> None`
    - `.compute(*, stats=True) -> M3Result`
    - `.reset(*, keep_real=False) -> None`
    - `.score(real, gen, *, n=None, stats=True, include=None, group_pattern=None, slice_band=None, window=None) -> M3Result`
    - attributes `device: torch.device`, `layers: tuple[int, int, int]`, `protocol: str`, `batch_size`, `seed`, `progress`
  - `M3Result`: frozen dataclass with the fields of spec section 6, plus `to_dict() -> dict` and `__str__`.
  - `m3score.score(real, gen, *, device=None, backbone=None, layers=None, batch_size=32, seed=0, progress=True, n=None, stats=True, include=None, group_pattern=None, slice_band=None, window=None) -> M3Result`
  - `metric.load_backbone` is looked up as a module global on each call, so tests and the CLI tests replace it with `monkeypatch.setattr(m3score.metric, "load_backbone", ...)`.
- Test fixture: `offline_backbone` makes the default backbone resolve to the tiny ViT under the name `"tiny-test-vit"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/conftest.py`:

```python
@pytest.fixture()
def offline_backbone(monkeypatch, tiny_model):
    """Resolve the default (m3-v1) backbone to the tiny random ViT, so nothing is downloaded."""
    from m3score import backbone, metric

    def load(spec=None, device="cpu"):
        if spec is None:
            return Backbone(tiny_model.to(device), "tiny-test-vit")
        return backbone.load_backbone(spec, device)

    monkeypatch.setattr(metric, "load_backbone", load)
```

Create `tests/test_metric.py`:

```python
import json

import numpy as np
import pytest
import torch

import m3score
from _images import write_png, write_pngs
from m3score import M3Result, M3Score, M3ScoreWarning

pytestmark = pytest.mark.filterwarnings("ignore::m3score.M3ScoreWarning")


def _images(seed, n, size=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, size, size, generator=g)


def _numbers(result):
    return (
        result.fidelity,
        result.memorization,
        result.coverage,
        result.memorization_nn_dist,
        *result.fidelity_ci,
        result.fidelity_p,
        result.fidelity_z,
        result.fidelity_floor,
        result.fidelity_floor_std,
    )


@pytest.fixture()
def metric(offline_backbone):
    return M3Score(device="cpu")


def test_streaming_equals_one_shot(offline_backbone):
    real, gen = _images(0, 40), _images(1, 30)
    streamed = M3Score(device="cpu", batch_size=7)
    for start in range(0, 40, 9):
        streamed.update(real[start : start + 9], real=True)
    streamed.update(gen, real=False)
    one_shot = M3Score(device="cpu").score(real, gen)
    assert _numbers(streamed.compute()) == pytest.approx(_numbers(one_shot), rel=1e-3, abs=1e-4)


def test_result_does_not_depend_on_batch_size(offline_backbone):
    real, gen = _images(0, 40), _images(1, 30)
    small = M3Score(device="cpu", batch_size=4).score(real, gen)
    large = M3Score(device="cpu", batch_size=64).score(real, gen)
    assert _numbers(small) == pytest.approx(_numbers(large), rel=1e-3, abs=1e-4)


def test_same_seed_same_result_and_the_seed_moves_only_the_statistics(offline_backbone):
    real, gen = _images(0, 40), _images(1, 30)
    a = M3Score(device="cpu", seed=3).score(real, gen)
    b = M3Score(device="cpu", seed=3).score(real, gen)
    c = M3Score(device="cpu", seed=4).score(real, gen)
    assert a == b
    assert a.fidelity == c.fidelity
    assert a.fidelity_ci != c.fidelity_ci


def test_reset_can_keep_the_real_features(metric):
    real, gen = _images(0, 40), _images(1, 30)
    metric.update(real, real=True)
    metric.update(gen, real=False)
    first = metric.compute()
    metric.reset(keep_real=True)
    metric.update(gen, real=False)
    assert metric.compute() == first
    metric.reset()
    with pytest.raises(ValueError, match="at least 6 real"):
        metric.compute()


def test_too_few_images_is_an_error(metric):
    metric.update(_images(0, 5), real=True)
    metric.update(_images(1, 5), real=False)
    with pytest.raises(ValueError, match="got 5 real and 5 generated"):
        metric.compute()


def test_statistics_can_be_skipped(metric):
    result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert result.fidelity_ci is None and result.fidelity_p is None
    assert result.fidelity_z is None and result.fidelity_floor is None
    assert 0.0 <= result.coverage <= 1.0
    assert 0.0 <= result.memorization <= 1.0


def test_default_configuration_is_labelled_m3_v1(metric):
    result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert result.protocol == "m3-v1"
    assert result.layers == (12, 9, 4)
    assert result.backbone == "tiny-test-vit"


def test_custom_backbone_requires_layers(tiny_model):
    with pytest.raises(ValueError, match="layers="):
        M3Score(device="cpu", backbone=tiny_model)


def test_custom_configuration_is_labelled_and_warns(tiny_model):
    metric = M3Score(device="cpu", backbone=tiny_model, layers=(12, 8, 3))
    with pytest.warns(M3ScoreWarning, match="not comparable"):
        result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert result.protocol == "custom"
    assert result.layers == (12, 8, 3)
    assert any("not comparable" in w for w in result.warnings)


def test_layers_beyond_the_backbone_depth_are_rejected(tiny_model):
    metric = M3Score(device="cpu", backbone=tiny_model, layers=(13, 9, 4))
    with pytest.raises(ValueError, match="12 blocks"):
        metric.update(_images(0, 8), real=True)


def test_small_sets_are_flagged(metric):
    result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert any("fewer than 500" in w for w in result.warnings)


def test_subjects_are_counted_and_low_diversity_is_flagged(metric):
    metric.update(_images(0, 20), real=True, groups=[f"s{i % 4}" for i in range(20)])
    metric.update(_images(1, 20), real=False)
    result = metric.compute()
    assert result.n_real_subjects == 4
    assert result.n_gen_subjects is None
    assert any("only 4 subjects" in w for w in result.warnings)


def test_partly_labelled_updates_leave_subjects_unknown(metric):
    metric.update(_images(0, 10), real=True, groups=["a"] * 10)
    metric.update(_images(2, 10), real=True)
    metric.update(_images(1, 20), real=False)
    assert metric.compute(stats=False).n_real_subjects is None


def test_group_labels_must_match_the_batch(metric):
    with pytest.raises(ValueError, match="group labels"):
        metric.update(_images(0, 10), real=True, groups=["a"] * 9)


def test_out_of_range_input_is_rejected(metric):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        metric.update(_images(0, 8) * 2 - 1, real=True)


def test_folders_are_scored_and_sampled(metric, tmp_path):
    write_pngs(tmp_path / "real", 30, seed=0)
    write_pngs(tmp_path / "gen", 24, seed=1)
    result = metric.score(tmp_path / "real", tmp_path / "gen", n=12, stats=False)
    assert (result.n_real, result.n_gen) == (12, 12)
    assert any("subjects unknown" in w for w in result.warnings)


def test_group_pattern_need_not_match_the_generated_set(metric, tmp_path):
    for subject in range(8):
        name = f"BraTS2021_{subject:05d}_slice{{:03d}}.png"
        write_pngs(tmp_path / "real", 3, seed=subject, name=name)
    write_pngs(tmp_path / "gen", 20, seed=99, name="generated_{:04d}.png")
    result = metric.score(
        tmp_path / "real", tmp_path / "gen", group_pattern=r"(BraTS2021_\d+)", stats=False
    )
    assert result.n_real_subjects == 8
    assert result.n_gen_subjects is None


def test_group_pattern_must_match_the_real_set(metric, tmp_path):
    write_pngs(tmp_path / "real", 10, name="scan_{:04d}.png")
    write_pngs(tmp_path / "gen", 10)
    with pytest.raises(ValueError, match="does not match"):
        metric.score(tmp_path / "real", tmp_path / "gen", group_pattern=r"(BraTS2021_\d+)")


def test_mixed_image_sizes_are_scored(metric, tmp_path):
    rng = np.random.default_rng(0)
    for i, size in enumerate([(240, 240), (256, 256), (64, 48), (17, 90)] * 3):
        pixels = rng.integers(0, 256, size=size, dtype=np.uint8)
        write_png(tmp_path / "real" / f"r{i:02d}.png", pixels)
    write_pngs(tmp_path / "gen", 10, size=(256, 256), seed=1)
    result = metric.score(tmp_path / "real", tmp_path / "gen", stats=False)
    assert (result.n_real, result.n_gen) == (12, 10)


def test_same_images_as_real_and_generated(metric):
    images = _images(0, 30)
    result = metric.score(images, images.clone())
    assert result.fidelity == 0.0
    assert result.coverage == 1.0
    assert result.memorization == 1.0
    assert all(np.isfinite(v) for v in _numbers(result))


def test_blank_generated_images_give_finite_values(metric):
    result = metric.score(_images(0, 30), torch.zeros(12, 1, 32, 32))
    assert all(np.isfinite(v) for v in _numbers(result))
    assert result.fidelity > 0.0


def test_unequal_set_sizes_and_oversized_n(metric):
    result = metric.score(_images(0, 8), _images(1, 60), n=1000)
    assert (result.n_real, result.n_gen) == (8, 60)
    swapped = metric.score(_images(1, 60), _images(0, 3), n=1000, stats=False)
    assert (swapped.n_real, swapped.n_gen) == (60, 3)


def test_cuda_request_without_cuda_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        M3Score(device="cuda")


def test_unreadable_files_are_reported_and_left_out(metric, tmp_path):
    write_pngs(tmp_path / "real", 10)
    (tmp_path / "real" / "broken.png").write_bytes(b"not a png")
    write_pngs(tmp_path / "gen", 10, seed=1)
    result = metric.score(tmp_path / "real", tmp_path / "gen", stats=False)
    assert result.n_real == 10
    assert any("broken.png" in w for w in result.warnings)


def test_result_serialises_and_prints(metric):
    result = metric.score(_images(0, 20), _images(1, 20))
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["protocol"] == "m3-v1"
    assert payload["layers"] == [12, 9, 4]
    assert isinstance(payload["fidelity_ci"], list)
    text = str(result)
    for word in ("fidelity", "memorization", "coverage", "floor", "permutation test", "warnings"):
        assert word in text
    assert text.isascii()


def test_module_level_score(offline_backbone):
    result = m3score.score(
        _images(0, 20), _images(1, 20), device="cpu", stats=False, progress=False
    )
    assert isinstance(result, M3Result)
```

Run: `pytest tests/test_metric.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'M3Result' from 'm3score'`.

- [ ] **Step 2: Implement `metric.py`**

Create `src/m3score/metric.py`:

```python
"""The M3-Score metric: accumulate features, then compute the three axes."""

from __future__ import annotations

import dataclasses
import logging
import warnings as _warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from . import protocol as _protocol
from . import stats as _stats
from ._errors import M3ScoreWarning
from ._version import __version__
from .backbone import Backbone, load_backbone
from .io import ImageSource, SkippedItem, open_source, sample_indices
from .io.arrays import check_images, to_float

_log = logging.getLogger("m3score")
_DEFAULT_BACKBONE = f"{_protocol.BACKBONE_ID}@{_protocol.BACKBONE_REVISION}"


@dataclass(frozen=True)
class M3Result:
    """The three M3 axes for one real/generated comparison, and what produced them.

    ``fidelity`` is the MMD^2 at the fidelity layer (lower is better),
    ``memorization`` the near-duplicate rate (lower is better), ``coverage`` the
    real-radius coverage (higher is better). The statistics are ``None`` when they
    were skipped.
    """

    fidelity: float
    memorization: float
    coverage: float
    memorization_nn_dist: float
    n_real: int
    n_gen: int
    fidelity_ci: tuple[float, float] | None = None
    fidelity_p: float | None = None
    fidelity_z: float | None = None
    fidelity_floor: float | None = None
    fidelity_floor_std: float | None = None
    n_real_subjects: int | None = None
    n_gen_subjects: int | None = None
    protocol: str = _protocol.PROTOCOL_VERSION
    backbone: str = _DEFAULT_BACKBONE
    layers: tuple[int, int, int] = _protocol.LAYERS
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """A JSON-serialisable copy."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dataclasses.asdict(self).items()
        }

    def __str__(self) -> str:
        return _format(self)


def _count_line(label: str, n: int, subjects: int | None) -> str:
    line = f"  {label:<10} {n:,} images"
    return line if subjects is None else f"{line}, {subjects:,} subjects"


def _format(result: M3Result) -> str:
    lines = [
        f"M3-Score  protocol {result.protocol}  (m3-score {__version__})",
        _count_line("real", result.n_real, result.n_real_subjects),
        _count_line("generated", result.n_gen, result.n_gen_subjects),
        "",
        f"  {'axis':<14}{'value':<9}{'better':<9}detail",
    ]
    detail = []
    if result.fidelity_ci is not None:
        low, high = result.fidelity_ci
        detail.append(f"95% CI [{low:.4f}, {high:.4f}]")
    if result.fidelity_floor is not None:
        detail.append(
            f"real-vs-real floor {result.fidelity_floor:.4f} +/- {result.fidelity_floor_std:.4f}"
        )
    lines.append(
        f"  {'fidelity':<14}{result.fidelity:<9.4f}{'lower':<9}{'; '.join(detail)}".rstrip()
    )
    lines.append(
        f"  {'memorization':<14}{result.memorization:<9.3f}{'lower':<9}"
        f"mean nearest-real distance {result.memorization_nn_dist:.3g}"
    )
    lines.append(
        f"  {'coverage':<14}{result.coverage:<9.3f}{'higher':<9}k = {_protocol.COVERAGE_K}"
    )
    if result.fidelity_p is not None:
        lines += [
            "",
            f"  permutation test: z = {result.fidelity_z:.1f}, p = {result.fidelity_p:.3g} "
            f"(descriptive, not a model-selection tool)",
        ]
    if result.warnings:
        lines += ["", "  warnings:", *(f"    - {note}" for note in result.warnings)]
    return "\n".join(lines)


class M3Score:
    """Accumulates image features, then computes fidelity, memorization and coverage.

    Use :meth:`score` for two complete image sets, or :meth:`update` batch by batch
    followed by :meth:`compute`. The defaults give the ``m3-v1`` protocol. Passing
    ``backbone`` or ``layers`` gives a result labelled ``custom`` that is not
    comparable with ``m3-v1`` values.
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        backbone: str | torch.nn.Module | None = None,
        layers: Sequence[int] | None = None,
        batch_size: int = 32,
        seed: int = 0,
        progress: bool = False,
    ) -> None:
        if backbone is not None and layers is None:
            raise ValueError(
                "a custom backbone needs layers=(fidelity, memorization, coverage); the m3-v1 "
                "depths 12/9/4 belong to RadioDINO-s16 and do not transfer to another encoder"
            )
        if layers is not None:
            layers = tuple(int(value) for value in layers)
            if len(layers) != 3 or min(layers) < 1:
                raise ValueError(
                    "layers must be three block numbers counted from 1: "
                    "(fidelity, memorization, coverage)"
                )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "device 'cuda' was requested but CUDA is not available; use device='cpu'"
            )

        custom = backbone is not None or layers is not None
        self.layers: tuple[int, int, int] = layers if layers is not None else _protocol.LAYERS
        self.protocol = _protocol.CUSTOM_PROTOCOL if custom else _protocol.PROTOCOL_VERSION
        self.batch_size = batch_size
        self.seed = seed
        self.progress = progress
        self._backbone_spec = backbone
        self._backbone: Backbone | None = None
        # keyed by True (real) / False (generated)
        self._features: dict[bool, dict[int, list[torch.Tensor]]] = {}
        self._groups: dict[bool, list[str]] = {}
        self._ungrouped: dict[bool, bool] = {}
        self._notes: dict[bool, list[str]] = {}
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self, *, keep_real: bool = False) -> None:
        """Forget accumulated images. ``keep_real`` keeps the real set for the next round."""
        for is_real in (False,) if keep_real else (True, False):
            self._features[is_real] = {layer: [] for layer in set(self.layers)}
            self._groups[is_real] = []
            self._ungrouped[is_real] = False
            self._notes[is_real] = []

    def _get_backbone(self) -> Backbone:
        if self._backbone is None:
            backbone = load_backbone(self._backbone_spec, self.device)
            if max(self.layers) > backbone.num_layers:
                raise ValueError(
                    f"layers {self.layers} reach beyond the backbone's "
                    f"{backbone.num_layers} blocks"
                )
            _log.info("loaded backbone %s on %s", backbone.name, self.device)
            self._backbone = backbone
        return self._backbone

    def _ingest(self, prepared: torch.Tensor, real: bool, groups: Sequence | None) -> None:
        features = self._get_backbone().extract(prepared, self.layers, self.device)
        for layer, value in features.items():
            self._features[real][layer].append(value)
        if groups is None:
            self._ungrouped[real] = True
        else:
            self._groups[real].extend(str(group) for group in groups)

    def _known_groups(self, real: bool) -> list[str] | None:
        """Subject of every accumulated image, or ``None`` unless all of them were labelled."""
        if self._ungrouped[real] or not self._groups[real]:
            return None
        return list(self._groups[real])

    # ------------------------------------------------------------------ input

    def update(self, images, *, real: bool, groups: Sequence | None = None) -> None:
        """Add a batch of images.

        ``images`` is a tensor or array shaped ``(N, H, W)``, ``(N, 1, H, W)`` or
        ``(N, 3, H, W)``: uint8, or floating point in [0, 1]. ``groups`` optionally
        gives one subject identifier per image.
        """
        tensor, divisor = check_images(images)
        if groups is not None and len(groups) != tensor.shape[0]:
            raise ValueError(f"got {len(groups)} group labels for {tensor.shape[0]} images")
        backbone = self._get_backbone()
        for start in range(0, tensor.shape[0], self.batch_size):
            stop = start + self.batch_size
            prepared = backbone.prepare(to_float(tensor[start:stop], divisor))
            self._ingest(prepared, real, None if groups is None else list(groups[start:stop]))

    def _ingest_source(self, source: ImageSource, indices: list[int], real: bool) -> None:
        backbone = self._get_backbone()
        description = "m3score: real" if real else "m3score: generated"
        starts = range(0, len(indices), self.batch_size)
        for start in tqdm(starts, desc=description, disable=not self.progress):
            images, groups = [], []
            for index in indices[start : start + self.batch_size]:
                try:
                    image = source.load(index)
                except SkippedItem as skipped:
                    self._notes[real].append(f"skipped {skipped}")
                    continue
                # each image is resized on its own, so sizes may differ within a source
                images.append(backbone.prepare(image[None]))
                if source.groups is not None:
                    groups.append(source.groups[index])
            if images:
                batch_groups = groups if source.groups is not None else None
                self._ingest(torch.cat(images), real, batch_groups)

    def score(
        self,
        real,
        gen,
        *,
        n: int | None = None,
        stats: bool = True,
        include: str | None = None,
        group_pattern: str | None = None,
        slice_band: tuple[float, float] | None = None,
        window: tuple[float, float] | None = None,
    ) -> M3Result:
        """Score a real set against a generated set. Resets the metric first.

        Each of ``real`` and ``gen`` is a folder or file path, a tensor, or an array.
        ``n`` images are drawn at random from each set (default: all, up to 5,000).
        ``group_pattern`` must match every file of the real set; it is applied to the
        generated set only if it matches there too.
        """
        self.reset()
        for is_real, origin, seed in ((True, real, self.seed), (False, gen, self.seed + 1)):
            source = open_source(
                origin,
                include=include,
                group_pattern=group_pattern,
                group_pattern_optional=not is_real,
                slice_band=slice_band,
                window=window,
            )
            self._notes[is_real].extend(source.warnings)
            indices = sample_indices(len(source), n, source.groups, seed)
            _log.info(
                "%s set: scoring %d of %d images",
                "real" if is_real else "generated",
                len(indices),
                len(source),
            )
            if is_real and source.groups is None and len(indices) < len(source):
                self._notes[True].append(
                    f"real set: {len(indices):,} of {len(source):,} images drawn at random with "
                    f"subjects unknown; pass group_pattern (--group-pattern) to balance the "
                    f"draw across subjects"
                )
            self._ingest_source(source, indices, is_real)
        return self.compute(stats=stats)

    # ---------------------------------------------------------------- compute

    def compute(self, *, stats: bool = True) -> M3Result:
        """The three axes for everything accumulated so far.

        ``stats=False`` skips the bootstrap interval, permutation test and
        real-vs-real floor.
        """
        fidelity_layer, memorization_layer, coverage_layer = self.layers
        n_real = sum(f.shape[0] for f in self._features[True][fidelity_layer])
        n_gen = sum(f.shape[0] for f in self._features[False][fidelity_layer])
        if n_real < _protocol.MIN_REAL_IMAGES or n_gen < _protocol.MIN_GEN_IMAGES:
            raise ValueError(
                f"need at least {_protocol.MIN_REAL_IMAGES} real and "
                f"{_protocol.MIN_GEN_IMAGES} generated images; "
                f"got {n_real} real and {n_gen} generated"
            )

        def features(real: bool, layer: int) -> torch.Tensor:
            return torch.cat(self._features[real][layer]).to(self.device)

        real_groups, gen_groups = self._known_groups(True), self._known_groups(False)
        notes = [*self._notes[True], *self._notes[False]]

        real_f, gen_f = features(True, fidelity_layer), features(False, fidelity_layer)
        fidelity = _stats.mmd2_rbf(real_f, gen_f)
        interval = p_value = z_score = floor = floor_std = None
        if stats:
            interval = _stats.bootstrap_ci(real_f, gen_f, seed=self.seed)
            p_value, z_score, _ = _stats.permutation_test(real_f, gen_f, seed=self.seed)
            floor_result = _stats.real_floor(real_f, real_groups, seed=self.seed)
            if floor_result is not None:
                floor, floor_std, subject_disjoint = floor_result
                if real_groups is not None and not subject_disjoint:
                    notes.append(
                        "the real-vs-real floor was split by image, not by subject, "
                        "because only one subject is known"
                    )
        memorization, nn_dist = _stats.memorization(
            features(True, memorization_layer), features(False, memorization_layer)
        )
        coverage = _stats.coverage(features(True, coverage_layer), features(False, coverage_layer))

        n_real_subjects = len(set(real_groups)) if real_groups is not None else None
        n_gen_subjects = len(set(gen_groups)) if gen_groups is not None else None
        if min(n_real, n_gen) < _protocol.MIN_IMAGES_WARN:
            notes.append(
                f"fewer than {_protocol.MIN_IMAGES_WARN} images in a set ({n_real:,} real, "
                f"{n_gen:,} generated): estimates are less repeatable at this size"
            )
        if n_real_subjects is not None and n_real_subjects < _protocol.MIN_SUBJECTS_WARN:
            notes.append(
                f"the real set spans only {n_real_subjects} subjects: low subject diversity "
                f"inflates distances and weakens the comparison"
            )
        if self.protocol != _protocol.PROTOCOL_VERSION:
            notes.append("custom backbone or layers: values are not comparable with m3-v1 results")

        unique_notes = tuple(dict.fromkeys(notes))
        for note in unique_notes:
            _warnings.warn(note, M3ScoreWarning, stacklevel=2)
        return M3Result(
            fidelity=fidelity,
            memorization=memorization,
            coverage=coverage,
            memorization_nn_dist=nn_dist,
            n_real=n_real,
            n_gen=n_gen,
            fidelity_ci=interval,
            fidelity_p=p_value,
            fidelity_z=z_score,
            fidelity_floor=floor,
            fidelity_floor_std=floor_std,
            n_real_subjects=n_real_subjects,
            n_gen_subjects=n_gen_subjects,
            protocol=self.protocol,
            backbone=self._get_backbone().name,
            layers=self.layers,
            warnings=unique_notes,
        )


def score(
    real,
    gen,
    *,
    device: torch.device | str | None = None,
    backbone: str | torch.nn.Module | None = None,
    layers: Sequence[int] | None = None,
    batch_size: int = 32,
    seed: int = 0,
    progress: bool = True,
    n: int | None = None,
    stats: bool = True,
    include: str | None = None,
    group_pattern: str | None = None,
    slice_band: tuple[float, float] | None = None,
    window: tuple[float, float] | None = None,
) -> M3Result:
    """Score a real image set against a generated one in a single call.

    Builds an :class:`M3Score`, which loads the backbone. To score several pairs, make
    one ``M3Score`` and call its ``score`` method repeatedly.
    """
    metric = M3Score(
        device,
        backbone=backbone,
        layers=layers,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
    )
    return metric.score(
        real,
        gen,
        n=n,
        stats=stats,
        include=include,
        group_pattern=group_pattern,
        slice_band=slice_band,
        window=window,
    )
```

Replace `src/m3score/__init__.py` with:

```python
"""M3-Score: fidelity, memorization and coverage for generative radiology image models."""

from ._errors import M3ScoreError, M3ScoreWarning
from ._version import __version__
from .metric import M3Result, M3Score, score
from .protocol import PROTOCOL_VERSION

__all__ = [
    "M3Result",
    "M3Score",
    "M3ScoreError",
    "M3ScoreWarning",
    "PROTOCOL_VERSION",
    "__version__",
    "score",
]
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_metric.py -v`
Expected: 26 passed.

Run: `pytest -q`
Expected: every test from Tasks 1-7 passes.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/m3score/metric.py src/m3score/__init__.py tests/conftest.py tests/test_metric.py
git commit -m "feat: add M3Score metric with streaming update, compute and one-shot score"
```

---

### Task 8: Command line

**Files:**
- Create: `src/m3score/cli.py`, `src/m3score/__main__.py`, `tests/test_cli.py`
- Modify: `pyproject.toml` (add the `[project.scripts]` table)

**Interfaces:**
- Consumes: `M3Score`, `M3ScoreError`, `M3ScoreWarning`, `__version__`; `M3Result.__str__`, `M3Result.to_dict`.
- Produces: `cli.build_parser() -> argparse.ArgumentParser`; `cli.main(argv: Sequence[str] | None = None) -> int` returning the exit status (0 success, 2 usage or input error, 1 `M3ScoreError`); console script `m3score`; `python -m m3score`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import json
import subprocess
import sys

import pytest

from _images import write_pngs
from m3score import M3Score, M3ScoreError, __version__, metric
from m3score.cli import main


@pytest.fixture()
def folders(tmp_path):
    write_pngs(tmp_path / "real", 14, seed=0)
    write_pngs(tmp_path / "gen", 12, seed=1)
    return str(tmp_path / "real"), str(tmp_path / "gen")


def test_prints_the_three_axes(offline_backbone, folders, capsys):
    assert main([*folders, "--device", "cpu", "--quiet"]) == 0
    out = capsys.readouterr().out
    for word in ("M3-Score", "m3-v1", "fidelity", "memorization", "coverage", "14 images"):
        assert word in out
    assert out.isascii()


def test_writes_json(offline_backbone, folders, tmp_path):
    target = tmp_path / "result.json"
    status = main([*folders, "--device", "cpu", "--quiet", "--no-stats", "--json", str(target)])
    assert status == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["m3score_version"] == __version__
    assert payload["n_real"] == 14
    assert payload["fidelity_ci"] is None


def test_n_is_passed_through(offline_backbone, folders, capsys):
    assert main([*folders, "--device", "cpu", "--quiet", "--no-stats", "--n", "8"]) == 0
    assert "8 images" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::m3score.M3ScoreWarning")
def test_command_line_and_python_api_agree(offline_backbone, folders, tmp_path):
    target = tmp_path / "result.json"
    arguments = [*folders, "--device", "cpu", "--quiet", "--seed", "5", "--n", "10"]
    assert main([*arguments, "--json", str(target)]) == 0
    from_cli = json.loads(target.read_text(encoding="utf-8"))
    from_python = M3Score(device="cpu", seed=5).score(*folders, n=10).to_dict()
    assert {key: from_cli[key] for key in from_python} == from_python


def test_input_errors_exit_2_with_one_line(offline_backbone, tmp_path, capsys):
    absent = str(tmp_path / "absent")
    assert main([absent, absent, "--device", "cpu", "--quiet"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("m3score: error: ")
    assert err.count("\n") == 1


def test_backbone_failure_exits_1_with_one_line(monkeypatch, folders, capsys):
    def refuse(spec=None, device="cpu"):
        raise M3ScoreError("could not load the M3-Score backbone")

    monkeypatch.setattr(metric, "load_backbone", refuse)
    assert main([*folders, "--device", "cpu", "--quiet"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("m3score: error: could not load")
    assert err.count("\n") == 1


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_module_entry_point():
    done = subprocess.run(
        [sys.executable, "-m", "m3score", "--version"], capture_output=True, text=True
    )
    assert done.returncode == 0
    assert __version__ in done.stdout
```

Run: `pytest tests/test_cli.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'm3score.cli'`.

- [ ] **Step 2: Implement the command line**

Create `src/m3score/cli.py`:

```python
"""Command line: ``m3score REAL GEN``."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence

from ._errors import M3ScoreError, M3ScoreWarning
from ._version import __version__
from .metric import M3Score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m3score",
        description=(
            "Score generated radiology images against real ones on the three M3 axes: "
            "fidelity, memorization and coverage."
        ),
    )
    parser.add_argument(
        "real", metavar="REAL", help="folder or file of real images: PNG/JPG/TIFF, NIfTI or DICOM"
    )
    parser.add_argument("gen", metavar="GEN", help="folder or file of generated images")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="images to score per set, drawn at random (default: all, up to 5000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling and statistics")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32, metavar="B")
    parser.add_argument(
        "--include",
        metavar="GLOB",
        help='only read files whose path inside the folder matches, e.g. "*_flair.nii.gz"',
    )
    parser.add_argument(
        "--group-pattern",
        metavar="REGEX",
        help=(
            "regular expression whose first capture group names the subject of each real "
            'file, e.g. "(BraTS2021_\\d+)"; enables subject-balanced sampling'
        ),
    )
    parser.add_argument(
        "--slice-band",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="fraction of each volume's depth to keep (default: 0.355 0.555)",
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help=(
            "intensity window for volumes, e.g. -1000 400 for CT "
            "(default: the stored window, else the 0.5-99.5 percentile range)"
        ),
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="skip the confidence interval, permutation test and real-vs-real floor",
    )
    parser.add_argument("--json", metavar="FILE", help="also write the result as JSON")
    parser.add_argument("--quiet", action="store_true", help="no progress bars")
    parser.add_argument("--version", action="version", version=f"m3-score {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metric = M3Score(
            None if args.device == "auto" else args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            progress=not args.quiet,
        )
        with warnings.catch_warnings():
            # warnings are listed under the result instead of interleaving with it
            warnings.simplefilter("ignore", M3ScoreWarning)
            result = metric.score(
                args.real,
                args.gen,
                n=args.n,
                stats=not args.no_stats,
                include=args.include,
                group_pattern=args.group_pattern,
                slice_band=tuple(args.slice_band) if args.slice_band else None,
                window=tuple(args.window) if args.window else None,
            )
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"m3score: error: {exc}", file=sys.stderr)
        return 2
    except M3ScoreError as exc:
        print(f"m3score: error: {exc}", file=sys.stderr)
        return 1

    print(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"m3score_version": __version__, **result.to_dict()}, handle, indent=2)
            handle.write("\n")
    return 0
```

Create `src/m3score/__main__.py`:

```python
"""``python -m m3score``."""

from .cli import main

raise SystemExit(main())
```

In `pyproject.toml`, insert this table between `[project.optional-dependencies]` and `[project.urls]`:

```toml
[project.scripts]
m3score = "m3score.cli:main"
```

- [ ] **Step 3: Reinstall so the command is registered, then run the tests**

Run: `pip install -e . --no-deps`
Expected: `Successfully installed m3-score-0.1.0`.

Run: `m3score --version`
Expected: `m3-score 0.1.0`

Run: `pytest tests/test_cli.py -v`
Expected: 8 passed.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 4: Run it on real images with the real backbone**

Run: `m3score output/generated_500_best output/generated_retinal --n 100`
Expected: progress bars, then the table with three axes, `protocol m3-v1`, and warnings that fewer than 500 images were used and that real subjects are unknown. The backbone loads from the local Hugging Face cache. Record the printed numbers in the task report.

- [ ] **Step 5: Commit**

```bash
git add src/m3score/cli.py src/m3score/__main__.py tests/test_cli.py pyproject.toml
git commit -m "feat: add m3score command line"
```

---

## Phase 2: volumes

### Task 9: NIfTI volumes

**Files:**
- Create: `src/m3score/io/volumes.py`, `src/m3score/io/nifti.py`, `tests/test_io_volumes.py`, `tests/test_io_nifti.py`
- Modify: `src/m3score/io/__init__.py` (imports, `_SUPPORTED`, and the `_open_path` function)

**Interfaces:**
- Consumes: `protocol.SLICE_BAND`, `INTENSITY_PERCENTILES`; `io.base.SkippedItem`; `_list_files`, `_groups_from_pattern`, `ItemSource`, `ImageFile`, `IMAGE_SUFFIXES` from `io/__init__.py`.
- Produces:
  - `io.volumes.check_band(band) -> tuple[float, float]` and `check_window(window) -> tuple[float, float]`, both raising `ValueError`
  - `io.volumes.band_indices(depth: int, band: tuple[float, float]) -> range`
  - `io.volumes.scale_intensity(volume: np.ndarray, window: tuple[float, float] | None) -> np.ndarray | None` (float32 in `[0, 1]`, or `None` when empty or constant)
  - `io.nifti.is_nifti(path: Path) -> bool`
  - `io.nifti.nifti_items(path: Path, label: str, band, window) -> list` of objects with `load() -> torch.Tensor` of shape `(1, H, W)`
  - `open_source` now reads `.nii` and `.nii.gz`; each file is one subject unless `group_pattern` says otherwise.

- [ ] **Step 1: Install nibabel on its own**

Run: `pip install "nibabel>=4.0"`
Expected: installs nibabel; `pip list | grep -i torch` still shows `torch 2.6.0+cu124`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_io_volumes.py`:

```python
import numpy as np
import pytest

from m3score import protocol
from m3score.io.volumes import band_indices, check_band, check_window, scale_intensity


def test_default_band_is_slices_55_to_85_of_155():
    assert band_indices(155, protocol.SLICE_BAND) == range(55, 86)


def test_band_scales_with_depth_and_is_never_empty():
    assert band_indices(10, (0.0, 1.0)) == range(0, 10)
    assert band_indices(64, protocol.SLICE_BAND) == range(23, 36)
    assert len(band_indices(1, protocol.SLICE_BAND)) == 1
    assert len(band_indices(3, protocol.SLICE_BAND)) == 1


@pytest.mark.parametrize("band", [(0.5, 0.5), (0.6, 0.4), (-0.1, 0.5), (0.2, 1.1)])
def test_invalid_bands_are_rejected(band):
    with pytest.raises(ValueError, match="slice band"):
        check_band(band)


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError, match="intensity window"):
        check_window((400, -1000))


def test_percentile_scaling_ignores_background_zeros():
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[2:8, 2:8, 2:8] = np.linspace(100, 200, 216).reshape(6, 6, 6)
    scaled = scale_intensity(volume, None)
    assert scaled.dtype == np.float32
    assert scaled.min() == 0.0 and scaled.max() == 1.0
    assert (scaled[volume == 0] == 0.0).all()
    assert 0.45 < scaled[volume > 0].mean() < 0.55


def test_an_explicit_window_is_applied_exactly():
    volume = np.array([[[-1000.0, -300.0, 400.0, 2000.0]]], dtype=np.float32)
    scaled = scale_intensity(volume, (-1000.0, 400.0))
    assert scaled.flatten().tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])


def test_empty_or_constant_volumes_cannot_be_scaled():
    assert scale_intensity(np.zeros((4, 4, 4), dtype=np.float32), None) is None
    assert scale_intensity(np.full((4, 4, 4), 7.0, dtype=np.float32), None) is None
```

Create `tests/test_io_nifti.py`:

```python
import sys

import numpy as np
import pytest
import torch

from m3score import M3Score
from m3score.io import SkippedItem, open_source
from m3score.io.nifti import _require_nibabel, is_nifti

nib = pytest.importorskip("nibabel")

SHAPE = (6, 7, 20)  # left-right, posterior-anterior, inferior-superior
BAND = range(7, 11)  # the default band for a depth of 20
WINDOW = (0.0, 30000.0)  # wider than any test volume, so scaling is a plain division


def _ras_volume(seed=0):
    """Values rise towards the patient's right, the anterior and the top, over a random base."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing="ij")
    return (100.0 + 10.0 * x + 100.0 * y + 1000.0 * z + rng.random(SHAPE)).astype(np.float32)


def _save(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))
    return path


def test_nifti_suffixes_are_recognised_in_any_case(tmp_path):
    assert is_nifti(tmp_path / "a.nii")
    assert is_nifti(tmp_path / "a.nii.gz")
    assert is_nifti(tmp_path / "A.NII.GZ")
    assert not is_nifti(tmp_path / "a.png")


def test_a_volume_yields_its_central_band(tmp_path):
    _save(tmp_path / "subject1.nii.gz", _ras_volume())
    source = open_source(tmp_path)
    assert len(source) == len(BAND)
    assert source.groups == ["subject1.nii.gz"] * len(BAND)
    image = source.load(0)
    assert image.shape == (1, SHAPE[1], SHAPE[0])
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_the_first_item_is_the_first_slice_of_the_band(tmp_path):
    data = _ras_volume()
    _save(tmp_path / "v.nii.gz", data)
    image = open_source(tmp_path, window=WINDOW).load(0)[0]
    expected = data[:, :, BAND.start][::-1, ::-1].T / WINDOW[1]
    assert np.allclose(image.numpy(), expected, atol=1e-6)


def test_slices_are_shown_anterior_up_with_the_patient_right_on_the_left(tmp_path):
    _save(tmp_path / "v.nii.gz", _ras_volume())
    image = open_source(tmp_path, window=WINDOW).load(0)[0]
    assert image[0, 0] > image[-1, 0]  # the top row is more anterior than the bottom row
    assert image[0, 0] > image[0, -1]  # the left column is further to the patient's right


def test_stored_orientation_does_not_change_the_slices(tmp_path):
    data = _ras_volume()
    _save(tmp_path / "ras" / "v.nii.gz", data)
    # the same anatomy stored left-posterior-superior, as BraTS files are
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    _save(tmp_path / "lps" / "v.nii.gz", data[::-1, ::-1, :].copy(), lps)
    # and with the array axes stored in the order superior, right, anterior
    sra = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    _save(tmp_path / "sra" / "v.nii.gz", np.transpose(data, (2, 0, 1)).copy(), sra)

    reference = open_source(tmp_path / "ras", window=WINDOW)
    for name in ("lps", "sra"):
        other = open_source(tmp_path / name, window=WINDOW)
        assert len(other) == len(reference) == len(BAND)
        for index in range(len(reference)):
            assert torch.allclose(other.load(index), reference.load(index), atol=1e-6)


def test_each_file_is_a_subject_and_include_filters_them(tmp_path):
    for name in ("a_flair.nii.gz", "b_flair.nii.gz", "a_seg.nii.gz"):
        _save(tmp_path / name, _ras_volume())
    assert len(set(open_source(tmp_path).groups)) == 3
    flair = open_source(tmp_path, include="*_flair.nii.gz")
    assert sorted(set(flair.groups)) == ["a_flair.nii.gz", "b_flair.nii.gz"]


def test_group_pattern_overrides_the_file_as_subject(tmp_path):
    for name in ("P01_t1.nii.gz", "P01_t2.nii.gz", "P02_t1.nii.gz"):
        _save(tmp_path / name, _ras_volume())
    source = open_source(tmp_path, group_pattern=r"(P\d+)")
    assert sorted(set(source.groups)) == ["P01", "P02"]


def test_the_slice_band_can_be_overridden(tmp_path):
    _save(tmp_path / "v.nii.gz", _ras_volume())
    assert len(open_source(tmp_path, slice_band=(0.0, 1.0))) == SHAPE[2]


def test_a_4d_file_is_an_error_naming_the_file(tmp_path):
    _save(tmp_path / "bold.nii.gz", np.zeros((4, 4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"bold\.nii\.gz.*3D"):
        open_source(tmp_path)


def test_a_trailing_singleton_dimension_is_accepted(tmp_path):
    _save(tmp_path / "v.nii.gz", _ras_volume()[..., None])
    assert open_source(tmp_path).load(0).shape == (1, SHAPE[1], SHAPE[0])


def test_a_constant_volume_is_skipped_with_its_name(tmp_path):
    _save(tmp_path / "blank.nii.gz", np.zeros(SHAPE, dtype=np.float32))
    with pytest.raises(SkippedItem, match="blank.nii.gz"):
        open_source(tmp_path).load(0)


def test_missing_nibabel_explains_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "nibabel", None)
    with pytest.raises(ImportError, match=r"m3-score\[nifti\]"):
        _require_nibabel()


@pytest.mark.filterwarnings("ignore::m3score.M3ScoreWarning")
def test_volumes_are_scored_end_to_end(offline_backbone, tmp_path):
    for i in range(3):
        _save(tmp_path / "real" / f"r{i}.nii.gz", _ras_volume(seed=i))
        _save(tmp_path / "gen" / f"g{i}.nii.gz", _ras_volume(seed=10 + i))
    _save(tmp_path / "real" / "blank.nii.gz", np.zeros(SHAPE, dtype=np.float32))
    result = M3Score(device="cpu").score(tmp_path / "real", tmp_path / "gen", stats=False)
    assert result.n_real == 3 * len(BAND)
    assert result.n_real_subjects == 3
    assert result.n_gen_subjects == 3
    assert any("blank.nii.gz" in note for note in result.warnings)
```

Run: `pytest tests/test_io_volumes.py tests/test_io_nifti.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'm3score.io.volumes'`.

- [ ] **Step 3: Implement the volume helpers and the NIfTI reader**

Create `src/m3score/io/volumes.py`:

```python
"""What NIfTI and DICOM volumes share: which slices are kept, how intensities are scaled."""

from __future__ import annotations

import math

import numpy as np

from .. import protocol


def check_band(band) -> tuple[float, float]:
    low, high = float(band[0]), float(band[1])
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"slice band must satisfy 0 <= LO < HI <= 1, got ({low:g}, {high:g})")
    return low, high


def check_window(window) -> tuple[float, float]:
    low, high = float(window[0]), float(window[1])
    if not high > low:
        raise ValueError(f"intensity window must satisfy LO < HI, got ({low:g}, {high:g})")
    return low, high


def band_indices(depth: int, band: tuple[float, float]) -> range:
    """Slice indices kept from a volume of ``depth`` slices: a fractional band, never empty.

    The default band (55/155, 86/155) keeps slices 55 to 85 of a 155-slice BraTS volume.
    """
    low, high = check_band(band)
    start = min(max(int(math.floor(low * depth + 0.5)), 0), depth - 1)
    stop = min(max(int(math.floor(high * depth + 0.5)), start + 1), depth)
    return range(start, stop)


def scale_intensity(
    volume: np.ndarray, window: tuple[float, float] | None
) -> np.ndarray | None:
    """Map intensities to [0, 1] as float32.

    With a ``window`` the values are clipped to it. Without one, the range is the
    0.5 to 99.5 percentile of the non-zero voxels, so a zero background stays black.
    Returns ``None`` when the volume is empty or constant.
    """
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if window is not None:
        low, high = window
    else:
        foreground = volume[volume != 0]
        if foreground.size == 0:
            return None
        low, high = np.percentile(foreground, protocol.INTENSITY_PERCENTILES)
    if not high > low:
        return None
    return np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32)
```

Create `src/m3score/io/nifti.py`:

```python
"""NIfTI volumes. Needs nibabel: ``pip install "m3-score[nifti]"``."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

from .base import SkippedItem
from .volumes import band_indices, scale_intensity

NIFTI_SUFFIXES = (".nii", ".nii.gz")
_SUPERIOR_AXIS = 2  # after reorientation to RAS+


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def _require_nibabel():
    try:
        import nibabel
    except ImportError as exc:
        raise ImportError('NIfTI input needs nibabel: pip install "m3-score[nifti]"') from exc
    return nibabel


@functools.lru_cache(maxsize=2)
def _load_volume(path: str, window: tuple[float, float] | None) -> np.ndarray | None:
    """The volume in RAS+ axes scaled to [0, 1], or ``None`` when empty or constant."""
    nibabel = _require_nibabel()
    image = nibabel.as_closest_canonical(nibabel.load(path))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim == 4 and volume.shape[3] == 1:
        volume = volume[..., 0]
    return scale_intensity(volume, window)


class NiftiSlice:
    """One axial slice of a volume, decoded on demand."""

    def __init__(
        self, path: Path, label: str, index: int, window: tuple[float, float] | None
    ) -> None:
        self.path = path
        self.label = label
        self.index = index
        self.window = window

    def load(self) -> torch.Tensor:
        volume = _load_volume(str(self.path), self.window)
        if volume is None:
            raise SkippedItem(f"{self.label}: volume is empty or constant")
        plane = volume[:, :, self.index]
        # A RAS+ plane is indexed [left->right, posterior->anterior]. Radiological
        # display puts anterior at the top and the patient's right on the image left.
        return torch.from_numpy(np.ascontiguousarray(plane[::-1, ::-1].T))[None]


def nifti_items(
    path: Path, label: str, band: tuple[float, float], window: tuple[float, float] | None
) -> list[NiftiSlice]:
    """The slices of one volume that fall inside ``band``. Reads only the header."""
    nibabel = _require_nibabel()
    image = nibabel.load(str(path))
    shape = tuple(image.shape)
    if len(shape) == 4 and shape[3] == 1:
        shape = shape[:3]
    if len(shape) != 3:
        raise ValueError(
            f"{label}: expected a 3D volume but found shape {tuple(image.shape)}; "
            f"split 4D files into 3D volumes first"
        )
    # which stored axis becomes inferior->superior once the volume is reoriented
    orientation = nibabel.orientations.io_orientation(image.affine)
    stored_axis = np.flatnonzero(orientation[:, 0] == _SUPERIOR_AXIS)
    depth = int(shape[int(stored_axis[0])]) if stored_axis.size else int(shape[2])
    return [NiftiSlice(path, label, index, window) for index in band_indices(depth, band)]
```

In `src/m3score/io/__init__.py`, replace the import block and `_SUPPORTED` with:

```python
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

import numpy as np
import torch

from .. import protocol
from .arrays import ArraySource
from .base import ImageSource, ItemSource, SkippedItem
from .images import IMAGE_SUFFIXES, ImageFile
from .nifti import is_nifti, nifti_items
from .sampling import sample_indices
from .volumes import check_band, check_window

__all__ = [
    "ArraySource",
    "ImageSource",
    "ItemSource",
    "SkippedItem",
    "open_source",
    "sample_indices",
]

_SUPPORTED = ", ".join(sorted(IMAGE_SUFFIXES)) + ", .nii, .nii.gz"
```

and replace the whole `_open_path` function with:

```python
def _open_path(
    root: Path,
    include: str | None,
    group_pattern: str | None,
    group_pattern_optional: bool,
    slice_band: tuple[float, float] | None,
    window: tuple[float, float] | None,
) -> ImageSource:
    band = check_band(slice_band) if slice_band is not None else protocol.SLICE_BAND
    window = check_window(window) if window is not None else None

    items: list = []
    labels: list[str] = []
    groups: list[str | None] = []
    notes: list[str] = []
    for path, label in _list_files(root, include):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(ImageFile(path, label))
            labels.append(label)
            groups.append(None)
        elif is_nifti(path):
            for item in nifti_items(path, label, band, window):
                items.append(item)
                labels.append(label)
                groups.append(label)  # one volume file is one subject
    if not items:
        matching = f" matching {include!r}" if include else ""
        raise FileNotFoundError(
            f"no readable images under {root}{matching}; supported: {_SUPPORTED}"
        )

    if group_pattern is not None:
        matched = _groups_from_pattern(labels, group_pattern, group_pattern_optional)
        if matched is not None:
            groups = list(matched)
    known = None if any(g is None for g in groups) else [str(g) for g in groups]
    return ItemSource(items, known, notes)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_io_volumes.py tests/test_io_nifti.py tests/test_io_images.py -v`
Expected: all pass.

Run: `pytest -q`
Expected: everything passes.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/m3score/io tests/test_io_volumes.py tests/test_io_nifti.py
git commit -m "feat: read NIfTI volumes as a central band of reoriented, scaled slices"
```

---

### Task 10: DICOM

**Files:**
- Create: `src/m3score/io/dicom.py`, `tests/_dicom.py`, `tests/test_io_dicom.py`
- Modify: `src/m3score/io/__init__.py` (one import line, `_SUPPORTED`, and the `_open_path` function)

**Interfaces:**
- Consumes: `io.volumes.band_indices`, `scale_intensity`; `io.base.SkippedItem`; `M3ScoreError`.
- Produces:
  - `io.dicom.is_dicom(path: Path) -> bool`: by suffix `.dcm`/`.dicom`/`.ima`, otherwise by `DICM` at byte 128
  - `io.dicom.dicom_items(files: list[tuple[Path, str]], band, window) -> tuple[list, list[str], list[str], list[str]]` returning `(items, labels, groups, notes)`, the first three aligned
  - `open_source` now reads DICOM. Subject is `PatientID`, else `StudyInstanceUID`, else the parent folder.
- Test helper: `tests/_dicom.py::write_dicom(path, pixels, *, series_uid=None, study_uid=None, patient_id="P1", position=None, instance=1, slope=1.0, intercept=0.0, window=None, photometric="MONOCHROME2", frames=None) -> Path`.

- [ ] **Step 1: Install pydicom on its own**

Run: `pip install "pydicom>=2.4"`
Expected: installs pydicom; `pip list | grep -i torch` still shows `torch 2.6.0+cu124`.

- [ ] **Step 2: Write the failing tests**

Create `tests/_dicom.py`:

```python
"""Helper that writes small uncompressed DICOM files for tests."""

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def write_dicom(
    path,
    pixels,
    *,
    series_uid=None,
    study_uid=None,
    patient_id="P1",
    position=None,
    instance=1,
    slope=1.0,
    intercept=0.0,
    window=None,
    photometric="MONOCHROME2",
    frames=None,
):
    """Write one DICOM file.

    ``pixels`` is ``(H, W)`` 16-bit, ``(F, H, W)`` 16-bit together with ``frames=F``,
    or ``(H, W, 3)`` uint8 for a colour image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    if patient_id is not None:
        ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.InstanceNumber = instance
    ds.PhotometricInterpretation = photometric

    colour = pixels.ndim == 3 and frames is None
    if colour:
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
        ds.Rows, ds.Columns = pixels.shape[:2]
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
        ds.PixelRepresentation = 0
    else:
        ds.SamplesPerPixel = 1
        ds.Rows, ds.Columns = pixels.shape[-2:]
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1 if pixels.dtype == np.int16 else 0
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
    if frames is not None:
        ds.NumberOfFrames = frames
    if position is not None:
        ds.ImagePositionPatient = [0.0, 0.0, float(position)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()

    try:
        ds.save_as(str(path), enforce_file_format=True)  # pydicom 3
    except TypeError:  # pydicom 2
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(str(path), write_like_original=False)
    return path
```

Create `tests/test_io_dicom.py`:

```python
import sys

import numpy as np
import pytest
import torch

from m3score import M3ScoreError
from m3score.io import open_source
from m3score.io.dicom import _require_pydicom, is_dicom

pydicom = pytest.importorskip("pydicom")

from _dicom import write_dicom  # noqa: E402  (imports pydicom, so it must follow importorskip)


def _ramp(shape=(8, 8)):
    return np.linspace(0, 4095, shape[0] * shape[1]).reshape(shape).astype(np.uint16)


def _write_series(folder, n_slices, *, series_uid, patient_id="P1"):
    """Slice k is filled with 100 * (k + 1). Files are written in shuffled order."""
    order = np.random.default_rng(0).permutation(n_slices)
    for file_number, k in enumerate(order):
        write_dicom(
            folder / f"file{file_number:03d}.dcm",
            np.full((8, 8), 100 * (int(k) + 1), dtype=np.uint16),
            series_uid=series_uid,
            study_uid="1.2.826.0.1.1",
            patient_id=patient_id,
            position=2.5 * int(k),
            instance=file_number + 1,  # deliberately unrelated to position
        )


def test_a_single_image_is_scored_on_its_own(tmp_path):
    write_dicom(tmp_path / "cxr.dcm", _ramp())
    source = open_source(tmp_path)
    assert len(source) == 1
    assert source.groups == ["P1"]
    image = source.load(0)
    assert image.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert float(image.min()) == 0.0 and float(image.max()) == 1.0


def test_rescale_and_the_stored_window_are_applied(tmp_path):
    pixels = np.array([[824, 1024], [1224, 3000]], dtype=np.uint16)  # HU -200, 0, 200, 1976
    write_dicom(tmp_path / "ct.dcm", pixels, intercept=-1024.0, window=(0.0, 400.0))
    image = open_source(tmp_path).load(0)
    assert image.flatten().tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])


def test_an_explicit_window_overrides_the_stored_one(tmp_path):
    pixels = np.array([[24, 1024], [1424, 3000]], dtype=np.uint16)  # HU -1000, 0, 400, 1976
    write_dicom(tmp_path / "ct.dcm", pixels, intercept=-1024.0, window=(0.0, 400.0))
    image = open_source(tmp_path, window=(-1000.0, 400.0)).load(0)
    assert image.flatten().tolist() == pytest.approx([0.0, 1000 / 1400, 1.0, 1.0])


def test_monochrome1_is_inverted(tmp_path):
    window = (2048.0, 4096.0)
    write_dicom(tmp_path / "a" / "x.dcm", _ramp(), window=window)
    write_dicom(tmp_path / "b" / "x.dcm", _ramp(), window=window, photometric="MONOCHROME1")
    normal = open_source(tmp_path / "a").load(0)
    inverted = open_source(tmp_path / "b").load(0)
    assert torch.allclose(normal + inverted, torch.ones_like(normal))


def test_a_series_is_a_volume_ordered_by_position_with_the_band_applied(tmp_path):
    _write_series(tmp_path, 20, series_uid="1.2.3")
    source = open_source(tmp_path, window=(0.0, 4000.0))
    assert len(source) == 4  # the default band of a 20-slice volume is slices 7 to 10
    values = [round(float(source.load(i).mean()) * 4000.0) for i in range(4)]
    assert values == [800, 900, 1000, 1100]
    assert source.groups == ["P1"] * 4


def test_series_are_grouped_by_patient(tmp_path):
    _write_series(tmp_path / "p1", 20, series_uid="1.2.3", patient_id="P1")
    _write_series(tmp_path / "p2", 20, series_uid="1.2.4", patient_id="P2")
    source = open_source(tmp_path)
    assert len(source) == 8
    assert sorted(set(source.groups)) == ["P1", "P2"]


def test_subject_falls_back_to_the_study_without_a_patient_id(tmp_path):
    write_dicom(tmp_path / "a.dcm", _ramp(), patient_id=None, study_uid="1.2.826.0.1")
    assert open_source(tmp_path).groups == ["1.2.826.0.1"]


def test_files_named_by_uid_are_found_by_their_magic_bytes(tmp_path):
    path = write_dicom(tmp_path / "1.2.840.113619.2.55", _ramp())
    (tmp_path / "README").write_text("not dicom")
    assert is_dicom(path)
    assert not is_dicom(tmp_path / "README")
    assert len(open_source(tmp_path)) == 1


def test_8bit_rgb_is_read_as_colour(tmp_path):
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    write_dicom(tmp_path / "us.dcm", pixels, photometric="RGB")
    image = open_source(tmp_path).load(0)
    assert image.shape == (3, 4, 4)
    assert float(image[0].min()) == 1.0
    assert float(image[1].max()) == 0.0


def test_other_colour_encodings_are_skipped_with_a_warning(tmp_path):
    write_dicom(tmp_path / "ybr.dcm", np.zeros((4, 4, 3), dtype=np.uint8), photometric="YBR_FULL")
    write_dicom(tmp_path / "ok.dcm", _ramp())
    source = open_source(tmp_path)
    assert len(source) == 1
    assert any("ybr.dcm" in note and "YBR_FULL" in note for note in source.warnings)


def test_a_multi_frame_file_is_a_volume(tmp_path):
    frames = np.stack([np.full((8, 8), 100 * (k + 1), dtype=np.uint16) for k in range(20)])
    write_dicom(tmp_path / "enhanced.dcm", frames, frames=20)
    source = open_source(tmp_path, window=(0.0, 4000.0))
    assert len(source) == 4
    assert round(float(source.load(0).mean()) * 4000.0) == 800


def test_undecodable_pixel_data_names_the_decoder_packages(tmp_path, monkeypatch):
    write_dicom(tmp_path / "a.dcm", _ramp())

    def no_handler(self):
        raise RuntimeError("no available image handler")

    monkeypatch.setattr(pydicom.dataset.Dataset, "pixel_array", property(no_handler))
    with pytest.raises(M3ScoreError, match="pylibjpeg"):
        open_source(tmp_path).load(0)


def test_missing_pydicom_explains_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pydicom", None)
    with pytest.raises(ImportError, match=r"m3-score\[dicom\]"):
        _require_pydicom()
```

Run: `pytest tests/test_io_dicom.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'm3score.io.dicom'`.

- [ ] **Step 3: Implement the DICOM reader**

Create `src/m3score/io/dicom.py`:

```python
"""DICOM files and series. Needs pydicom: ``pip install "m3-score[dicom]"``."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

from .._errors import M3ScoreError
from .base import SkippedItem
from .volumes import band_indices, scale_intensity

DICOM_SUFFIXES = frozenset({".dcm", ".dicom", ".ima"})
_DECODER_HINT = (
    "compressed transfer syntaxes need a decoder plugin: "
    "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
)


def is_dicom(path: Path) -> bool:
    """By suffix, otherwise by the ``DICM`` marker (files are often named by UID)."""
    if path.suffix.lower() in DICOM_SUFFIXES:
        return True
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _require_pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError('DICOM input needs pydicom: pip install "m3-score[dicom]"') from exc
    return pydicom


def _number(value, default: float | None) -> float | None:
    """A DICOM numeric element as a float; the first value when it is multi-valued."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value[0])
        except (TypeError, ValueError, IndexError):
            return default


def _stored_window(ds) -> tuple[float, float] | None:
    center = _number(getattr(ds, "WindowCenter", None), None)
    width = _number(getattr(ds, "WindowWidth", None), None)
    if center is None or width is None or width <= 0:
        return None
    return center - width / 2.0, center + width / 2.0


def _raw_pixels(ds, label: str) -> np.ndarray:
    try:
        return ds.pixel_array
    except Exception as exc:
        raise M3ScoreError(
            f"{label}: could not decode DICOM pixel data ({exc}); {_DECODER_HINT}"
        ) from exc


def _pixels(ds, label: str) -> np.ndarray:
    """Stored values after the modality rescale, as float32."""
    slope = _number(getattr(ds, "RescaleSlope", None), 1.0)
    intercept = _number(getattr(ds, "RescaleIntercept", None), 0.0)
    return _raw_pixels(ds, label).astype(np.float32) * slope + intercept


def _to_unit_range(array: np.ndarray, ds, window: tuple[float, float] | None):
    """Explicit window, else the stored window, else the percentile rule; MONOCHROME1 inverted."""
    scaled = scale_intensity(array, window if window is not None else _stored_window(ds))
    if scaled is None:
        return None
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        scaled = 1.0 - scaled
    return scaled


def _subject(ds, label: str) -> str:
    for keyword in ("PatientID", "StudyInstanceUID"):
        value = str(getattr(ds, keyword, "") or "").strip()
        if value:
            return value
    return label.rsplit("/", 1)[0] if "/" in label else "."


def _placement(ds) -> tuple[float, tuple[float, ...]] | None:
    """Position along the slice normal and the rounded orientation, when both are stored."""
    position = getattr(ds, "ImagePositionPatient", None)
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if position is None or orientation is None or len(position) != 3 or len(orientation) != 6:
        return None
    axes = np.asarray(orientation, dtype=np.float64)
    normal = np.cross(axes[:3], axes[3:])
    along = float(np.dot(np.asarray(position, dtype=np.float64), normal))
    return along, tuple(float(v) for v in np.round(axes, 2))


class DicomImage:
    """One single-frame instance scored on its own, such as a chest radiograph."""

    def __init__(self, path: Path, label: str, window: tuple[float, float] | None) -> None:
        self.path = path
        self.label = label
        self.window = window

    def load(self) -> torch.Tensor:
        pydicom = _require_pydicom()
        ds = pydicom.dcmread(str(self.path))
        if int(_number(getattr(ds, "SamplesPerPixel", None), 1.0)) == 3:
            colour = np.ascontiguousarray(_raw_pixels(ds, self.label))
            return torch.from_numpy(colour).permute(2, 0, 1).float() / 255.0
        scaled = _to_unit_range(_pixels(ds, self.label), ds, self.window)
        if scaled is None:
            raise SkippedItem(f"{self.label}: image is empty or constant")
        return torch.from_numpy(scaled)[None]


@functools.lru_cache(maxsize=2)
def _load_stack(
    paths: tuple[str, ...], label: str, window: tuple[float, float] | None
) -> np.ndarray | None:
    """A whole volume as (slices, H, W) in [0, 1], scaled as one unit."""
    pydicom = _require_pydicom()
    datasets = [pydicom.dcmread(path) for path in paths]
    frames = [_pixels(ds, label) for ds in datasets]
    # a multi-frame instance already decodes to (frames, H, W)
    stack = frames[0] if len(frames) == 1 else np.stack(frames)
    return _to_unit_range(stack, datasets[0], window)


class DicomSlice:
    """One slice of a series, or one frame of a multi-frame instance."""

    def __init__(
        self, paths: tuple[str, ...], label: str, index: int, window: tuple[float, float] | None
    ) -> None:
        self.paths = paths
        self.label = label
        self.index = index
        self.window = window

    def load(self) -> torch.Tensor:
        volume = _load_stack(self.paths, self.label, self.window)
        if volume is None:
            raise SkippedItem(f"{self.label}: volume is empty or constant")
        return torch.from_numpy(np.ascontiguousarray(volume[self.index]))[None]


def dicom_items(
    files: list[tuple[Path, str]], band: tuple[float, float], window: tuple[float, float] | None
) -> tuple[list, list[str], list[str], list[str]]:
    """Items for a set of DICOM files, read from their headers only.

    Returns ``(items, labels, groups, notes)``; the first three are aligned. Single
    frames that share a series, an image size and an orientation, and that carry
    positions, form a volume: sorted along the slice normal, with ``band`` applied.
    """
    pydicom = _require_pydicom()
    items: list = []
    labels: list[str] = []
    groups: list[str] = []
    notes: list[str] = []
    series: dict[tuple, list[tuple[float, float, str, str, str]]] = {}

    def add(item, label: str, subject: str) -> None:
        items.append(item)
        labels.append(label)
        groups.append(subject)

    for path, label in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            notes.append(f"skipped {label}: not a readable DICOM file ({exc})")
            continue
        if "Rows" not in ds or "Columns" not in ds:
            notes.append(f"skipped {label}: DICOM object without an image")
            continue
        subject = _subject(ds, label)
        frames = int(_number(getattr(ds, "NumberOfFrames", None), 1.0))
        if int(_number(getattr(ds, "SamplesPerPixel", None), 1.0)) == 3:
            photometric = str(getattr(ds, "PhotometricInterpretation", ""))
            eight_bit = int(_number(getattr(ds, "BitsAllocated", None), 8.0)) == 8
            if photometric != "RGB" or not eight_bit or frames > 1:
                notes.append(f"skipped {label}: unsupported colour encoding {photometric}")
                continue
            add(DicomImage(path, label, window), label, subject)
        elif frames > 1:
            for index in band_indices(frames, band):
                add(DicomSlice((str(path),), label, index, window), label, subject)
        else:
            placement = _placement(ds)
            uid = getattr(ds, "SeriesInstanceUID", None)
            if placement is None or uid is None:
                add(DicomImage(path, label, window), label, subject)
                continue
            along, orientation = placement
            key = (str(uid), int(ds.Rows), int(ds.Columns), orientation)
            instance = _number(getattr(ds, "InstanceNumber", None), 0.0)
            series.setdefault(key, []).append((along, instance, str(path), label, subject))

    for members in series.values():
        if len(members) < 2:
            _, _, path, label, subject = members[0]
            add(DicomImage(Path(path), label, window), label, subject)
            continue
        members.sort(key=lambda member: (member[0], member[1]))
        paths = tuple(member[2] for member in members)
        label, subject = members[0][3], members[0][4]
        for index in band_indices(len(members), band):
            add(DicomSlice(paths, label, index, window), label, subject)
    return items, labels, groups, notes
```

In `src/m3score/io/__init__.py`, add this import after the `.base` import:

```python
from .dicom import dicom_items, is_dicom
```

change `_SUPPORTED` to:

```python
_SUPPORTED = ", ".join(sorted(IMAGE_SUFFIXES)) + ", .nii, .nii.gz and DICOM"
```

and replace the whole `_open_path` function with:

```python
def _open_path(
    root: Path,
    include: str | None,
    group_pattern: str | None,
    group_pattern_optional: bool,
    slice_band: tuple[float, float] | None,
    window: tuple[float, float] | None,
) -> ImageSource:
    band = check_band(slice_band) if slice_band is not None else protocol.SLICE_BAND
    window = check_window(window) if window is not None else None

    items: list = []
    labels: list[str] = []
    groups: list[str | None] = []
    notes: list[str] = []
    dicom_files: list[tuple[Path, str]] = []
    for path, label in _list_files(root, include):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            items.append(ImageFile(path, label))
            labels.append(label)
            groups.append(None)
        elif is_nifti(path):
            for item in nifti_items(path, label, band, window):
                items.append(item)
                labels.append(label)
                groups.append(label)  # one volume file is one subject
        elif is_dicom(path):
            dicom_files.append((path, label))
    if dicom_files:
        # DICOM files are handled together because slices of one series form a volume
        found, found_labels, found_groups, found_notes = dicom_items(dicom_files, band, window)
        items += found
        labels += found_labels
        groups += found_groups
        notes += found_notes
    if not items:
        matching = f" matching {include!r}" if include else ""
        raise FileNotFoundError(
            f"no readable images under {root}{matching}; supported: {_SUPPORTED}"
        )

    if group_pattern is not None:
        matched = _groups_from_pattern(labels, group_pattern, group_pattern_optional)
        if matched is not None:
            groups = list(matched)
    known = None if any(g is None for g in groups) else [str(g) for g in groups]
    return ItemSource(items, known, notes)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_io_dicom.py -v`
Expected: 13 passed. If `write_dicom` fails inside pydicom with a message about the installed version's `save_as` or `FileDataset` signature, adjust only `tests/_dicom.py` to that version's API; the reader in `src/` uses nothing version-specific.

Run: `pytest -q`
Expected: everything passes.

Run: `ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/m3score/io tests/_dicom.py tests/test_io_dicom.py
git commit -m "feat: read DICOM images and series, with windowing and series-as-volume"
```

---

## Phase 3: verification, release tooling, documentation

### Task 11: Golden values and the end-to-end comparison with the research pipeline

**Files:**
- Create: `tests/_synthetic.py`, `tools/make_golden.py`, `tests/golden_m3_v1.json` (generated), `tests/test_golden.py`, `tools/check_package_parity.py`

**Interfaces:**
- Consumes: `m3score.score`, `M3Result`, `protocol.BACKBONE_ID`, `BACKBONE_REVISION`; from the research code `M3EntropyMetric.compute_axes`, `_extract_cls_features`, `_compute_coverage`, `COVERAGE_LAYER`.
- Produces: `tests/_synthetic.py::synthetic_sets(n=64) -> tuple[torch.Tensor, torch.Tensor]`; a network-marked regression test; a script that exits non-zero when the package and the research pipeline disagree beyond the spec's tolerances.

- [ ] **Step 1: Write the synthetic sets and the golden-value generator**

Create `tests/_synthetic.py`:

```python
"""Two fixed synthetic image sets for the golden-value regression test."""

import torch
import torch.nn.functional as F


def synthetic_sets(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth random textures at 224x224; the second set has lower contrast."""

    def make(seed: int, contrast: float) -> torch.Tensor:
        coarse = torch.rand(n, 1, 14, 14, generator=torch.Generator().manual_seed(seed))
        smooth = F.interpolate(coarse, size=(224, 224), mode="bilinear", align_corners=False)
        return (smooth * contrast).clamp(0.0, 1.0)

    return make(0, 1.0), make(1, 0.7)
```

Create `tools/make_golden.py`:

```python
"""Regenerate tests/golden_m3_v1.json from the real m3-v1 backbone.

Run only when the protocol is changed on purpose:

    python tools/make_golden.py
"""

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import m3score  # noqa: E402
from _synthetic import synthetic_sets  # noqa: E402


def main() -> None:
    real, gen = synthetic_sets()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", m3score.M3ScoreWarning)
        result = m3score.score(real, gen, device="cpu", progress=False)
    payload = {
        "protocol": result.protocol,
        "backbone": result.backbone,
        "fidelity": result.fidelity,
        "fidelity_ci": list(result.fidelity_ci),
        "memorization": result.memorization,
        "memorization_nn_dist": result.memorization_nn_dist,
        "coverage": result.coverage,
    }
    target = ROOT / "tests" / "golden_m3_v1.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the regression test**

Create `tests/test_golden.py`:

```python
"""Regression on the real backbone: catches silent drift from a timm or weights change."""

import json
from pathlib import Path

import pytest

import m3score
from _synthetic import synthetic_sets
from m3score import protocol

pytestmark = [
    pytest.mark.network,
    pytest.mark.filterwarnings("ignore::m3score.M3ScoreWarning"),
]

GOLDEN_PATH = Path(__file__).parent / "golden_m3_v1.json"


def test_the_real_backbone_reproduces_the_golden_values():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    real, gen = synthetic_sets()
    result = m3score.score(real, gen, device="cpu", progress=False)

    pinned = f"{protocol.BACKBONE_ID}@{protocol.BACKBONE_REVISION}"
    assert result.protocol == golden["protocol"] == "m3-v1"
    assert result.backbone == golden["backbone"] == pinned
    assert result.fidelity == pytest.approx(golden["fidelity"], abs=1e-3)
    assert result.fidelity_ci == pytest.approx(tuple(golden["fidelity_ci"]), abs=2e-3)
    # each rate moves in steps of 1/64, so allow two images to change side
    assert result.memorization == pytest.approx(golden["memorization"], abs=0.04)
    assert result.coverage == pytest.approx(golden["coverage"], abs=0.04)
    assert result.memorization_nn_dist == pytest.approx(golden["memorization_nn_dist"], rel=1e-2)
```

Run: `pytest tests/test_golden.py -v -m network`
Expected: FAIL with `FileNotFoundError` for `golden_m3_v1.json`.

- [ ] **Step 3: Generate the golden file and run the test**

Run: `python tools/make_golden.py`
Expected: prints `wrote ...tests/golden_m3_v1.json` and the values, with `"protocol": "m3-v1"` and `"backbone": "Snarcy/RadioDino-s16@773d28d873455153495d39179fc42ea816aa1212"`. The backbone loads from the local Hugging Face cache.

Sanity-check the values before trusting them as a reference: the fidelity must be positive and its interval must contain it; memorization and coverage must lie in `[0, 1]`. If any of these fails, stop and report; do not commit the file.

Run: `pytest tests/test_golden.py -v -m network`
Expected: 1 passed.

Run: `pytest -q -m "not network"`
Expected: everything else passes and the golden test is deselected.

- [ ] **Step 4: Write the end-to-end comparison script**

Create `tools/check_package_parity.py`:

```python
"""Compare the m3score package with the research pipeline, end to end, on two image folders.

The research pipeline decodes with PIL and resizes with torchvision; the package
resizes in torch. This script measures what that difference does to the three axes
and exits non-zero when it exceeds the tolerances in the design spec:
fidelity within 1% relative, memorization and coverage within 0.02 absolute.

    python tools/check_package_parity.py REAL_DIR GEN_DIR [--device cuda]
"""

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import m3score  # noqa: E402
from evaluation.m3_score_v2 import COVERAGE_LAYER, M3EntropyMetric  # noqa: E402

FIDELITY_RELATIVE = 0.01
RATE_ABSOLUTE = 0.02


def research_load(directory: str) -> torch.Tensor:
    """Every PNG under ``directory``, as the research evaluation pipeline loads them."""
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.png"), recursive=True))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("real")
    parser.add_argument("gen")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    real, gen = research_load(args.real), research_load(args.gen)
    research = M3EntropyMetric(device=args.device)
    axes = research.compute_axes(real, gen, n_permutations=1, n_bootstrap=1)
    real_l4 = research._extract_cls_features(real, {COVERAGE_LAYER})[COVERAGE_LAYER - 1]
    gen_l4 = research._extract_cls_features(gen, {COVERAGE_LAYER})[COVERAGE_LAYER - 1]
    research_coverage = research._compute_coverage(real_l4, gen_l4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", m3score.M3ScoreWarning)
        package = m3score.score(args.real, args.gen, device=args.device, stats=False, progress=False)

    rows = [
        ("fidelity", axes["fidelity/mmd2"], package.fidelity),
        ("memorization", axes["memorization/rate"], package.memorization),
        ("coverage", research_coverage, package.coverage),
    ]
    print(f"\n{len(real)} real, {len(gen)} generated images; package scored "
          f"{package.n_real} and {package.n_gen}")
    print(f"{'axis':<14}{'research':>12}{'package':>12}{'difference':>14}")
    failed = False
    for name, expected, actual in rows:
        difference = actual - expected
        if name == "fidelity":
            within = abs(difference) <= FIDELITY_RELATIVE * max(abs(expected), 1e-12)
        else:
            within = abs(difference) <= RATE_ABSOLUTE
        failed = failed or not within
        print(f"{name:<14}{expected:>12.6f}{actual:>12.6f}{difference:>+14.6f}"
              f"{'' if within else '   OUTSIDE TOLERANCE'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the comparison on the local image sets**

Run: `python tools/check_package_parity.py output/generated_500_best output/generated_retinal`
Run: `python tools/check_package_parity.py output/generated_500_best output/generated_train70`
Expected for both: a three-row table and exit status 0 (`echo $?` prints `0`). Record both tables in the task report.

If either run is outside tolerance, **stop and report the table to the maintainer.** Do not change tolerances, and do not switch the package to PIL resizing without their decision: the spec says a larger gap is reported, not tuned away.

- [ ] **Step 6: Measure the run time at the default cap**

Run:

```bash
python - <<'PY'
import time, torch
from m3score import stats
g = torch.Generator().manual_seed(0)
real = torch.randn(5000, 384, generator=g)
gen = torch.randn(5000, 384, generator=g) + 0.1
for name, fn in [
    ("fidelity", lambda: stats.mmd2_rbf(real, gen)),
    ("bootstrap", lambda: stats.bootstrap_ci(real, gen)),
    ("permutation", lambda: stats.permutation_test(real, gen)),
    ("floor", lambda: stats.real_floor(real)),
    ("memorization", lambda: stats.memorization(real, gen)),
    ("coverage", lambda: stats.coverage(real, gen)),
]:
    start = time.time(); fn(); print(f"{name:<13}{time.time() - start:6.1f} s")
PY
```

Expected: every line finishes; the total is on the order of a minute on CPU. Record the figures in the task report; Task 13 quotes the total in the README.

- [ ] **Step 7: Commit**

```bash
git add tests/_synthetic.py tests/test_golden.py tests/golden_m3_v1.json tools/make_golden.py tools/check_package_parity.py
git commit -m "test: add golden-value regression and end-to-end comparison with the research pipeline"
```

---

### Task 12: Build, CI and the release workflow

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`
- Modify: `RELEASE.md` (append)

**Interfaces:**
- Consumes: `pyproject.toml`; the `network` pytest marker.
- Produces: a wheel and sdist in `dist/` verified locally; CI on push and pull request; a workflow that publishes to PyPI when a GitHub release is published.

- [ ] **Step 1: Write the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install CPU PyTorch
        run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - name: Install the package with development extras
        run: pip install -e ".[dev]"
      - name: Lint
        run: ruff check src tests
      - name: Test
        run: pytest -m "not network"

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install build twine
      - run: python -m build
      - run: twine check dist/*
```

- [ ] **Step 2: Write the release workflow**

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Check that the release tag matches the package version
        run: |
          version=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
          test "v$version" = "$GITHUB_REF_NAME"
      - run: python -m pip install build twine
      - run: python -m build
      - run: twine check dist/*
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # PyPI trusted publishing; no API token is stored anywhere
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 3: Document the release procedure**

Append to `RELEASE.md`:

```markdown

## Python package

The `m3-score` distribution contains only `src/m3score/`. Experiment code, figures and
result reports stay in the repository and are not part of the wheel.

### One-time setup

1. Create an account on https://pypi.org and enable two-factor authentication.
2. Under "Publishing", add a pending trusted publisher with:
   - PyPI project name: `m3-score`
   - Owner: `Nishan-Charlie`
   - Repository: `M3-Score`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In the GitHub repository, create an environment named `pypi` (Settings -> Environments).

### Each release

1. Set `version` in `pyproject.toml` and `CITATION.cff`, and merge to `main` with CI green.
2. Locally: `python -m build` and `twine check dist/*`.
3. On GitHub, publish a release whose tag is `v` followed by the version, for example
   `v0.1.0`. The "Publish to PyPI" workflow builds and uploads it. The workflow refuses
   to run when the tag and the version differ.

A version number can be published only once, and a project name cannot be released
by anyone else afterwards. Check the built wheel before publishing the release.

### Changing the protocol

Any change to a constant in `src/m3score/protocol.py`, or to the arithmetic in
`src/m3score/stats.py`, changes what an M3-Score means. Give it a new
`PROTOCOL_VERSION`, regenerate the golden values with `python tools/make_golden.py`,
and say so in the release notes.
```

- [ ] **Step 4: Build and inspect the distributions**

Run: `rm -rf dist && python -m build`
Expected: ends with `Successfully built m3_score-0.1.0.tar.gz and m3_score-0.1.0-py3-none-any.whl`.

Run: `twine check dist/*`
Expected: `PASSED` for both files.

Run: `python -m zipfile -l dist/m3_score-0.1.0-py3-none-any.whl`
Expected: only `m3score/...` files (including `m3score/py.typed` and `m3score/io/...`) and `m3_score-0.1.0.dist-info/...` including `licenses/LICENSE` and `entry_points.txt`. Nothing from `evaluation/`, `results/`, `figures/` or `tests/`.

Run: `tar -tzf dist/m3_score-0.1.0.tar.gz`
Expected: `src/`, `tests/`, `README.md`, `LICENSE`, `CITATION.cff`, `pyproject.toml`, `PKG-INFO`. Nothing from `evaluation/`, `results/`, `figures/`, `paper_release/` or `output/`.

- [ ] **Step 5: Install the wheel into a clean environment and run it from outside the repository**

This is the check behind success criterion 1. It downloads CPU PyTorch into the new environment and takes several minutes.

```bash
REPO="$(pwd)"
WORK="$(mktemp -d)"
python -m venv "$WORK/venv"
"$WORK/venv/Scripts/python" -m pip install "$REPO/dist/m3_score-0.1.0-py3-none-any.whl[all]"
cd "$WORK"
"$WORK/venv/Scripts/m3score" --version
"$WORK/venv/Scripts/python" -c "import m3score, pathlib; print(pathlib.Path(m3score.__file__).parent)"
"$WORK/venv/Scripts/python" -m pip check
"$WORK/venv/Scripts/m3score" "$REPO/output/generated_500_best" "$REPO/output/generated_retinal" --n 50 --device cpu --quiet
cd "$REPO"
```

On Linux or macOS the interpreter is `"$WORK/venv/bin/python"` and the command `"$WORK/venv/bin/m3score"`.

Expected:
- `m3-score 0.1.0`
- the import path is inside `$WORK/venv`, not inside the repository
- `No broken requirements found.`
- the result table, with three axes and `protocol m3-v1`

Record the output in the task report.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/publish.yml RELEASE.md
git commit -m "ci: add test matrix, build check and PyPI trusted-publishing workflow"
```

---

### Task 13: Documentation

**Files:**
- Modify: `README.md` (replace whole file), `CITATION.cff` (replace whole file), `docs/index.html` (two code blocks and one paragraph), `RELEASE.md` (append the project-page section moved out of the README)

**Interfaces:**
- Consumes: everything above. Every command and code sample in the README must run as written against the finished package.
- Produces: the page PyPI renders as the project description.

- [ ] **Step 1: Replace `README.md`**

All links and images are absolute, because PyPI does not resolve relative ones. Replace the file with:

````markdown
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
  fidelity      0.0731   lower    95% CI [0.0689, 0.0774]; real-vs-real floor 0.0015 +/- 0.0004
  memorization  0.012    lower    mean nearest-real distance 14.2
  coverage      0.380    higher   k = 5

  permutation test: z = 347.2, p = 0.00498 (descriptive, not a model-selection tool)
```

The numbers above illustrate the layout; they are not a published result.

| Option | Meaning |
| --- | --- |
| `--n N` | Images scored per set, drawn at random. Default: all, up to 5,000. |
| `--seed S` | Seed for sampling and statistics. Default 0. |
| `--device auto\|cpu\|cuda` | Default `auto`. |
| `--include GLOB` | Only read files whose path inside the folder matches, e.g. `"*_flair.nii.gz"`. |
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
for batch in real_loader:
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

## NIfTI and DICOM

```bash
pip install "m3-score[all]"
m3score BraTS2021/ generated/ --include "*_flair.nii.gz" --group-pattern "(BraTS2021_\d+)"
```

- **NIfTI.** Each volume is reoriented to one display convention (anterior up,
  radiological), so stored orientation does not matter. The central axial band is kept
  (slices 55 to 85 of a 155-slice BraTS volume, scaled to other depths) and each volume is
  scaled by the 0.5 to 99.5 percentile of its non-zero voxels. One file is one subject.
- **DICOM.** A series of positioned slices is a volume and follows the same rules, ordered
  by position. Single images such as chest radiographs are scored as they are. The stored
  window is used when present; `MONOCHROME1` is inverted. The subject is `PatientID`.
  Compressed files need a pydicom decoder plugin such as `pylibjpeg`.
- **CT.** Pass an explicit window, for example `--window -1000 400`.

A dataset folder that holds several modalities and segmentations per subject must be
filtered with `--include`, or they will be mixed.

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
- The result reports distinct subjects next to the image count, and warns below 50 subjects
  or 500 images.

Report both counts with any M3-Score you publish, and keep generator training data and the
reference set subject-disjoint.

## Reading the result

- **Fidelity** is an unbiased MMD² and has no natural scale. Read it against the
  **real-vs-real floor**: the same statistic between random, subject-disjoint halves of your
  real set. A generator at a few times the floor is close to the real distribution; one at
  fifty times is not.
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
exactly on identical features (`tests/test_parity.py`). It differs deliberately in four places:

- **Confidence interval.** The research code's bootstrap counted duplicated images as
  distinct pairs, which shifts the interval upward. The package leaves those pairs out. The
  paper's interval [0.071, 0.080] was computed with the earlier procedure.
- **Permutation p-value** is `(1 + b) / (1 + B)` and so is never exactly 0.
- **`m3score.stats.precision_recall`** uses the standard any-ball definition of
  Kynkäänniemi et al. (2019); the research code tested only the nearest reference ball.
- **Inputs outside `[0, 1]`** are rejected.

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
````

If the Task 11 timing differed materially from "about a minute", correct that sentence to the measured figure.

- [ ] **Step 2: Check every sample in the README**

Run each of these and confirm it behaves as the README says:

```bash
m3score output/generated_500_best output/generated_retinal --n 100 --quiet
python - <<'PY'
import torch, m3score
real, gen = torch.rand(40, 1, 64, 64), torch.rand(40, 1, 64, 64)
result = m3score.score(real, gen, device="cpu", progress=False)
print(result.fidelity, result.memorization, result.coverage)
print(sorted(result.to_dict()))
metric = m3score.M3Score(device="cpu")
metric.update(real, real=True)
metric.update(gen, real=False)
first = metric.compute()
metric.reset(keep_real=True)
metric.update(gen, real=False)
assert metric.compute() == first
try:
    m3score.score(real * 2 - 1, gen, device="cpu", progress=False)
except ValueError as exc:
    print("rejected:", exc)
PY
```

Expected: the table; three floats; the field names; no assertion error; a `rejected:` line mentioning `(x + 1) / 2`.

- [ ] **Step 3: Replace `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use M3-Score or this codebase, please cite the following."
title: "M3-Score: Fidelity, Memorization and Coverage as Separate Axes for Evaluating Generative Radiology Image Models"
authors:
  - family-names: Sathiyamohan
    given-names: Nishankar
type: software
version: "0.1.0"
license: MIT
repository-code: "https://github.com/Nishan-Charlie/M3-Score"
url: "https://nishan-charlie.github.io/M3-Score/"
keywords:
  - generative model evaluation
  - medical imaging
  - radiology
  - diffusion models
  - maximum mean discrepancy
  - MRI
preferred-citation:
  type: article
  title: "M3-Score: Fidelity, Memorization and Coverage as Separate Axes for Evaluating Generative Radiology Image Models"
  authors:
    - family-names: Sathiyamohan
      given-names: Nishankar
  year: 2026
  journal: "arXiv preprint"
```

- [ ] **Step 4: Update the landing page**

In `docs/index.html`, replace:

```html
            <h3>Install</h3>
            <pre class="code">conda create -n m3score python=3.10
conda activate m3score
pip install -r requirements.txt</pre>
```

with:

```html
            <h3>Install</h3>
            <pre class="code">pip install m3-score

# with NIfTI and DICOM support
pip install "m3-score[all]"</pre>
```

Replace:

```html
            <h3>Evaluate two image directories</h3>
            <pre class="code">python evaluation/eval_pipeline.py \
  --real_dir /path/to/real_images \
  --gen_dir /path/to/generated_images \
  --output_dir results/evaluation \
  --metrics fid kid ssim m3 alpha</pre>
```

with:

```html
            <h3>Score two image folders</h3>
            <pre class="code">m3score /path/to/real /path/to/generated</pre>
```

And replace the paragraph that begins `<p class="section-intro">This GitHub Pages site is a presentation layer for the repository.` with:

```html
        <p class="section-intro">M3-Score is available as the Python package <code>m3-score</code>, with a command line and an API for 2D images, NIfTI and DICOM. The repository also holds the research implementation, figures, numerical reports, and retained logs behind the paper. Medical datasets and model checkpoints are intentionally not redistributed.</p>
```

- [ ] **Step 5: Move the project-page instructions into `RELEASE.md`**

The old README section "Publish the project page" is maintainer documentation. Append to `RELEASE.md`:

```markdown

## Project page

GitHub Pages publishes the site from the `docs/` folder:

1. Open the repository on GitHub and go to **Settings -> Pages**.
2. Select **Deploy from a branch**.
3. Choose the `main` branch and the `/docs` folder.
4. Save the setting and open the generated Pages URL.

The entry point is `docs/index.html`, served at
`https://nishan-charlie.github.io/M3-Score/`. The page uses its own copy of selected
figures under `docs/assets/`.
```

- [ ] **Step 6: Rebuild and confirm the description renders**

Run: `rm -rf dist && python -m build && twine check dist/*`
Expected: `PASSED` for both files. `twine check` fails when the README cannot be rendered as a PyPI description.

Run: `pytest -q -m "not network" && ruff check src tests`
Expected: all pass; `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add README.md CITATION.cff docs/index.html RELEASE.md
git commit -m "docs: document the pip package, reference-set guidance and release procedure"
```

---

## Final verification

- [ ] `pytest -q` passes, including the network-marked golden test.
- [ ] `ruff check src tests` reports `All checks passed!`.
- [ ] `git status --short evaluation/` prints nothing: the research code is untouched.
- [ ] `git diff main --stat -- requirements.txt` prints nothing.
- [ ] `grep -rn "print(" src/m3score --include="*.py" | grep -v "src/m3score/cli.py"` prints nothing.
- [ ] `grep -rn "sorted(.*)\[:" src/m3score` prints nothing: no head-of-sorted-list selection.
- [ ] The final report to the maintainer states: the two end-to-end comparison tables from Task 11; the measured timings; the clean-environment output from Task 12; that nothing was uploaded to PyPI or pushed; that the GitHub Actions workflows have therefore never run (success criterion 6 is verified locally for the build only); and that the paper's headline values (0.073, 38%) were not re-derived because the BraTS reference set is not on this machine.

