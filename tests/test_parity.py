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
