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
