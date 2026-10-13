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
