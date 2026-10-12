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
