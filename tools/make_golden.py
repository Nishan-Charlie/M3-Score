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
