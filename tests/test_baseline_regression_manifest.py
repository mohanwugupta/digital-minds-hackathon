import json
from pathlib import Path

from analysis.check_baseline_regression import check_baseline


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_retained_sprint_outputs_match_frozen_regression_manifest():
    result = check_baseline(
        _load("config/baseline_regression.json"),
        _load("artifacts/linear_probes/metrics.json"),
        _load("artifacts/value_probes/episode_split.json"),
        _load(
            "artifacts/value_dissociation/publication/"
            "value_dissociation_summary.json"
        ),
        _load(
            "artifacts/causal_steering/publication/causal_steering_summary.json"
        ),
    )

    assert result == {"passed": True, "failures": []}
