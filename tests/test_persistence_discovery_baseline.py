import json
from pathlib import Path

from analysis.check_persistence_discovery_baseline import check_discovery_baseline


ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_track_a_and_track_b_outputs_match_discovery_regression_manifest():
    result = check_discovery_baseline(
        _load("config/persistence_discovery_regression.json"),
        _load(
            "artifacts/value_dissociation/layerwise_publication_track_a_v1/"
            "factorial_layerwise_summary.json"
        ),
        _load(
            "artifacts/cross_task/track_b_shared_v3/shared_transfer/"
            "shared_persistence_transfer_summary.json"
        ),
        _load("artifacts/value_probes/episode_split.json"),
    )
    assert result == {"passed": True, "failures": []}

