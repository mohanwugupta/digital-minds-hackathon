import json

import yaml

from experiments.persistence_battery.base_environment import COMMON_RECORD_FIELDS
from experiments.persistence_battery.collection import build_specs, collect_pair
from experiments.persistence_battery.registry import TASKS


class FixedBinaryModel:
    model_id = "mock/behavior-only"

    def binary_decision(self, _messages, labels, *, positive_label, **_kwargs):
        negative = next(label for label in labels if label != positive_label)
        return {
            "p_positive": 0.72,
            "p_negative": 0.28,
            "choice_logit": 0.9444616088,
            f"logit_{positive_label}": 1.0,
            f"logit_{negative}": 0.0555383912,
            "top_token_is_action": True,
            "p_action_mass_raw": 1.0,
        }


def _config():
    with open("config/persistence_battery.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_counterbalanced_pair_replays_identical_semantic_history_without_hidden_states():
    config = _config()
    specs = build_specs(config, "voluntary_waiting", mode="pilot", smoke=True)
    pair = [spec for spec in specs if spec.pair_id == specs[0].pair_id]
    records = collect_pair(
        FixedBinaryModel(),
        TASKS["voluntary_waiting"],
        pair,
        config["tasks"]["voluntary_waiting"],
    )
    by_mapping = {}
    for record in records:
        by_mapping.setdefault(record["mapping_id"], []).append(record)
        assert set(COMMON_RECORD_FIELDS) <= set(record)
        assert "hidden_states" not in record
        json.loads(record["condition"])
    left, right = by_mapping.values()
    assert [row["semantic_action"] for row in left] == [
        row["semantic_action"] for row in right
    ]
    assert [row["subsequent_outcome"] for row in left] == [
        row["subsequent_outcome"] for row in right
    ]
    assert left[0]["raw_label"] != right[0]["raw_label"]
    assert {left[0]["action_source"], right[0]["action_source"]} == {
        "sampled_primary_mapping",
        "matched_semantic_replay",
    }


def test_sequential_control_does_not_receive_persistence_fields():
    config = _config()
    specs = build_specs(
        config, "independent_effort_control", mode="pilot", smoke=True
    )
    pair = [spec for spec in specs if spec.pair_id == specs[0].pair_id]
    records = collect_pair(
        FixedBinaryModel(),
        TASKS["independent_effort_control"],
        pair,
        config["tasks"]["independent_effort_control"],
    )
    assert len(records) == 2 * config["tasks"]["independent_effort_control"]["rounds"]
    assert all(record["same_goal_across_steps"] is False for record in records)
    assert all(record["p_continue"] is None for record in records)
    assert all(record["persistence_logit"] is None for record in records)
