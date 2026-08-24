import math

import pytest

from analysis.persistence_contrasts import (
    ContrastDefinition,
    behavioral_validity_gate,
    classify_pair,
    equal_task_manipulation_weights,
    validate_contrast_pair,
)


def _record(**updates):
    record = {
        "task": "foraging",
        "state_id": "state",
        "episode_id": "episode",
        "pair_id": "pair",
        "mapping_id": "stay_x",
        "round": 2,
        "initial_quality": 0.55,
        "depletion": 0.12,
        "outside_option": 0,
        "stay_cost": 0,
        "choice_history": ["STAY", "STAY"],
        "reward_history": [3, -1],
        "persistence_logit": 1.0,
    }
    record.update(updates)
    return record


def test_mismatched_persistence_pair_is_rejected():
    definition = ContrastDefinition(
        task="foraging",
        manipulation="search_cost",
        factor="stay_cost",
        higher_promotes_persistence=False,
        matched_fields=(
            "round",
            "mapping_id",
            "initial_quality",
            "depletion",
            "outside_option",
            "choice_history",
            "reward_history",
        ),
    )
    promoting = _record(stay_cost=0)
    discouraging = _record(
        state_id="other", stay_cost=1, reward_history=[3, 3]
    )

    with pytest.raises(ValueError, match="reward_history"):
        validate_contrast_pair(promoting, discouraging, definition)


def test_reversed_orientation_is_detected():
    definition = ContrastDefinition(
        task="foraging",
        manipulation="search_cost",
        factor="stay_cost",
        higher_promotes_persistence=False,
        matched_fields=("round", "mapping_id"),
    )
    with pytest.raises(ValueError, match="orientation"):
        validate_contrast_pair(
            _record(stay_cost=1),
            _record(state_id="other", stay_cost=0),
            definition,
        )


def test_label_only_pair_is_classified_as_nuisance():
    left = _record(mapping_id="stay_x")
    right = _record(state_id="other", mapping_id="stay_y")
    assert classify_pair(left, right) == "label_identity_nuisance"


def test_task_and_manipulation_families_receive_equal_aggregate_weight():
    rows = []
    for index in range(100):
        rows.append({"task": "bandit", "manipulation": "continue", "id": index})
    for index in range(10):
        rows.append({"task": "bandit", "manipulation": "stop", "id": index})
    for index in range(10):
        rows.append({"task": "foraging", "manipulation": "cost", "id": index})

    weights = equal_task_manipulation_weights(rows)
    by_task = {}
    by_family = {}
    for row, weight in zip(rows, weights):
        by_task[row["task"]] = by_task.get(row["task"], 0.0) + weight
        key = (row["task"], row["manipulation"])
        by_family[key] = by_family.get(key, 0.0) + weight

    assert by_task == pytest.approx({"bandit": 0.5, "foraging": 0.5})
    assert by_family[("bandit", "continue")] == pytest.approx(0.25)
    assert by_family[("bandit", "stop")] == pytest.approx(0.25)
    assert by_family[("foraging", "cost")] == pytest.approx(0.5)
    assert math.isclose(sum(weights), 1.0)


def test_behavioral_gate_requires_positive_clustered_interval():
    passing = behavioral_validity_gate(
        [
            {"behavior_effect": 0.8, "cluster_id": "a"},
            {"behavior_effect": 1.1, "cluster_id": "b"},
            {"behavior_effect": 0.6, "cluster_id": "c"},
        ],
        bootstrap_samples=200,
        seed=3,
    )
    failing = behavioral_validity_gate(
        [
            {"behavior_effect": -0.8, "cluster_id": "a"},
            {"behavior_effect": -1.1, "cluster_id": "b"},
            {"behavior_effect": -0.6, "cluster_id": "c"},
        ],
        bootstrap_samples=200,
        seed=3,
    )
    assert passing["passed"] is True
    assert failing["passed"] is False

