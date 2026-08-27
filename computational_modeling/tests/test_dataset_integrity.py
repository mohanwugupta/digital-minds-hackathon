import pytest

from computational_modeling.data.build_cross_task_behavioral_dataset import (
    harmonize_record,
    validate_behavioral_records,
)


def test_semantic_continuation_is_harmonized_across_tasks():
    cases = (
        ("bandit", {"sampled_action": "A"}, 1),
        ("bandit", {"sampled_action": "C"}, 0),
        ("foraging", {"semantic_choice": "STAY"}, 1),
        ("foraging", {"semantic_choice": "LEAVE"}, 0),
        ("solvability", {"semantic_choice": "TRY_AGAIN"}, 1),
        ("solvability", {"semantic_choice": "GIVE_UP"}, 0),
    )
    common = {
        "episode_id": "e",
        "state_id": "e:0",
        "round": 0,
        "p_continue": 0.7,
        "p_stop": 0.3,
        "persistence_logit": 0.8472978604,
    }
    for task, choice, expected in cases:
        row = harmonize_record(task, {**common, **choice}, pair_id="pair")
        assert row["continue"] == expected


def test_state_order_must_be_contiguous_within_episode():
    records = [
        {"task": "bandit", "episode_id": "e", "pair_id": "e", "state_id": "e:0", "round": 0},
        {"task": "bandit", "episode_id": "e", "pair_id": "e", "state_id": "e:2", "round": 2},
    ]
    with pytest.raises(ValueError, match="state order"):
        validate_behavioral_records(records, require_targets=False)
