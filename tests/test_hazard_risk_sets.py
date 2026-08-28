import pytest

from analysis.persistence_hazard.build_risk_sets import (
    build_risk_set,
    validate_episode_and_pair_splits,
)


def _row(episode, pair, split, round_index, continued):
    return {
        "task": "synthetic",
        "episode_id": episode,
        "pair_id": pair,
        "state_id": f"{episode}:{round_index}",
        "split": split,
        "round": round_index,
        "continue": int(continued),
        "outcome_after_choice": float(round_index),
    }


def test_risk_set_rejects_post_termination_states():
    records = [
        _row("e1", "p1", "train", 0, True),
        _row("e1", "p1", "train", 1, False),
        _row("e1", "p1", "train", 2, True),
    ]
    with pytest.raises(ValueError, match="post-termination"):
        build_risk_set(records)


def test_risk_set_contains_only_at_risk_states_and_causal_target():
    records = [
        _row("e1", "p1", "train", 0, True),
        _row("e1", "p1", "train", 1, False),
        _row("e2", "p2", "test", 0, True),
    ]
    risk = build_risk_set(records)
    assert [row["hazard_event"] for row in risk] == [0, 1, 0]
    assert all(row["at_risk"] == 1 for row in risk)
    assert "remaining_episode_length" not in risk[0]


def test_episode_and_counterbalanced_pair_cannot_cross_splits():
    records = [
        _row("e1-x", "p1", "train", 0, True),
        _row("e1-y", "p1", "test", 0, True),
    ]
    with pytest.raises(ValueError, match="pair crosses splits"):
        validate_episode_and_pair_splits(records)

