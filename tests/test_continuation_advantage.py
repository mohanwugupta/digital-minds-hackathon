import json

import pytest

from experiments.collect_continuation_advantage import (
    estimate_state_advantage,
    rollout_forced_action,
    select_states,
)


class StopAfterForcedAction:
    action_labels = "ABC"

    def decision(self, _messages):
        return {
            "p_A": 0.0,
            "p_B": 0.0,
            "p_stop": 1.0,
        }


def source_record(state_id="episode:0"):
    return {
        "episode_id": "episode",
        "state_id": state_id,
        "round": 0,
        "p_A_true": 1.0,
        "p_B_true": 0.0,
        "choice_history": json.dumps([]),
        "reward_history": json.dumps([]),
        "conversation": json.dumps(
            [{"role": "user", "content": "Choose A, B, or C."}]
        ),
        "previous_outcome": None,
        "cumulative_score": 0,
        "persistence_logit": 0.0,
    }


def test_forced_rollout_includes_forced_reward_then_policy_stop():
    model = StopAfterForcedAction()
    record = source_record()

    assert rollout_forced_action(
        model, record, "A", outcome_seed=1, action_seed=2
    ) == 3
    assert rollout_forced_action(
        model, record, "B", outcome_seed=1, action_seed=2
    ) == -2


def test_advantage_uses_maximum_forced_action_value():
    result = estimate_state_advantage(
        StopAfterForcedAction(), source_record(), rollouts=4, seed=11
    )

    assert result["q_A"] == 3
    assert result["q_B"] == -2
    assert result["q_STOP"] == 0
    assert result["continuation_advantage"] == 3
    assert result["best_forced_action"] == "A"


def test_state_selection_is_deterministic_and_stratified():
    rows = []
    for episode in range(10):
        for round_index in range(3):
            rows.append(
                {
                    **source_record(f"episode-{episode}:{round_index}"),
                    "episode_id": f"episode-{episode}",
                    "round": round_index,
                    "previous_outcome": None if round_index == 0 else -2,
                    "reward_history": json.dumps([] if round_index == 0 else [-2]),
                }
            )

    first = select_states(rows, 12, seed=7)
    second = select_states(rows, 12, seed=7)
    assert [row["state_id"] for row in first] == [row["state_id"] for row in second]
    assert len(first) == 12
    assert len({row["round"] for row in first}) == 3


def test_advantage_requires_replicates_for_standard_error():
    with pytest.raises(ValueError, match="at least two"):
        estimate_state_advantage(
            StopAfterForcedAction(), source_record(), rollouts=1, seed=11
        )
