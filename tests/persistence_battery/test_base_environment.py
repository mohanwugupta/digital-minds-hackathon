from dataclasses import replace

import pytest

from experiments.persistence_battery.base_environment import (
    SemanticHistory,
    assign_pair_splits,
)
from experiments.persistence_battery.voluntary_waiting import WaitingCondition


def test_history_accounting_is_exact():
    history = SemanticHistory()
    history.record("WAIT", outcome=-1.0, effort=2.0, reward=0.0, progress=0.1)
    history.record("WAIT", outcome=-2.0, effort=3.0, reward=0.0, progress=0.2)
    state = history.state()
    assert state["previous_action"] == "WAIT"
    assert state["previous_outcome"] == -2.0
    assert state["failure_streak"] == 2
    assert state["success_streak"] == 0
    assert state["elapsed_steps"] == 2
    assert state["cumulative_effort"] == 5.0
    assert state["cumulative_reward"] == 0.0
    assert state["current_progress"] == 0.2


def test_pair_split_keeps_both_mappings_together():
    split = assign_pair_splits([f"pair-{index}" for index in range(40)], seed=11)
    assert set(split.values()) == {"train", "validation", "test"}
    assert split == assign_pair_splits(list(reversed(list(split))), seed=11)


def test_three_pair_smoke_populates_every_split():
    split = assign_pair_splits(["pair-0", "pair-1", "pair-2"], seed=11)
    assert sorted(split.values()) == ["test", "train", "validation"]


def test_factor_replacement_changes_only_requested_field():
    condition = WaitingCondition("short_wait_optimal", 8, 1, 0)
    changed = replace(condition, opportunity_cost=2)
    assert changed.opportunity_cost != condition.opportunity_cost
    assert changed.timing_environment == condition.timing_environment
    assert changed.reward_magnitude == condition.reward_magnitude
    assert changed.quit_payoff == condition.quit_payoff


def test_terminated_environment_rejects_later_states():
    from experiments.persistence_battery.voluntary_waiting import (
        QUIT,
        VoluntaryWaitingEnvironment,
    )

    environment = VoluntaryWaitingEnvironment(
        WaitingCondition("short_wait_optimal", 8, 1, 0), 4
    )
    environment.step(QUIT)
    with pytest.raises(RuntimeError, match="terminated"):
        environment.step(QUIT)
