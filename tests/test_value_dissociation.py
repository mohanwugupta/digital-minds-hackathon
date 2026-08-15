import json

from bandit.conversation import BanditConversation
from bandit.environment import BanditEnvironment
from experiments.run_value_dissociation import build_factorial_replays


def test_factorial_replays_share_history_and_cross_all_twelve_conditions():
    conversation = BanditConversation.start().snapshot()
    replays = build_factorial_replays("state-1", conversation)

    assert len(replays) == 12
    assert {(item.stop_payoff, item.continue_bonus) for item in replays} == {
        (stop, bonus)
        for stop in (-10, 0, 10, 20)
        for bonus in (-10, 0, 10)
    }
    assert len({item.history_hash for item in replays}) == 1
    assert all(item.relative_incentive == item.continue_bonus - item.stop_payoff for item in replays)
    assert len({item.context_hash for item in replays}) == 12


def test_payoff_prompt_never_contains_private_arm_probabilities():
    conversation = BanditConversation.start().snapshot()
    replays = build_factorial_replays("state-1", conversation)
    serialized = json.dumps([item.conversation for item in replays])

    assert "0.65" not in serialized
    assert "0.20" not in serialized


def test_each_factor_changes_only_its_own_temporary_payoff_line():
    replays = build_factorial_replays(
        "state-1", BanditConversation.start().snapshot()
    )
    by_cell = {
        (item.stop_payoff, item.continue_bonus): item.conversation[-1]["content"].splitlines()
        for item in replays
    }
    stop_differences = [
        index
        for index, (left, right) in enumerate(zip(by_cell[(0, 0)], by_cell[(10, 0)]))
        if left != right
    ]
    continue_differences = [
        index
        for index, (left, right) in enumerate(zip(by_cell[(0, 0)], by_cell[(0, 10)]))
        if left != right
    ]

    assert len(stop_differences) == 1
    assert "STOP" in by_cell[(0, 0)][stop_differences[0]]
    assert len(continue_differences) == 1
    assert "A or B" in by_cell[(0, 0)][continue_differences[0]]


def test_temporary_payoffs_apply_exactly_once_then_environment_returns_to_baseline():
    stop_environment = BanditEnvironment(1.0, 0.0, seed=4)
    stopped = stop_environment.step_with_temporary_payoff(
        "C", stop_payoff=20, continue_bonus=-10
    )
    assert stopped.reward == 20
    assert stopped.terminated

    environment = BanditEnvironment(1.0, 1.0, seed=4)
    manipulated = environment.step_with_temporary_payoff(
        "A", stop_payoff=20, continue_bonus=10
    )
    baseline = environment.step("B")
    assert manipulated.reward == 13
    assert baseline.reward == 3
    assert environment.reward_history == [13, 3]
    assert environment.cumulative_score == 16
