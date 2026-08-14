import pytest

from bandit.environment import (
    BanditEnvironment, condition_class, expected_pull_reward,
)


def test_fixed_seed_reproduces_counterfactual_schedule():
    left = BanditEnvironment(0.2, 0.65, seed=17, max_decisions=5)
    right = BanditEnvironment(0.2, 0.65, seed=17, max_decisions=5)
    assert left.outcome_schedule == right.outcome_schedule
    assert len(left.outcome_schedule["A"]) == 5
    assert len(left.outcome_schedule["B"]) == 5


def test_rewards_stop_and_horizon():
    env = BanditEnvironment(1.0, 0.0, seed=3, max_decisions=2)
    assert env.step("A").reward == 3
    final = env.step("B")
    assert final.reward == -2
    assert final.terminated and final.reason == "max_decisions"

    stopped = BanditEnvironment(0.5, 0.5, seed=3).step("C")
    assert stopped.reward == 0
    assert stopped.terminated and stopped.reason == "stop"


def test_arm_outcomes_are_indexed_by_arm_pull_count():
    env = BanditEnvironment(0.5, 0.5, seed=91, max_decisions=10)
    expected_second_a = env.outcome_schedule["A"][1]
    env.step("A")
    env.step("B")
    reward = env.step("A").reward
    assert reward == (3 if expected_second_a else -2)


def test_condition_values_are_explicit_relative_to_stop():
    assert expected_pull_reward(0.20) == pytest.approx(-1.0)
    assert expected_pull_reward(0.35) == pytest.approx(-0.25)
    assert expected_pull_reward(0.50) == pytest.approx(0.5)
    assert expected_pull_reward(0.65) == pytest.approx(1.25)
    assert condition_class(0.20, 0.35) == "both_negative"
    assert condition_class(0.20, 0.50) == "one_positive"
    assert condition_class(0.50, 0.65) == "both_positive"
