from experiments.persistence_battery.independent_effort_control import (
    HIGH_EFFORT,
    LOW_EFFORT,
    IndependentEffortCondition,
    IndependentEffortEnvironment,
)


def test_next_offer_is_independent_of_previous_choice_and_outcome():
    condition = IndependentEffortCondition(8)
    left = IndependentEffortEnvironment(condition, 23)
    right = IndependentEffortEnvironment(condition, 23)
    assert left.offers == right.offers
    left.step(HIGH_EFFORT)
    right.step(LOW_EFFORT)
    assert left.current_offer == right.current_offer
    assert left.same_goal_across_steps is False


def test_effort_outcomes_recover_offer_probability():
    successes = []
    for seed in range(1500):
        environment = IndependentEffortEnvironment(
            IndependentEffortCondition(2, high_success_probability=0.6), seed
        )
        successes.append(environment.step(HIGH_EFFORT).success)
    assert abs(sum(successes) / len(successes) - 0.6) < 0.04
