from experiments.persistence_battery.sunk_cost import (
    CONTINUE_WAITING,
    SunkCostEnvironment,
    matched_sunk_cost_conditions,
)


def test_sunk_cost_match_changes_prior_investment_only():
    low, high = matched_sunk_cost_conditions(
        prior_investments=(1, 8),
        remaining_steps=4,
        reward_magnitude=12,
        outside_option=2,
        step_cost=1,
        success_probability=0.75,
    )
    assert low.prior_investment != high.prior_investment
    assert low.remaining_steps == high.remaining_steps
    assert low.reward_magnitude == high.reward_magnitude
    assert low.outside_option == high.outside_option
    assert low.step_cost == high.step_cost
    assert low.success_probability == high.success_probability
    left, right = SunkCostEnvironment(low, 9), SunkCostEnvironment(high, 9)
    assert left.success_uniform == right.success_uniform


def test_sunk_cost_completion_recovers_success_probability():
    condition = matched_sunk_cost_conditions(
        prior_investments=(1,),
        remaining_steps=1,
        reward_magnitude=12,
        outside_option=0,
        step_cost=0,
        success_probability=0.65,
    )[0]
    success = [
        SunkCostEnvironment(condition, seed).step(CONTINUE_WAITING).success
        for seed in range(2000)
    ]
    assert abs(sum(success) / len(success) - 0.65) < 0.04
