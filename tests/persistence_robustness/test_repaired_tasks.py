import pandas as pd

from experiments.persistence_battery.progressive_ratio import (
    ProgressiveRatioCondition,
    ProgressiveRatioEnvironment,
)
from experiments.persistence_battery.sunk_cost import matched_sunk_cost_conditions
from experiments.persistence_battery.voluntary_waiting import WaitingCondition
from experiments.persistence_battery.validation import manipulation_checks


def test_repaired_waiting_profiles_delay_most_first_step_arrivals():
    early = WaitingCondition("moderate_early", 8, 1, 0)
    late = WaitingCondition("moderate_late", 8, 1, 0)
    assert early.arrival_hazard(0) <= 0.10
    assert late.arrival_hazard(0) <= 0.10
    assert early.arrival_hazard(1) > late.arrival_hazard(1)


def test_repaired_progressive_schedules_create_material_cost_growth():
    moderate = ProgressiveRatioEnvironment(
        ProgressiveRatioCondition("moderate_repair", 5, 6, 8), 1
    )
    sharp = ProgressiveRatioEnvironment(
        ProgressiveRatioCondition("sharp_repair", 5, 6, 8), 1
    )
    assert moderate.schedule != sharp.schedule
    assert sharp.schedule[-1] > moderate.schedule[-1]
    assert 6 * sharp.schedule[-1] > 5


def test_repaired_progressive_schedule_names_have_finite_non_gating_diagnostic():
    rows = []
    for schedule, breakpoint in (
        ("moderate_repair", 4),
        ("sharp_repair", 2),
    ):
        for cost in (2, 6):
            for reward in (5, 11):
                rows.append(
                    {
                        "episode_id": f"{schedule}-{cost}-{reward}",
                        "step": 0,
                        "ratio_schedule": schedule,
                        "effort_cost": cost,
                        "reward_magnitude": reward,
                        "breakpoint": breakpoint,
                        "p_positive_semantic": 0.7 - 0.04 * cost + 0.01 * reward,
                    }
                )
    checks = manipulation_checks(
        {"progressive_ratio": pd.DataFrame(rows)},
        {"validation": {"minimum_expected_probability_effect": 0.02}},
    )
    schedule = checks[
        checks.check == "more gradual effort growth increases breakpoint"
    ].iloc[0]
    assert pd.notna(schedule.effect)
    assert schedule.gate_role == "diagnostic"
    assert checks[checks.check == "lower effort cost increases work"].iloc[0].passed


def test_sunk_repair_keeps_prospective_state_exactly_matched():
    conditions = matched_sunk_cost_conditions(
        prior_investments=(0, 4, 10),
        remaining_steps=4,
        reward_magnitude=14,
        outside_option=6,
        step_cost=4,
        success_probability=0.45,
    )
    prospective = {
        (
            condition.remaining_steps,
            condition.reward_magnitude,
            condition.outside_option,
            condition.step_cost,
            condition.success_probability,
        )
        for condition in conditions
    }
    assert len(prospective) == 1
    assert {condition.prior_investment for condition in conditions} == {0, 4, 10}
