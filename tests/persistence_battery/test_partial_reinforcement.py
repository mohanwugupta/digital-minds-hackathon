import numpy as np

from experiments.persistence_battery.partial_reinforcement import (
    STOP,
    TRY_AGAIN,
    PartialReinforcementCondition,
    PartialReinforcementEnvironment,
)


def test_acquisition_schedules_and_extinction_are_correct():
    continuous = PartialReinforcementEnvironment(
        PartialReinforcementCondition("continuous", 8, 0.5, 1), 7
    )
    partial = PartialReinforcementEnvironment(
        PartialReinforcementCondition("partial", 8, 0.5, 1), 7
    )
    assert all(continuous.acquisition_reinforced)
    assert 0 < sum(partial.acquisition_reinforced) < 8
    for _ in range(3):
        result = partial.step(TRY_AGAIN)
        assert result.reward == -1
        assert result.task_values["phase"] == "extinction"
    assert partial.step(STOP).terminated


def test_partial_schedule_empirically_matches_probability():
    values = []
    for seed in range(500):
        environment = PartialReinforcementEnvironment(
            PartialReinforcementCondition("partial", 20, 0.4, 0), seed
        )
        values.extend(environment.acquisition_reinforced)
    assert abs(np.mean(values) - 0.4) < 0.03
