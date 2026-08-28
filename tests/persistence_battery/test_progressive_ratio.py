from experiments.persistence_battery.progressive_ratio import (
    QUIT,
    WORK,
    ProgressiveRatioCondition,
    ProgressiveRatioEnvironment,
)


def test_progressive_ratio_follows_schedule_exactly_and_tracks_breakpoint():
    condition = ProgressiveRatioCondition("steep", 6, 1, 0)
    environment = ProgressiveRatioEnvironment(condition, 2)
    observed = []
    for expected in [1, 2, 4, 6]:
        observed.append(environment.current_requirement)
        for _ in range(expected):
            environment.step(WORK)
    assert observed == [1, 2, 4, 6]
    assert environment.rewards_completed == 4
    result = environment.step(QUIT)
    assert result.terminated
    assert result.task_values["breakpoint"] == 4
