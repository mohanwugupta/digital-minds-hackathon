import numpy as np

from experiments.persistence_battery.voluntary_waiting import (
    QUIT,
    WAIT,
    WaitingCondition,
    VoluntaryWaitingEnvironment,
    optimal_policy,
)


def test_waiting_replay_and_configured_probability_are_correct():
    condition = WaitingCondition("short_wait_optimal", 8, 0, 0)
    left = VoluntaryWaitingEnvironment(condition, 17, max_steps=6)
    right = VoluntaryWaitingEnvironment(condition, 17, max_steps=6)
    assert left.arrival_uniforms == right.arrival_uniforms
    assert left.step(WAIT) == right.step(WAIT)

    arrivals = []
    for seed in range(2000):
        env = VoluntaryWaitingEnvironment(condition, seed, max_steps=1)
        arrivals.append(env.step(WAIT).success)
    assert abs(np.mean(arrivals) - condition.arrival_hazard(0)) < 0.04


def test_short_and_long_wait_environments_have_different_optimal_actions():
    short = WaitingCondition("short_wait_optimal", 8, 2, 0)
    long = WaitingCondition("long_wait_optimal", 8, 2, 0)
    assert optimal_policy(short, max_steps=8)[0] == WAIT
    assert optimal_policy(long, max_steps=8)[0] == QUIT
