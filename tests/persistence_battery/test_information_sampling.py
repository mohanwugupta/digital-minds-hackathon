import math
import numpy as np

from experiments.persistence_battery.information_sampling import (
    SAMPLE,
    InformationSamplingCondition,
    InformationSamplingEnvironment,
)


def test_information_sampling_posterior_updates_by_bayes_rule():
    condition = InformationSamplingCondition(
        evidence_accuracy=0.75,
        sample_cost=1,
        error_penalty=8,
        prior_a=0.5,
        true_state="A",
        max_samples=5,
    )
    environment = InformationSamplingEnvironment(condition, 3)
    environment.observations[0] = "A"
    environment.step(SAMPLE)
    assert math.isclose(environment.posterior_a, 0.75)
    environment.observations[1] = "B"
    environment.step(SAMPLE)
    assert math.isclose(environment.posterior_a, 0.5)


def test_information_signals_recover_configured_accuracy_and_replay():
    condition = InformationSamplingCondition(0.7, 0, 4, 0.5, "A", 20)
    values = []
    for seed in range(500):
        left = InformationSamplingEnvironment(condition, seed)
        right = InformationSamplingEnvironment(condition, seed)
        assert left.observations == right.observations
        values.extend(value == "A" for value in left.observations)
    assert abs(np.mean(values) - 0.7) < 0.03
