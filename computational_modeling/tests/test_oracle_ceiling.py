import numpy as np

from computational_modeling.analysis.evaluate_models import choice_metrics


def test_oracle_log_loss_matches_analytical_expectation():
    probabilities = np.array([0.8, 0.8, 0.2, 0.2])
    choices = np.array([1, 0, 1, 0])
    metrics = choice_metrics(choices, probabilities, np.ones(4))
    expected = -np.mean(
        choices * np.log(probabilities)
        + (1 - choices) * np.log(1 - probabilities)
    )
    assert np.isclose(metrics["log_loss"], expected)
    assert np.isclose(metrics["brier"], np.mean((choices - probabilities) ** 2))
