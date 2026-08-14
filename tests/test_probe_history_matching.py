import random

import pytest

from analysis.probe_history_matching import (
    exact_history_match_analysis,
    variance_decomposition,
)


def test_exact_matching_recovers_older_history_beyond_recent_state():
    rng = random.Random(91)
    rows = []
    for episode in range(40):
        older_history = rng.gauss(0, 1)
        for round_index in (8, 9, 10):
            previous_outcome = -2.0
            sparse_probe = older_history + rng.gauss(0, 0.05)
            full_probe = 1.5 * older_history + rng.gauss(0, 0.05)
            persistence = 0.8 * older_history + rng.gauss(0, 0.05)
            rows.append(
                {
                    "episode_id": f"episode-{episode}",
                    "round": round_index,
                    "previous_outcome": previous_outcome,
                    "loss_streak": 1,
                    "cumulative_score": older_history + previous_outcome,
                    "probe_value": sparse_probe,
                    "probe_value_full": full_probe,
                    "persistence_logit": persistence,
                }
            )

    result = exact_history_match_analysis(rows)

    assert result["eligible_strata"] == 3
    assert result["matched_states"] == 120
    history_probe = result["simple_regressions"]["older_history_to_sparse_probe"]
    history_persistence = result["simple_regressions"]["older_history_to_persistence"]
    assert history_probe["coefficients"]["prior_score"]["standardized_beta"] > 0.9
    assert history_persistence["coefficients"]["prior_score"]["standardized_beta"] > 0.9


def test_variance_decomposition_separates_shared_and_unique_prediction():
    rows = [
        {"probe_value": value, "persistence_logit": value}
        for value in (-2.0, -1.0, 1.0, 2.0)
    ]
    mechanism = {
        "primary_pruned_probe": {
            "control_r_squared": 0.75,
            "augmented_r_squared": 0.80,
        }
    }

    result = variance_decomposition(rows, mechanism, "probe_value")

    assert result["probe_only_r_squared"] == 1.0
    assert result["unique_probe"] == pytest.approx(0.05)
    assert result["unique_history"] == pytest.approx(-0.20)
    assert result["shared_probe_history"] == pytest.approx(0.95)
