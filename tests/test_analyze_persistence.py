import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from analysis.analyze_persistence import analyze_frame


def _matched_rows():
    rows = []
    for episode in range(6):
        for state in range(3):
            baseline = 0.1 * episode - 0.05 * state
            value_step = 0.35 + 0.02 * episode
            for alpha in (-1.0, 0.0, 1.0):
                rows.append(
                    {
                        "episode_id": f"episode-{episode}",
                        "state_id": f"episode-{episode}:{state}",
                        "intervention_type": "value",
                        "neuron_set": "value",
                        "alpha": alpha,
                        "persistence_logit": baseline + alpha * value_step,
                    }
                )
            for random_index in range(20):
                random_step = 0.005 * (random_index + 1) + 0.001 * episode
                for alpha in (-1.0, 0.0, 1.0):
                    rows.append(
                        {
                            "episode_id": f"episode-{episode}",
                            "state_id": f"episode-{episode}:{state}",
                            "intervention_type": "random",
                            "neuron_set": f"random_{random_index:02d}",
                            "alpha": alpha,
                            "persistence_logit": baseline + alpha * random_step,
                        }
                    )
    return pd.DataFrame(rows)


def test_matched_analysis_uses_episodes_and_passes_specific_value_effect():
    result = analyze_frame(_matched_rows())

    assert result["inference_unit"] == "episode"
    assert result["n_matched_states"] == 18
    assert result["n_matched_episodes"] == 6
    assert result["primary_monotonic_ordering"] is True
    assert result["primary_ordering_passed"] is True
    assert result["random_set_count"] == 20
    assert result["random_control_empirical_p"] == pytest.approx(1 / 21)
    assert result["random_control_passed"] is True
    total = result["episode_clustered_inference"]["positive_vs_negative"]
    assert total["episodes"] == 6
    assert total["confidence_interval_95"][0] > 0
