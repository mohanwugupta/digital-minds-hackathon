import pytest

pytest.importorskip("numpy")

from analysis.analyze_factorial_layerwise import analyze_rows


def synthetic_rows():
    rows = []
    for episode in range(12):
        state_id = f"episode-{episode}:0"
        state_offset = episode * 0.03
        for stop in (-10, 0, 10, 20):
            for continued in (-10, 0, 10):
                relative = continued - stop
                rows.append(
                    {
                        "episode_id": f"episode-{episode}",
                        "state_id": state_id,
                        "stop_payoff": stop,
                        "continue_bonus": continued,
                        "relative_incentive": relative,
                        "common_incentive": continued + stop,
                        "history_hash": f"history-{state_id}",
                        "context_hash": f"context-{state_id}-{stop}-{continued}",
                        "persistence_logit": state_offset + 0.2 * relative,
                        "layer_00_projection": state_offset + 0.05 * relative,
                        "layer_01_projection": state_offset + 0.2 * relative,
                    }
                )
    return rows


def test_layerwise_analysis_recovers_signed_trajectory_without_refitting():
    result = analyze_rows(synthetic_rows(), [0, 1])

    assert result["audit"]["complete_states"] == 12
    assert all(row["stop"]["raw_slope"] < 0 for row in result["layers"])
    assert all(row["continue"]["raw_slope"] > 0 for row in result["layers"])
    assert result["layers"][0]["normalized_to_behavior"]["stop_raw_slope"] == pytest.approx(0.25)
    assert result["layers"][1]["normalized_to_behavior"]["continue_raw_slope"] == pytest.approx(1.0)
