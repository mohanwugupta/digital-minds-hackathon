from analysis.analyze_causal_steering import analyze


def synthetic_rows():
    rows = []
    effects = {"persistence": 0.8, "generic_return": 0.2, "advantage": 0.0}
    for episode in range(12):
        for state_index in range(2):
            state_id = f"episode-{episode}:{state_index}"
            baseline = episode * 0.01 + state_index * 0.02
            for direction, effect in effects.items():
                for control_type, control_ids in (
                    ("target", ("target",)),
                    ("random", tuple(f"random_{index:02d}" for index in range(20))),
                ):
                    for control_id in control_ids:
                        selected_effect = effect if control_type == "target" else 0.01
                        for alpha in (-1.0, 0.0, 1.0):
                            rows.append(
                                {
                                    "episode_id": f"episode-{episode}",
                                    "state_id": state_id,
                                    "direction_name": direction,
                                    "control_type": control_type,
                                    "control_id": control_id,
                                    "alpha": alpha,
                                    "context_hash": f"context-{state_id}",
                                    "logit_A": "1.0" if alpha == 0 else str(1 + alpha * selected_effect),
                                    "logit_B": "0.5" if alpha == 0 else str(0.5 + alpha * selected_effect),
                                    "logit_C": "0.0",
                                    "persistence_logit": baseline + alpha * selected_effect,
                                    "p_continue": 0.8,
                                    "probe_value_pre": 0.0,
                                    "probe_value_post": alpha,
                                    "direction_l2_norm": 1.0,
                                    "intervention_relative_rms": 0.1,
                                }
                            )
    return rows


def test_causal_analysis_requires_and_recovers_persistence_positive_control():
    result = analyze(synthetic_rows())

    assert result["audit"]["alpha_zero_exact_across_conditions"]
    assert result["audit"]["target_probe_ordering_failures"] == 0
    assert result["positive_control_passed"]
    assert result["directions"]["persistence"]["target"]["mean"] > 1.5
    assert result["directions"]["advantage"]["target"]["mean"] == 0

