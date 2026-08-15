from analysis.analyze_value_dissociation import analyze_rows


def synthetic_rows():
    rows = []
    for episode in range(12):
        for state_index in range(2):
            state_id = f"episode-{episode}:{state_index}"
            baseline = episode * 0.1 + state_index * 0.2
            for stop in (-10, 0, 10, 20):
                for bonus in (-10, 0, 10):
                    persistence = baseline + 0.2 * bonus - 0.3 * stop
                    rows.append(
                        {
                            "episode_id": f"episode-{episode}",
                            "state_id": state_id,
                            "stop_payoff": stop,
                            "continue_bonus": bonus,
                            "relative_incentive": bonus - stop,
                            "common_incentive": bonus + stop,
                            "persistence_logit": persistence,
                            "p_continue": 0.5,
                            "generic_return_projection": baseline + 0.4 * bonus,
                            "advantage_projection": baseline + 0.5 * (bonus - stop),
                            "persistence_projection": persistence,
                            "previous_outcome": -2,
                            "loss_streak": 1,
                            "round": 6,
                            "history_hash": f"history-{state_id}",
                            "context_hash": f"context-{state_id}-{stop}-{bonus}",
                            "logit_A": persistence,
                            "logit_B": persistence - 0.1,
                        }
                    )
    return rows


def test_factorial_analysis_recovers_stop_continue_and_representation_effects():
    result = analyze_rows(synthetic_rows())
    behavior = result["state_fixed_effects"]["persistence_logit"]["stop_and_continue"]
    representations = result["state_fixed_effects"]

    assert result["audit"]["complete_states"] == 24
    assert behavior["coefficients"]["stop_payoff"]["standardized_beta"] < 0
    assert behavior["coefficients"]["continue_bonus"]["standardized_beta"] > 0
    assert abs(
        representations["generic_return_projection"]["stop_and_continue"]
        ["coefficients"]["stop_payoff"]["standardized_beta"]
    ) < 1e-10
    assert (
        representations["advantage_projection"]["relative_only"]["r_squared"]
        > 0.99
    )
