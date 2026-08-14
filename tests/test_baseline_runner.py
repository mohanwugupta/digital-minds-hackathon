from collections import Counter

from experiments.run_bandit_baseline import episode_conditions, run_episode


class MockDecisionModel:
    model_id = "mock/model"

    def decision(self, messages):
        # Uniform metrics make sampling depend only on action_seed.
        return {
            "logit_A": 0.0,
            "logit_B": 0.0,
            "logit_C": 0.0,
            "p_A": 1 / 3,
            "p_B": 1 / 3,
            "p_stop": 1 / 3,
            "p_continue": 2 / 3,
            "persistence_logit": 0.6931471805599453,
        }


def test_mock_episode_is_reproducible_and_stops_cleanly():
    first = run_episode(MockDecisionModel(), 0.2, 0.65, seed=4, action_seed=9)
    second = run_episode(MockDecisionModel(), 0.2, 0.65, seed=4, action_seed=9)
    assert [row.to_row() for row in first] == [row.to_row() for row in second]
    assert first[-1].terminated
    assert first[-1].sampled_action == "C"
    assert first[-1].conversation[-1]["role"] == "user"


def test_probability_cells_are_balanced_and_seeded_in_random_order():
    first = list(episode_conditions(200, 2026))
    second = list(episode_conditions(200, 2026))
    assert first == second
    counts = Counter((p_a, p_b) for p_a, p_b, _, _ in first)
    assert len(counts) == 16
    assert max(counts.values()) - min(counts.values()) <= 1
    assert [(p_a, p_b) for p_a, p_b, _, _ in first[:16]] != sorted(counts)
