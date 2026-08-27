from computational_modeling.models.rw import rw_states


def test_rw_updates_only_chosen_arm_after_observed_outcome():
    rows = [
        {"episode_id": "e", "round": 0, "semantic_choice": "A", "outcome_after_choice": 3.0},
        {"episode_id": "e", "round": 1, "semantic_choice": "A", "outcome_after_choice": -2.0},
        {"episode_id": "e", "round": 2, "semantic_choice": "B", "outcome_after_choice": -2.0},
    ]
    states = rw_states(rows, alpha=0.5)
    assert states[0]["rw_a"] == 0.5
    assert states[1]["rw_a"] == 1.75
    assert states[1]["rw_b"] == 0.5
    assert states[2]["rw_a"] == -0.125
    assert states[2]["rw_b"] == 0.5
