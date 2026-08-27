import numpy as np

from computational_modeling.models.accumulator import accumulated_state
from computational_modeling.models.termination import choice_kernel


def test_hand_constructed_latent_recursions_are_exact_and_reset_by_episode():
    rows = [
        {"episode_id": "a", "round": 0, "evidence": 1.0, "continue": 1},
        {"episode_id": "a", "round": 1, "evidence": 2.0, "continue": 0},
        {"episode_id": "b", "round": 0, "evidence": 4.0, "continue": 1},
    ]
    assert np.allclose(accumulated_state(rows, "evidence", rho=0.5), [1.0, 2.5, 4.0])
    # K_t contains choices strictly before t.
    assert np.allclose(choice_kernel(rows, decay=0.5), [0.0, 1.0, 0.0])
