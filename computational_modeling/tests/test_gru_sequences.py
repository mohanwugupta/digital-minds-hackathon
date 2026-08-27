import numpy as np

from computational_modeling.models.gru import build_sequences


def test_gru_sequences_reset_and_never_include_future_state():
    rows = [
        {"episode_id": "a", "round": 0, "x": 1.0, "persistence_logit": 10.0},
        {"episode_id": "a", "round": 1, "x": 2.0, "persistence_logit": 20.0},
        {"episode_id": "b", "round": 0, "x": 9.0, "persistence_logit": 90.0},
    ]
    sequences = build_sequences(rows, ["x"], target="persistence_logit")
    assert len(sequences) == 2
    assert np.allclose(sequences[0].features, [[1.0], [2.0]])
    assert np.allclose(sequences[0].targets, [10.0, 20.0])
    assert np.allclose(sequences[1].features, [[9.0]])
    assert sequences[0].episode_id == "a"
    assert sequences[1].episode_id == "b"
