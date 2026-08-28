import pandas as pd

from analysis.persistence_convergence.persistence_vs_controls import (
    compare_persistence_controls,
    sequence_support,
)


def test_one_shot_controls_are_not_given_fabricated_history():
    records = [
        {"episode_id": "c1", "round": 0},
        {"episode_id": "c2", "round": 0},
    ]
    support = sequence_support(records)
    assert support == {
        "history": False,
        "recurrence": False,
        "reason": "one_shot_control",
    }


def test_control_comparison_preserves_layer_and_control_identity():
    persistence = pd.DataFrame(
        {"layer": [0, 1], "test_r_squared": [0.1, 0.4]}
    )
    controls = pd.DataFrame(
        {
            "layer": [0, 1, 0, 1],
            "control": ["generic", "generic", "terminality", "terminality"],
            "test_r_squared": [0.2, 0.5, 0.3, 0.6],
        }
    )
    result = compare_persistence_controls(persistence, controls)
    assert set(result.control) == {"generic", "terminality"}
    assert set(result.layer) == {0, 1}
    assert (result.delta_r_squared_persistence_minus_control < 0).all()
