import numpy as np
import pandas as pd
import pytest

from analysis.comparative_persistence.flexible.neural import select_and_fit_neural
from analysis.comparative_persistence.semantic_features import ALL_OBSERVABLE_FEATURES


def _split_frame(split):
    rows = []
    for task_index, task in enumerate(("task_a", "task_b")):
        for episode_index in range(4):
            for round_index in range(3):
                row = {
                    "task": task,
                    "episode_id": f"{split}-{task}-{episode_index}",
                    "round": round_index,
                    "hazard_event": int((task_index + episode_index + round_index) % 4 == 0),
                }
                row.update(
                    {
                        name: float(
                            (task_index + 2 * episode_index + round_index + offset) % 5
                        )
                        / 5.0
                        for offset, name in enumerate(ALL_OBSERVABLE_FEATURES)
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_name", ["mlp", "gru"])
def test_neural_hazard_models_return_ordered_probabilities(model_name):
    pytest.importorskip("torch")
    config = {
        "seed": 11,
        "task_specs": {},
        "mlp": {
            "hidden_sizes": [4],
            "learning_rates": [0.01],
            "dropout": [0.0],
            "max_epochs": 3,
            "patience": 2,
        },
        "gru": {
            "hidden_sizes": [2],
            "learning_rate": 0.01,
            "dropout": 0.0,
            "max_epochs": 3,
            "patience": 2,
        },
    }
    application = _split_frame("test")
    fit = select_and_fit_neural(
        _split_frame("train"),
        _split_frame("validation"),
        application,
        model_name,
        "fully_shared",
        config,
    )
    assert fit.application_records.episode_id.tolist() == application.episode_id.tolist()
    assert fit.prediction.shape == (len(application),)
    assert np.isfinite(fit.prediction).all()
    assert ((fit.prediction > 0) & (fit.prediction < 1)).all()
    assert fit.parameter_count > 0
