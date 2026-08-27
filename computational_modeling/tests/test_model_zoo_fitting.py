import numpy as np
import pytest

from computational_modeling.analysis.model_fitting import fit_interpretable_model
from computational_modeling.data.feature_schema import FEATURE_SCHEMA
from computational_modeling.models.baselines import MODEL_DEFINITIONS
from computational_modeling.models.gru import fit_gru_ceiling
from computational_modeling.models.mlp import fit_mlp_ceiling


def _records():
    rows = []
    tasks = ("bandit", "foraging", "solvability")
    for split_index, split in enumerate(("train", "validation", "test")):
        for task_index, task in enumerate(tasks):
            for episode in range(4):
                for decision in range(3):
                    value = 0.2 * episode + 0.1 * decision - 0.1 * task_index
                    row = {
                        name: value + 0.01 * feature_index
                        for feature_index, name in enumerate(
                            FEATURE_SCHEMA[task]["observable"]
                            + FEATURE_SCHEMA[task]["oracle"]
                        )
                    }
                    row.update(
                        {
                            "task": task,
                            "episode_id": f"{split}-{task}-{episode}",
                            "pair_id": f"{split}-{task}-{episode}",
                            "state_id": f"{split}-{task}-{episode}:{decision}",
                            "round": decision,
                            "split": split,
                            "semantic_choice": "A" if task == "bandit" else "STAY" if task == "foraging" else "TRY_AGAIN",
                            "continue": int(decision < 2),
                            "outcome_after_choice": 1.0 if decision % 2 == 0 else -1.0,
                            "persistence_logit": 1.2 * value - 0.4 * decision + 0.05 * split_index,
                        }
                    )
                    rows.append(row)
    return rows


@pytest.mark.parametrize("definition", MODEL_DEFINITIONS, ids=lambda value: value.name)
def test_every_interpretable_model_fits_without_touching_selection_test(definition):
    records = _records()
    split = {
        name: [row for row in records if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    split["test"] = [
        row
        for row in split["test"]
        if not (row["episode_id"].endswith("-0") and row["round"] == 2)
    ]
    config = {
        "finite_history": {"lags": [1, 2]},
        "learning": {"rw_alphas": [0.2, 0.5]},
        "dynamics": {"decays": [0.2, 0.7], "rho_grid": [0.2, 0.7]},
    }
    fit = fit_interpretable_model(
        split["train"],
        split["validation"],
        split["test"],
        definition,
        information_set="observable",
        sharing="shared_architecture_task_observation",
        config=config,
    )
    assert len(fit["prediction"]) == len(fit["test_records"])
    assert np.isfinite(fit["prediction"]).all()


def test_small_mlp_and_gru_ceilings_train_and_predict():
    records = _records()
    split = {
        name: [row for row in records if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    split["test"] = [
        row
        for row in split["test"]
        if not (row["episode_id"].endswith("-0") and row["round"] == 2)
    ]
    features = ["task_bandit", "task_foraging", "task_solvability", "log_round", "previous_outcome"]
    mlp = fit_mlp_ceiling(
        split["train"], split["validation"], split["test"], features,
        max_epochs=3, patience=2, seed=3,
    )
    gru = fit_gru_ceiling(
        split["train"], split["validation"], split["test"], features,
        hidden_size=4, max_epochs=3, patience=2, seed=3,
    )
    assert len(mlp["prediction"]) == len(split["test"])
    assert len(gru["prediction"]) == len(split["test"])
    assert np.isfinite(mlp["prediction"]).all()
    assert np.isfinite(gru["prediction"]).all()


def test_task_specific_dynamics_select_hyperparameters_independently():
    records = _records()
    split = {
        name: [row for row in records if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    definition = next(
        value for value in MODEL_DEFINITIONS if value.name == "latent_commitment"
    )
    fit = fit_interpretable_model(
        split["train"], split["validation"], split["test"], definition,
        information_set="observable", sharing="task_specific",
        config={
            "finite_history": {"lags": [1]},
            "learning": {"rw_alphas": [0.5]},
            "dynamics": {"decays": [0.5], "rho_grid": [0.2, 0.7]},
        },
    )
    assert set(fit["selected_hyperparameters"]) == {
        "bandit", "foraging", "solvability"
    }
