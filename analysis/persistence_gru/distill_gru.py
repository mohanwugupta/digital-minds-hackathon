"""Distill a recurrent policy into explicit history and stay/switch variables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.persistence_convergence.task_specific_readouts import fit_task_readout
from analysis.persistence_gru.memory_ablations import (
    CHOICE_HISTORY,
    OUTCOME_HISTORY,
    TASK_EVIDENCE,
    TIME_FEATURES,
    fit_windowed_gru,
)


def run_gru_distillation(records, features, settings, *, logger=None):
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    full = fit_windowed_gru(
        train,
        validation,
        records,
        features,
        hidden_size=int(settings["hidden_size"]),
        learning_rate=float(settings["learning_rate"]),
        dropout=float(settings.get("dropout", 0.0)),
        max_epochs=int(settings["max_epochs"]),
        patience=int(settings["patience"]),
        seed=int(settings["seed"]),
    )
    targets = np.asarray(full["prediction"], dtype=float)
    task_names = sorted({row["task"] for row in records})
    task_features = tuple(f"task::{task}" for task in task_names)

    def matrix(local_features):
        return np.asarray(
            [
                [float(str(row["task"]) == task) for task in task_names]
                + [float(row[name]) for name in local_features]
                for row in records
            ],
            dtype=float,
        )

    candidates = {
        "history_only": tuple(name for name in features if name in CHOICE_HISTORY | OUTCOME_HISTORY),
        "stay_switch_variables": tuple(name for name in features if name in CHOICE_HISTORY | OUTCOME_HISTORY | TIME_FEATURES | TASK_EVIDENCE),
        "full_observable": tuple(features),
    }
    split_indices = {
        split: np.asarray([index for index, row in enumerate(records) if row["split"] == split], dtype=int)
        for split in ("train", "validation", "test")
    }
    rows = []
    for name, local_features in candidates.items():
        values = matrix(local_features)
        fit = fit_task_readout(
            values[split_indices["train"]],
            targets[split_indices["train"]],
            values[split_indices["validation"]],
            targets[split_indices["validation"]],
            values[split_indices["test"]],
            targets[split_indices["test"]],
            alphas=settings.get("ridge_grid", (0.01, 0.1, 1.0, 10.0)),
        )
        rows.append(
            {
                "model": name,
                "r_squared_to_gru": fit["test_r_squared"],
                "mse_to_gru": fit["test_mse"],
                "pearson_r_to_gru": fit["test_pearson_r"],
                "selected_alpha": fit["selected_alpha"],
                "features": ";".join(task_features + local_features),
            }
        )
    return pd.DataFrame(rows)

