"""Finite/exponential history kernels and between-task shape similarity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .modeling import select_and_fit_linear_model
from ..evaluation.metrics import task_metrics


def run_history_analysis(records, config, *, logger=None):
    finite_rows, exponential_rows, vectors = [], [], {}
    for task, local in records.groupby("task"):
        train = local[local.split == "train"]
        validation = local[local.split == "validation"]
        test = local[local.split == "test"]
        for lag in config["history_lags"]:
            local_config = {**config, "history_lags": [int(lag)]}
            fit = select_and_fit_linear_model(
                train, validation, test, "finite_history", "task_specific", local_config
            )
            metric = task_metrics(
                pd.DataFrame({"task": test.task, "observed": test.hazard_event, "predicted": fit.prediction})
            ).iloc[0]
            coefficients = np.asarray(fit.coefficients[task])
            value_coefficients = coefficients[1:][::2]
            finite_rows.append(
                {
                    "task": task,
                    "history_lag": int(lag),
                    "validation_log_loss": fit.validation_macro_log_loss,
                    "log_loss": metric.log_loss,
                    "selected_penalty": fit.selected_hyperparameters["penalty"],
                    "feature_names": ";".join(fit.feature_names),
                    "value_coefficients": json.dumps(value_coefficients.tolist()),
                }
            )
        selected = min(
            (row for row in finite_rows if row["task"] == task),
            key=lambda row: (row["validation_log_loss"], row["history_lag"]),
        )
        selected["validation_selected"] = True
        values = json.loads(selected["value_coefficients"])
        names = list(selected["feature_names"].split(";"))[::2]
        vectors[task] = {
            name: value for name, value in zip(names, values) if name.startswith("outcome_lag_")
        }
        for decay in config["history_decay"]:
            local_config = {**config, "history_decay": [float(decay)]}
            fit = select_and_fit_linear_model(
                train, validation, test, "dual_history", "task_specific", local_config
            )
            metric = task_metrics(
                pd.DataFrame({"task": test.task, "observed": test.hazard_event, "predicted": fit.prediction})
            ).iloc[0]
            exponential_rows.append(
                {
                    "task": task,
                    "decay": float(decay),
                    "validation_log_loss": fit.validation_macro_log_loss,
                    "log_loss": metric.log_loss,
                    "selected_penalty": fit.selected_hyperparameters["penalty"],
                }
            )
        if logger is not None:
            logger.note("history", f"completed kernels for {task}")
    finite = pd.DataFrame(finite_rows)
    finite["validation_selected"] = finite["validation_selected"].eq(True)
    exponential = pd.DataFrame(exponential_rows)
    exponential["validation_selected"] = False
    for _task, indices in exponential.groupby("task").groups.items():
        best_index = min(
            indices,
            key=lambda index: (
                exponential.loc[index, "validation_log_loss"],
                exponential.loc[index, "decay"],
            ),
        )
        exponential.loc[best_index, "validation_selected"] = True
    similarity = []
    tasks = sorted(vectors)
    lag_names = [f"outcome_lag_{lag}" for lag in (1, 2, 3, 5, 8)]
    for left_index, left in enumerate(tasks):
        for right in tasks[left_index + 1 :]:
            a = np.asarray([vectors[left].get(name, 0.0) for name in lag_names])
            b = np.asarray([vectors[right].get(name, 0.0) for name in lag_names])
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            similarity.append(
                {
                    "task_a": left,
                    "task_b": right,
                    "correlation": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan"),
                    "cosine_similarity": float(a @ b / denominator) if denominator > 0 else float("nan"),
                }
            )
    return finite, exponential, pd.DataFrame(similarity)
