"""Action-versus-outcome finite-history decomposition across persistence tasks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.comparative_persistence.evaluation.metrics import task_metrics
from analysis.comparative_persistence.hazard_models.baselines import (
    fit_ridge_logistic,
    predict_ridge_logistic,
)
from analysis.comparative_persistence.semantic_features import (
    IMMEDIATE_FEATURES,
    build_feature_matrix,
)


CURRENT = ("time_norm", "effort_norm", *IMMEDIATE_FEATURES)
ACTION = (
    "continue_streak",
    *(f"action_lag_{lag}" for lag in (1, 2, 3, 5)),
)
OUTCOME = (
    "success_streak",
    "failure_streak",
    *(f"outcome_lag_{lag}" for lag in (1, 2, 3, 5)),
)
FEATURES = {
    "current_state": CURRENT,
    "action_only": (*CURRENT, *ACTION),
    "outcome_only": (*CURRENT, *OUTCOME),
    "joint_history": (*CURRENT, *ACTION, *OUTCOME),
}


def _fit(train, validation, test, features, penalties):
    x_train, names = build_feature_matrix(train, features)
    x_validation, _ = build_feature_matrix(validation, features)
    x_test, _ = build_feature_matrix(test, features)
    candidates = []
    for penalty in penalties:
        coefficient = fit_ridge_logistic(
            x_train, train.hazard_event, penalty=float(penalty)
        )
        probability = predict_ridge_logistic(coefficient, x_validation)
        score = task_metrics(
            pd.DataFrame(
                {
                    "task": validation.task,
                    "observed": validation.hazard_event,
                    "predicted": probability,
                }
            )
        ).iloc[0].log_loss
        candidates.append((float(score), float(penalty)))
    _score, selected = min(candidates)
    refit = pd.concat((train, validation), ignore_index=True)
    x_refit, _ = build_feature_matrix(refit, features)
    coefficient = fit_ridge_logistic(
        x_refit, refit.hazard_event, penalty=selected
    )
    prediction = predict_ridge_logistic(coefficient, x_test)
    metric = task_metrics(
        pd.DataFrame(
            {
                "task": test.task,
                "observed": test.hazard_event,
                "predicted": prediction,
            }
        )
    ).iloc[0]
    return float(metric.log_loss), coefficient, names, selected


def _coefficient_map(coefficient, names):
    return {
        name: float(coefficient[index + 1])
        for index, name in enumerate(names)
        if not name.endswith("__present")
    }


def run_history_decomposition(records, config, *, smoke=False, logger=None):
    records = records[records.is_persistence_task.astype(bool)]
    penalties = config["smoke"]["ridge_penalties"] if smoke else config["ridge_penalties"]
    rows, kernels = [], []
    for task, local in records.groupby("task"):
        train = local[local.split == "train"]
        validation = local[local.split == "validation"]
        test = local[local.split == "test"]
        fits = {}
        for model, features in FEATURES.items():
            fits[model] = _fit(train, validation, test, features, penalties)
        current_loss = fits["current_state"][0]
        for model, (loss, _coefficient, _names, selected) in fits.items():
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "log_loss": loss,
                    "history_gain": current_loss - loss,
                    "selected_penalty": selected,
                    "states": len(test),
                }
            )
        joint_loss, coefficient, names, _selected = fits["joint_history"]
        coefficients = _coefficient_map(coefficient, names)
        action = {
            name: value
            for name, value in coefficients.items()
            if name.startswith("action_lag_")
        }
        outcome = {
            name: value
            for name, value in coefficients.items()
            if name.startswith("outcome_lag_")
        }
        action_norm = float(np.linalg.norm(list(action.values())))
        outcome_norm = float(np.linalg.norm(list(outcome.values())))
        kernels.append(
            {
                "task": task,
                "joint_log_loss": joint_loss,
                "action_kernel": json.dumps(action, sort_keys=True),
                "outcome_kernel": json.dumps(outcome, sort_keys=True),
                "action_kernel_norm": action_norm,
                "outcome_kernel_norm": outcome_norm,
                "action_outcome_norm_ratio": (
                    action_norm / outcome_norm if outcome_norm > 0 else float("nan")
                ),
                "action_recent_sign": np.sign(action.get("action_lag_1", np.nan)),
                "outcome_recent_sign": np.sign(outcome.get("outcome_lag_1", np.nan)),
                "action_decay_monotone": all(
                    abs(left) >= abs(right)
                    for left, right in zip(list(action.values()), list(action.values())[1:])
                ),
                "outcome_decay_monotone": all(
                    abs(left) >= abs(right)
                    for left, right in zip(list(outcome.values()), list(outcome.values())[1:])
                ),
            }
        )
        if logger is not None:
            logger.note("history_decomposition", f"completed {task}")
    kernel_frame = pd.DataFrame(kernels)
    similarities = []
    for left_index, left in enumerate(kernel_frame.itertuples()):
        for right in list(kernel_frame.itertuples())[left_index + 1 :]:
            for kind in ("action", "outcome"):
                a = np.asarray(list(json.loads(getattr(left, f"{kind}_kernel")).values()))
                b = np.asarray(list(json.loads(getattr(right, f"{kind}_kernel")).values()))
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                similarities.append(
                    {
                        "task_a": left.task,
                        "task_b": right.task,
                        "kernel_type": kind,
                        "cosine_similarity": (
                            float(a @ b / denominator)
                            if denominator > 0
                            else float("nan")
                        ),
                        "directional_agreement": float(
                            np.mean(np.sign(a) == np.sign(b))
                        ),
                    }
                )
    return pd.DataFrame(rows), kernel_frame, pd.DataFrame(similarities)

