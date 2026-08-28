"""Validation-selected linear hazard fits under three sharing assumptions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ..evaluation.metrics import summarize_predictions
from ..semantic_features import (
    HISTORY_FEATURES,
    add_causal_history,
    build_feature_matrix,
)
from .baselines import MODEL_SPECS, fit_ridge_logistic, predict_ridge_logistic


def task_balanced_weights(frame):
    counts = frame.groupby("task").size().to_dict()
    weights = np.asarray([1.0 / counts[task] for task in frame.task], dtype=float)
    return weights / weights.mean()


def _history_features(lag):
    return tuple(
        ["continue_streak", "failure_streak", "success_streak"]
        + [f"action_lag_{value}" for value in (1, 2, 3, 5, 8) if value <= int(lag)]
        + [f"outcome_lag_{value}" for value in (1, 2, 3, 5, 8) if value <= int(lag)]
    )


def hyperparameter_grid(model_name, config):
    spec = MODEL_SPECS[model_name]
    if spec.kind == "finite_history":
        structural = [{"history_lag": int(value)} for value in config["history_lags"]]
    elif spec.kind == "decay":
        structural = [{"history_decay": float(value)} for value in config["history_decay"]]
    elif spec.kind == "latent":
        structural = [{"rho": float(value)} for value in config["latent_rho"]]
    else:
        structural = [{}]
    return [
        {**structure, "penalty": float(penalty)}
        for structure in structural
        for penalty in config["ridge_penalties"]
    ]


def _filtered_matrix(frame, matrix, rho):
    filtered = np.zeros_like(matrix)
    for (_task, _episode), indices in frame.groupby(
        ["task", "episode_id"], sort=False
    ).groups.items():
        state = np.zeros(matrix.shape[1], dtype=float)
        ordered = sorted(indices, key=lambda index: int(frame.loc[index, "round"]))
        for index in ordered:
            state = float(rho) * state + matrix[index]
            filtered[index] = state
    return filtered


def model_matrix(frame, model_name, hyperparameters, config):
    frame = frame.reset_index(drop=True).copy()
    spec = MODEL_SPECS[model_name]
    if spec.kind == "decay":
        payoff = {
            task: details["payoff_scale"]
            for task, details in config["task_specs"].items()
        }
        frame = add_causal_history(
            frame,
            decay=float(hyperparameters["history_decay"]),
            payoff_scales=payoff,
        ).reset_index(drop=True)
    features = (
        _history_features(hyperparameters["history_lag"])
        if spec.kind == "finite_history"
        else spec.features
    )
    matrix, names = build_feature_matrix(frame, features)
    if spec.kind == "latent":
        matrix = _filtered_matrix(frame, matrix, hyperparameters["rho"])
        names = tuple(f"latent::{name}" for name in names)
    if spec.kind == "flexible":
        lookup = {name: index for index, name in enumerate(names)}
        interactions = (
            ("outcome_lag_1", "cost_norm"),
            ("action_lag_1", "progress_norm"),
            ("time_norm", "progress_norm"),
        )
        extra, extra_names = [], []
        for left, right in interactions:
            if left not in lookup or right not in lookup:
                continue
            present_left = lookup[f"{left}__present"]
            present_right = lookup[f"{right}__present"]
            extra.extend(
                (
                    matrix[:, lookup[left]] * matrix[:, lookup[right]],
                    matrix[:, present_left] * matrix[:, present_right],
                )
            )
            extra_names.extend((f"{left}*{right}", f"{left}*{right}__present"))
        if extra:
            matrix = np.column_stack((matrix, *extra))
            names = (*names, *extra_names)
    return matrix, tuple(names)


def _sharing_design(matrix, tasks, sharing, known_tasks, deviation_multiplier):
    tasks = np.asarray(tasks, dtype=str)
    known_tasks = tuple(known_tasks)
    onehot = np.column_stack(
        [(tasks == task).astype(float) for task in known_tasks]
    ) if known_tasks else np.empty((len(tasks), 0))
    if sharing == "fully_shared":
        return np.column_stack((matrix, onehot)), np.r_[
            np.ones(matrix.shape[1]), np.ones(onehot.shape[1])
        ]
    if sharing == "hierarchical":
        deviations = [matrix * onehot[:, [index]] for index in range(len(known_tasks))]
        design = np.column_stack((matrix, onehot, *deviations))
        penalties = np.r_[
            np.ones(matrix.shape[1]),
            np.full(onehot.shape[1], deviation_multiplier),
            np.full(matrix.shape[1] * len(known_tasks), deviation_multiplier),
        ]
        return design, penalties
    if sharing == "task_specific":
        return matrix, np.ones(matrix.shape[1])
    raise ValueError(f"unknown sharing assumption: {sharing}")


def _macro_loss(frame, prediction):
    scored = pd.DataFrame(
        {
            "task": frame.task.to_numpy(),
            "observed": frame.hazard_event.to_numpy(dtype=float),
            "predicted": prediction,
        }
    )
    return summarize_predictions(scored)["macro_log_loss"]


def _fit_shared(train, application, model_name, sharing, hyper, config):
    known_tasks = sorted(train.task.unique())
    x_train, names = model_matrix(train, model_name, hyper, config)
    x_application, application_names = model_matrix(application, model_name, hyper, config)
    if names != application_names:
        raise RuntimeError("train/application feature definitions differ")
    multiplier = math.sqrt(float(config.get("hierarchical_deviation_penalty", 2.0)))
    design_train, penalty_weights = _sharing_design(
        x_train, train.task, sharing, known_tasks, multiplier
    )
    design_application, _ = _sharing_design(
        x_application, application.task, sharing, known_tasks, multiplier
    )
    coefficient = fit_ridge_logistic(
        design_train,
        train.hazard_event,
        penalty=float(hyper["penalty"]),
        weights=task_balanced_weights(train),
        penalty_weights=penalty_weights,
    )
    return predict_ridge_logistic(coefficient, design_application), {
        "global": coefficient.tolist()
    }, names


def _fit_task_specific(train, application, model_name, hyper, config):
    prediction = np.full(len(application), np.nan)
    coefficients, names = {}, None
    for task in sorted(application.task.unique()):
        local_train = train[train.task == task]
        local_application = application[application.task == task]
        if local_train.empty:
            raise ValueError(f"task-specific model cannot predict unseen task {task}")
        x_train, local_names = model_matrix(local_train, model_name, hyper, config)
        x_application, application_names = model_matrix(
            local_application, model_name, hyper, config
        )
        if local_names != application_names:
            raise RuntimeError("task-specific feature definitions differ")
        coefficient = fit_ridge_logistic(
            x_train,
            local_train.hazard_event,
            penalty=float(hyper["penalty"]),
        )
        positions = application.index.get_indexer(local_application.index)
        prediction[positions] = predict_ridge_logistic(coefficient, x_application)
        coefficients[task] = coefficient.tolist()
        names = local_names
    if not np.isfinite(prediction).all():
        raise RuntimeError("task-specific predictions are incomplete")
    return prediction, coefficients, names or tuple()


@dataclass(frozen=True)
class HazardFit:
    model: str
    sharing: str
    prediction: np.ndarray
    application_records: pd.DataFrame
    selected_hyperparameters: dict
    validation_macro_log_loss: float
    coefficients: dict
    feature_names: tuple[str, ...]

    @property
    def parameter_count(self):
        return int(sum(len(values) for values in self.coefficients.values()))


def select_and_fit_linear_model(
    train,
    validation,
    application,
    model_name,
    sharing,
    config,
):
    if MODEL_SPECS[model_name].kind in {"mlp", "gru"}:
        raise ValueError("neural model requested from linear fitter")
    candidates = []
    for hyper in hyperparameter_grid(model_name, config):
        if sharing == "task_specific":
            prediction, _coef, _names = _fit_task_specific(
                train, validation, model_name, hyper, config
            )
        else:
            prediction, _coef, _names = _fit_shared(
                train, validation, model_name, sharing, hyper, config
            )
        candidates.append((_macro_loss(validation, prediction), dict(hyper)))
    best_loss = min(value for value, _hyper in candidates)
    tolerance = 0.002
    selected_loss, selected = min(
        (item for item in candidates if item[0] <= best_loss + tolerance),
        key=lambda item: (
            item[1].get("history_lag", 0),
            item[1].get("history_decay", 0),
            item[1].get("rho", 0),
            -item[1]["penalty"],
        ),
    )
    refit = pd.concat((train, validation), ignore_index=True)
    application = application.reset_index(drop=True)
    if sharing == "task_specific":
        prediction, coefficients, names = _fit_task_specific(
            refit, application, model_name, selected, config
        )
    else:
        prediction, coefficients, names = _fit_shared(
            refit, application, model_name, sharing, selected, config
        )
    return HazardFit(
        model_name,
        sharing,
        np.asarray(prediction, dtype=float),
        application,
        selected,
        float(selected_loss),
        coefficients,
        tuple(names),
    )


def fit_fixed_linear_model(train, application, model_name, sharing, hyperparameters, config):
    """Fit a frozen, source-selected architecture without new target selection."""

    train = train.reset_index(drop=True)
    application = application.reset_index(drop=True)
    if sharing == "task_specific":
        prediction, coefficients, names = _fit_task_specific(
            train, application, model_name, hyperparameters, config
        )
    else:
        prediction, coefficients, names = _fit_shared(
            train, application, model_name, sharing, hyperparameters, config
        )
    return HazardFit(
        model_name,
        sharing,
        np.asarray(prediction),
        application,
        dict(hyperparameters),
        float("nan"),
        coefficients,
        tuple(names),
    )
