"""Validation-selected fitting for interpretable computational models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from computational_modeling.analysis.evaluate_models import persistence_metrics
from computational_modeling.data.feature_schema import select_features
from computational_modeling.models.accumulator import accumulated_design, accumulated_state
from computational_modeling.models.base import (
    TrainStandardizer,
    assert_selection_blind,
    balanced_weights,
    linear_predict,
    weighted_ridge_fit,
)
from computational_modeling.models.history import finite_history_features
from computational_modeling.models.latent_commitment import (
    commitment_inputs,
    generic_value_input,
)
from computational_modeling.models.rw import bayesian_bandit_states, rw_states
from computational_modeling.models.termination import choice_kernel


def _tasks(records, supported_tasks):
    return tuple(task for task in supported_tasks if any(row["task"] == task for row in records))


def _replace_oracle(features, information_set):
    if information_set != "oracle":
        return tuple(features)
    replacements = {
        "estimated_continue_value": "oracle_continue_value",
        "estimated_outside_value": "oracle_outside_value",
        "termination_advantage": "oracle_termination_advantage",
        "relative_value": "oracle_relative_value",
    }
    return tuple(replacements.get(name, name) for name in features)


def _augment_learning(records, *, alpha=None, bayesian=False):
    output = [dict(row) for row in records]
    if alpha is not None:
        for row, states in zip(output, rw_states(output, alpha=float(alpha))):
            row.update(states)
    if bayesian:
        for row, states in zip(output, bayesian_bandit_states(output)):
            row.update(states)
    return output


def _design(records, definition, information_set, hyperparameters):
    records = [dict(row) for row in records]
    state_columns = {}
    if definition.family == "finite_history":
        features = finite_history_features(int(hyperparameters["history_lag"]))
    else:
        features = _replace_oracle(definition.features, information_set)
    if definition.family in {"rw", "rw_bayesian"}:
        records = _augment_learning(
            records,
            alpha=float(hyperparameters["alpha"]),
            bayesian=definition.family == "rw_bayesian",
        )
    elif definition.family == "bayesian":
        records = _augment_learning(records, bayesian=True)
    if definition.family == "sticky":
        advantage = (
            "oracle_termination_advantage"
            if information_set == "oracle"
            else "termination_advantage"
        )
        kernel = choice_kernel(records, decay=float(hyperparameters["decay"]))
        matrix = np.column_stack(([float(row[advantage]) for row in records], kernel))
        state_columns["choice_kernel"] = kernel
        return matrix, (advantage, "choice_kernel"), state_columns
    if definition.family == "accumulator":
        source = "disengagement_evidence"
        if information_set == "oracle":
            source_values = [-float(row["oracle_termination_advantage"]) for row in records]
            for row, value in zip(records, source_values):
                row["oracle_disengagement_evidence"] = value
            source = "oracle_disengagement_evidence"
        state = accumulated_state(records, source, rho=float(hyperparameters["rho"]))
        # Standardize saved orientation: higher means more persistence.
        persistence_state = -state
        state_columns["disengagement_accumulator"] = state
        state_columns["latent_persistence_state"] = persistence_state
        return persistence_state[:, None], ("latent_persistence_state",), state_columns
    if definition.family == "commitment":
        features = commitment_inputs(information_set)
        matrix = accumulated_design(records, features, rho=float(hyperparameters["rho"]))
        state_columns["commitment_input_state"] = matrix
        return matrix, features, state_columns
    if definition.family == "generic_value":
        feature = generic_value_input(information_set)
        state = accumulated_state(records, feature, rho=float(hyperparameters["rho"]))
        state_columns["generic_latent_value"] = state
        return state[:, None], ("generic_latent_value",), state_columns
    if not features:
        return np.empty((len(records), 0)), (), state_columns
    matrices = []
    for task in _tasks(records, definition.supported_tasks):
        task_rows = [row for row in records if row["task"] == task]
        matrix = select_features(task_rows, task, information_set, features)
        matrices.extend(matrix)
    # Preserve original row order; supported multi-task records are normally grouped,
    # but reconstruct explicitly to make that assumption unnecessary.
    if len(set(row["task"] for row in records)) > 1:
        matrices = [
            select_features([row], row["task"], information_set, features)[0]
            for row in records
        ]
    return np.asarray(matrices, dtype=float), features, state_columns


def _fit_apply(train, application, definition, information_set, hyperparameters, sharing):
    x_train, feature_names, _ = _design(train, definition, information_set, hyperparameters)
    x_apply, _, state_columns = _design(application, definition, information_set, hyperparameters)
    prediction = np.empty(len(application), dtype=float)
    fitted = {}
    if sharing == "fully_shared":
        if x_train.shape[1]:
            normalizer = TrainStandardizer.fit(x_train, feature_names)
            train_design = np.asarray(normalizer.transform(x_train))
            apply_design = np.asarray(normalizer.transform(x_apply))
        else:
            normalizer, train_design, apply_design = None, x_train, x_apply
        coefficient = weighted_ridge_fit(
            train_design,
            [row["persistence_logit"] for row in train],
            balanced_weights(train, task_balanced=True),
        )
        prediction[:] = linear_predict(apply_design, coefficient)
        fitted["shared"] = {
            "coefficient": coefficient.tolist(),
            "normalizer": None if normalizer is None else normalizer.__dict__,
        }
    elif sharing in {"task_specific", "shared_architecture_task_observation"}:
        for task in sorted({row["task"] for row in train}):
            train_indices = [index for index, row in enumerate(train) if row["task"] == task]
            apply_indices = [index for index, row in enumerate(application) if row["task"] == task]
            if not apply_indices:
                continue
            local_train, local_apply = x_train[train_indices], x_apply[apply_indices]
            if local_train.shape[1]:
                normalizer = TrainStandardizer.fit(local_train, feature_names)
                local_train = np.asarray(normalizer.transform(local_train))
                local_apply = np.asarray(normalizer.transform(local_apply))
            else:
                normalizer = None
            coefficient = weighted_ridge_fit(
                local_train,
                [train[index]["persistence_logit"] for index in train_indices],
                balanced_weights([train[index] for index in train_indices], task_balanced=False),
            )
            prediction[apply_indices] = linear_predict(local_apply, coefficient)
            fitted[task] = {
                "coefficient": coefficient.tolist(),
                "normalizer": None if normalizer is None else normalizer.__dict__,
            }
    else:
        raise ValueError(f"unknown sharing architecture: {sharing!r}")
    return prediction, fitted, state_columns, list(feature_names)


def _hyperparameter_grid(definition, config):
    if definition.family == "finite_history":
        return [{"history_lag": int(value)} for value in config["finite_history"]["lags"]]
    if definition.family in {"rw", "rw_bayesian"}:
        return [{"alpha": float(value)} for value in config["learning"]["rw_alphas"]]
    if definition.family == "sticky":
        return [{"decay": float(value)} for value in config["dynamics"]["decays"]]
    if definition.family in {"accumulator", "commitment", "generic_value"}:
        return [{"rho": float(value)} for value in config["dynamics"]["rho_grid"]]
    return [{}]


def fit_interpretable_model(
    train: Sequence[Mapping],
    validation: Sequence[Mapping],
    test: Sequence[Mapping],
    definition,
    *,
    information_set: str,
    sharing: str,
    config: Mapping,
):
    """Select all hyperparameters on validation, then freeze and evaluate test."""

    train = [dict(row) for row in train if row["task"] in definition.supported_tasks]
    validation = [dict(row) for row in validation if row["task"] in definition.supported_tasks]
    test = [dict(row) for row in test if row["task"] in definition.supported_tasks]
    assert_selection_blind(train, validation)
    observed_tasks = sorted({row["task"] for row in train})
    if sharing == "task_specific" and len(observed_tasks) > 1:
        task_fits = [
            fit_interpretable_model(
                [row for row in train if row["task"] == task],
                [row for row in validation if row["task"] == task],
                [row for row in test if row["task"] == task],
                definition,
                information_set=information_set,
                sharing="shared_architecture_task_observation",
                config=config,
            )
            for task in observed_tasks
        ]
        state_names = sorted(
            {name for fit in task_fits for name in fit["state_columns"]}
        )
        state_columns = {}
        for name in state_names:
            values = [
                np.asarray(fit["state_columns"][name])
                for fit in task_fits
                if name in fit["state_columns"]
            ]
            state_columns[name] = np.concatenate(values, axis=0)
        return {
            "model": definition.name,
            "code": definition.code,
            "information_set": information_set,
            "sharing": "task_specific",
            "prediction": np.concatenate(
                [np.asarray(fit["prediction"]) for fit in task_fits]
            ),
            "test_records": [
                row for fit in task_fits for row in fit["test_records"]
            ],
            "selected_hyperparameters": {
                task: fit["selected_hyperparameters"]
                for task, fit in zip(observed_tasks, task_fits)
            },
            "validation_macro_mse": float(
                np.mean([fit["validation_macro_mse"] for fit in task_fits])
            ),
            "fitted_parameters": {
                task: fit["fitted_parameters"]
                for task, fit in zip(observed_tasks, task_fits)
            },
            "feature_names": task_fits[0]["feature_names"],
            "state_columns": state_columns,
            "parameter_count": sum(fit["parameter_count"] for fit in task_fits),
            "hyperparameter_candidates": {
                task: fit["hyperparameter_candidates"]
                for task, fit in zip(observed_tasks, task_fits)
            },
        }
    candidates = []
    for hyperparameters in _hyperparameter_grid(definition, config):
        prediction, _, _, _ = _fit_apply(
            train, validation, definition, information_set, hyperparameters, sharing
        )
        task_scores = []
        for task in sorted({row["task"] for row in validation}):
            indices = [index for index, row in enumerate(validation) if row["task"] == task]
            metrics = persistence_metrics(
                [validation[index]["persistence_logit"] for index in indices],
                prediction[indices],
                balanced_weights([validation[index] for index in indices], task_balanced=False),
            )
            task_scores.append(metrics["mse"])
        candidates.append((float(np.mean(task_scores)), hyperparameters))
    best_score = min(item[0] for item in candidates)
    tolerance = max(1e-8, 0.01 * best_score)
    contenders = [item for item in candidates if item[0] <= best_score + tolerance]
    selected_score, selected_hyperparameters = min(
        contenders,
        key=lambda item: (
            int(item[1].get("history_lag", 0)),
            sorted(item[1].items()),
        ),
    )
    fit_records = train + validation
    prediction, fitted, state_columns, features = _fit_apply(
        fit_records, test, definition, information_set, selected_hyperparameters, sharing
    )
    return {
        "model": definition.name,
        "code": definition.code,
        "information_set": information_set,
        "sharing": sharing,
        "prediction": prediction,
        "test_records": test,
        "selected_hyperparameters": selected_hyperparameters,
        "validation_macro_mse": selected_score,
        "fitted_parameters": fitted,
        "feature_names": features,
        "state_columns": state_columns,
        "parameter_count": sum(len(row["coefficient"]) for row in fitted.values()),
        "hyperparameter_candidates": [
            {"validation_macro_mse": score, **parameters}
            for score, parameters in candidates
        ],
    }
