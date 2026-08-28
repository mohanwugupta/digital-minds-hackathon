"""Validation-selected shared and task-specific discrete-time hazard models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from analysis.persistence_hazard.build_risk_sets import build_risk_set
from analysis.persistence_hazard.history_kernels import finite_history_features
from computational_modeling.analysis.evaluate_models import choice_metrics
from computational_modeling.models.base import balanced_weights


ARCHITECTURES = (
    "baseline",
    "task_specific",
    "shared_history",
    "shared_stay_switch",
    "fully_shared",
)

CURRENT_FEATURES = (
    "log_round",
    "normalized_time",
    "cost_pressure",
    "progress_evidence",
    "estimated_continue_value",
    "estimated_outside_value",
)


@dataclass(frozen=True)
class TaskStandardizer:
    features: tuple[str, ...]
    task_mean: dict[str, list[float]]
    task_scale: dict[str, list[float]]
    fit_split: str = "train"

    @classmethod
    def fit(cls, records, features, *, pooled=False):
        means, scales = {}, {}
        tasks = sorted({str(row["task"]) for row in records})
        if pooled:
            values = np.asarray(
                [[float(row[name]) for name in features] for row in records],
                dtype=float,
            )
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale[scale == 0] = 1.0
            for task in tasks:
                means[task] = mean.tolist()
                scales[task] = scale.tolist()
        else:
            for task in tasks:
                values = np.asarray(
                    [
                        [float(row[name]) for name in features]
                        for row in records
                        if str(row["task"]) == task
                    ],
                    dtype=float,
                )
                means[task] = values.mean(axis=0).tolist()
                scale = values.std(axis=0)
                scale[scale == 0] = 1.0
                scales[task] = scale.tolist()
        return cls(tuple(features), means, scales)

    def transform(self, records):
        rows = []
        for row in records:
            task = str(row["task"])
            values = np.asarray([float(row[name]) for name in self.features])
            rows.append(
                (values - np.asarray(self.task_mean[task]))
                / np.asarray(self.task_scale[task])
            )
        return np.asarray(rows, dtype=float)


def _weighted_log_loss(y, probability, weights):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(
        np.average(-(y * np.log(probability) + (1 - y) * np.log(1 - probability)), weights=weights)
    )


def _fit_logistic(x, y, weights, penalty, unpenalized):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    mask = np.ones(x.shape[1], dtype=float)
    mask[: int(unpenalized)] = 0.0

    def objective(coefficient):
        score = x @ coefficient
        loss = np.average(np.logaddexp(0.0, score) - y * score, weights=weights)
        return float(loss + 0.5 * float(penalty) * np.sum((mask * coefficient) ** 2))

    def gradient(coefficient):
        error = expit(x @ coefficient) - y
        data = x.T @ (weights * error) / weights.sum()
        return data + float(penalty) * mask * coefficient

    result = minimize(
        objective,
        np.zeros(x.shape[1], dtype=float),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"hazard optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _design(values, task_indices, task_count, architecture, history_count):
    count, feature_count = values.shape
    intercepts = np.eye(task_count)[task_indices]
    if architecture == "baseline":
        return intercepts, task_count
    if architecture == "fully_shared":
        return np.column_stack((np.ones(count), values)), 1
    if architecture == "task_specific":
        blocks = np.zeros((count, task_count * feature_count), dtype=float)
        for task_index in range(task_count):
            mask = task_indices == task_index
            blocks[mask, task_index * feature_count : (task_index + 1) * feature_count] = values[mask]
        return np.column_stack((intercepts, blocks)), task_count
    if architecture == "shared_history":
        history = values[:, :history_count]
        current = values[:, history_count:]
        current_count = current.shape[1]
        blocks = np.zeros((count, task_count * current_count), dtype=float)
        for task_index in range(task_count):
            mask = task_indices == task_index
            blocks[mask, task_index * current_count : (task_index + 1) * current_count] = current[mask]
        return np.column_stack((intercepts, history, blocks)), task_count
    raise ValueError(f"unsupported linear hazard architecture: {architecture}")


def _fit_shared_rule(values, task_indices, y, weights, penalty, task_count):
    feature_count = values.shape[1]

    def unpack(parameters):
        intercept = parameters[:task_count]
        beta = parameters[task_count : task_count + feature_count]
        scales = np.ones(task_count)
        if task_count > 1:
            scales[1:] = np.exp(parameters[task_count + feature_count :])
        return intercept, beta, scales

    weights = np.asarray(weights, dtype=float)
    weights /= weights.mean()

    def objective(parameters):
        intercept, beta, scales = unpack(parameters)
        score = intercept[task_indices] + scales[task_indices] * (values @ beta)
        loss = np.average(np.logaddexp(0.0, score) - y * score, weights=weights)
        return float(loss + 0.5 * float(penalty) * np.sum(beta**2))

    initial = np.zeros(task_count + feature_count + max(0, task_count - 1))
    bounds = [(None, None)] * (task_count + feature_count) + [(-3.0, 3.0)] * max(
        0, task_count - 1
    )
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1200, "ftol": 1e-9},
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(f"shared-rule optimization failed: {result.message}")
    return np.asarray(result.x), unpack


def _metrics(records, probability):
    weights = balanced_weights(records, task_balanced=True)
    observed = [int(row["hazard_event"]) for row in records]
    return choice_metrics(observed, probability, weights)


def fit_hazard_architecture(
    records: Sequence[Mapping],
    architecture: str,
    *,
    history_features: Sequence[str],
    current_features: Sequence[str] = CURRENT_FEATURES,
    penalties=(0.01, 0.1, 1.0),
):
    """Select ridge on validation and evaluate one architecture on test episodes."""

    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown hazard architecture: {architecture}")
    risk = build_risk_set(records)
    split = {
        name: [row for row in risk if str(row["split"]) == name]
        for name in ("train", "validation", "test")
    }
    if any(not rows for rows in split.values()):
        raise ValueError("hazard fitting requires train, validation, and test risk sets")
    features = tuple(history_features) + tuple(current_features)
    standardizer = TaskStandardizer.fit(
        split["train"],
        features,
        pooled=architecture in {"shared_stay_switch", "fully_shared"},
    )
    tasks = sorted({str(row["task"]) for row in risk})
    task_lookup = {task: index for index, task in enumerate(tasks)}

    def prepared(rows):
        values = standardizer.transform(rows)
        task_indices = np.asarray([task_lookup[str(row["task"])] for row in rows], dtype=int)
        y = np.asarray([int(row["hazard_event"]) for row in rows], dtype=float)
        weights = balanced_weights(rows, task_balanced=True)
        return values, task_indices, y, weights

    train = prepared(split["train"])
    validation = prepared(split["validation"])
    candidates = []
    for penalty in penalties:
        if architecture == "shared_stay_switch":
            coefficient, unpack = _fit_shared_rule(
                train[0], train[1], train[2], train[3], penalty, len(tasks)
            )
            intercept, beta, scales = unpack(coefficient)
            validation_score = intercept[validation[1]] + scales[validation[1]] * (validation[0] @ beta)
        else:
            x_train, unpenalized = _design(
                train[0], train[1], len(tasks), architecture, len(history_features)
            )
            coefficient = _fit_logistic(
                x_train, train[2], train[3], penalty, unpenalized
            )
            x_validation, _ = _design(
                validation[0], validation[1], len(tasks), architecture, len(history_features)
            )
            validation_score = x_validation @ coefficient
        candidates.append(
            (
                _weighted_log_loss(validation[2], expit(validation_score), validation[3]),
                float(penalty),
            )
        )
    validation_loss, selected_penalty = min(candidates)
    fit_rows = split["train"] + split["validation"]
    fit_values, fit_tasks, fit_y, fit_weights = prepared(fit_rows)
    test_values, test_tasks, _test_y, _test_weights = prepared(split["test"])
    if architecture == "shared_stay_switch":
        coefficient, unpack = _fit_shared_rule(
            fit_values,
            fit_tasks,
            fit_y,
            fit_weights,
            selected_penalty,
            len(tasks),
        )
        intercept, beta, scales = unpack(coefficient)
        test_score = intercept[test_tasks] + scales[test_tasks] * (test_values @ beta)
    else:
        x_fit, unpenalized = _design(
            fit_values, fit_tasks, len(tasks), architecture, len(history_features)
        )
        coefficient = _fit_logistic(
            x_fit, fit_y, fit_weights, selected_penalty, unpenalized
        )
        x_test, _ = _design(
            test_values, test_tasks, len(tasks), architecture, len(history_features)
        )
        test_score = x_test @ coefficient
    probability = expit(test_score)
    metrics = _metrics(split["test"], probability)
    return {
        "model": architecture,
        "selected_penalty": selected_penalty,
        "validation_log_loss": validation_loss,
        "test_probability": probability,
        "test_records": split["test"],
        "coefficient": coefficient,
        "normalizer": standardizer,
        **{f"test_{name}": value for name, value in metrics.items()},
    }


def _calibration_rows(fit, bins):
    frame = pd.DataFrame(
        {
            "task": [row["task"] for row in fit["test_records"]],
            "observed": [row["hazard_event"] for row in fit["test_records"]],
            "predicted": fit["test_probability"],
        }
    )
    rows = []
    for task, part in frame.groupby("task"):
        local = part.copy()
        local["bin"] = pd.cut(local.predicted, np.linspace(0, 1, int(bins) + 1), include_lowest=True, labels=False)
        for bin_index, group in local.groupby("bin", dropna=False):
            rows.append(
                {
                    "model": fit["model"],
                    "task": task,
                    "bin": int(bin_index),
                    "states": len(group),
                    "mean_predicted_hazard": float(group.predicted.mean()),
                    "observed_hazard": float(group.observed.mean()),
                }
            )
    return rows


def fit_hazard_architectures(records: Sequence[Mapping], config: Mapping):
    lag = max(int(value) for value in config.get("history_lags", [5]))
    history = finite_history_features(lag)
    fits = {
        architecture: fit_hazard_architecture(
            records,
            architecture,
            history_features=history,
            penalties=config.get("ridge_grid", (0.01, 0.1, 1.0)),
        )
        for architecture in ARCHITECTURES
    }
    baseline_loss = fits["baseline"]["test_log_loss"]
    ceiling_loss = fits["task_specific"]["test_log_loss"]
    denominator = baseline_loss - ceiling_loss
    comparison_rows, task_rows, calibration_rows = [], [], []
    for name, fit in fits.items():
        # A ceiling that fails to beat the intercept-only model does not define
        # a meaningful fraction.  This most often happens in tiny smoke subsets.
        fraction = (
            float("nan")
            if denominator <= 1e-12
            else (baseline_loss - fit["test_log_loss"]) / denominator
        )
        comparison_rows.append(
            {
                "model": name,
                "selected_penalty": fit["selected_penalty"],
                "validation_log_loss": fit["validation_log_loss"],
                "test_log_loss": fit["test_log_loss"],
                "test_brier": fit["test_brier"],
                "test_auc": fit["test_auc"],
                "performance_fraction": fraction,
                "states": len(fit["test_records"]),
                "episodes": len({row["episode_id"] for row in fit["test_records"]}),
            }
        )
        for task in sorted({row["task"] for row in fit["test_records"]}):
            indices = [index for index, row in enumerate(fit["test_records"]) if row["task"] == task]
            local_records = [fit["test_records"][index] for index in indices]
            metrics = _metrics(local_records, fit["test_probability"][indices])
            task_rows.append(
                {
                    "model": name,
                    "task": task,
                    **metrics,
                    "states": len(local_records),
                    "episodes": len({row["episode_id"] for row in local_records}),
                }
            )
        calibration_rows.extend(_calibration_rows(fit, config.get("calibration_bins", 10)))
    return {
        "comparison": pd.DataFrame(comparison_rows),
        "taskwise": pd.DataFrame(task_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "fits": fits,
    }


def simulate_hazard_records(
    architecture: str, *, episodes_per_task: int = 120, decisions: int = 8, seed: int = 0
):
    """Generate absorbing synthetic records for architecture-recovery tests."""

    if architecture not in {"shared_history", "task_specific", "fully_shared"}:
        raise ValueError("unsupported generating architecture")
    rng = np.random.default_rng(seed)
    tasks = ("bandit", "foraging", "solvability")
    rows = []
    for task_index, task in enumerate(tasks):
        for episode_index in range(int(episodes_per_task)):
            fraction = episode_index / int(episodes_per_task)
            split = "train" if fraction < 0.6 else "validation" if fraction < 0.8 else "test"
            episode = f"{task}-{episode_index}"
            prior_choice = 0.0
            prior_outcome = 0.0
            outcome_history = []
            action_history = []
            for round_index in range(int(decisions)):
                cost = rng.normal()
                progress = rng.normal()
                continue_value = rng.normal()
                outside = rng.normal()
                if architecture == "fully_shared":
                    hazard_logit = -1.7 + 1.0 * prior_outcome - 0.8 * prior_choice + 0.8 * cost - 0.6 * progress
                elif architecture == "shared_history":
                    current_scale = (1.4, -1.1, 0.5)[task_index]
                    hazard_logit = (-1.9, -1.5, -1.7)[task_index] + 1.2 * prior_outcome - 0.9 * prior_choice + current_scale * cost - 0.4 * progress
                else:
                    history_scale = (2.5, -2.5, 0.1)[task_index]
                    hazard_logit = (-1.8, -1.6, -1.7)[task_index] + history_scale * prior_outcome + (1.0 - task_index * 0.6) * prior_choice + (1.2, -1.0, 0.5)[task_index] * cost
                event = int(rng.random() < expit(hazard_logit))
                outcome = float(rng.normal())
                row = {
                    "task": task,
                    "episode_id": episode,
                    "pair_id": episode,
                    "state_id": f"{episode}:{round_index}",
                    "round": round_index,
                    "split": split,
                    "continue": 1 - event,
                    "previous_choice": prior_choice,
                    "previous_outcome": prior_outcome,
                    "failure_streak": float(sum(value < 0 for value in outcome_history[-3:])),
                    "success_streak": float(sum(value > 0 for value in outcome_history[-3:])),
                    "log_round": float(np.log1p(round_index)),
                    "normalized_time": round_index / max(1, decisions - 1),
                    "cost_pressure": cost,
                    "progress_evidence": progress,
                    "estimated_continue_value": continue_value,
                    "estimated_outside_value": outside,
                    "outcome_after_choice": outcome,
                }
                for lag in (1, 2, 3, 5):
                    row[f"action_lag_{lag}"] = action_history[-lag] if len(action_history) >= lag else 0.0
                    row[f"outcome_lag_{lag}"] = outcome_history[-lag] if len(outcome_history) >= lag else 0.0
                rows.append(row)
                action_history.append(float(1 - event))
                outcome_history.append(outcome)
                prior_choice = float(1 - event)
                prior_outcome = outcome
                if event:
                    break
    return rows
