"""Episode-balanced policy and sampled-choice evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from computational_modeling.models.base import balanced_weights


def persistence_metrics(observed, predicted, weights) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mse = float(np.average((observed - predicted) ** 2, weights=weights))
    baseline = float(np.average((observed - np.average(observed, weights=weights)) ** 2, weights=weights))
    r_squared = 0.0 if baseline <= 0 else 1.0 - mse / baseline
    centered_x = observed - np.average(observed, weights=weights)
    centered_y = predicted - np.average(predicted, weights=weights)
    denominator = np.sqrt(
        np.average(centered_x**2, weights=weights)
        * np.average(centered_y**2, weights=weights)
    )
    correlation = 0.0 if denominator == 0 else float(
        np.average(centered_x * centered_y, weights=weights) / denominator
    )
    return {"r_squared": r_squared, "mse": mse, "pearson_r": correlation}


def _auc(observed, probability) -> float:
    observed = np.asarray(observed, dtype=int)
    positives = observed == 1
    n_positive, n_negative = int(positives.sum()), int((~positives).sum())
    if not n_positive or not n_negative:
        return float("nan")
    ranks = rankdata(np.asarray(probability, dtype=float))
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def choice_metrics(observed, probability, weights) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    weights = np.asarray(weights, dtype=float)
    row_loss = -(
        observed * np.log(probability) + (1 - observed) * np.log(1 - probability)
    )
    return {
        "log_loss": float(np.average(row_loss, weights=weights)),
        "brier": float(np.average((observed - probability) ** 2, weights=weights)),
        "auc": _auc(observed, probability),
    }


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def evaluate_predictions(fit: dict) -> tuple[dict, list[dict]]:
    records = fit["test_records"]
    prediction = np.asarray(fit["prediction"], dtype=float)
    weights = balanced_weights(records, task_balanced=True)
    pooled = {
        **persistence_metrics(
            [row["persistence_logit"] for row in records], prediction, weights
        ),
        **choice_metrics(
            [row["continue"] for row in records], sigmoid(prediction), weights
        ),
    }
    aggregate = {
        "model": fit["model"],
        "code": fit["code"],
        "information_set": fit["information_set"],
        "sharing": fit["sharing"],
        "parameter_count": int(fit["parameter_count"]),
        "task_count": len({row["task"] for row in records}),
        "states": len(records),
        "episodes": len({row["episode_id"] for row in records}),
    }
    taskwise = []
    for task in sorted({row["task"] for row in records}):
        indices = [index for index, row in enumerate(records) if row["task"] == task]
        local_records = [records[index] for index in indices]
        local_weights = balanced_weights(local_records, task_balanced=False)
        taskwise.append(
            {
                **persistence_metrics(
                    [row["persistence_logit"] for row in local_records],
                    prediction[indices],
                    local_weights,
                ),
                **choice_metrics(
                    [row["continue"] for row in local_records],
                    sigmoid(prediction[indices]),
                    local_weights,
                ),
                "model": fit["model"],
                "code": fit["code"],
                "information_set": fit["information_set"],
                "sharing": fit["sharing"],
                "task": task,
                "states": len(local_records),
                "episodes": len({row["episode_id"] for row in local_records}),
            }
        )
    for metric in ("r_squared", "mse", "pearson_r", "log_loss", "brier", "auc"):
        aggregate[metric] = float(
            np.nanmean([float(row[metric]) for row in taskwise])
        )
        aggregate[f"pooled_task_balanced_{metric}"] = float(pooled[metric])
    return aggregate, taskwise


def normalized_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in metrics.groupby(["information_set", "sharing"], dropna=False):
        information_set, sharing = keys
        baseline_rows = part[part.model == "intercept"]
        flexible_rows = part[
            part.model.isin(["flexible_linear", "mlp", "gru"])
        ].sort_values(["r_squared", "mse"], ascending=[False, True])
        oracle_rows = part[part.model == "oracle_policy"]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        flexible = None if flexible_rows.empty else flexible_rows.iloc[0]
        oracle = None if oracle_rows.empty else oracle_rows.iloc[0]
        for row in part.itertuples():
            fraction_flexible = (
                float("nan")
                if flexible is None or abs(float(flexible.r_squared)) < 1e-12
                else float(row.r_squared) / float(flexible.r_squared)
            )
            denominator = (
                float("nan")
                if oracle is None
                else float(baseline.log_loss) - float(oracle.log_loss)
            )
            fraction_reducible = (
                float("nan")
                if not np.isfinite(denominator) or abs(denominator) < 1e-12
                else (float(baseline.log_loss) - float(row.log_loss)) / denominator
            )
            rows.append(
                {
                    "model": row.model,
                    "information_set": information_set,
                    "sharing": sharing,
                    "best_flexible_reference": (
                        "" if flexible is None else str(flexible.model)
                    ),
                    "fraction_best_flexible_r_squared_raw": fraction_flexible,
                    "delta_r_squared_best_flexible_minus_model": (
                        float("nan")
                        if flexible is None
                        else float(flexible.r_squared) - float(row.r_squared)
                    ),
                    "fraction_reducible_choice_log_loss": fraction_reducible,
                    "raw_r_squared": float(row.r_squared),
                    "raw_log_loss": float(row.log_loss),
                }
            )
    return pd.DataFrame(rows)


def model_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in metrics.groupby(["information_set", "sharing"], dropna=False):
        flexible = part[
            part.model.isin(["flexible_linear", "mlp", "gru"])
        ].sort_values(["r_squared", "mse"], ascending=[False, True])
        flexible_reference = None if flexible.empty else str(flexible.iloc[0].model)
        reference_names = ["time", "choice_inertia"]
        if flexible_reference is not None:
            reference_names.append(flexible_reference)
        references = {
            name: part[part.model == name].iloc[0]
            for name in dict.fromkeys(reference_names)
            if not part[part.model == name].empty
        }
        for candidate in part.itertuples():
            for reference_name, reference in references.items():
                if candidate.model == reference_name:
                    continue
                rows.append(
                    {
                        "model": candidate.model,
                        "reference": reference_name,
                        "information_set": keys[0],
                        "sharing": keys[1],
                        "delta_mse_reference_minus_model": float(reference.mse)
                        - float(candidate.mse),
                        "delta_r_squared_model_minus_reference": float(candidate.r_squared)
                        - float(reference.r_squared),
                        "delta_log_loss_reference_minus_model": float(reference.log_loss)
                        - float(candidate.log_loss),
                    }
                )
    return pd.DataFrame(rows)


def clustered_bootstrap_intervals(
    fits: list[dict], *, samples: int, seed: int
) -> pd.DataFrame:
    """Pair/episode-cluster bootstrap for each model's primary metrics."""

    if int(samples) < 20:
        raise ValueError("cluster bootstrap requires at least 20 samples")
    rows = []
    for fit_index, fit in enumerate(fits):
        records = fit["test_records"]
        prediction = np.asarray(fit["prediction"], dtype=float)
        probability = sigmoid(prediction)
        episode_stats: dict[tuple[str, str, str], list[list[float]]] = {}
        for index, row in enumerate(records):
            observed = float(row["persistence_logit"])
            choice = float(row["continue"])
            key = (
                str(row["task"]),
                str(row.get("pair_id", row["episode_id"])),
                str(row["episode_id"]),
            )
            episode_stats.setdefault(key, []).append(
                [
                    observed,
                    observed**2,
                    (observed - prediction[index]) ** 2,
                    -(
                        choice * np.log(np.clip(probability[index], 1e-9, 1))
                        + (1 - choice)
                        * np.log(np.clip(1 - probability[index], 1e-9, 1))
                    ),
                    (choice - probability[index]) ** 2,
                ]
            )
        pair_stats: dict[str, dict[str, list[np.ndarray]]] = {}
        for (task, pair, _episode), values in episode_stats.items():
            pair_stats.setdefault(task, {}).setdefault(pair, []).append(
                np.asarray(values, dtype=float).mean(axis=0)
            )
        task_arrays = {
            task: np.stack(
                [np.mean(episodes, axis=0) for _pair, episodes in sorted(pairs.items())]
            )
            for task, pairs in pair_stats.items()
        }
        local_rng = np.random.default_rng(int(seed) + fit_index * 104729)
        task_boot = {}
        for task, task_values in task_arrays.items():
            indices = local_rng.integers(
                0, len(task_values), size=(int(samples), len(task_values))
            )
            task_boot[task] = task_values[indices].mean(axis=1)
        task_metric_draws = {
            metric: [] for metric in ("r_squared", "mse", "log_loss", "brier")
        }
        for task, task_values in task_boot.items():
            task_variance = np.maximum(
                task_values[:, 1] - task_values[:, 0] ** 2, 1e-12
            )
            task_draws = {
                "r_squared": 1.0 - task_values[:, 2] / task_variance,
                "mse": task_values[:, 2],
                "log_loss": task_values[:, 3],
                "brier": task_values[:, 4],
            }
            for metric, values in task_draws.items():
                task_metric_draws[metric].append(values)
                rows.append(
                    {
                        "model": fit["model"],
                        "kind": "metric",
                        "reference": "",
                        "task": task,
                        "information_set": fit["information_set"],
                        "sharing": fit["sharing"],
                        "metric": metric,
                        "estimate": float(np.mean(values)),
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "samples": int(samples),
                        "clusters": len(task_arrays[task]),
                    }
                )
        draws = {
            metric: np.stack(values).mean(axis=0)
            for metric, values in task_metric_draws.items()
        }
        for metric, values in draws.items():
            rows.append(
                {
                    "model": fit["model"],
                    "kind": "metric",
                    "reference": "",
                    "task": "macro",
                    "information_set": fit["information_set"],
                    "sharing": fit["sharing"],
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "samples": int(samples),
                    "clusters": sum(len(values) for values in task_arrays.values()),
                }
            )
    return pd.DataFrame(rows)


def clustered_bootstrap_differences(
    fits: list[dict], *, samples: int, seed: int
) -> pd.DataFrame:
    """Paired intervals versus time, inertia, and the best flexible reference."""

    if int(samples) < 20:
        raise ValueError("cluster bootstrap requires at least 20 samples")
    lookup = {
        (fit["information_set"], fit["sharing"], fit["model"]): fit for fit in fits
    }
    flexible_references = {}
    for fit in fits:
        if fit["model"] not in {"flexible_linear", "mlp", "gru"}:
            continue
        key = (fit["information_set"], fit["sharing"])
        aggregate, _ = evaluate_predictions(fit)
        contender = (float(aggregate["r_squared"]), -float(aggregate["mse"]), fit)
        if key not in flexible_references or contender[:2] > flexible_references[key][:2]:
            flexible_references[key] = contender
    rows = []
    rng = np.random.default_rng(seed + 7919)
    for candidate in fits:
        if candidate["model"] in {"oracle_policy", "time", "choice_inertia", "gru"}:
            continue
        flexible = flexible_references.get(
            (candidate["information_set"], candidate["sharing"])
        )
        reference_names = ["time", "choice_inertia"]
        if flexible is not None:
            reference_names.append(str(flexible[2]["model"]))
        for reference_name in dict.fromkeys(reference_names):
            reference = lookup.get(
                (candidate["information_set"], candidate["sharing"], reference_name)
            )
            if reference is None:
                continue
            candidate_by_state = {
                str(row["state_id"]): (row, float(prediction))
                for row, prediction in zip(
                    candidate["test_records"], candidate["prediction"]
                )
            }
            reference_by_state = {
                str(row["state_id"]): float(prediction)
                for row, prediction in zip(
                    reference["test_records"], reference["prediction"]
                )
            }
            common = sorted(set(candidate_by_state) & set(reference_by_state))
            if not common:
                continue
            episode_values: dict[tuple[str, str, str], dict[str, list[float]]] = {}
            for state_id in common:
                record, candidate_prediction = candidate_by_state[state_id]
                reference_prediction = reference_by_state[state_id]
                observed = float(record["persistence_logit"])
                choice = float(record["continue"])
                candidate_probability = float(sigmoid([candidate_prediction])[0])
                reference_probability = float(sigmoid([reference_prediction])[0])
                key = (
                    str(record["task"]),
                    str(record["pair_id"]),
                    str(record["episode_id"]),
                )
                values = episode_values.setdefault(
                    key, {"mse": [], "log_loss": []}
                )
                values["mse"].append(
                    (observed - reference_prediction) ** 2
                    - (observed - candidate_prediction) ** 2
                )
                values["log_loss"].append(
                    -(
                        choice * np.log(np.clip(reference_probability, 1e-9, 1))
                        + (1 - choice)
                        * np.log(np.clip(1 - reference_probability, 1e-9, 1))
                    )
                    + (
                        choice * np.log(np.clip(candidate_probability, 1e-9, 1))
                        + (1 - choice)
                        * np.log(np.clip(1 - candidate_probability, 1e-9, 1))
                    )
                )
            cluster_values: dict[str, dict[str, dict[str, list[float]]]] = {}
            for (task, pair, _episode), values in episode_values.items():
                target = cluster_values.setdefault(task, {}).setdefault(
                    pair, {"mse": [], "log_loss": []}
                )
                for metric in target:
                    target[metric].append(float(np.mean(values[metric])))
            task_draws = {"mse": {}, "log_loss": {}}
            for task, clusters in cluster_values.items():
                cluster_ids = sorted(clusters)
                for metric in task_draws:
                    values = np.asarray(
                        [np.mean(clusters[cluster][metric]) for cluster in cluster_ids]
                    )
                    indices = rng.integers(
                        0, len(values), size=(int(samples), len(values))
                    )
                    draws = values[indices].mean(axis=1)
                    task_draws[metric][task] = draws
                    rows.append(
                        {
                            "kind": "difference",
                            "model": candidate["model"],
                            "reference": reference_name,
                            "information_set": candidate["information_set"],
                            "sharing": candidate["sharing"],
                            "task": task,
                            "metric": f"delta_{metric}_reference_minus_model",
                            "estimate": float(values.mean()),
                            "ci_low": float(np.quantile(draws, 0.025)),
                            "ci_high": float(np.quantile(draws, 0.975)),
                            "samples": int(samples),
                            "clusters": len(values),
                        }
                    )
            for metric, by_task in task_draws.items():
                macro = np.mean(np.stack(list(by_task.values())), axis=0)
                rows.append(
                    {
                        "kind": "difference",
                        "model": candidate["model"],
                        "reference": reference_name,
                        "information_set": candidate["information_set"],
                        "sharing": candidate["sharing"],
                        "task": "macro",
                        "metric": f"delta_{metric}_reference_minus_model",
                        "estimate": float(np.mean(macro)),
                        "ci_low": float(np.quantile(macro, 0.025)),
                        "ci_high": float(np.quantile(macro, 0.975)),
                        "samples": int(samples),
                        "clusters": sum(
                            len(clusters) for clusters in cluster_values.values()
                        ),
                    }
                )
    return pd.DataFrame(rows)
