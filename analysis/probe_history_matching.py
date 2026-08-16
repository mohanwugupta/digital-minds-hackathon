"""Variance decomposition and exact-history matching for held-out probe states."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator < 1e-15:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def variance_decomposition(rows: list[dict], mechanism: dict, probe_key: str) -> dict:
    """Partition probe/history/joint R² using held-out model fits."""
    probe_r_squared = pearson(
        [float(row[probe_key]) for row in rows],
        [float(row["persistence_logit"]) for row in rows],
    ) ** 2
    if probe_key == "probe_value":
        joint = mechanism["primary_pruned_probe"]
    elif probe_key == "probe_value_full":
        joint = mechanism["full_probe"]
    else:
        raise ValueError(f"unsupported probe key: {probe_key}")
    history_r_squared = float(joint["control_r_squared"])
    joint_r_squared = float(joint["augmented_r_squared"])
    return {
        "probe_only_r_squared": probe_r_squared,
        "history_only_r_squared": history_r_squared,
        "joint_r_squared": joint_r_squared,
        "unique_probe": joint_r_squared - history_r_squared,
        "unique_history": joint_r_squared - probe_r_squared,
        "shared_probe_history": probe_r_squared
        + history_r_squared
        - joint_r_squared,
        "unexplained": 1.0 - joint_r_squared,
    }


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        [*row, *[float(index == column) for column in range(size)]]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            augmented[column][column] += 1e-8
            pivot = column
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[size:] for row in augmented]


def _matrix_multiply(left, right):
    return [
        [
            sum(left_value * right_value for left_value, right_value in zip(row, column))
            for column in zip(*right)
        ]
        for row in left
    ]


def _standardize(values: list[float]) -> tuple[list[float], float]:
    # ``statistics.pstdev`` preserves exact numeric ratios internally. That is
    # useful for small mixed numeric inputs, but makes the repeated factorial
    # regressions unnecessarily slow for tens of thousands of float rows.
    # These analyses are floating-point throughout, so use the equivalent
    # population moments directly.
    mean = sum(values) / len(values)
    scale = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if scale < 1e-12:
        return [0.0] * len(values), 0.0
    return [(value - mean) / scale for value in values], scale


def _clustered_regression(
    outcome: list[float],
    predictors: dict[str, list[float]],
    clusters: list[str],
) -> dict:
    standardized_outcome, outcome_scale = _standardize(outcome)
    names = []
    standardized_predictors = []
    predictor_scales = {}
    for name, values in predictors.items():
        standardized, scale = _standardize(values)
        predictor_scales[name] = scale
        if scale > 0:
            names.append(name)
            standardized_predictors.append(standardized)
    design = [
        [1.0, *[values[index] for values in standardized_predictors]]
        for index in range(len(outcome))
    ]
    parameter_count = len(design[0])
    xtx = [
        [sum(row[left] * row[right] for row in design) for right in range(parameter_count)]
        for left in range(parameter_count)
    ]
    bread = _inverse(xtx)
    xty = [
        sum(row[column] * value for row, value in zip(design, standardized_outcome))
        for column in range(parameter_count)
    ]
    beta = [sum(row[index] * xty[index] for index in range(parameter_count)) for row in bread]
    fitted = [sum(value * coefficient for value, coefficient in zip(row, beta)) for row in design]
    residual = [value - prediction for value, prediction in zip(standardized_outcome, fitted)]
    outcome_mean = sum(standardized_outcome) / len(standardized_outcome)
    total = sum((value - outcome_mean) ** 2 for value in standardized_outcome)
    r_squared = 1.0 - sum(value * value for value in residual) / total if total else 0.0

    grouped = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[str(cluster)].append(index)
    meat = [[0.0] * parameter_count for _ in range(parameter_count)]
    for indices in grouped.values():
        score = [
            sum(design[index][column] * residual[index] for index in indices)
            for column in range(parameter_count)
        ]
        for left in range(parameter_count):
            for right in range(parameter_count):
                meat[left][right] += score[left] * score[right]
    cluster_count = len(grouped)
    sample_count = len(outcome)
    correction = 1.0
    if cluster_count > 1 and sample_count > parameter_count:
        correction = (cluster_count / (cluster_count - 1)) * (
            (sample_count - 1) / (sample_count - parameter_count)
        )
    covariance = _matrix_multiply(_matrix_multiply(bread, meat), bread)
    standard_errors = [
        math.sqrt(max(0.0, correction * covariance[index][index]))
        for index in range(parameter_count)
    ]
    coefficients = {}
    for index, name in enumerate(names, 1):
        standard_error = standard_errors[index]
        z_value = beta[index] / standard_error if standard_error else None
        coefficients[name] = {
            "standardized_beta": beta[index],
            "cluster_robust_standard_error": standard_error,
            "z_value": z_value,
            "normal_approximation_p_value": (
                math.erfc(abs(z_value) / math.sqrt(2)) if z_value is not None else None
            ),
            "predictor_scale": predictor_scales[name],
        }
    for name, scale in predictor_scales.items():
        if scale == 0:
            coefficients[name] = {
                "standardized_beta": 0.0,
                "cluster_robust_standard_error": None,
                "z_value": None,
                "normal_approximation_p_value": None,
                "predictor_scale": 0.0,
            }
    return {
        "states": sample_count,
        "episode_clusters": cluster_count,
        "outcome_scale": outcome_scale,
        "r_squared": r_squared,
        "coefficients": coefficients,
    }


def exact_history_match_analysis(rows: list[dict], min_stratum_size: int = 4) -> dict:
    """Compare older histories at identical round, outcome, and loss streak."""
    grouped = defaultdict(list)
    for row in rows:
        previous_outcome = row.get("previous_outcome")
        if previous_outcome in (None, ""):
            continue
        key = (
            int(float(row["round"])),
            float(previous_outcome),
            int(float(row["loss_streak"])),
        )
        copied = dict(row)
        copied["prior_score"] = float(row["cumulative_score"]) - float(previous_outcome)
        grouped[key].append(copied)
    eligible = {
        key: selected
        for key, selected in grouped.items()
        if len(selected) >= min_stratum_size
        and max(row["prior_score"] for row in selected)
        > min(row["prior_score"] for row in selected)
    }
    matched_rows = [row for selected in eligible.values() for row in selected]
    if not matched_rows:
        raise ValueError("no exact-match strata have older-history variation")

    keys = ("prior_score", "probe_value", "probe_value_full", "persistence_logit")
    residuals = {key: [] for key in keys}
    clusters = []
    for selected in eligible.values():
        means = {
            key: statistics.mean(float(row[key]) for row in selected) for key in keys
        }
        for row in selected:
            for key in keys:
                residuals[key].append(float(row[key]) - means[key])
            clusters.append(str(row["episode_id"]))

    simple = {
        "older_history_to_sparse_probe": _clustered_regression(
            residuals["probe_value"], {"prior_score": residuals["prior_score"]}, clusters
        ),
        "older_history_to_full_probe": _clustered_regression(
            residuals["probe_value_full"],
            {"prior_score": residuals["prior_score"]},
            clusters,
        ),
        "older_history_to_persistence": _clustered_regression(
            residuals["persistence_logit"],
            {"prior_score": residuals["prior_score"]},
            clusters,
        ),
        "sparse_probe_to_persistence": _clustered_regression(
            residuals["persistence_logit"],
            {"probe_value": residuals["probe_value"]},
            clusters,
        ),
        "full_probe_to_persistence": _clustered_regression(
            residuals["persistence_logit"],
            {"probe_value_full": residuals["probe_value_full"]},
            clusters,
        ),
    }
    joint = {
        "sparse": _clustered_regression(
            residuals["persistence_logit"],
            {
                "prior_score": residuals["prior_score"],
                "probe_value": residuals["probe_value"],
            },
            clusters,
        ),
        "full": _clustered_regression(
            residuals["persistence_logit"],
            {
                "prior_score": residuals["prior_score"],
                "probe_value_full": residuals["probe_value_full"],
            },
            clusters,
        ),
    }
    history_standardized, _ = _standardize(residuals["prior_score"])
    standardized = {
        key: _standardize(values)[0]
        for key, values in residuals.items()
        if key != "prior_score"
    }
    order = sorted(range(len(history_standardized)), key=history_standardized.__getitem__)
    deciles = []
    for decile in range(10):
        start = decile * len(order) // 10
        end = (decile + 1) * len(order) // 10
        selected = order[start:end]
        deciles.append(
            {
                "decile": decile + 1,
                "states": len(selected),
                "prior_score": statistics.mean(history_standardized[index] for index in selected),
                **{
                    key: statistics.mean(values[index] for index in selected)
                    for key, values in standardized.items()
                },
            }
        )
    return {
        "matching_variables": ["round", "previous_outcome", "loss_streak"],
        "older_history_measure": "cumulative_score minus previous_outcome",
        "minimum_stratum_size": min_stratum_size,
        "eligible_strata": len(eligible),
        "matched_states": len(matched_rows),
        "episode_clusters": len(set(clusters)),
        "simple_regressions": simple,
        "joint_older_history_and_probe": joint,
        "within_stratum_deciles": deciles,
    }
