"""Minimal interpretable one-dimensional policy-commitment state models.

The initial model is a deterministic-input state-space approximation:

    w[t] = rho * w[t-1] + beta_task @ X[t]
    standardized D[t] = alpha_task + w[t] + error

Taskwise standardization absorbs task-specific positive emission scales while
preserving latent ordering. The model is deliberately small and must pass
synthetic architecture-confusion and future-behavior gates before real-state
representational decoding is permitted.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence


DEFAULT_FEATURES = (
    "relative_value",
    "cost_pressure",
    "progress_evidence",
)


def _torch():
    import torch

    return torch


def _normal(generator):
    torch = _torch()
    return float(torch.randn((), generator=generator))


def simulate_latent_trajectories(
    *,
    architecture: str,
    tasks=("bandit", "foraging", "solvability"),
    episodes_per_task: int = 24,
    decisions: int = 12,
    rho: float = 0.7,
    emission_scales: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict:
    """Generate blinded synthetic records for model-recovery/confusion gates."""

    torch = _torch()
    allowed = {"immediate", "choice_inertia", "generic_value", "latent_commitment"}
    if architecture not in allowed:
        raise ValueError(f"unknown synthetic architecture: {architecture!r}")
    if episodes_per_task < 4 or decisions < 4 or not 0 <= rho < 1:
        raise ValueError("invalid synthetic trajectory dimensions or rho")
    tasks = tuple(str(task) for task in tasks)
    if len(tasks) < 1 or len(set(tasks)) != len(tasks):
        raise ValueError("synthetic tasks must be unique")
    emission_scales = dict(emission_scales or {})
    generator = torch.Generator().manual_seed(int(seed))
    records = []
    base_betas = {
        task: torch.tensor(
            [1.15 + 0.08 * index, -0.85 + 0.05 * index, 0.75 + 0.04 * index],
            dtype=torch.float64,
        )
        for index, task in enumerate(tasks)
    }
    for task_index, task in enumerate(tasks):
        scale = float(emission_scales.get(task, 0.8 + 0.6 * (task_index + 1)))
        if scale <= 0:
            raise ValueError("synthetic emission scales must be positive")
        beta = base_betas[task]
        for episode in range(episodes_per_task):
            state = 0.0
            previous_choice = 0.0
            second_previous_choice = 0.0
            episode_id = f"synthetic-{architecture}-{task}-{episode:04d}"
            for decision in range(decisions):
                features = torch.randn(3, generator=generator, dtype=torch.float64)
                immediate = float(torch.dot(beta, features))
                if architecture == "latent_commitment":
                    state = rho * state + immediate
                    decision_logit = scale * state + 0.04 * _normal(generator)
                elif architecture == "generic_value":
                    state = rho * state + 1.8 * float(features[0])
                    decision_logit = scale * state + 0.04 * _normal(generator)
                elif architecture == "choice_inertia":
                    state = 0.0
                    decision_logit = (
                        0.25 * immediate
                        + 3.0 * (previous_choice - 0.5)
                        + 1.2 * (second_previous_choice - 0.5)
                        + 0.04 * _normal(generator)
                    )
                else:
                    state = 0.0
                    decision_logit = 1.7 * immediate + 0.04 * _normal(generator)
                probability = float(torch.sigmoid(torch.tensor(decision_logit)))
                choice = float(torch.rand((), generator=generator) < probability)
                row = {
                    "task": task,
                    "episode_id": episode_id,
                    "pair_id": episode_id,
                    "round": decision,
                    "relative_value": float(features[0]),
                    "cost_pressure": float(features[1]),
                    "progress_evidence": float(features[2]),
                    "previous_choice": previous_choice,
                    "second_previous_choice": second_previous_choice,
                    "choice": choice,
                    "persistence_logit": float(decision_logit),
                    "true_w": float(state),
                    "synthetic_architecture": architecture,
                }
                records.append(row)
                second_previous_choice = previous_choice
                previous_choice = choice
    return {
        "records": records,
        "feature_names": list(DEFAULT_FEATURES),
        "tasks": list(tasks),
        "architecture": architecture,
        "rho": float(rho),
        "beta_by_task": {task: base_betas[task].tolist() for task in tasks},
        "emission_scales": {
            task: float(emission_scales.get(task, 0.8 + 0.6 * (index + 1)))
            for index, task in enumerate(tasks)
        },
    }


def _validate_records(records: Sequence[Mapping], feature_names: Sequence[str]):
    if not records or not feature_names:
        raise ValueError("latent model requires records and input features")
    required = {"task", "episode_id", "round", "persistence_logit", *feature_names}
    for index, row in enumerate(records):
        missing = required - set(row)
        if missing:
            raise ValueError(f"latent record {index} is missing {sorted(missing)}")
        values = [float(row[name]) for name in (*feature_names, "persistence_logit")]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("latent records must contain finite numeric values")


def _episode_accumulated_design(records, indices, feature_names, rho):
    torch = _torch()
    output = torch.zeros((len(indices), len(feature_names)), dtype=torch.float64)
    positions = {original: position for position, original in enumerate(indices)}
    episodes: dict[str, list[int]] = {}
    for original in indices:
        episodes.setdefault(str(records[original]["episode_id"]), []).append(original)
    for episode_indices in episodes.values():
        state = torch.zeros(len(feature_names), dtype=torch.float64)
        ordered = sorted(episode_indices, key=lambda index: int(records[index]["round"]))
        for original in ordered:
            current = torch.tensor(
                [float(records[original][name]) for name in feature_names],
                dtype=torch.float64,
            )
            state = float(rho) * state + current
            output[positions[original]] = state
    return output


def _ridge_fit(train_x, train_y, *, penalty=1e-6):
    torch = _torch()
    design = torch.cat((torch.ones((len(train_x), 1), dtype=train_x.dtype), train_x), dim=1)
    eye = torch.eye(design.shape[1], dtype=design.dtype)
    eye[0, 0] = 0.0
    coefficient = torch.linalg.solve(
        design.T @ design + float(penalty) * eye, design.T @ train_y
    )
    prediction = design @ coefficient
    mse = float(((prediction - train_y) ** 2).mean())
    return coefficient, prediction, mse


def _predict(design_without_intercept, coefficient):
    torch = _torch()
    design = torch.cat(
        (
            torch.ones((len(design_without_intercept), 1), dtype=design_without_intercept.dtype),
            design_without_intercept,
        ),
        dim=1,
    )
    return design @ coefficient


def _task_standardized_targets(records, train_indices, apply_indices):
    torch = _torch()
    tasks = sorted({str(records[index]["task"]) for index in train_indices})
    if {str(records[index]["task"]) for index in apply_indices} - set(tasks):
        raise ValueError("latent target application contains an unseen task")
    moments, target = {}, torch.empty(len(apply_indices), dtype=torch.float64)
    apply_positions = {original: position for position, original in enumerate(apply_indices)}
    for task in tasks:
        train_values = torch.tensor(
            [
                float(records[index]["persistence_logit"])
                for index in train_indices
                if str(records[index]["task"]) == task
            ],
            dtype=torch.float64,
        )
        if len(train_values) < 3:
            raise ValueError(f"task {task!r} has too few latent training states")
        mean = float(train_values.mean())
        std = max(float(train_values.std(unbiased=False)), 1e-6)
        moments[task] = {"mean": mean, "std": std, "fit_split": "train"}
        for index in apply_indices:
            if str(records[index]["task"]) == task:
                target[apply_positions[index]] = (
                    float(records[index]["persistence_logit"]) - mean
                ) / std
    return target, moments


def _fit_recurrent_for_indices(
    records,
    feature_names,
    train_indices,
    apply_indices,
    rho,
    *,
    penalty=1e-6,
):
    torch = _torch()
    train_target, moments = _task_standardized_targets(records, train_indices, train_indices)
    apply_target, _ = _task_standardized_targets(records, train_indices, apply_indices)
    train_design = _episode_accumulated_design(
        records, train_indices, feature_names, rho
    )
    apply_design = _episode_accumulated_design(
        records, apply_indices, feature_names, rho
    )
    tasks = sorted({str(records[index]["task"]) for index in train_indices})
    train_positions = {original: position for position, original in enumerate(train_indices)}
    apply_positions = {original: position for position, original in enumerate(apply_indices)}
    prediction = torch.empty(len(apply_indices), dtype=torch.float64)
    coefficients, train_mses, apply_mses = {}, {}, {}
    for task in tasks:
        local_train = [
            train_positions[index]
            for index in train_indices
            if str(records[index]["task"]) == task
        ]
        local_apply = [
            apply_positions[index]
            for index in apply_indices
            if str(records[index]["task"]) == task
        ]
        coefficient, _train_prediction, train_mse = _ridge_fit(
            train_design[local_train], train_target[local_train], penalty=penalty
        )
        local_prediction = _predict(apply_design[local_apply], coefficient)
        prediction[local_apply] = local_prediction
        coefficients[task] = coefficient.tolist()
        train_mses[task] = train_mse
        apply_mses[task] = float(
            ((local_prediction - apply_target[local_apply]) ** 2).mean()
        )
    return {
        "prediction": prediction,
        "target": apply_target,
        "macro_train_mse": sum(train_mses.values()) / len(train_mses),
        "macro_apply_mse": sum(apply_mses.values()) / len(apply_mses),
        "mse_by_task": apply_mses,
        "coefficients_by_task": coefficients,
        "target_moments": moments,
    }


def fit_latent_state_model(
    records: Sequence[Mapping],
    *,
    feature_names: Sequence[str],
    rho_grid: Sequence[float] = tuple(index / 20 for index in range(20)),
    ridge_penalty: float = 1e-6,
    fit_episode_ids: set[str] | None = None,
) -> dict:
    """Fit rho by task-balanced error and infer consistently oriented w[t]."""

    _validate_records(records, feature_names)
    rho_grid = tuple(float(value) for value in rho_grid)
    if not rho_grid or any(not 0 <= value < 1 for value in rho_grid):
        raise ValueError("rho grid values must fall in [0, 1)")
    indices = list(range(len(records)))
    fit_indices = (
        indices
        if fit_episode_ids is None
        else [
            index
            for index, row in enumerate(records)
            if str(row["episode_id"]) in fit_episode_ids
        ]
    )
    if not fit_indices:
        raise ValueError("latent fit episode set contains no records")
    candidates = []
    for rho in rho_grid:
        fit = _fit_recurrent_for_indices(
            records,
            feature_names,
            fit_indices,
            indices,
            rho,
            penalty=ridge_penalty,
        )
        candidates.append((fit["macro_train_mse"], rho, fit))
    _score, selected_rho, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "model": "latent_commitment",
        "rho": selected_rho,
        "feature_names": list(feature_names),
        "latent_state": selected["prediction"].tolist(),
        "standardized_observed_logit": selected["target"].tolist(),
        "coefficients_by_task": selected["coefficients_by_task"],
        "target_moments": selected["target_moments"],
        "macro_mse": selected["macro_apply_mse"],
        "rho_candidates": [
            {"rho": rho, "macro_mse": score} for score, rho, _fit in candidates
        ],
        "orientation": "higher_means_more_persistence",
        "parameters_fit_on_future_behavior": 0,
        "fit_episodes": len({records[index]["episode_id"] for index in fit_indices}),
        "fit_states": len(fit_indices),
        "application_states": len(indices),
    }


def _episode_split_indices(records, *, validation_fraction=0.25):
    episodes = sorted({str(row["episode_id"]) for row in records})
    validation = {
        episode
        for episode in episodes
        if int.from_bytes(hashlib.sha256(episode.encode()).digest()[:8], "big") % 1000
        < int(1000 * validation_fraction)
    }
    # Deterministic hashing can produce an empty side in tiny tests.
    if not validation or validation == set(episodes):
        cut = max(1, min(len(episodes) - 1, int(len(episodes) * validation_fraction)))
        validation = set(episodes[:cut])
    train = [index for index, row in enumerate(records) if str(row["episode_id"]) not in validation]
    test = [index for index, row in enumerate(records) if str(row["episode_id"]) in validation]
    return train, test


def _fit_design_model(records, feature_names, train_indices, test_indices):
    torch = _torch()
    train_target, _moments = _task_standardized_targets(records, train_indices, train_indices)
    test_target, _ = _task_standardized_targets(records, train_indices, test_indices)
    tasks = sorted({str(records[index]["task"]) for index in train_indices})
    train_positions = {original: position for position, original in enumerate(train_indices)}
    test_positions = {original: position for position, original in enumerate(test_indices)}
    train_x = torch.tensor(
        [[float(records[index].get(name, 0.0)) for name in feature_names] for index in train_indices],
        dtype=torch.float64,
    )
    test_x = torch.tensor(
        [[float(records[index].get(name, 0.0)) for name in feature_names] for index in test_indices],
        dtype=torch.float64,
    )
    task_mses = {}
    for task in tasks:
        local_train = [train_positions[index] for index in train_indices if str(records[index]["task"]) == task]
        local_test = [test_positions[index] for index in test_indices if str(records[index]["task"]) == task]
        if not local_test:
            raise ValueError("episode split omitted a task from model-comparison test")
        coefficient, _prediction, _mse = _ridge_fit(train_x[local_train], train_target[local_train])
        prediction = _predict(test_x[local_test], coefficient)
        task_mses[task] = float(((prediction - test_target[local_test]) ** 2).mean())
    return sum(task_mses.values()) / len(task_mses)


def compare_behavioral_architectures(
    records: Sequence[Mapping],
    *,
    feature_names: Sequence[str],
    generic_value_feature: str,
    rho_grid: Sequence[float] = tuple(index / 20 for index in range(20)),
) -> dict:
    """Distinguish immediate, inertia, recurrent, and generic-value models."""

    _validate_records(records, feature_names)
    if generic_value_feature not in feature_names:
        raise ValueError("generic-value feature must be one of the measured inputs")
    train_indices, test_indices = _episode_split_indices(records)
    results = {
        "immediate_decision": {
            "test_mse": _fit_design_model(
                records, feature_names, train_indices, test_indices
            ),
            "parameters_per_task": len(feature_names) + 1,
        },
        "choice_history_inertia": {
            "test_mse": _fit_design_model(
                records,
                (*feature_names, "previous_choice", "second_previous_choice"),
                train_indices,
                test_indices,
            ),
            "parameters_per_task": len(feature_names) + 3,
        },
    }
    for model_name, recurrent_features in (
        ("latent_commitment", tuple(feature_names)),
        ("generic_latent_value", (generic_value_feature,)),
    ):
        candidates = []
        for rho in rho_grid:
            fit = _fit_recurrent_for_indices(
                records,
                recurrent_features,
                train_indices,
                test_indices,
                float(rho),
            )
            candidates.append((fit["macro_train_mse"], float(rho), fit))
        _train_score, selected_rho, selected = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        results[model_name] = {
            "test_mse": selected["macro_apply_mse"],
            "train_mse": selected["macro_train_mse"],
            "rho": selected_rho,
            "parameters_per_task": len(recurrent_features) + 2,
        }
    best_mse = min(row["test_mse"] for row in results.values())
    tolerance = max(1e-5, 0.01 * best_mse)
    contenders = [
        name for name, row in results.items() if row["test_mse"] <= best_mse + tolerance
    ]
    # Complexity breaks empirical prediction ties. The ordering only resolves
    # exact equal-complexity numerical ties and is fixed before real analysis.
    priority = {
        "immediate_decision": 0,
        "generic_latent_value": 1,
        "choice_history_inertia": 2,
        "latent_commitment": 3,
    }
    selected_model = min(
        contenders,
        key=lambda name: (results[name]["parameters_per_task"], priority[name]),
    )
    return {
        "selected_model": selected_model,
        "models": results,
        "train_episodes": len({records[index]["episode_id"] for index in train_indices}),
        "test_episodes": len({records[index]["episode_id"] for index in test_indices}),
        "selection": "heldout_episode_prediction_with_fixed_complexity_tie_break",
        "latent_commitment_supported": selected_model == "latent_commitment",
    }


def real_task_latent_inputs(records: Sequence[Mapping]) -> tuple[list[dict], list[str]]:
    """Map existing task records into explicit, documented commitment inputs."""

    converted = []
    for source in records:
        task = str(source["task"])
        row = dict(source)
        if task == "bandit":
            expected_a = 5.0 * float(source.get("p_A_true", 0.0)) - 2.0
            expected_b = 5.0 * float(source.get("p_B_true", 0.0)) - 2.0
            row["relative_value"] = max(expected_a, expected_b)
            row["cost_pressure"] = -float(source.get("round", 0.0))
            row["progress_evidence"] = float(source.get("previous_outcome", 0.0) or 0.0)
        elif task == "foraging":
            patch_probability = float(source.get("patch_probability_private", 0.0))
            row["relative_value"] = 4.0 * patch_probability - float(source.get("outside_option", 0.0))
            row["cost_pressure"] = -float(source.get("stay_cost", 0.0))
            row["progress_evidence"] = float(source.get("previous_outcome", 0.0) or 0.0)
        elif task == "solvability":
            row["relative_value"] = float(source.get("progress_probability", 0.0)) - float(source.get("give_up_value", 0.0))
            row["cost_pressure"] = -float(source.get("attempt_cost", 0.0))
            previous = source.get("previous_progress")
            row["progress_evidence"] = 0.0 if previous is None else float(bool(previous))
        else:
            raise ValueError(f"unsupported persistence task for latent inputs: {task!r}")
        row["choice"] = float(
            str(source.get("semantic_choice", source.get("sampled_action", "")))
            in {"A", "B", "STAY", "TRY_AGAIN"}
        )
        converted.append(row)
    by_episode: dict[str, list[int]] = {}
    for index, row in enumerate(converted):
        by_episode.setdefault(str(row["episode_id"]), []).append(index)
    for indices in by_episode.values():
        previous = second_previous = 0.0
        for index in sorted(indices, key=lambda value: int(converted[value]["round"])):
            converted[index]["previous_choice"] = previous
            converted[index]["second_previous_choice"] = second_previous
            second_previous = previous
            previous = float(converted[index]["choice"])
    return converted, list(DEFAULT_FEATURES)
