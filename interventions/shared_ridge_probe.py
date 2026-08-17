"""Task-balanced ridge fitting for a shared semantic persistence direction."""

from __future__ import annotations

from typing import Mapping, Optional

from analysis.shared_persistence_integrity import macro_average
from interventions.ridge_probe import RidgeProbe, regression_metrics


def _validate_task_data(
    train_by_task: Mapping[str, dict], validation_by_task: Mapping[str, dict]
) -> tuple[str, ...]:
    tasks = tuple(sorted(train_by_task))
    if len(tasks) < 2 or set(tasks) != set(validation_by_task):
        raise ValueError("shared ridge requires the same two or more train/validation tasks")
    widths = set()
    for task in tasks:
        for split_name, split in (
            ("train", train_by_task[task]),
            ("validation", validation_by_task[task]),
        ):
            states, target = split["states"], split["target"]
            if states.ndim != 2 or target.ndim != 1 or len(states) != len(target):
                raise ValueError(f"malformed {task} {split_name} shared-ridge data")
            if len(states) < (2 if split_name == "train" else 1):
                raise ValueError(f"too few {task} {split_name} states")
            widths.add(int(states.shape[1]))
    if len(widths) != 1:
        raise ValueError("all shared-ridge tasks must have the same activation width")
    return tasks


def fit_balanced_shared_ridge(
    train_by_task: Mapping[str, dict],
    validation_by_task: Mapping[str, dict],
    *,
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    device: Optional[str] = None,
) -> tuple[RidgeProbe, dict]:
    """Fit one direction with equal task loss and train-only target scaling.

    Each task's semantic persistence logit is standardized using only that
    task's training episodes. Activation moments and squared-error loss give
    each discovery task equal weight, regardless of its number of states.
    """
    import torch

    tasks = _validate_task_data(train_by_task, validation_by_task)
    if not alphas or any(float(alpha) <= 0 for alpha in alphas):
        raise ValueError("ridge alphas must be positive")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_states = {
        task: train_by_task[task]["states"].float() for task in tasks
    }
    task_state_means = {
        task: states.mean(dim=0) for task, states in train_states.items()
    }
    state_mean = torch.stack(list(task_state_means.values())).mean(dim=0)
    state_variance = torch.stack(
        [
            ((states - state_mean).square()).mean(dim=0)
            for states in train_states.values()
        ]
    ).mean(dim=0)
    state_std = state_variance.sqrt().clamp_min(1e-6)

    target_moments = {}
    train_x, train_y, validation_x, validation_y = {}, {}, {}, {}
    for task in tasks:
        train_target = train_by_task[task]["target"].float()
        mean = float(train_target.mean())
        std = max(float(train_target.std(unbiased=False)), 1e-6)
        target_moments[task] = {
            "mean": mean,
            "std": std,
            "fit_split": "train",
            "states": len(train_target),
        }
        train_x[task] = ((train_states[task] - state_mean) / state_std).to(target_device)
        train_y[task] = ((train_target - mean) / std).to(target_device)
        validation_x[task] = (
            (validation_by_task[task]["states"].float() - state_mean) / state_std
        ).to(target_device)
        validation_y[task] = (
            (validation_by_task[task]["target"].float() - mean) / std
        ).to(target_device)

    # Multiplying each task's rows by sqrt(1 / (T*n_t)) turns an ordinary
    # least-squares solve into mean_t mean_i loss, so large tasks cannot win by
    # contributing more states.
    task_count = len(tasks)
    weighted_x = torch.cat(
        [train_x[task] / (task_count * len(train_x[task])) ** 0.5 for task in tasks]
    )
    weighted_y = torch.cat(
        [train_y[task] / (task_count * len(train_y[task])) ** 0.5 for task in tasks]
    )
    sample_count, feature_count = weighted_x.shape
    mode = "primal" if feature_count <= sample_count else "dual"
    if mode == "primal":
        gram = weighted_x.T @ weighted_x
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0)
        projected_target = eigenvectors.T @ (weighted_x.T @ weighted_y)
    else:
        gram = weighted_x @ weighted_x.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0)
        projected_target = eigenvectors.T @ weighted_y

    candidates, selected = [], None
    for alpha_value in alphas:
        alpha = float(alpha_value)
        if mode == "primal":
            weight = eigenvectors @ (projected_target / (eigenvalues + alpha))
        else:
            dual = eigenvectors @ (projected_target / (eigenvalues + alpha))
            weight = weighted_x.T @ dual
        per_task = {}
        for task in tasks:
            prediction = validation_x[task] @ weight
            per_task[task] = regression_metrics(prediction, validation_y[task])
        row = {
            "alpha": alpha,
            "macro_mse": macro_average(
                {task: metrics["mse"] for task, metrics in per_task.items()}
            ),
            "macro_correlation": macro_average(
                {task: metrics["correlation"] for task, metrics in per_task.items()}
            ),
            "per_task": per_task,
        }
        candidates.append(row)
        if selected is None or row["macro_mse"] < selected["metrics"]["macro_mse"]:
            selected = {
                "alpha": alpha,
                "weight": weight.detach().cpu().clone(),
                "metrics": row,
            }

    assert selected is not None
    probe = RidgeProbe(
        weight=selected["weight"],
        state_mean=state_mean.cpu(),
        state_std=state_std.cpu(),
        target_mean=0.0,
        target_std=1.0,
        alpha=selected["alpha"],
        target="shared_standardized_semantic_persistence_logit",
    )
    return probe, {
        "solver": mode,
        "tasks": list(tasks),
        "task_weighting": "equal_macro_weight",
        "target_standardization": "per_task_training_episodes_only",
        "target_moments": target_moments,
        "state_moments": "equal_task_weight_training_episodes_only",
        "train_states": {task: len(train_x[task]) for task in tasks},
        "validation_states": {task: len(validation_x[task]) for task in tasks},
        "input_dim": feature_count,
        "alpha_candidates": candidates,
        "selected": selected["metrics"],
    }
