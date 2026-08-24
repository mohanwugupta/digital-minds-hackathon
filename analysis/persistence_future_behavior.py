"""Future-engagement validation beyond the immediate semantic choice logit."""

from __future__ import annotations


def _vector(value, *, name: str):
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float64).flatten()
    if len(tensor) < 5 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain at least five finite values")
    return tensor


def _ols_metrics(design, target):
    import torch

    coefficient = torch.linalg.pinv(design) @ target
    prediction = design @ coefficient
    residual = target - prediction
    total = ((target - target.mean()) ** 2).sum()
    r_squared = 1.0 - float((residual**2).sum() / total.clamp_min(1e-12))
    return coefficient, prediction, r_squared


def future_behavior_validation(
    *,
    current_choice,
    latent_state,
    future_outcome,
    covariates=None,
    minimum_incremental_r_squared: float = 0.01,
) -> dict:
    """Test whether commitment adds prediction beyond current choice/covariates."""

    import torch

    current = _vector(current_choice, name="current choice")
    latent = _vector(latent_state, name="latent state")
    future = _vector(future_outcome, name="future outcome")
    if not len(current) == len(latent) == len(future):
        raise ValueError("future-validation vectors must have equal lengths")
    columns = [torch.ones_like(current), current]
    if covariates is not None:
        nuisance = torch.as_tensor(covariates, dtype=torch.float64)
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        if nuisance.ndim != 2 or len(nuisance) != len(current):
            raise ValueError("future covariates must be observations x features")
        columns.extend(nuisance[:, index] for index in range(nuisance.shape[1]))
    baseline = torch.stack(columns, dim=1)
    full = torch.cat((baseline, latent[:, None]), dim=1)
    _, baseline_prediction, baseline_r_squared = _ols_metrics(baseline, future)
    coefficient, full_prediction, full_r_squared = _ols_metrics(full, future)
    incremental = max(0.0, full_r_squared - baseline_r_squared)
    latent_residual = latent - baseline @ (torch.linalg.pinv(baseline) @ latent)
    outcome_residual = future - baseline_prediction
    denominator = torch.linalg.vector_norm(latent_residual) * torch.linalg.vector_norm(
        outcome_residual
    )
    partial_correlation = (
        float(torch.dot(latent_residual, outcome_residual) / denominator)
        if float(denominator) > 1e-12
        else 0.0
    )
    return {
        "passed": incremental >= float(minimum_incremental_r_squared)
        and abs(partial_correlation) > 0,
        "baseline_r_squared": baseline_r_squared,
        "full_r_squared": full_r_squared,
        "incremental_r_squared": incremental,
        "latent_coefficient": float(coefficient[-1]),
        "partial_correlation": partial_correlation,
        "minimum_incremental_r_squared": float(minimum_incremental_r_squared),
        "observations": len(current),
        "target": "future_persistence_beyond_immediate_choice",
    }


def add_future_behavior_targets(records: list[dict], *, k_values=(2, 5)) -> list[dict]:
    """Annotate sequential records without using future information as model input."""

    by_episode: dict[str, list[tuple[int, dict]]] = {}
    for index, record in enumerate(records):
        by_episode.setdefault(str(record["episode_id"]), []).append((index, record))
    output = [dict(record) for record in records]
    for episode_rows in by_episode.values():
        ordered = sorted(episode_rows, key=lambda item: int(item[1]["round"]))
        total = len(ordered)
        for position, (original_index, _record) in enumerate(ordered):
            remaining = total - position - 1
            output[original_index]["remaining_episode_length"] = remaining
            output[original_index]["future_run_length"] = remaining
            for k in k_values:
                output[original_index][f"persists_at_least_{int(k)}"] = int(
                    remaining >= int(k)
                )
            output[original_index]["later_disengagement"] = int(remaining > 0)
    return output


def heldout_future_behavior_validation(
    *,
    records: list[dict],
    current_choice,
    latent_state,
    future_outcome,
    train_episode_ids: set[str],
    test_episode_ids: set[str],
    minimum_incremental_r_squared: float = 0.01,
) -> dict:
    """Fit future models on training episodes and score untouched episodes."""

    import torch

    current = _vector(current_choice, name="current choice")
    latent = _vector(latent_state, name="latent state")
    future = _vector(future_outcome, name="future outcome")
    if len(records) != len(current) or not len(current) == len(latent) == len(future):
        raise ValueError("held-out future vectors do not align with records")
    train = torch.tensor(
        [
            index
            for index, row in enumerate(records)
            if str(row["episode_id"]) in train_episode_ids
        ],
        dtype=torch.long,
    )
    test = torch.tensor(
        [
            index
            for index, row in enumerate(records)
            if str(row["episode_id"]) in test_episode_ids
        ],
        dtype=torch.long,
    )
    if len(train) < 5 or len(test) < 5:
        raise ValueError("held-out future validation has too few train/test states")
    baseline_train = torch.stack((torch.ones_like(current[train]), current[train]), dim=1)
    full_train = torch.cat((baseline_train, latent[train, None]), dim=1)
    baseline_coefficient = torch.linalg.pinv(baseline_train) @ future[train]
    full_coefficient = torch.linalg.pinv(full_train) @ future[train]
    baseline_test = torch.stack((torch.ones_like(current[test]), current[test]), dim=1)
    full_test = torch.cat((baseline_test, latent[test, None]), dim=1)
    centered = ((future[test] - future[test].mean()) ** 2).sum().clamp_min(1e-12)
    baseline_r2 = 1.0 - float(
        ((future[test] - baseline_test @ baseline_coefficient) ** 2).sum() / centered
    )
    full_r2 = 1.0 - float(
        ((future[test] - full_test @ full_coefficient) ** 2).sum() / centered
    )
    incremental = full_r2 - baseline_r2
    return {
        "passed": incremental >= float(minimum_incremental_r_squared),
        "baseline_test_r_squared": baseline_r2,
        "full_test_r_squared": full_r2,
        "incremental_test_r_squared": incremental,
        "latent_train_coefficient": float(full_coefficient[-1]),
        "train_states": len(train),
        "test_states": len(test),
        "parameters_fit_on_test": 0,
        "minimum_incremental_r_squared": float(minimum_incremental_r_squared),
    }
