"""Validation-only recovery of sharing, finite history, and latent timescale."""

from __future__ import annotations

import numpy as np

from ..hazard_models.baselines import (
    binary_log_loss,
    fit_ridge_logistic,
    predict_ridge_logistic,
)


def recover_sharing(records):
    train = records[records.split == "train"]
    validation = records[records.split == "validation"]
    features = ("x1", "x2")
    shared = fit_ridge_logistic(train[list(features)], train.hazard_event, penalty=0.01)
    shared_loss = binary_log_loss(
        validation.hazard_event,
        predict_ridge_logistic(shared, validation[list(features)]),
    )
    task_losses, sizes = [], []
    for task in sorted(records.task.unique()):
        local_train = train[train.task == task]
        local_validation = validation[validation.task == task]
        coefficient = fit_ridge_logistic(
            local_train[list(features)], local_train.hazard_event, penalty=0.01
        )
        task_losses.append(
            binary_log_loss(
                local_validation.hazard_event,
                predict_ridge_logistic(coefficient, local_validation[list(features)]),
            )
        )
        sizes.append(len(local_validation))
    specific_loss = float(np.average(task_losses, weights=sizes))
    # Require a material advantage before paying for separate algorithms.
    return "task_specific" if specific_loss < shared_loss - 0.03 else "fully_shared"


def recover_history_lag(records, *, candidates=(1, 2, 3, 5, 8)):
    train = records[records.split == "train"]
    validation = records[records.split == "validation"]
    rows = []
    for maximum in candidates:
        features = [
            f"outcome_lag_{lag}" for lag in candidates if int(lag) <= int(maximum)
        ]
        coefficient = fit_ridge_logistic(
            train[features], train.hazard_event, penalty=0.03
        )
        loss = binary_log_loss(
            validation.hazard_event,
            predict_ridge_logistic(coefficient, validation[features]),
        )
        rows.append((loss, len(features), int(maximum)))
    best = min(row[0] for row in rows)
    tolerance = 0.005
    return min(
        (row for row in rows if row[0] <= best + tolerance),
        key=lambda row: (row[1], row[2]),
    )[2]


def _latent_feature(records, rho):
    values = np.zeros(len(records), dtype=float)
    for _episode, indices in records.groupby("episode_id", sort=False).groups.items():
        state = 0.0
        for index in sorted(indices, key=lambda item: int(records.loc[item, "round"])):
            state = float(rho) * state + float(records.loc[index, "drive"])
            values[records.index.get_loc(index)] = state
    return values


def recover_latent_rho(records, *, candidates=(0.0, 0.5, 0.9)):
    local = records.reset_index(drop=True).copy()
    train_mask = local.split == "train"
    validation_mask = local.split == "validation"
    rows = []
    for rho in candidates:
        feature = _latent_feature(local, rho).reshape(-1, 1)
        coefficient = fit_ridge_logistic(
            feature[train_mask], local.loc[train_mask, "hazard_event"], penalty=0.01
        )
        loss = binary_log_loss(
            local.loc[validation_mask, "hazard_event"],
            predict_ridge_logistic(coefficient, feature[validation_mask]),
        )
        rows.append((loss, float(rho)))
    best = min(value for value, _rho in rows)
    return min(
        (row for row in rows if row[0] <= best + 0.002),
        key=lambda row: row[1],
    )[1]


def run_recovery_experiment(config, *, smoke=False, logger=None):
    """Repeat mandatory H1-H4 recovery and return long/confusion tables."""

    from .generators import (
        generate_history_data,
        generate_latent_data,
        generate_sharing_data,
    )

    repetitions = int(
        config["smoke"]["synthetic_repetitions"]
        if smoke
        else config["synthetic_recovery"]["repetitions"]
    )
    episodes = int(config["synthetic_recovery"]["episodes_per_task"])
    rows = []
    for repetition in range(repetitions):
        seed = int(config["seed"]) + repetition * 17
        rho = recover_latent_rho(
            generate_latent_data(rho=0.9, episodes=episodes, seed=seed),
            candidates=(0.0, 0.5, 0.9),
        )
        rows.append(
            {
                "repetition": repetition,
                "generating_model": "H1_latent_commitment",
                "selected_model": "latent_commitment" if rho >= 0.5 else "immediate_state",
                "recovered_parameter": rho,
            }
        )
        sharing = recover_sharing(
            generate_sharing_data("shared", episodes_per_task=episodes, seed=seed + 1)
        )
        rows.append(
            {
                "repetition": repetition,
                "generating_model": "H2_shared_rule",
                "selected_model": "shared_rule" if sharing == "fully_shared" else "task_specific",
                "recovered_parameter": sharing,
            }
        )
        sharing = recover_sharing(
            generate_sharing_data(
                "task_specific", episodes_per_task=episodes, seed=seed + 2
            )
        )
        rows.append(
            {
                "repetition": repetition,
                "generating_model": "H3_task_specific_evaluation",
                "selected_model": "task_specific" if sharing == "task_specific" else "shared_rule",
                "recovered_parameter": sharing,
            }
        )
        lag = recover_history_lag(
            generate_history_data(lag=3, episodes=episodes, seed=seed + 3),
            candidates=(1, 2, 3, 5, 8),
        )
        rows.append(
            {
                "repetition": repetition,
                "generating_model": "H4_generic_sequential_choice",
                "selected_model": "finite_history",
                "recovered_parameter": lag,
            }
        )
        if logger is not None and (
            repetition + 1 == repetitions or (repetition + 1) % 10 == 0
        ):
            logger.note("synthetic_recovery", f"completed {repetition + 1}/{repetitions}")
    recovery = __import__("pandas").DataFrame(rows)
    confusion = (
        recovery.groupby(["generating_model", "selected_model"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return recovery, confusion
