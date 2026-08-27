"""Blinded synthetic architecture recovery for interpretable model families."""

from __future__ import annotations

import numpy as np
import pandas as pd

from computational_modeling.models.accumulator import accumulated_design, accumulated_state
from computational_modeling.models.base import (
    assert_selection_blind,
    balanced_weights,
    linear_predict,
    weighted_ridge_fit,
)
from computational_modeling.models.termination import choice_kernel


GENERATING_ARCHITECTURES = (
    "immediate",
    "choice_inertia",
    "finite_history",
    "rw_value",
    "generic_latent_value",
    "latent_commitment",
    "sticky_termination",
    "disengagement_accumulator",
)


def simulate_architecture(
    architecture: str, *, episodes: int = 48, decisions: int = 10, seed: int = 0
) -> pd.DataFrame:
    if architecture not in GENERATING_ARCHITECTURES:
        raise ValueError(f"unknown generating architecture: {architecture!r}")
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(episodes):
        previous = second = 0.0
        lagged = [0.0] * 5
        recurrent_value = commitment = accumulator = kernel = 0.0
        rw_value = 0.0
        train_end = max(1, int(episodes * 0.6))
        validation_end = max(train_end + 1, int(episodes * 0.8))
        split = "train" if episode < train_end else "validation" if episode < validation_end else "test"
        for decision in range(decisions):
            value = float(rng.normal())
            cost = float(rng.normal())
            progress = float(rng.normal())
            immediate = float(rng.normal())
            advantage = float(rng.normal())
            disengagement = float(rng.normal())
            recurrent_value = 0.7 * recurrent_value + value
            commitment = 0.7 * commitment + 1.2 * value - 0.9 * cost + 0.7 * progress
            accumulator = 0.7 * accumulator + disengagement
            if architecture == "immediate":
                target = 3.0 * immediate
            elif architecture == "choice_inertia":
                target = 3.5 * previous + 1.8 * second + 0.1 * immediate
            elif architecture == "finite_history":
                target = 3.5 * lagged[2] - 2.0 * lagged[0] + 0.1 * immediate
            elif architecture == "rw_value":
                target = 3.0 * rw_value
            elif architecture == "generic_latent_value":
                target = 2.5 * recurrent_value
            elif architecture == "latent_commitment":
                target = 2.0 * commitment
            elif architecture == "sticky_termination":
                target = 2.0 * advantage + 2.5 * kernel
            else:
                target = -2.5 * accumulator
            target += float(rng.normal(scale=0.03))
            probability = 1.0 / (1.0 + np.exp(-np.clip(target, -20, 20)))
            choice = float(rng.random() < probability)
            outcome = float(rng.choice((-1.0, 1.0)))
            rows.append(
                {
                    "task": "synthetic",
                    "episode_id": f"episode-{episode:04d}",
                    "pair_id": f"episode-{episode:04d}",
                    "round": decision,
                    "split": split,
                    "continue": choice,
                    "persistence_logit": target,
                    "immediate_evidence": immediate,
                    "previous_choice": previous,
                    "second_previous_choice": second,
                    "action_lag_1": lagged[0],
                    "action_lag_3": lagged[2],
                    "rw_value": rw_value,
                    "relative_value": value,
                    "cost_pressure": cost,
                    "progress_evidence": progress,
                    "termination_advantage": advantage,
                    "disengagement_evidence": disengagement,
                }
            )
            kernel = 0.8 * kernel + choice
            rw_value += 0.5 * (outcome - rw_value)
            second, previous = previous, choice
            lagged = [choice, *lagged[:4]]
    return pd.DataFrame(rows)


def _design(records, family: str, parameter: float | None = None):
    rows = records.to_dict(orient="records") if hasattr(records, "to_dict") else records
    if family == "immediate":
        return np.asarray([[row["immediate_evidence"]] for row in rows])
    if family == "choice_inertia":
        return np.asarray([[row["previous_choice"], row["second_previous_choice"]] for row in rows])
    if family == "finite_history":
        return np.asarray([[row["action_lag_1"], row["action_lag_3"]] for row in rows])
    if family == "rw_value":
        return np.asarray([[row["rw_value"]] for row in rows])
    if family == "generic_latent_value":
        return accumulated_design(rows, ["relative_value"], rho=float(parameter))
    if family == "latent_commitment":
        return accumulated_design(
            rows,
            ["relative_value", "cost_pressure", "progress_evidence"],
            rho=float(parameter),
        )
    if family == "sticky_termination":
        return np.column_stack(
            (
                [row["termination_advantage"] for row in rows],
                choice_kernel(rows, decay=float(parameter)),
            )
        )
    if family == "disengagement_accumulator":
        return accumulated_state(rows, "disengagement_evidence", rho=float(parameter))[:, None]
    raise ValueError(f"unknown recovery family: {family}")


def recover_architecture(frame: pd.DataFrame) -> dict:
    """Select architecture on validation only, then report its final test error."""

    train = frame[frame.split == "train"].to_dict(orient="records")
    validation = frame[frame.split == "validation"].to_dict(orient="records")
    test = frame[frame.split == "test"].to_dict(orient="records")
    assert_selection_blind(train, validation)
    grids = {
        family: ((None,) if family in {"immediate", "choice_inertia", "finite_history", "rw_value"} else (0.0, 0.5, 0.7, 0.8, 0.9))
        for family in GENERATING_ARCHITECTURES
    }
    candidates = []
    for family in GENERATING_ARCHITECTURES:
        for parameter in grids[family]:
            x_train = _design(train, family, parameter)
            coefficient = weighted_ridge_fit(
                x_train,
                [row["persistence_logit"] for row in train],
                balanced_weights(train),
                penalty=1e-3,
            )
            prediction = linear_predict(_design(validation, family, parameter), coefficient)
            mse = float(np.mean((prediction - [row["persistence_logit"] for row in validation]) ** 2))
            candidates.append((mse, family, parameter))
    best_mse = min(row[0] for row in candidates)
    tolerance = max(1e-6, 0.01 * best_mse)
    complexity = {
        "immediate": 1,
        "rw_value": 1,
        "generic_latent_value": 2,
        "disengagement_accumulator": 2,
        "choice_inertia": 2,
        "finite_history": 2,
        "sticky_termination": 3,
        "latent_commitment": 4,
    }
    selected = min(
        (row for row in candidates if row[0] <= best_mse + tolerance),
        key=lambda row: (complexity[row[1]], row[1], -1 if row[2] is None else row[2]),
    )
    _validation_mse, family, parameter = selected
    fit_records = train + validation
    coefficient = weighted_ridge_fit(
        _design(fit_records, family, parameter),
        [row["persistence_logit"] for row in fit_records],
        balanced_weights(fit_records),
    )
    test_prediction = linear_predict(_design(test, family, parameter), coefficient)
    test_mse = float(np.mean((test_prediction - [row["persistence_logit"] for row in test]) ** 2))
    equivalence = {
        "generic_latent_value": ["generic_latent_value", "latent_commitment"],
        "latent_commitment": ["latent_commitment", "generic_latent_value"],
        "sticky_termination": ["sticky_termination", "latent_commitment"],
        "disengagement_accumulator": ["disengagement_accumulator", "generic_latent_value"],
    }
    acceptable = equivalence.get(family, [family])
    # Include all validation-near-ties so observational equivalence is explicit.
    acceptable = sorted(
        set(acceptable)
        | {row[1] for row in candidates if row[0] <= best_mse + tolerance}
    )
    return {
        "selected_family": family,
        "selected_hyperparameter": parameter,
        "validation_mse": _validation_mse,
        "test_mse": test_mse,
        "acceptable_families": acceptable,
        "selection_split": "validation",
    }
