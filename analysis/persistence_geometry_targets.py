"""Prespecified behavioral targets for persistence-subspace geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from computational_modeling.analysis.model_fitting import (
    _fit_apply,
    fit_interpretable_model,
)
from computational_modeling.models.baselines import MODEL_DEFINITIONS


@dataclass(frozen=True)
class GeometryTarget:
    name: str
    family: str
    priority: int
    role: str = "primary"
    geometry_target: bool = False
    description: str = ""


TARGET_SPECS = (
    GeometryTarget(
        "history_finite_prediction",
        "history",
        1,
        geometry_target=True,
        description="Train/validation-frozen task-specific finite-history model prediction.",
    ),
    GeometryTarget(
        "history_summary",
        "history",
        1,
        description="Train-standardized composite of recent outcomes, choices, and streak balance.",
    ),
    GeometryTarget(
        "history_recent_outcomes",
        "history",
        1,
        description="Exponentially weighted past-only outcome lags.",
    ),
    GeometryTarget(
        "history_recent_choices",
        "history",
        1,
        description="Exponentially weighted past-only continuation-choice lags.",
    ),
    GeometryTarget(
        "time_effort",
        "time_effort",
        2,
        geometry_target=True,
        description="Train-standardized mean of log round and normalized time.",
    ),
    GeometryTarget(
        "cost_pressure",
        "cost",
        3,
        geometry_target=True,
        description="Harmonized displayed task cost.",
    ),
    GeometryTarget(
        "progress_solvability",
        "progress_solvability",
        4,
        geometry_target=True,
        description="Train-standardized progress-evidence/cumulative-progress composite.",
    ),
    GeometryTarget(
        "estimated_continue_value",
        "continuation_value",
        5,
        role="control",
        description="Past-derived observable continuation value.",
    ),
    GeometryTarget(
        "estimated_outside_value",
        "outside_option",
        5,
        role="control",
        description="Observable outside-option value.",
    ),
    GeometryTarget(
        "termination_advantage",
        "derived_termination",
        5,
        role="control",
        description="Observable continuation minus outside-option value.",
    ),
)


def load_behavior_records(
    records_dir: str | Path,
    tasks: Sequence[str],
    *,
    max_pairs_per_task_split: int | None = None,
) -> pd.DataFrame:
    frames = []
    for task in tasks:
        frame = pd.read_csv(Path(records_dir) / f"{task}_records.csv")
        frame["task"] = str(task)
        if max_pairs_per_task_split is not None:
            retained = []
            for _split, part in frame.groupby("split", sort=False):
                pairs = sorted(part.pair_id.astype(str).unique())[
                    : int(max_pairs_per_task_split)
                ]
                retained.append(part[part.pair_id.astype(str).isin(pairs)])
            frame = pd.concat(retained, ignore_index=True)
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.state_id.astype(str).duplicated().any():
        raise ValueError("behavioral geometry records contain duplicate state IDs")
    return output


def _zscore_from_train(frame: pd.DataFrame, values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    train = frame.split.astype(str).to_numpy() == "train"
    mean = float(np.mean(values[train]))
    scale = float(np.std(values[train]))
    if scale < 1e-8:
        scale = 1.0
    return (values - mean) / scale


def _finite_history_target(frame: pd.DataFrame, zoo_config: Mapping) -> np.ndarray:
    definition = next(
        item for item in MODEL_DEFINITIONS if item.name == "finite_history"
    )
    predictions = np.empty(len(frame), dtype=float)
    for task in sorted(frame.task.astype(str).unique()):
        indices = np.flatnonzero(frame.task.astype(str).to_numpy() == task)
        records = frame.iloc[indices].to_dict(orient="records")
        train = [row for row in records if row["split"] == "train"]
        validation = [row for row in records if row["split"] == "validation"]
        test = [row for row in records if row["split"] == "test"]
        fit = fit_interpretable_model(
            train,
            validation,
            test,
            definition,
            information_set="observable",
            sharing="shared_architecture_task_observation",
            config=zoo_config,
        )
        selected = fit["selected_hyperparameters"]
        application_prediction, _parameters, _states, _features = _fit_apply(
            train + validation,
            records,
            definition,
            "observable",
            selected,
            "shared_architecture_task_observation",
        )
        predictions[indices] = application_prediction
    return predictions


def build_geometry_targets(
    frame: pd.DataFrame, zoo_config: Mapping
) -> tuple[dict[str, np.ndarray], tuple[GeometryTarget, ...]]:
    """Build only preregistered, past-only observable targets."""

    outcome = sum(
        weight * frame[column].to_numpy(dtype=float)
        for weight, column in zip(
            (1.0, 0.5, 0.25, 0.125),
            ("outcome_lag_1", "outcome_lag_2", "outcome_lag_3", "outcome_lag_5"),
        )
    )
    choice = sum(
        weight * frame[column].to_numpy(dtype=float)
        for weight, column in zip(
            (1.0, 0.5, 0.25, 0.125),
            ("action_lag_1", "action_lag_2", "action_lag_3", "action_lag_5"),
        )
    )
    streak = (
        frame.success_streak.to_numpy(dtype=float)
        - frame.failure_streak.to_numpy(dtype=float)
    )
    outcome_z = _zscore_from_train(frame, outcome)
    choice_z = _zscore_from_train(frame, choice)
    streak_z = _zscore_from_train(frame, streak)
    time = np.mean(
        np.column_stack(
            [
                _zscore_from_train(frame, frame.log_round),
                _zscore_from_train(frame, frame.normalized_time),
            ]
        ),
        axis=1,
    )
    progress = np.mean(
        np.column_stack(
            [
                _zscore_from_train(frame, frame.progress_evidence),
                _zscore_from_train(frame, frame.cumulative_progress),
            ]
        ),
        axis=1,
    )
    targets = {
        "history_finite_prediction": _finite_history_target(frame, zoo_config),
        "history_summary": np.mean(
            np.column_stack((outcome_z, choice_z, streak_z)), axis=1
        ),
        "history_recent_outcomes": outcome,
        "history_recent_choices": choice,
        "time_effort": time,
        "cost_pressure": frame.cost_pressure.to_numpy(dtype=float),
        "progress_solvability": progress,
        "estimated_continue_value": frame.estimated_continue_value.to_numpy(dtype=float),
        "estimated_outside_value": frame.estimated_outside_value.to_numpy(dtype=float),
        "termination_advantage": frame.termination_advantage.to_numpy(dtype=float),
    }
    if set(targets) != {spec.name for spec in TARGET_SPECS}:
        raise AssertionError("geometry target registry and construction differ")
    if any(not np.isfinite(values).all() for values in targets.values()):
        raise ValueError("geometry targets must be finite")
    return targets, TARGET_SPECS
