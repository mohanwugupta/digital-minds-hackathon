"""Shared leakage, normalization, weighting, and linear-fit primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrainStandardizer:
    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    fit_split: str = "train"

    @classmethod
    def fit(cls, values: Sequence[Sequence[float]], feature_names: Sequence[str]):
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(feature_names) or not len(array):
            raise ValueError("standardizer requires a nonempty rows-by-features matrix")
        mean = array.mean(axis=0)
        scale = array.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(list(feature_names), mean.tolist(), scale.tolist())

    def transform(self, values: Sequence[Sequence[float]]) -> list[list[float]]:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("standardizer input shape does not match fitted features")
        return ((array - np.asarray(self.mean)) / np.asarray(self.scale)).tolist()


def balanced_weights(
    records: Sequence[Mapping], *, task_balanced: bool = True
) -> np.ndarray:
    """Give episodes equal mass and, optionally, tasks equal aggregate mass."""

    if not records:
        raise ValueError("cannot weight empty records")
    episode_counts: dict[tuple[str, str], int] = {}
    task_episodes: dict[str, set[str]] = {}
    for row in records:
        task = str(row.get("task", "single_task"))
        episode = str(row["episode_id"])
        episode_counts[(task, episode)] = episode_counts.get((task, episode), 0) + 1
        task_episodes.setdefault(task, set()).add(episode)
    values = []
    total_episodes = sum(len(value) for value in task_episodes.values())
    for row in records:
        task = str(row.get("task", "single_task"))
        episode = str(row["episode_id"])
        episode_mass = (
            1.0 / len(task_episodes[task])
            if task_balanced
            else 1.0 / total_episodes
        )
        values.append(episode_mass / episode_counts[(task, episode)])
    weights = np.asarray(values, dtype=float)
    return weights / weights.mean()


def assert_selection_blind(*record_sets: Sequence[Mapping]) -> None:
    """Runtime guard: hyperparameter selection may never receive test rows."""

    for records in record_sets:
        if any(str(row.get("split", "")) == "test" for row in records):
            raise ValueError("hyperparameter selection received the final test split")


def weighted_ridge_fit(x, y, weights, *, penalty: float = 1e-3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    gram = design.T @ (weights[:, None] * design)
    regularizer = np.eye(design.shape[1]) * float(penalty)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(gram + regularizer, design.T @ (weights * y))


def linear_predict(x, coefficient):
    x = np.asarray(x, dtype=float)
    return np.column_stack((np.ones(len(x)), x)) @ np.asarray(coefficient)
