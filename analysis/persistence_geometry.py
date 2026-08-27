"""Frozen-subspace geometry and leakage-safe decoding primitives.

The module deliberately separates representation loading/projection from every
computational decoder.  A :class:`FrozenPersistenceSubspace` can project hidden
states but has no fitting path, which prevents computational labels from
silently redefining the previously discovered persistence basis.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from computational_modeling.analysis.evaluate_models import persistence_metrics
from computational_modeling.models.base import balanced_weights


STAGES = ("l21", "displacement", "l22")


@dataclass(frozen=True)
class FrozenPersistenceSubspace:
    """Read-only rank-k basis loaded from the prior contrast search."""

    basis: np.ndarray
    source: str
    key: str = "displacement-L21-k4"
    sha256: str = ""

    @classmethod
    def from_array(cls, basis, *, source: str, key: str = "synthetic"):
        array = np.asarray(basis, dtype=np.float32).copy()
        if array.ndim != 2 or not len(array) or array.shape[1] > array.shape[0]:
            raise ValueError("persistence basis must be hidden_width x rank")
        if not np.isfinite(array).all():
            raise ValueError("persistence basis must be finite")
        gram = array.T @ array
        if not np.allclose(gram, np.eye(array.shape[1]), atol=2e-5):
            raise ValueError("persistence basis must have orthonormal columns")
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        array.setflags(write=False)
        return cls(array, str(source), str(key), digest)

    @classmethod
    def load(cls, path: str | Path, key: str = "displacement-L21-k4"):
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        try:
            candidate = payload["wide_candidates"][key]
        except KeyError as error:
            raise KeyError(f"frozen persistence candidate is absent: {key}") from error
        if int(candidate.get("rank", -1)) != 4 or key != "displacement-L21-k4":
            raise ValueError("geometry follow-up requires frozen displacement-L21-k4")
        return cls.from_array(candidate["basis"].numpy(), source=str(path), key=key)

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def width(self) -> int:
        return int(self.basis.shape[0])

    def project(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.width:
            raise ValueError("hidden states do not match frozen persistence width")
        return array @ self.basis

    def fit(self, *_args, **_kwargs):
        raise RuntimeError("persistence subspace is frozen")

    def refit(self, *_args, **_kwargs):
        raise RuntimeError("persistence subspace is frozen")


def stage_representation(h21, h22, stage: str) -> np.ndarray:
    """Return exact static/transition representations with fixed indexing."""

    h21 = np.asarray(h21)
    h22 = np.asarray(h22)
    if h21.shape != h22.shape or h21.ndim != 2:
        raise ValueError("L21 and L22 representations must share a rows-by-width shape")
    if stage == "l21":
        return h21
    if stage == "displacement":
        return h22 - h21
    if stage == "l22":
        return h22
    raise ValueError(f"unknown layer stage: {stage!r}")


def validate_episode_splits(frame: pd.DataFrame) -> None:
    required = {"episode_id", "pair_id", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"split metadata is missing columns: {missing}")
    valid = {"train", "validation", "test"}
    observed = set(frame.split.astype(str))
    if not observed <= valid:
        raise ValueError(f"invalid split labels: {sorted(observed - valid)}")
    episode_counts = frame.groupby("episode_id").split.nunique()
    if (episode_counts > 1).any():
        raise ValueError("episode crosses split boundaries")
    pair_counts = frame.groupby("pair_id").split.nunique()
    if (pair_counts > 1).any():
        raise ValueError("counterbalanced pair crosses split boundaries")


def _metadata_records(frame: pd.DataFrame, indices: np.ndarray) -> list[dict]:
    return frame.iloc[np.asarray(indices)].to_dict(orient="records")


def _macro_metrics(observed, predicted, frame: pd.DataFrame, indices) -> dict:
    indices = np.asarray(indices, dtype=int)
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    local = frame.iloc[indices].reset_index(drop=True)
    task_rows = []
    for task in sorted(local.task.astype(str).unique()):
        selected = np.flatnonzero(local.task.astype(str).to_numpy() == task)
        records = _metadata_records(local, selected)
        task_rows.append(
            persistence_metrics(
                observed[selected],
                predicted[selected],
                balanced_weights(records, task_balanced=False),
            )
        )
    output = {
        metric: float(np.nanmean([row[metric] for row in task_rows]))
        for metric in ("r_squared", "mse", "pearson_r")
    }
    output["tasks"] = int(len(task_rows))
    output["states"] = int(len(indices))
    output["episodes"] = int(local.episode_id.nunique())
    return output


def _weighted_location_scale(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    variance = np.sum((values - mean) ** 2 * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _ridge_coefficients(x, y, weights, alphas):
    """Weighted standardized ridge path using one eigendecomposition."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(len(x), -1)
    weights = np.asarray(weights, dtype=np.float64)
    x_mean, x_scale = _weighted_location_scale(x, weights)
    y_mean, y_scale = _weighted_location_scale(y, weights)
    xn = (x - x_mean) / x_scale
    yn = (y - y_mean) / y_scale
    normalized_weights = weights / weights.mean()
    root = np.sqrt(normalized_weights)
    xw = xn * root[:, None]
    yw = yn * root[:, None]
    gram = xw.T @ xw
    cross = xw.T @ yw
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected = eigenvectors.T @ cross
    coefficients = {}
    for alpha in map(float, alphas):
        if alpha < 0:
            raise ValueError("ridge alpha must be nonnegative")
        normalized = eigenvectors @ (
            projected / (eigenvalues[:, None] + alpha)
        )
        coefficients[alpha] = normalized
    return coefficients, {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def _predict_ridge(x, normalized_coefficient, normalizer):
    x = np.asarray(x, dtype=np.float64)
    prediction = (
        (x - normalizer["x_mean"]) / normalizer["x_scale"]
    ) @ normalized_coefficient
    return prediction * normalizer["y_scale"] + normalizer["y_mean"]


def _decode_masks(
    values,
    target,
    metadata: pd.DataFrame,
    *,
    train_indices,
    validation_indices,
    test_indices,
    alphas,
    forbidden_fit_tasks: Sequence[str] = (),
):
    values = np.asarray(values)
    target = np.asarray(target, dtype=float)
    if values.ndim != 2 or len(values) != len(metadata) or len(target) != len(metadata):
        raise ValueError("decoder arrays and metadata must have the same rows")
    if not np.isfinite(target).all() or not np.isfinite(values).all():
        raise ValueError("decoder inputs and targets must be finite")
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if set(train_indices) & set(validation_indices) or set(train_indices) & set(test_indices):
        raise ValueError("decoder row splits overlap")
    fit_tasks = set(metadata.iloc[train_indices].task.astype(str))
    forbidden = set(map(str, forbidden_fit_tasks))
    if fit_tasks & forbidden:
        raise ValueError("held-out task entered decoder fitting")
    weights = balanced_weights(
        _metadata_records(metadata, train_indices), task_balanced=True
    )
    coefficients, normalizer = _ridge_coefficients(
        values[train_indices], target[train_indices], weights, alphas
    )
    candidates = []
    for alpha, coefficient in coefficients.items():
        prediction = _predict_ridge(
            values[validation_indices], coefficient, normalizer
        )[:, 0]
        metrics = _macro_metrics(
            target[validation_indices], prediction, metadata, validation_indices
        )
        candidates.append((metrics["mse"], alpha, coefficient, metrics))
    _score, selected_alpha, coefficient, validation_metrics = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    test_prediction = _predict_ridge(
        values[test_indices], coefficient, normalizer
    )[:, 0]
    return {
        "selected_alpha": float(selected_alpha),
        "validation_metrics": validation_metrics,
        "test_metrics": _macro_metrics(
            target[test_indices], test_prediction, metadata, test_indices
        ),
        "test_prediction": test_prediction,
        "test_indices": test_indices,
        "coefficient": coefficient[:, 0].copy(),
        "normalizer": normalizer,
        "fit_tasks": sorted(fit_tasks),
        "fit_indices": train_indices,
    }


def decode_continuous_targets(
    values,
    targets: Mapping[str, Sequence[float]],
    metadata: pd.DataFrame,
    *,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    train_indices=None,
    validation_indices=None,
    test_indices=None,
    forbidden_fit_tasks: Sequence[str] = (),
) -> dict[str, dict]:
    """Decode many finite targets while sharing one ridge eigendecomposition."""

    validate_episode_splits(metadata)
    values = np.asarray(values)
    names = list(targets)
    if not names:
        raise ValueError("multi-target decoder requires at least one target")
    matrix = np.column_stack(
        [np.asarray(targets[name], dtype=float) for name in names]
    )
    if values.ndim != 2 or len(values) != len(metadata) or len(matrix) != len(metadata):
        raise ValueError("decoder arrays and metadata must have the same rows")
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise ValueError("decoder inputs and targets must be finite")
    split = metadata.split.astype(str).to_numpy()
    if train_indices is None:
        train_indices = np.flatnonzero(split == "train")
    if validation_indices is None:
        validation_indices = np.flatnonzero(split == "validation")
    if test_indices is None:
        test_indices = np.flatnonzero(split == "test")
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if set(train_indices) & set(validation_indices) or set(train_indices) & set(test_indices):
        raise ValueError("decoder row splits overlap")
    fit_tasks = set(metadata.iloc[train_indices].task.astype(str))
    if fit_tasks & set(map(str, forbidden_fit_tasks)):
        raise ValueError("held-out task entered decoder fitting")
    weights = balanced_weights(
        _metadata_records(metadata, train_indices), task_balanced=True
    )
    coefficients, normalizer = _ridge_coefficients(
        values[train_indices], matrix[train_indices], weights, alphas
    )
    validation_predictions = {
        alpha: _predict_ridge(values[validation_indices], coefficient, normalizer)
        for alpha, coefficient in coefficients.items()
    }
    output = {}
    for column, name in enumerate(names):
        candidates = []
        for alpha, prediction in validation_predictions.items():
            metrics = _macro_metrics(
                matrix[validation_indices, column],
                prediction[:, column],
                metadata,
                validation_indices,
            )
            candidates.append((metrics["mse"], alpha, metrics))
        _score, selected_alpha, validation_metrics = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        coefficient = coefficients[selected_alpha][:, column]
        test_prediction = _predict_ridge(
            values[test_indices],
            coefficients[selected_alpha],
            normalizer,
        )[:, column]
        output[name] = {
            "selected_alpha": float(selected_alpha),
            "validation_metrics": validation_metrics,
            "test_metrics": _macro_metrics(
                matrix[test_indices, column],
                test_prediction,
                metadata,
                test_indices,
            ),
            "test_prediction": test_prediction,
            "test_indices": test_indices,
            "coefficient": coefficient.copy(),
            "normalizer": {
                "x_mean": normalizer["x_mean"],
                "x_scale": normalizer["x_scale"],
                "y_mean": normalizer["y_mean"][column : column + 1],
                "y_scale": normalizer["y_scale"][column : column + 1],
            },
            "fit_tasks": sorted(fit_tasks),
            "fit_indices": train_indices,
        }
    return output


def decode_continuous_target(
    values,
    target,
    metadata: pd.DataFrame,
    *,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
):
    """Validation-select a train-only ridge decoder and evaluate test episodes."""

    validate_episode_splits(metadata)
    splits = metadata.split.astype(str).to_numpy()
    return _decode_masks(
        values,
        target,
        metadata,
        train_indices=np.flatnonzero(splits == "train"),
        validation_indices=np.flatnonzero(splits == "validation"),
        test_indices=np.flatnonzero(splits == "test"),
        alphas=alphas,
    )


def leave_one_task_out_decode(
    values,
    target,
    metadata: pd.DataFrame,
    *,
    heldout_task: str,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
):
    """Fit/normalize/select only on source tasks, then test one unseen task."""

    validate_episode_splits(metadata)
    task = metadata.task.astype(str).to_numpy()
    split = metadata.split.astype(str).to_numpy()
    heldout_task = str(heldout_task)
    source = task != heldout_task
    result = _decode_masks(
        values,
        target,
        metadata,
        train_indices=np.flatnonzero(source & (split == "train")),
        validation_indices=np.flatnonzero(source & (split == "validation")),
        test_indices=np.flatnonzero((task == heldout_task) & (split == "test")),
        alphas=alphas,
        forbidden_fit_tasks=(heldout_task,),
    )
    result["heldout_task"] = heldout_task
    return result


def matched_random_bases(width: int, rank: int, count: int, seed: int) -> np.ndarray:
    if width < rank or rank < 1 or count < 1:
        raise ValueError("invalid matched-random subspace dimensions")
    rng = np.random.default_rng(int(seed))
    bases = []
    for _ in range(int(count)):
        basis, _ = np.linalg.qr(rng.normal(size=(int(width), int(rank))))
        bases.append(basis[:, :rank].astype(np.float32))
    return np.stack(bases)


def compare_matched_random_subspaces(
    hidden,
    frozen: FrozenPersistenceSubspace,
    target,
    metadata: pd.DataFrame,
    *,
    count: int,
    seed: int,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
):
    hidden = np.asarray(hidden, dtype=np.float32)
    persistence = decode_continuous_target(
        frozen.project(hidden), target, metadata, alphas=alphas
    )
    random_scores = []
    for basis in matched_random_bases(
        frozen.width, frozen.rank, count=count, seed=seed
    ):
        decoded = decode_continuous_target(
            hidden @ basis, target, metadata, alphas=alphas
        )
        random_scores.append(float(decoded["test_metrics"]["r_squared"]))
    return {
        "persistence_r_squared": float(persistence["test_metrics"]["r_squared"]),
        "random_r_squared": random_scores,
        "random_r_squared_mean": float(np.mean(random_scores)),
        "random_r_squared_95th": float(np.quantile(random_scores, 0.95)),
        "empirical_p_value": float(
            (1 + sum(score >= persistence["test_metrics"]["r_squared"] for score in random_scores))
            / (len(random_scores) + 1)
        ),
    }


def task_identity_confound(target, metadata: pd.DataFrame, cross_task: pd.DataFrame) -> dict:
    target = np.asarray(target, dtype=float)
    if len(target) != len(metadata):
        raise ValueError("target and metadata rows differ")
    overall = float(np.var(target))
    means = metadata.assign(_target=target).groupby("task")._target.mean()
    counts = metadata.groupby("task").size().reindex(means.index).to_numpy()
    between = float(np.average((means.to_numpy() - np.mean(target)) ** 2, weights=counts))
    fraction = 0.0 if overall <= 1e-12 else between / overall
    scores = pd.to_numeric(cross_task["r_squared"], errors="coerce").to_numpy()
    finite = scores[np.isfinite(scores)]
    median_transfer = float(np.median(finite)) if len(finite) else float("nan")
    return {
        "between_task_variance_fraction": float(fraction),
        "median_leave_one_task_out_r_squared": median_transfer,
        "flagged": bool(fraction >= 0.90 and (not len(finite) or median_transfer <= 0.0)),
    }
