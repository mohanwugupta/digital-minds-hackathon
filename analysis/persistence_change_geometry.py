"""Leakage-safe primitives for matched persistence-change geometry.

The functions in this module never fit or rotate the persistence basis.  They
operate on semantically oriented matched-pair differences and keep all feature
and target normalization inside the source training fold.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from analysis.persistence_contrasts import ContrastDefinition, validate_contrast_pair
from analysis.persistence_geometry import (
    STAGES,
    _decode_masks,
    decode_continuous_target,
    validate_episode_splits,
)
from computational_modeling.analysis.evaluate_models import persistence_metrics
from computational_modeling.models.base import balanced_weights


def oriented_difference(promoting, discouraging):
    """Return ``P+ - P-`` without changing the scientific orientation."""

    return np.asarray(promoting) - np.asarray(discouraging)


def audit_exact_pair(
    promoting: Mapping,
    discouraging: Mapping,
    definition: ContrastDefinition,
) -> dict:
    """Validate an exact pair and return the required audit fields.

    ``validate_contrast_pair`` deliberately raises on any mismatch.  Keeping
    that behavior here prevents invalid pairs from being silently converted to
    analysis rows.
    """

    validation = validate_contrast_pair(promoting, discouraging, definition)
    return {
        "task": definition.task,
        "contrast_family": definition.manipulation,
        "matched": True,
        "unmatched_fields": "",
        "orientation_valid": validation["orientation"]
        == "positive_is_more_persistence_promoting",
    }


def construct_pair_change(
    positive_l21,
    positive_l22,
    negative_l21,
    negative_l22,
    basis,
) -> dict[str, np.ndarray]:
    """Construct and project exact static and L21→L22 pair differences."""

    p21 = np.asarray(positive_l21, dtype=np.float32)
    p22 = np.asarray(positive_l22, dtype=np.float32)
    n21 = np.asarray(negative_l21, dtype=np.float32)
    n22 = np.asarray(negative_l22, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if p21.ndim != 1 or not (p21.shape == p22.shape == n21.shape == n22.shape):
        raise ValueError("pair endpoints must be equal-width hidden-state vectors")
    if basis.ndim != 2 or basis.shape[0] != p21.shape[0]:
        raise ValueError("frozen basis does not match endpoint hidden width")
    changes = {
        "l21": p21 - n21,
        "displacement": (p22 - p21) - (n22 - n21),
        "l22": p22 - n22,
    }
    return {stage: values @ basis for stage, values in changes.items()}


def _weighted_sign_accuracy(observed, predicted, records) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    informative = np.flatnonzero(np.abs(observed) > 1e-12)
    if not len(informative):
        return float("nan")
    local_records = [records[index] for index in informative]
    weights = balanced_weights(local_records, task_balanced=True)
    correct = np.sign(observed[informative]) == np.sign(predicted[informative])
    return float(np.average(correct, weights=weights))


def _attach_change_metrics(result: dict, target, metadata: pd.DataFrame) -> dict:
    output = dict(result)
    metrics = dict(result["test_metrics"])
    indices = np.asarray(result["test_indices"], dtype=int)
    records = metadata.iloc[indices].to_dict(orient="records")
    metrics["sign_accuracy"] = _weighted_sign_accuracy(
        np.asarray(target, dtype=float)[indices], result["test_prediction"], records
    )
    output["test_metrics"] = metrics
    return output


def fit_change_decoder(
    values,
    target,
    metadata: pd.DataFrame,
    *,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """Validation-select a train-only linear change decoder."""

    result = decode_continuous_target(values, target, metadata, alphas=alphas)
    return _attach_change_metrics(result, target, metadata)


def strict_group_transfer(
    values,
    target,
    metadata: pd.DataFrame,
    *,
    group_column: str,
    heldout: str,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """Fit on source groups and evaluate one unseen task or manipulation."""

    validate_episode_splits(metadata)
    if group_column not in metadata:
        raise ValueError(f"transfer metadata lacks group column {group_column!r}")
    groups = metadata[group_column].astype(str).to_numpy()
    split = metadata.split.astype(str).to_numpy()
    heldout = str(heldout)
    source = groups != heldout
    train = np.flatnonzero(source & (split == "train"))
    validation = np.flatnonzero(source & (split == "validation"))
    test = np.flatnonzero((groups == heldout) & (split == "test"))
    if not len(train) or not len(validation) or not len(test):
        raise ValueError(
            f"strict {group_column} transfer has an empty source/test split for {heldout}"
        )
    result = _decode_masks(
        values,
        target,
        metadata,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        alphas=alphas,
        forbidden_fit_tasks=(heldout,) if group_column == "task" else (),
    )
    fit_groups = sorted(set(groups[result["fit_indices"]]))
    if heldout in fit_groups:
        raise RuntimeError(f"held-out {group_column} entered decoder fitting")
    output = _attach_change_metrics(result, target, metadata)
    output.update(
        {
            "group_column": group_column,
            "heldout_group": heldout,
            "fit_groups": fit_groups,
        }
    )
    return output


def clustered_bootstrap_predictions(
    observed,
    predicted,
    metadata: pd.DataFrame,
    indices,
    *,
    samples: int,
    seed: int,
    cluster_column: str = "pair_id",
) -> dict[str, dict[str, float]]:
    """Pair-clustered bootstrap CIs for held-out prediction metrics."""

    if int(samples) < 20:
        raise ValueError("pair-clustered bootstrap requires at least 20 samples")
    indices = np.asarray(indices, dtype=int)
    observed = np.asarray(observed, dtype=float)[indices]
    predicted = np.asarray(predicted, dtype=float)
    local = metadata.iloc[indices].reset_index(drop=True)
    if len(predicted) != len(local) or cluster_column not in local:
        raise ValueError("bootstrap predictions or cluster metadata are malformed")
    groups = {
        str(cluster): np.asarray(rows, dtype=int)
        for cluster, rows in local.groupby(cluster_column).groups.items()
    }
    cluster_ids = sorted(groups)
    rng = np.random.default_rng(int(seed))

    def metrics(local_observed, local_predicted, frame):
        task_metrics = []
        for task in sorted(frame.task.astype(str).unique()):
            selected = np.flatnonzero(frame.task.astype(str).to_numpy() == task)
            records = frame.iloc[selected].to_dict(orient="records")
            weights = balanced_weights(records, task_balanced=False)
            result = persistence_metrics(
                local_observed[selected], local_predicted[selected], weights
            )
            result["sign_accuracy"] = _weighted_sign_accuracy(
                local_observed[selected], local_predicted[selected], records
            )
            task_metrics.append(result)
        return {
            name: float(np.nanmean([row[name] for row in task_metrics]))
            for name in task_metrics[0]
        }

    point = metrics(observed, predicted, local)
    draws = {name: [] for name in point}
    for _ in range(int(samples)):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        observed_parts, predicted_parts, frame_parts = [], [], []
        for draw_index, cluster in enumerate(sampled):
            selected = groups[cluster]
            part = local.iloc[selected].copy()
            # A cluster drawn twice is two bootstrap observations.  Give each
            # draw a unique episode/pair suffix so episode balancing preserves
            # that multiplicity instead of collapsing duplicate draws.
            part["episode_id"] = part.episode_id.astype(str) + f"#boot-{draw_index}"
            part["pair_id"] = part.pair_id.astype(str) + f"#boot-{draw_index}"
            frame_parts.append(part)
            observed_parts.append(observed[selected])
            predicted_parts.append(predicted[selected])
        row = metrics(
            np.concatenate(observed_parts),
            np.concatenate(predicted_parts),
            pd.concat(frame_parts, ignore_index=True),
        )
        for name, value in row.items():
            draws[name].append(float(value))
    output = {}
    for name, values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        output[name] = {
            "estimate": float(point[name]),
            "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
            "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
            "samples": int(samples),
            "clusters": len(cluster_ids),
        }
    return output


def direction_alignment_rows(
    projected: Mapping[str, np.ndarray], metadata: pd.DataFrame
) -> list[dict]:
    """Summarize family directions and shared task subspaces at every stage."""

    rows: list[dict] = []
    for stage in STAGES:
        values = np.asarray(projected[stage], dtype=float)
        directions = {}
        for (task, manipulation), indices in metadata.groupby(
            ["task", "manipulation"]
        ).groups.items():
            direction = values[np.asarray(list(indices), dtype=int)].mean(axis=0)
            directions[(str(task), str(manipulation))] = direction
            norm = float(np.linalg.norm(direction))
            rows.append(
                {
                    "stage": stage,
                    "kind": "mean_direction",
                    "task_a": str(task),
                    "manipulation_a": str(manipulation),
                    "task_b": "",
                    "manipulation_b": "",
                    "component": 0,
                    "value": norm,
                    **{f"direction_{index + 1}": float(value) for index, value in enumerate(direction)},
                }
            )
        for (left_key, left), (right_key, right) in combinations(directions.items(), 2):
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            cosine = 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)
            rows.append(
                {
                    "stage": stage,
                    "kind": "direction_cosine",
                    "task_a": left_key[0],
                    "manipulation_a": left_key[1],
                    "task_b": right_key[0],
                    "manipulation_b": right_key[1],
                    "component": 0,
                    "value": cosine,
                }
            )
            rows.append(
                {
                    "stage": stage,
                    "kind": "sign_agreement",
                    "task_a": left_key[0],
                    "manipulation_a": left_key[1],
                    "task_b": right_key[0],
                    "manipulation_b": right_key[1],
                    "component": 0,
                    "value": float(np.mean(np.sign(left) == np.sign(right))),
                }
            )
        task_bases = {}
        for task in sorted(metadata.task.astype(str).unique()):
            task_directions = np.stack(
                [value for (local_task, _), value in directions.items() if local_task == task]
            )
            _u, singular, vt = np.linalg.svd(task_directions, full_matrices=False)
            rank = max(1, int(np.sum(singular > max(singular[0], 1e-12) * 1e-6)))
            task_bases[task] = vt[:rank].T
        for left_task, right_task in combinations(sorted(task_bases), 2):
            singular = np.linalg.svd(
                task_bases[left_task].T @ task_bases[right_task],
                compute_uv=False,
            )
            for component, cosine in enumerate(np.clip(singular, 0, 1), start=1):
                rows.append(
                    {
                        "stage": stage,
                        "kind": "task_subspace_principal_angle_degrees",
                        "task_a": left_task,
                        "manipulation_a": "all",
                        "task_b": right_task,
                        "manipulation_b": "all",
                        "component": component,
                        "value": float(np.degrees(np.arccos(cosine))),
                    }
                )
            rows.append(
                {
                    "stage": stage,
                    "kind": "task_subspace_overlap",
                    "task_a": left_task,
                    "manipulation_a": "all",
                    "task_b": right_task,
                    "manipulation_b": "all",
                    "component": 0,
                    "value": float(np.mean(singular**2)),
                }
            )
    return rows
