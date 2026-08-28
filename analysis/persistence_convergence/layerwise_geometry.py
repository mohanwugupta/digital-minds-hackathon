"""Literal and low-rank geometry of independently trained task readouts."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _unit(values):
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)
    if norm <= 0:
        raise ValueError("readout direction has zero norm")
    return values / norm


def direction_similarity(directions_by_layer):
    rows = []
    for layer in sorted(directions_by_layer):
        directions = {
            str(task): _unit(values)
            for task, values in directions_by_layer[layer].items()
        }
        pair_values = []
        for left, right in combinations(sorted(directions), 2):
            cosine = float(directions[left] @ directions[right])
            pair_values.append(cosine)
            rows.append(
                {
                    "layer": int(layer),
                    "metric": "pairwise_cosine",
                    "task_a": left,
                    "task_b": right,
                    "value": cosine,
                }
            )
            rows.append(
                {
                    "layer": int(layer),
                    "metric": "principal_angle_degrees",
                    "task_a": left,
                    "task_b": right,
                    "value": float(np.degrees(np.arccos(np.clip(abs(cosine), 0, 1)))),
                }
            )
        matrix = np.column_stack([directions[task] for task in sorted(directions)])
        singular = np.linalg.svd(matrix, compute_uv=False)
        probability = singular**2 / np.sum(singular**2)
        effective_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))
        rows.extend(
            [
                {
                    "layer": int(layer),
                    "metric": "mean_pairwise_cosine",
                    "task_a": "all",
                    "task_b": "all",
                    "value": float(np.mean(pair_values)),
                },
                {
                    "layer": int(layer),
                    "metric": "task_span_effective_rank",
                    "task_a": "all",
                    "task_b": "all",
                    "value": effective_rank,
                },
            ]
        )
    return pd.DataFrame(rows)

