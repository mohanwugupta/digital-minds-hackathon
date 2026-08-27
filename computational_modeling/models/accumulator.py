"""Leaky evidence and commitment-state recursions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def accumulated_state(
    records: Sequence[Mapping], feature: str, *, rho: float
) -> np.ndarray:
    return accumulated_design(records, [feature], rho=rho)[:, 0]


def accumulated_design(
    records: Sequence[Mapping], features: Sequence[str], *, rho: float
) -> np.ndarray:
    if not features or not 0 <= float(rho) < 1:
        raise ValueError("accumulation requires features and rho in [0, 1)")
    output = np.zeros((len(records), len(features)), dtype=float)
    episodes: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        episodes.setdefault(str(row["episode_id"]), []).append(index)
    for indices in episodes.values():
        state = np.zeros(len(features), dtype=float)
        for index in sorted(indices, key=lambda item: int(records[item]["round"])):
            current = np.asarray([float(records[index][name]) for name in features])
            state = float(rho) * state + current
            output[index] = state
    return output
