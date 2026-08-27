"""Termination-advantage, sticky-kernel, and meta-control utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def termination_advantage(record: Mapping, information_set: str) -> float:
    if information_set == "observable":
        return float(record["estimated_continue_value"]) - float(
            record["estimated_outside_value"]
        )
    if information_set == "oracle":
        return float(record["oracle_continue_value"]) - float(
            record["oracle_outside_value"]
        )
    raise ValueError("information_set must be observable or oracle")


def choice_kernel(records: Sequence[Mapping], *, decay: float) -> np.ndarray:
    """Return K_t using choices strictly before the current decision."""

    if not 0 <= float(decay) < 1:
        raise ValueError("choice-kernel decay must fall in [0, 1)")
    result = np.zeros(len(records), dtype=float)
    episodes: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        episodes.setdefault(str(row["episode_id"]), []).append(index)
    for indices in episodes.values():
        state = 0.0
        for index in sorted(indices, key=lambda item: int(records[item]["round"])):
            result[index] = state
            state = float(decay) * state + float(records[index]["continue"])
    return result
