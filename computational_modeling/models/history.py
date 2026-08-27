"""Finite-history feature construction."""

from __future__ import annotations


def finite_history_features(lag: int) -> tuple[str, ...]:
    if int(lag) not in {1, 2, 3, 5}:
        raise ValueError("finite history lag must be one of 1, 2, 3, or 5")
    features = ["log_round"]
    for index in range(1, int(lag) + 1):
        if index in {1, 2, 3, 5}:
            features.extend((f"action_lag_{index}", f"outcome_lag_{index}"))
    return tuple(features)
