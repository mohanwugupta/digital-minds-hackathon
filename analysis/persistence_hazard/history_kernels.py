"""Causal finite and exponential summaries of prior engagement history."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


VALID_LAGS = (1, 2, 3, 5)


def finite_history_features(lag: int) -> tuple[str, ...]:
    lag = int(lag)
    if lag not in VALID_LAGS:
        raise ValueError(f"history lag must be one of {VALID_LAGS}")
    features = ["previous_choice", "previous_outcome", "failure_streak", "success_streak"]
    for index in VALID_LAGS:
        if 1 < index <= lag:
            features.extend((f"action_lag_{index}", f"outcome_lag_{index}"))
    return tuple(features)


def add_exponential_history(
    records: Sequence[Mapping], *, decay: float
) -> list[dict]:
    """Add pre-decision kernels; the current choice/outcome updates only the future."""

    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError("history decay must lie in [0, 1)")
    output = [dict(row) for row in records]
    by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        by_episode[str(row["episode_id"])].append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda index: int(output[index]["round"]))
        choice_state = 0.0
        outcome_state = 0.0
        for index in indices:
            row = output[index]
            row["choice_kernel"] = choice_state
            row["outcome_kernel"] = outcome_state
            outcome = row.get("outcome_after_choice", 0.0)
            outcome = 0.0 if outcome is None else float(outcome)
            choice_state = decay * choice_state + float(row.get("continue", 0.0))
            outcome_state = decay * outcome_state + outcome
    return output

