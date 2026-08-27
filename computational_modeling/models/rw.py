"""Rescorla--Wagner and Beta--Bernoulli state construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def rw_states(
    records: Sequence[Mapping], *, alpha: float, initial_value: float = 0.5
) -> list[dict[str, float]]:
    if not 0 <= float(alpha) <= 1:
        raise ValueError("RW alpha must fall in [0, 1]")
    output: list[dict[str, float] | None] = [None] * len(records)
    episodes: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        episodes.setdefault(str(row["episode_id"]), []).append(index)
    for indices in episodes.values():
        q = {"A": float(initial_value), "B": float(initial_value)}
        for index in sorted(indices, key=lambda item: int(records[item]["round"])):
            output[index] = {
                "rw_a": q["A"],
                "rw_b": q["B"],
                "rw_best": max(q.values()),
                "rw_gap": abs(q["A"] - q["B"]),
            }
            action = str(records[index].get("semantic_choice", "")).upper()
            if action in q:
                reward = float(records[index]["outcome_after_choice"])
                q[action] += float(alpha) * (reward - q[action])
    return [dict(row) for row in output]


def bayesian_bandit_states(records: Sequence[Mapping]) -> list[dict[str, float]]:
    output: list[dict[str, float] | None] = [None] * len(records)
    episodes: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        episodes.setdefault(str(row["episode_id"]), []).append(index)
    for indices in episodes.values():
        successes = {"A": 1.0, "B": 1.0}
        failures = {"A": 1.0, "B": 1.0}
        for index in sorted(indices, key=lambda item: int(records[item]["round"])):
            value = {
                arm: 5.0 * successes[arm] / (successes[arm] + failures[arm]) - 2.0
                for arm in ("A", "B")
            }
            output[index] = {
                "bayes_a": value["A"],
                "bayes_b": value["B"],
                "bayes_best": max(value.values()),
                "bayes_gap": abs(value["A"] - value["B"]),
            }
            action = str(records[index].get("semantic_choice", "")).upper()
            if action in value:
                if float(records[index]["outcome_after_choice"]) > 0:
                    successes[action] += 1
                else:
                    failures[action] += 1
    return [dict(row) for row in output]
