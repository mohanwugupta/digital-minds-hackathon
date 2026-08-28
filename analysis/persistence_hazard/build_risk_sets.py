"""Construct absorbing discrete-time stopping-risk sets from existing records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def validate_episode_and_pair_splits(records: Sequence[Mapping]) -> None:
    """Reject episode or counterbalanced-pair leakage across persisted splits."""

    episode_splits: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split = str(row["split"])
        episode_splits[str(row["episode_id"])].add(split)
        pair_splits[str(row.get("pair_id", row["episode_id"]))].add(split)
    crossing_episodes = sorted(
        episode for episode, splits in episode_splits.items() if len(splits) != 1
    )
    if crossing_episodes:
        raise ValueError(f"episode crosses splits: {crossing_episodes[:5]}")
    crossing_pairs = sorted(
        pair for pair, splits in pair_splits.items() if len(splits) != 1
    )
    if crossing_pairs:
        raise ValueError(f"pair crosses splits: {crossing_pairs[:5]}")


def build_risk_set(records: Sequence[Mapping]) -> list[dict]:
    """Return one row per at-risk decision and reject post-event observations."""

    copied = [dict(row) for row in records]
    validate_episode_and_pair_splits(copied)
    by_episode: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for original_index, row in enumerate(copied):
        key = (str(row.get("task", "single_task")), str(row["episode_id"]))
        by_episode[key].append((original_index, row))

    output: list[tuple[int, dict]] = []
    for (_task, episode), indexed_rows in by_episode.items():
        indexed_rows.sort(key=lambda item: int(item[1]["round"]))
        rounds = [int(row["round"]) for _, row in indexed_rows]
        if rounds != list(range(len(rounds))):
            raise ValueError(f"non-contiguous risk set for episode {episode}: {rounds[:8]}")
        terminated = False
        for original_index, row in indexed_rows:
            if terminated:
                raise ValueError(f"post-termination state in episode {episode}")
            event = 1 - int(row["continue"])
            if event not in {0, 1}:
                raise ValueError("continue must be binary")
            output.append(
                (
                    original_index,
                    {
                        **row,
                        "hazard_event": event,
                        "at_risk": 1,
                    },
                )
            )
            terminated = bool(event)
    return [row for _, row in sorted(output, key=lambda item: item[0])]


def read_behavior_records(directory: str | Path, tasks) -> list[dict]:
    """Read the persisted records-only model-zoo dataset."""

    directory = Path(directory)
    records = []
    for task in tasks:
        path = directory / f"{task}_records.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        records.extend(pd.read_csv(path).to_dict(orient="records"))
    return build_risk_set(records)

