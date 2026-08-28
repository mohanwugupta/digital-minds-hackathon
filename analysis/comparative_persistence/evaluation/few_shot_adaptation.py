"""Pair-safe few-shot target-task adaptation partitions."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class FewShotPartition:
    adaptation: object
    evaluation: object
    requested_pairs: int
    selected_pairs: int


def few_shot_partition(records, task, *, pair_count, seed):
    task = str(task)
    candidates = sorted(
        records[(records.task == task) & (records.split == "train")].pair_id.unique()
    )
    random.Random(int(seed)).shuffle(candidates)
    selected = set(candidates[: min(int(pair_count), len(candidates))])
    adaptation = records[
        (records.task == task) & records.pair_id.isin(selected)
    ].copy()
    evaluation = records[(records.task == task) & (records.split == "test")].copy()
    if not set(adaptation.pair_id).isdisjoint(evaluation.pair_id):
        raise RuntimeError("few-shot adaptation/evaluation leakage")
    return FewShotPartition(
        adaptation, evaluation, int(pair_count), len(selected)
    )
