"""Strict entire-task holdout partitions with no target calibration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LOTOPartition:
    fit: object
    selection: object
    evaluation: object
    heldout_task: str


def loto_partition(records, heldout_task):
    heldout_task = str(heldout_task)
    fit = records[(records.task != heldout_task) & (records.split == "train")].copy()
    selection = records[
        (records.task != heldout_task) & (records.split == "validation")
    ].copy()
    evaluation = records[
        (records.task == heldout_task) & (records.split == "test")
    ].copy()
    if fit.empty or selection.empty or evaluation.empty:
        raise ValueError(f"LOTO partition is empty for {heldout_task}")
    if heldout_task in set(fit.task) | set(selection.task):
        raise RuntimeError("held-out-task leakage")
    if not set(fit.state_id).isdisjoint(evaluation.state_id):
        raise RuntimeError("LOTO state leakage")
    return LOTOPartition(fit, selection, evaluation, heldout_task)
