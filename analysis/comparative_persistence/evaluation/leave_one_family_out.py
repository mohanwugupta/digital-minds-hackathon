"""Predefined whole-family holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LOFOPartition:
    fit: object
    selection: object
    evaluation: object
    heldout_family: str


def lofo_partition(records, heldout_family):
    family = str(heldout_family)
    fit = records[(records.family != family) & (records.split == "train")].copy()
    selection = records[(records.family != family) & (records.split == "validation")].copy()
    evaluation = records[(records.family == family) & (records.split == "test")].copy()
    if fit.empty or selection.empty or evaluation.empty:
        raise ValueError(f"LOFO partition is empty for {family}")
    if family in set(fit.family) | set(selection.family):
        raise RuntimeError("held-out-family leakage")
    return LOFOPartition(fit, selection, evaluation, family)
