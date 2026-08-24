"""Transparent persistence-versus-nuisance candidate decision rules."""

from __future__ import annotations

import math


NUISANCE_NAMES = ("label", "arbitrary_choice", "terminality", "generic_value")


def classify_candidate(
    *,
    persistence_sensitivity: float,
    cross_manipulation_transfer: float,
    cross_task_transfer: float,
    nuisance_sensitivity: dict[str, float],
    minimum_transfer: float,
    maximum_nuisance_fraction: float,
    positive_projection_fraction: float = 1.0,
    minimum_positive_projection_fraction: float = 0.5,
) -> dict:
    values = {
        "persistence_sensitivity": float(persistence_sensitivity),
        "cross_manipulation_transfer": float(cross_manipulation_transfer),
        "cross_task_transfer": float(cross_task_transfer),
        **{name: float(nuisance_sensitivity[name]) for name in NUISANCE_NAMES},
        "positive_projection_fraction": float(positive_projection_fraction),
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("candidate sensitivities must be finite and nonnegative")
    if not 0 <= maximum_nuisance_fraction <= 1 or minimum_transfer < 0:
        raise ValueError("invalid specificity thresholds")
    bound = maximum_nuisance_fraction * max(values["persistence_sensitivity"], 1e-12)
    criteria = {
        "persistence_direction": values["persistence_sensitivity"] >= minimum_transfer
        and values["positive_projection_fraction"]
        > float(minimum_positive_projection_fraction),
        "cross_manipulation_transfer": values["cross_manipulation_transfer"] >= minimum_transfer,
        "cross_task_transfer": values["cross_task_transfer"] >= minimum_transfer,
        **{
            f"{name}_specificity": values[name] <= bound for name in NUISANCE_NAMES
        },
    }
    passed = all(criteria.values())
    nuisance_dominates = max(values[name] for name in NUISANCE_NAMES) >= values[
        "persistence_sensitivity"
    ]
    return {
        "classification": (
            "persistence_specific_candidate"
            if passed
            else "no_persistence_specific_candidate"
        ),
        "criteria": criteria,
        "metrics": values,
        "nuisance_bound": bound,
        "alternative_hypothesis": (
            "domain_general_decision_or_value" if nuisance_dominates else None
        ),
        "causal_gate_passed": passed,
    }


def select_candidates(rows: list[dict], **thresholds) -> dict:
    """Classify every candidate without hiding component metrics in one score."""

    evaluated = []
    for row in rows:
        result = classify_candidate(
            persistence_sensitivity=row["persistence_sensitivity"],
            cross_manipulation_transfer=row["cross_manipulation_transfer"],
            cross_task_transfer=row["cross_task_transfer"],
            nuisance_sensitivity=row["nuisance_sensitivity"],
            **thresholds,
        )
        evaluated.append({**row, "decision": result})
    passing = [row for row in evaluated if row["decision"]["causal_gate_passed"]]
    return {
        "classification": (
            "persistence_specific_candidate_found"
            if passing
            else "no_persistence_specific_candidate"
        ),
        "candidates": evaluated,
        "passing_candidates": passing,
        "causal_gate_passed": bool(passing),
    }
