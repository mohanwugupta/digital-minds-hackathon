"""Leakage and execution gates for shared cross-task persistence probes."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


def macro_average(task_metrics: dict[str, float]) -> float:
    """Give each task one vote, independent of its number of activation states."""
    if len(task_metrics) < 2:
        raise ValueError("a shared metric requires at least two tasks")
    values = [float(value) for value in task_metrics.values()]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("task metrics must be finite")
    return statistics.mean(values)


def validate_discovery_plan(
    *,
    discovery_tasks: tuple[str, ...] | list[str],
    heldout_task: str,
    layer_selection_tasks: tuple[str, ...] | list[str],
) -> dict:
    discovery = tuple(str(task) for task in discovery_tasks)
    selection = tuple(str(task) for task in layer_selection_tasks)
    heldout = str(heldout_task)
    if len(discovery) < 2 or len(set(discovery)) != len(discovery):
        raise ValueError("shared discovery requires at least two distinct tasks")
    if heldout in discovery:
        raise ValueError("the held-out task cannot be a discovery task")
    if set(selection) != set(discovery):
        raise ValueError(
            "layer and regularization selection must use exactly the discovery tasks"
        )
    if heldout in selection:
        raise ValueError("held-out task leaked into model selection")
    return {
        "discovery_tasks": list(discovery),
        "heldout_task": heldout,
        "layer_selection_tasks": list(selection),
        "heldout_task_parameters_fit": 0,
        "task_weighting": "equal_macro_weight",
    }


def validate_loto_folds(tasks: tuple[str, ...] | list[str], folds: list[dict]) -> list[dict]:
    """Require one exact complement fold for every task."""
    universe = {str(task) for task in tasks}
    if len(universe) < 3 or len(folds) != len(universe):
        raise ValueError("leave-one-task-out requires one fold per task")
    plans = []
    for fold in folds:
        heldout = str(fold["heldout"])
        discovery = tuple(str(task) for task in fold["discovery"])
        if heldout not in universe or set(discovery) != universe - {heldout}:
            raise ValueError("each LOTO fold must discover on the exact task complement")
        plans.append(
            validate_discovery_plan(
                discovery_tasks=discovery,
                heldout_task=heldout,
                layer_selection_tasks=discovery,
            )
        )
    if {plan["heldout_task"] for plan in plans} != universe:
        raise ValueError("each task must be held out exactly once")
    return plans


def source_task_gate(
    per_task_metrics: dict[str, dict], random_correlation_95: dict[str, float]
) -> dict:
    """Prevent a nominally shared candidate from being carried by one source task."""
    if len(per_task_metrics) < 2 or set(per_task_metrics) != set(random_correlation_95):
        raise ValueError("source gate requires matched metrics and controls for every task")
    criteria = {
        task: {
            "positive_correlation": float(metrics["correlation"]) > 0,
            "positive_r_squared": float(metrics["r_squared"]) > 0,
            "exceeds_matched_random_95th": float(metrics["correlation"])
            > float(random_correlation_95[task]),
        }
        for task, metrics in per_task_metrics.items()
    }
    return {
        "passed": all(all(values.values()) for values in criteria.values()),
        "criteria_by_task": criteria,
        "random_correlation_95th_by_task": {
            task: float(value) for task, value in random_correlation_95.items()
        },
    }


def _load(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        return source
    with Path(source).open(encoding="utf-8") as handle:
        return json.load(handle)


def require_shared_clearance(source: str | Path | dict) -> dict:
    result = _load(source)
    if result.get("classification") not in {
        "strong_shared_transfer",
        "partial_shared_transfer",
    }:
        raise RuntimeError(
            "held-out causal work requires strong or partial shared transfer; "
            f"observed {result.get('classification')!r}"
        )
    if int(result.get("heldout_task_parameters_fit", -1)) != 0:
        raise RuntimeError("shared-transfer artifact is not strict held-out transfer")
    if result.get("primary_heldout_task") != "solvability" or set(
        result.get("primary_discovery_tasks", ())
    ) != {"bandit", "foraging"}:
        raise RuntimeError(
            "causal work requires the Bandit+Foraging to Solvability primary test"
        )
    return result
