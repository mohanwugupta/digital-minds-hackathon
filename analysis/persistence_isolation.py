"""Leakage guards for exploratory cross-task and cross-manipulation search."""

from __future__ import annotations


def _validate_holdout(
    *, discovery, heldout, selection, unit: str, minimum_discovery: int
) -> dict:
    discovery = tuple(str(value) for value in discovery)
    selection = tuple(str(value) for value in selection)
    heldout = str(heldout)
    if len(discovery) < minimum_discovery or len(set(discovery)) != len(discovery):
        raise ValueError(f"{unit} holdout requires distinct discovery units")
    if heldout in discovery or heldout in selection:
        raise ValueError(f"held-out {unit} leaked into discovery or model selection")
    if set(selection) != set(discovery):
        raise ValueError(f"selection must use exactly the discovery {unit}s")
    return {
        f"discovery_{unit}s": list(discovery),
        f"heldout_{unit}": heldout,
        f"selection_{unit}s": list(selection),
        "heldout_parameters_fit": 0,
    }


def validate_task_holdout(
    *, discovery_tasks, heldout_task: str, selection_tasks
) -> dict:
    return _validate_holdout(
        discovery=discovery_tasks,
        heldout=heldout_task,
        selection=selection_tasks,
        unit="task",
        minimum_discovery=2,
    )


def validate_manipulation_holdout(
    *, discovery_manipulations, heldout_manipulation: str, selection_manipulations
) -> dict:
    return _validate_holdout(
        discovery=discovery_manipulations,
        heldout=heldout_manipulation,
        selection=selection_manipulations,
        unit="manipulation",
        minimum_discovery=1,
    )

