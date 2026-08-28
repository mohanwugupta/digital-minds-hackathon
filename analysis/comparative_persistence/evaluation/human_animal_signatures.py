"""Prespecified qualitative human/animal signatures; never validity gates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _difference(frame, column, low, high, *, outcome="model_p_continue"):
    left = frame[frame[column] == low][outcome].mean()
    right = frame[frame[column] == high][outcome].mean()
    return float(right - left)


def run_signature_analysis(records, inclusion):
    rows = []
    included = dict(zip(inclusion.task, inclusion.included))

    def add(task, signature, effect=None, expected=None, note=""):
        available = bool(included.get(task, False)) and effect is not None and np.isfinite(effect)
        rows.append(
            {
                "task": task,
                "signature": signature,
                "available": available,
                "effect": float(effect) if available else float("nan"),
                "expected_direction": expected,
                "direction_reproduced": bool(effect > 0) if available and expected == "positive" else bool(effect < 0) if available and expected == "negative" else None,
                "validity_gate": False,
                "note": note if available else "task excluded by frozen PRD-1 gate or construct unavailable",
            }
        )

    for task in ("voluntary_waiting", "progressive_ratio", "sunk_cost"):
        add(task, task.replace("_", " "), note="scientific output, not a validity gate")

    pree = records[records.task == "partial_reinforcement"]
    if not pree.empty:
        initial = pree[pree["round"] == 0]
        effect = _difference(initial, "reinforcement_schedule", "continuous", "partial")
        add("partial_reinforcement", "partial reinforcement extinction effect", effect, "positive")

    information = records[records.task == "information_sampling"]
    if not information.empty:
        initial = information[information["round"] == 0]
        costs = sorted(initial.sample_cost.dropna().unique())
        penalties = sorted(initial.error_penalty.dropna().unique())
        if len(costs) >= 2:
            add(
                "information_sampling",
                "sampling-cost sensitivity",
                -_difference(initial, "sample_cost", costs[0], costs[-1]),
                "positive",
            )
        if len(penalties) >= 2:
            add(
                "information_sampling",
                "error-penalty sensitivity",
                _difference(initial, "error_penalty", penalties[0], penalties[-1]),
                "positive",
            )
        evidence = information[["current_success_evidence", "model_p_continue"]].dropna()
        if len(evidence) > 2:
            add(
                "information_sampling",
                "decisive evidence reduces sampling",
                -float(evidence.corr().iloc[0, 1]),
                "positive",
            )
    add("controllability", "controllability transfer", note="optional stretch task not collected")
    return pd.DataFrame(rows)
