"""Non-gating comparative-cognition outputs for the repaired battery."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _difference(frame, column, low=None, high=None, outcome="model_p_continue"):
    values = sorted(frame[column].dropna().unique())
    if low is None:
        low = values[0]
    if high is None:
        high = values[-1]
    return float(
        frame[frame[column] == high][outcome].mean()
        - frame[frame[column] == low][outcome].mean()
    )


def run_robustness_signatures(records, inclusion, history_kernels):
    included = dict(zip(inclusion.task, inclusion.included))
    rows = []

    def add(task, signature, effect, estimand, expected=None):
        available = bool(included.get(task, False)) and np.isfinite(effect)
        direction = None
        if available and expected == "positive":
            direction = bool(effect > 0)
        elif available and expected == "negative":
            direction = bool(effect < 0)
        rows.append(
            {
                "task": task,
                "signature": signature,
                "available": available,
                "effect": effect if available else float("nan"),
                "estimand": estimand,
                "expected_direction": expected,
                "direction_reproduced": direction,
                "validity_gate": False,
            }
        )

    waiting = records[records.task == "voluntary_waiting"]
    if not waiting.empty:
        initial = waiting[waiting["round"] == 0]
        timing = initial.groupby("timing_environment").model_p_continue.mean()
        add(
            "voluntary_waiting",
            "temporal-context adaptation",
            float(timing.max() - timing.min()),
            "maximum minus minimum initial waiting probability across timing environments",
            "positive",
        )

    progressive = records[records.task == "progressive_ratio"]
    if not progressive.empty:
        breakpoints = progressive.groupby("episode_id")["rewards_completed"].max()
        add(
            "progressive_ratio",
            "progressive-ratio breakpoint",
            float(breakpoints.mean()),
            "mean completed reward requirements before quitting/censoring",
        )

    sunk = records[(records.task == "sunk_cost") & (records["round"] == 0)]
    if not sunk.empty:
        columns = [
            "prior_investment",
            "remaining_steps",
            "reward_magnitude",
            "outside_option",
            "step_cost",
            "success_probability",
        ]
        local = sunk[columns + ["model_p_continue"]].dropna()
        design = np.column_stack((np.ones(len(local)), local[columns].to_numpy(float)))
        coefficient = np.linalg.lstsq(
            design, local.model_p_continue.to_numpy(float), rcond=None
        )[0]
        add(
            "sunk_cost",
            "prospectively controlled sunk-cost coefficient",
            float(coefficient[1]),
            "linear probability coefficient on prior investment controlling all prospective variables",
        )

    pree = records[(records.task == "partial_reinforcement") & (records["round"] == 0)]
    if not pree.empty:
        add(
            "partial_reinforcement",
            "partial-reinforcement extinction effect",
            _difference(pree, "reinforcement_schedule", "continuous", "partial"),
            "partial minus continuous initial retry probability",
            "positive",
        )

    information = records[records.task == "information_sampling"]
    if not information.empty:
        initial = information[information["round"] == 0]
        add(
            "information_sampling",
            "sampling-cost sensitivity",
            -_difference(initial, "sample_cost"),
            "negative high-minus-low sampling-cost effect, sign reversed",
            "positive",
        )
        add(
            "information_sampling",
            "error-penalty sensitivity",
            _difference(initial, "error_penalty"),
            "high-minus-low error-penalty effect",
            "positive",
        )

    controllability = records[(records.task == "controllability") & (records["round"] == 0)]
    if not controllability.empty:
        add(
            "controllability",
            "prior controllability transfer",
            _difference(
                controllability,
                "exposure_type",
                "uncontrollable",
                "controllable",
            ),
            "controllable minus uncontrollable exposure",
            "positive",
        )

    if not history_kernels.empty:
        add(
            "cross_task",
            "finite history kernel directional agreement",
            float(
                history_kernels.action_recent_sign.value_counts(normalize=True).max()
            ),
            "majority fraction sharing the modal recent-action sign",
            "positive",
        )
    return pd.DataFrame(rows)

