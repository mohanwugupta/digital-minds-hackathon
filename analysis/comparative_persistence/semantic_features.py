"""Causal semantic history and explicit missingness-aware feature matrices."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd


FORBIDDEN_FUTURE_FIELDS = {
    "subsequent_reward",
    "subsequent_outcome",
    "subsequent_effort",
    "subsequent_success",
    "outcome_after_choice",
    "termination_reason",
    "episode_duration",
    "realized_future_reward",
}

HISTORY_FEATURES = tuple(
    ["continue_streak", "failure_streak", "success_streak"]
    + [f"action_lag_{lag}" for lag in (1, 2, 3, 5, 8)]
    + [f"outcome_lag_{lag}" for lag in (1, 2, 3, 5, 8)]
)

IMMEDIATE_FEATURES = (
    "cost_norm",
    "outside_norm",
    "progress_norm",
    "success_evidence",
    "remaining_effort_norm",
    "remaining_time_norm",
    "continue_payoff_norm",
    "futility_norm",
)

ALL_OBSERVABLE_FEATURES = (
    "time_norm",
    "effort_norm",
    "invested_norm",
    *IMMEDIATE_FEATURES,
    *HISTORY_FEATURES,
    "reward_kernel",
    "choice_kernel",
    "reevaluation_advantage",
    "option_advantage",
)


def _number(value):
    if value is None:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def add_causal_history(frame, *, decay=0.7, payoff_scales=None):
    """Reconstruct only information available before each current decision."""

    frame = frame.copy()
    output = []
    payoff_scales = payoff_scales or {}
    for (_task, _episode), episode in frame.groupby(
        ["task", "episode_id"], sort=False
    ):
        episode = episode.sort_values("round")
        first = episode.iloc[0]
        raw_actions = first.get("prehistory_actions", [])
        raw_outcomes = first.get("prehistory_outcomes", [])
        actions = list(raw_actions) if isinstance(raw_actions, (list, tuple)) else []
        outcomes = [
            float(value) / float(payoff_scales.get(str(first["task"]), 1.0))
            for value in (raw_outcomes if isinstance(raw_outcomes, (list, tuple)) else [])
        ]
        reward_kernel = choice_kernel = 0.0
        for action, outcome in zip(actions, outcomes):
            reward_kernel = float(decay) * reward_kernel + float(outcome)
            choice_kernel = float(decay) * choice_kernel + float(action)
        continue_streak = 0
        for action in reversed(actions):
            if not action:
                break
            continue_streak += 1
        for _, source in episode.iterrows():
            row = source.to_dict()
            task = str(row["task"])
            scale = float(payoff_scales.get(task, 1.0))
            initial_previous = _number(row.get("previous_outcome_raw")) / scale
            initial_action = _number(row.get("previous_action_raw"))
            for lag in (1, 2, 3, 5, 8):
                if len(actions) >= lag:
                    row[f"action_lag_{lag}"] = actions[-lag]
                    row[f"outcome_lag_{lag}"] = outcomes[-lag]
                elif lag == 1:
                    row[f"action_lag_{lag}"] = initial_action
                    row[f"outcome_lag_{lag}"] = initial_previous
                else:
                    row[f"action_lag_{lag}"] = float("nan")
                    row[f"outcome_lag_{lag}"] = float("nan")
            row["continue_streak"] = continue_streak
            row["reward_kernel"] = reward_kernel
            row["choice_kernel"] = choice_kernel
            output.append(row)

            action = float(bool(row["continued"]))
            outcome = _number(row.get("outcome_after_choice")) / scale
            outcome = 0.0 if not math.isfinite(outcome) else outcome
            actions.append(action)
            outcomes.append(outcome)
            continue_streak = continue_streak + 1 if action else 0
            reward_kernel = float(decay) * reward_kernel + outcome
            choice_kernel = float(decay) * choice_kernel + action
    return pd.DataFrame(output)


def build_feature_matrix(frame, features):
    """Encode every nullable construct as value plus an explicit presence bit."""

    features = tuple(str(name) for name in features)
    forbidden = sorted(set(features) & FORBIDDEN_FUTURE_FIELDS)
    if forbidden:
        raise ValueError(f"future leakage fields requested: {forbidden}")
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"requested semantic features are absent: {missing}")
    columns, names = [], []
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        present = np.isfinite(values)
        columns.extend((np.where(present, values, 0.0), present.astype(float)))
        names.extend((name, f"{name}__present"))
    if not columns:
        return np.empty((len(frame), 0), dtype=float), tuple()
    return np.column_stack(columns), tuple(names)
