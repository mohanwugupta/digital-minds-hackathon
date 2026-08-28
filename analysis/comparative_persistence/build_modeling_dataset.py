"""Harmonize approved persistence records into an absorbing hazard dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.persistence_battery.storage import read_records_frame

from .normalization import StaticNormalizer
from .semantic_features import add_causal_history


def validate_split_integrity(frame):
    required = {"task", "episode_id", "pair_id", "state_id", "round", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"modeling records missing identifiers: {sorted(missing)}")
    if frame.state_id.duplicated().any():
        raise ValueError("duplicate state identifiers")
    for column, label in (("episode_id", "episode"), ("pair_id", "pair")):
        counts = frame.groupby(["task", column]).split.nunique()
        if (counts > 1).any():
            raise ValueError(f"{label} crosses splits")
    unknown = set(frame.split) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")


def build_hazard_risk_set(frame):
    frame = frame.copy()
    validate_split_integrity(frame)
    output = []
    for (task, episode_id), episode in frame.groupby(
        ["task", "episode_id"], sort=False
    ):
        episode = episode.sort_values("round")
        rounds = episode["round"].astype(int).tolist()
        if rounds != list(range(len(rounds))):
            raise ValueError(f"non-contiguous episode {task}/{episode_id}: {rounds}")
        absorbed = False
        for _, source in episode.iterrows():
            if absorbed:
                raise ValueError(f"post-termination state in episode {task}/{episode_id}")
            row = source.to_dict()
            continued = bool(row["continued"])
            row["hazard_event"] = int(not continued)
            row["at_risk"] = 1
            row["target_kind"] = (
                "termination_hazard"
                if bool(row.get("is_persistence_task", True))
                else "independent_binary_choice"
            )
            output.append(row)
            if bool(row.get("is_persistence_task", True)) and not continued:
                absorbed = True
    return pd.DataFrame(output)


def _prefixed(task, value):
    value = str(value)
    return value if value.startswith(f"{task}::") else f"{task}::{value}"


def _original_rows(frame, task, normalizer):
    rows = []
    spec = normalizer.task_specs[task]
    for _, source in frame.iterrows():
        row = source.to_dict()
        round_index = int(row["round"])
        horizon = int(spec["horizon"])
        if task == "bandit":
            effort = invested = progress = None
            success = np.clip((float(row["bayes_best"]) + 2.0) / 5.0, 0, 1)
            remaining_effort = None
        elif task == "foraging":
            effort = invested = float(row.get("search_count", round_index)) * float(row["stay_cost"])
            progress = None
            success = float(row["bayes_patch_probability"])
            remaining_effort = None
        else:
            effort = invested = float(row["cumulative_cost"])
            progress = float(row["progress_count"]) / max(1.0, float(row["max_attempts"]))
            success = float(row["bayes_progress_probability"])
            remaining_effort = max(0.0, float(row["max_attempts"]) - round_index) * float(row["attempt_cost"])
        raw = {
            "current_continue_cost": row["cost_pressure"],
            "current_outside_option": row["estimated_outside_value"],
            "elapsed_steps": round_index,
            "cumulative_effort": effort,
            "already_invested_cost": invested,
            "current_progress": progress,
            "current_success_evidence": success,
            "expected_remaining_effort": remaining_effort,
            "remaining_time": max(0, horizon - round_index),
            "expected_continue_payoff": row["estimated_continue_value"],
            "futility_evidence": float(row["estimated_outside_value"]) - float(row["estimated_continue_value"]),
        }
        normalized = normalizer.transform_row(task, raw)
        oracle_advantage = row.get("oracle_termination_advantage")
        oracle_advantage = (
            float(oracle_advantage) / float(spec["payoff_scale"])
            if oracle_advantage is not None and not pd.isna(oracle_advantage)
            else float("nan")
        )
        previous_action = row.get("previous_choice", np.nan)
        rows.append(
            {
                **row,
                **normalized,
                "task": task,
                "episode_id": _prefixed(task, row["episode_id"]),
                "pair_id": _prefixed(task, row["pair_id"]),
                "state_id": _prefixed(task, row["state_id"]),
                "continued": bool(row["continue"]),
                "is_persistence_task": True,
                "model_p_continue": float(row["p_continue"]),
                "model_choice_logit": float(row["persistence_logit"]),
                "previous_action_raw": previous_action,
                "previous_outcome_raw": row.get("previous_outcome", np.nan),
                "family": spec["family"],
                "reevaluation_advantage": normalized["continue_payoff_norm"] - normalized["outside_norm"],
                "option_advantage": success - normalized["cost_norm"] - normalized["outside_norm"],
                "oracle_reevaluation_advantage": oracle_advantage,
            }
        )
    return rows


def _semantic_previous_action(value, positive):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return float(str(value) == str(positive))


def _battery_remaining(task, row, spec):
    step = int(row["step"])
    horizon = int(spec["horizon"])
    remaining_time = max(0, horizon - step)
    cost = float(row.get("current_continue_cost", 0.0) or 0.0)
    if task == "progressive_ratio":
        remaining_effort = float(row.get("distance_to_goal", 0.0)) * cost
    elif task == "sunk_cost":
        remaining_effort = float(row.get("remaining_steps", 0.0)) * cost
    else:
        remaining_effort = remaining_time * cost
    reward = float(row.get("reward_magnitude", row.get("expected_reward_per_trial", 0.0)) or 0.0)
    evidence = row.get("current_success_evidence")
    evidence = 0.5 if pd.isna(evidence) else float(evidence)
    expected = evidence * reward - cost
    if task == "information_sampling":
        expected = float(row.get("error_penalty", 0.0)) * (1.0 - evidence) - cost
    return remaining_time, remaining_effort, expected


def _battery_rows(frame, task, normalizer, persistence):
    rows = []
    spec = normalizer.task_specs[task]
    for _, source in frame.iterrows():
        row = source.to_dict()
        remaining_time, remaining_effort, expected = _battery_remaining(task, row, spec)
        if task == "independent_effort_control":
            high_effort = float(row["high_effort"])
            low_effort = float(row["low_effort"])
            high_value = float(row["high_reward"]) * float(
                row["high_success_probability"]
            ) - high_effort
            low_value = float(row["low_reward"]) * float(
                row["low_success_probability"]
            ) - low_effort
            current_cost = high_effort - low_effort
            outside = low_value
            success_evidence = float(row["high_success_probability"])
            remaining_time = remaining_effort = None
            expected = high_value
        else:
            current_cost = row.get("current_continue_cost")
            outside = row.get("current_outside_option")
            success_evidence = row.get("current_success_evidence")
        prehistory_actions, prehistory_outcomes = [], []
        if task == "partial_reinforcement":
            try:
                messages = json.loads(row["conversation_json"])
                prompt = messages[0]["content"]
                outcome_line = prompt.split("recent outcomes were:\n", 1)[1].split("\n", 1)[0]
                tokens = [token.strip() for token in outcome_line.split(",")]
                prehistory_outcomes = [
                    0.0 if token == "no reward" else float(token.lstrip("+"))
                    for token in tokens
                ]
                prehistory_actions = [1.0] * len(prehistory_outcomes)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                prehistory_actions, prehistory_outcomes = [], []
        raw = {
            "current_continue_cost": current_cost,
            "current_outside_option": outside,
            "elapsed_steps": row["step"],
            "cumulative_effort": row.get("cumulative_effort"),
            "already_invested_cost": (
                float(row.get("prior_investment", 0.0)) * float(row.get("step_cost", 0.0))
                if task == "sunk_cost"
                else row.get("cumulative_effort")
            ),
            "current_progress": row.get("current_progress"),
            "current_success_evidence": success_evidence,
            "expected_remaining_effort": remaining_effort,
            "remaining_time": remaining_time,
            "expected_continue_payoff": expected,
            "futility_evidence": (
                float(outside or 0.0) - expected
            ),
        }
        normalized = normalizer.transform_row(task, raw)
        if task == "partial_reinforcement":
            oracle_advantage = -(
                0.0 if np.isnan(normalized["cost_norm"]) else normalized["cost_norm"]
            ) - (0.0 if np.isnan(normalized["outside_norm"]) else normalized["outside_norm"])
        else:
            oracle_advantage = (
                (0.0 if np.isnan(normalized["success_evidence"]) else normalized["success_evidence"])
                - (0.0 if np.isnan(normalized["cost_norm"]) else normalized["cost_norm"])
                - (0.0 if np.isnan(normalized["outside_norm"]) else normalized["outside_norm"])
            )
        continued = str(row["semantic_action"]) == str(row["positive_semantic"])
        rows.append(
            {
                **row,
                **normalized,
                "task": task,
                "episode_id": _prefixed(task, row["episode_id"]),
                "pair_id": _prefixed(task, row["pair_id"]),
                "state_id": _prefixed(task, row["state_id"]),
                "round": int(row["step"]),
                "continued": continued,
                "is_persistence_task": bool(persistence),
                "outcome_after_choice": row.get("subsequent_outcome"),
                "model_p_continue": float(row["p_positive_semantic"]),
                "model_choice_logit": float(row["choice_logit"]),
                "previous_action_raw": _semantic_previous_action(
                    row.get("previous_action"), row["positive_semantic"]
                ),
                "previous_outcome_raw": row.get("previous_outcome"),
                "prehistory_actions": prehistory_actions,
                "prehistory_outcomes": prehistory_outcomes,
                "family": spec["family"],
                "reevaluation_advantage": normalized["continue_payoff_norm"] - (
                    0.0 if np.isnan(normalized["outside_norm"]) else normalized["outside_norm"]
                ),
                "option_advantage": (
                    (0.0 if np.isnan(normalized["success_evidence"]) else normalized["success_evidence"])
                    - (0.0 if np.isnan(normalized["cost_norm"]) else normalized["cost_norm"])
                    - (0.0 if np.isnan(normalized["outside_norm"]) else normalized["outside_norm"])
                ),
                "oracle_reevaluation_advantage": oracle_advantage,
            }
        )
    return rows


def load_modeling_dataset(config, *, smoke=False):
    """Load original tasks and only PRD-1-approved battery tasks."""

    normalizer = StaticNormalizer(config["task_specs"])
    rows, inclusion = [], []
    original_root = Path(config["inputs"]["original_records"])
    for task in config["original_tasks"]:
        path = original_root / f"{task}_records.csv"
        frame = pd.read_csv(path)
        rows.extend(_original_rows(frame, task, normalizer))
        inclusion.append({"task": task, "included": True, "reason": "frozen original task"})

    run_root = Path(
        config["inputs"]["battery_smoke_run" if smoke else "battery_run"]
    )
    approval_path = run_root / "validation/pilot_approval.json"
    if not approval_path.exists():
        raise FileNotFoundError(
            f"battery approval is missing: {approval_path}; finalize the battery run first"
        )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    records_root = run_root / "pilot/records"
    for task in config["battery_tasks"]:
        approved = bool(approval.get("tasks", {}).get(task, False))
        is_control = task == config["control_task"]
        included = approved or bool(smoke)
        reason = "approved PRD-1 task" if approved else "smoke plumbing override" if smoke else "failed PRD-1 basic gate"
        inclusion.append({"task": task, "included": included, "reason": reason})
        if not included:
            continue
        frame = read_records_frame(records_root, task)
        rows.extend(_battery_rows(frame, task, normalizer, not is_control))

    frame = pd.DataFrame(rows)
    payoff_scales = {
        task: spec["payoff_scale"] for task, spec in config["task_specs"].items()
    }
    frame = add_causal_history(
        frame,
        decay=float(config["history_decay"][0]),
        payoff_scales=payoff_scales,
    )
    risk = build_hazard_risk_set(frame)
    return risk.reset_index(drop=True), pd.DataFrame(inclusion)
