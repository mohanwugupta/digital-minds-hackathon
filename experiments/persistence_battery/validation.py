"""Pilot manipulation, label-bias, and behavioral nondegeneracy checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import TASKS
from .storage import read_records_frame
from .voluntary_waiting import WaitingCondition, optimal_policy


def _difference(frame, column, *, outcome="p_positive_semantic", high_minus_low=False):
    values = sorted(frame[column].dropna().unique())
    if len(values) < 2:
        return float("nan")
    low = frame[frame[column] == values[0]][outcome].mean()
    high = frame[frame[column] == values[-1]][outcome].mean()
    return float(high - low if high_minus_low else low - high)


def _correlation(frame, left, right="p_positive_semantic"):
    local = frame[[left, right]].dropna()
    if len(local) < 3 or local[left].std() == 0 or local[right].std() == 0:
        return float("nan")
    return float(local[left].corr(local[right]))


def manipulation_checks(frames, config):
    rows = []
    minimum_effect = float(
        config["validation"].get("minimum_expected_probability_effect", 0.0)
    )

    def add(task, check, effect, *, role="validity_gate", expected="positive"):
        passed = bool(
            np.isfinite(effect)
            and (
                abs(effect) >= minimum_effect
                if expected == "nonzero"
                else effect >= minimum_effect
            )
        )
        rows.append(
            {
                "task": task,
                "check": check,
                "effect": effect,
                "expected_direction": expected,
                "passed": passed,
                "gate_role": role,
            }
        )

    for task, frame in frames.items():
        initial = frame[frame.step == 0]
        if task == "voluntary_waiting":
            add(task, "lower opportunity cost increases waiting", _difference(initial, "opportunity_cost"))
            add(task, "higher reward increases waiting", _difference(initial, "reward_magnitude", high_minus_low=True))
            add(task, "lower quit payoff increases waiting", _difference(initial, "quit_payoff"))
            timing_means = initial.groupby("timing_environment").p_positive_semantic.mean()
            add(
                task,
                "timing environment changes waiting",
                float(timing_means.max() - timing_means.min()),
                expected="nonzero",
            )
            add(
                task,
                "elapsed time changes waiting",
                _correlation(frame, "step"),
                expected="nonzero",
            )
            policy_rows = []
            maximum_steps = int(config["tasks"][task]["max_steps"])
            condition_columns = [
                "timing_environment",
                "reward_magnitude",
                "opportunity_cost",
                "quit_payoff",
            ]
            for source in initial[condition_columns].drop_duplicates().itertuples(
                index=False
            ):
                condition = WaitingCondition(
                    source.timing_environment,
                    int(source.reward_magnitude),
                    int(source.opportunity_cost),
                    int(source.quit_payoff),
                )
                policy_rows.append(
                    {
                        **source._asdict(),
                        "initial_policy": optimal_policy(
                            condition, max_steps=maximum_steps
                        )[0],
                    }
                )
            policy_frame = pd.DataFrame(policy_rows)
            policy_varies = (
                policy_frame.groupby(
                    ["reward_magnitude", "opportunity_cost", "quit_payoff"]
                ).initial_policy.nunique().ge(2).any()
            )
            add(
                task,
                "timing environments have different normative initial policies",
                float(policy_varies),
                role="design_gate",
            )
        elif task == "progressive_ratio":
            add(task, "lower effort cost increases work", _difference(initial, "effort_cost"))
            add(
                task,
                "higher reward increases work",
                _difference(initial, "reward_magnitude", high_minus_low=True),
                role="diagnostic",
            )
            breakpoints = (
                frame.groupby(["episode_id", "ratio_schedule"], as_index=False)
                .breakpoint.max()
            )
            available_schedules = set(breakpoints.ratio_schedule)
            schedule_pair = next(
                (
                    pair
                    for pair in (
                        ("shallow", "steep"),
                        ("moderate_repair", "sharp_repair"),
                    )
                    if set(pair) <= available_schedules
                ),
                None,
            )
            gradual, sharp = schedule_pair or (None, None)
            gradual_breakpoint = breakpoints[
                breakpoints.ratio_schedule == gradual
            ].breakpoint.mean()
            sharp_breakpoint = breakpoints[
                breakpoints.ratio_schedule == sharp
            ].breakpoint.mean()
            add(
                task,
                "more gradual effort growth increases breakpoint",
                float(gradual_breakpoint - sharp_breakpoint),
                role="diagnostic",
            )
        elif task == "sunk_cost":
            add(task, "lower remaining cost increases continuation", _difference(initial, "remaining_steps"))
            prospective = [
                "remaining_steps",
                "reward_magnitude",
                "outside_option",
                "step_cost",
                "success_probability",
            ]
            matched = (
                initial.groupby(prospective).prior_investment.nunique().ge(2).all()
            )
            add(
                task,
                "past investment varies within exactly matched prospective states",
                float(matched),
                role="design_gate",
            )
            add(
                task,
                "greater sunk investment increases persistence",
                _difference(initial, "prior_investment", high_minus_low=True),
                role="scientific_hypothesis",
            )
        elif task == "information_sampling":
            add(task, "lower sampling cost increases sampling", _difference(frame, "sample_cost"))
            add(task, "decisive evidence reduces sampling", -_correlation(frame, "current_success_evidence"))
            add(task, "higher error penalty increases sampling", _difference(initial, "error_penalty", high_minus_low=True))
        elif task == "partial_reinforcement":
            add(task, "lower extinction cost increases trying", _difference(initial, "extinction_try_cost"))
            partial = initial[initial.reinforcement_schedule == "partial"].p_positive_semantic.mean()
            continuous = initial[initial.reinforcement_schedule == "continuous"].p_positive_semantic.mean()
            add(
                task,
                "partial training increases extinction persistence",
                float(partial - continuous),
                role="scientific_hypothesis",
            )
        elif task == "independent_effort_control":
            local = frame.copy()
            local["high_utility_advantage"] = (
                local.high_reward * local.high_success_probability - local.high_effort
                - (local.low_reward * local.low_success_probability - local.low_effort)
            )
            add(task, "high-effort choice tracks offer utility", _correlation(local, "high_utility_advantage"))
        elif task == "controllability":
            add(task, "lower transfer cost increases trying", _difference(initial, "transfer_cost"))
            uncontrollable = initial[initial.exposure_type == "uncontrollable"].p_positive_semantic.mean()
            controllable = initial[initial.exposure_type == "controllable"].p_positive_semantic.mean()
            add(
                task,
                "uncontrollable exposure reduces transfer persistence",
                float(controllable - uncontrollable),
                role="scientific_hypothesis",
            )
        elif task == "debugging_persistence":
            add(task, "lower attempt cost increases debugging", _difference(initial, "attempt_cost"))
            add(task, "higher solution reward increases debugging", _difference(initial, "solution_reward", high_minus_low=True))
            add(task, "lower restart value increases debugging", _difference(initial, "restart_value"))
    return pd.DataFrame(rows)


def label_bias_checks(frames, config):
    rows = []
    minimum_correlation = float(
        config["validation"]["minimum_mapping_balance_correlation"]
    )
    maximum_gap = float(config["validation"]["maximum_mean_absolute_mapping_gap"])
    for task, frame in frames.items():
        pivot = frame.pivot_table(
            index=["pair_id", "step"],
            columns="mapping_id",
            values="p_positive_semantic",
            aggfunc="first",
        ).dropna()
        if pivot.shape[1] != 2 or len(pivot) < 2:
            correlation, gap = float("nan"), float("nan")
        else:
            left, right = pivot.iloc[:, 0], pivot.iloc[:, 1]
            if left.std() == 0 and right.std() == 0:
                correlation = 1.0 if np.allclose(left, right) else 0.0
            else:
                correlation = float(left.corr(right))
            gap = float((pivot.iloc[:, 0] - pivot.iloc[:, 1]).abs().mean())
        rows.append(
            {
                "task": task,
                "paired_states": len(pivot),
                "mapping_probability_correlation": correlation,
                "mean_absolute_mapping_gap": gap,
                "minimum_correlation": minimum_correlation,
                "maximum_gap": maximum_gap,
                "passed": bool(
                    np.isfinite(correlation)
                    and correlation >= minimum_correlation
                    and gap <= maximum_gap
                ),
            }
        )
    return pd.DataFrame(rows)


def nondegeneracy_checks(frames, manipulations, label_bias, config, *, model_free):
    rows = []
    lower, upper = config["validation"]["persistence_probability_bounds"]
    minimum_sd = float(config["validation"]["minimum_persistence_logit_sd"])
    minimum_length = int(config["validation"]["minimum_median_episode_length"])
    minimum_parse_rate = float(
        config["validation"].get("minimum_top_token_action_rate", 0.9)
    )
    for task, frame in frames.items():
        probability = frame.p_positive_semantic.astype(float)
        logits = frame.choice_logit.astype(float)
        episode_lengths = frame.groupby("episode_id").size()
        probability_sd = float(probability.std())
        logit_sd = float(logits.std())
        within_episode_logit_sd = float(
            (logits - logits.groupby(frame.episode_id).transform("mean")).std()
        )
        choice_rate = float(
            (frame.semantic_action == TASKS[task].positive_action).mean()
        )
        task_manipulations = manipulations[manipulations.task == task]
        validity_checks = task_manipulations[
            task_manipulations.gate_role == "validity_gate"
        ]
        design_checks = task_manipulations[
            task_manipulations.gate_role == "design_gate"
        ]
        # The PRD requires at least one basic incentive effect.  Static design
        # checks (currently the waiting-policy contrast) must all pass, while
        # rows marked scientific_hypothesis are deliberately ignored here.
        require_all = bool(
            config["validation"].get("require_all_validity_checks", False)
        )
        validity_passed = (
            validity_checks.passed.all()
            if require_all
            else validity_checks.passed.any()
        )
        manipulation_passed = bool(
            validity_passed
            and (design_checks.empty or design_checks.passed.all())
        )
        label_passed = bool(label_bias[label_bias.task == task].passed.iloc[0])
        probability_gate = bool(lower <= probability.mean() <= upper)
        variability_gate = bool(within_episode_logit_sd >= minimum_sd)
        history_gate = bool(episode_lengths.median() >= minimum_length)
        parse_rate = float(
            frame.get("top_token_is_action", pd.Series(True, index=frame.index))
            .fillna(False)
            .astype(bool)
            .mean()
        )
        parsing_gate = bool(parse_rate >= minimum_parse_rate)
        approved = bool(
            probability_gate
            and variability_gate
            and history_gate
            and manipulation_passed
            and label_passed
            and parsing_gate
            and not model_free
        )
        rows.append(
            {
                "task": task,
                "states": len(frame),
                "episodes": frame.episode_id.nunique(),
                "semantic_pairs": frame.pair_id.nunique(),
                "mean_positive_probability": float(probability.mean()),
                "positive_probability_sd": probability_sd,
                "choice_logit_sd": logit_sd,
                "within_episode_choice_logit_sd": within_episode_logit_sd,
                "semantic_positive_action_rate": choice_rate,
                "median_episode_length": float(episode_lengths.median()),
                "probability_non_degenerate": probability_gate,
                "statewise_variability": variability_gate,
                "sufficient_history_depth": history_gate,
                "basic_manipulation_passed": manipulation_passed,
                "label_balance_passed": label_passed,
                "top_token_action_rate": parse_rate,
                "response_parsing_passed": parsing_gate,
                "approved_for_full_collection": approved,
                "approval_note": (
                    "model-free smoke is non-scientific"
                    if model_free
                    else "all pilot gates passed"
                    if approved
                    else "revise task parameters or parsing before full collection"
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_records(record_directory, output, config, tasks, *, model_free=False):
    record_directory, output = Path(record_directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        task: read_records_frame(record_directory, task)
        for task in tasks
    }
    manipulations = manipulation_checks(frames, config)
    label_bias = label_bias_checks(frames, config)
    nondegeneracy = nondegeneracy_checks(
        frames, manipulations, label_bias, config, model_free=model_free
    )
    manipulations.to_csv(output / "manipulation_checks.csv", index=False)
    label_bias.to_csv(output / "label_bias.csv", index=False)
    nondegeneracy.to_csv(output / "behavioral_non_degeneracy.csv", index=False)
    approval = {
        "all_tasks_approved": bool(nondegeneracy.approved_for_full_collection.all()),
        "model_free": bool(model_free),
        "tasks": {
            row.task: bool(row.approved_for_full_collection)
            for row in nondegeneracy.itertuples()
        },
    }
    (output / "pilot_approval.json").write_text(
        json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frames, manipulations, label_bias, nondegeneracy, approval
