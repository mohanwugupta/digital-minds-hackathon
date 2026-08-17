"""Dependency-free integrity and preregistered gates for Track B."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


FORAGING = "foraging"
CONTROL = "binary_control"
SOLVABILITY = "solvability"
TERMINALITY = "terminality_control"
_SEMANTICS = {
    FORAGING: ("STAY", "LEAVE"),
    SOLVABILITY: ("TRY_AGAIN", "GIVE_UP"),
    CONTROL: ("LEFT_GREATER", "RIGHT_GREATER"),
    TERMINALITY: ("PROCEED", "END"),
}
_ISSUES = (
    "task",
    "episode_count",
    "duplicate_episode_id",
    "duplicate_state_id",
    "empty_episode",
    "record_episode",
    "record_pair",
    "record_mapping_id",
    "round_or_state_id",
    "mapping_schema",
    "mapping_record_fields",
    "raw_label_semantics",
    "terminal_position",
    "terminal_semantics",
    "probability_geometry",
    "control_episode_shape",
    "counterbalance_pair_size",
    "counterbalance_mapping",
    "paired_condition",
)


def _task_name(value: str) -> str:
    if value in {"control", CONTROL}:
        return CONTROL
    return TERMINALITY if value in {"terminality", TERMINALITY} else value


def _mapping(record: dict) -> dict[str, str] | None:
    value = record.get("label_mapping")
    try:
        parsed = json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return {str(key): str(label) for key, label in parsed.items()}


def _conversation(record: dict) -> list[dict]:
    value = record.get("conversation", [])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def audit_cross_task_shards(
    shards: list[dict],
    expected_task: str,
    *,
    response_labels: tuple[str, str] = ("X", "Y"),
    expected_episodes: int | None = None,
) -> dict:
    """Audit episode semantics and exact label reversal without loading tensors."""
    expected_task = _task_name(expected_task)
    if expected_task not in _SEMANTICS:
        raise ValueError(f"unknown cross-task type: {expected_task!r}")
    positive, negative = _SEMANTICS[expected_task]
    expected_labels = set(response_labels)
    if len(expected_labels) != 2:
        raise ValueError("response_labels must contain two distinct labels")

    issues: Counter[str] = Counter()
    seen_episodes: set[str] = set()
    seen_states: set[str] = set()
    pairs: dict[str, list[dict]] = defaultdict(list)

    for shard in shards:
        episode_id = str(shard.get("episode_id", ""))
        pair_id = str(shard.get("pair_id", ""))
        if _task_name(str(shard.get("task", ""))) != expected_task:
            issues["task"] += 1
        if episode_id in seen_episodes:
            issues["duplicate_episode_id"] += 1
        seen_episodes.add(episode_id)
        pairs[pair_id].append(shard)
        records = shard.get("records", [])
        if not records:
            issues["empty_episode"] += 1
            continue
        if expected_task in {CONTROL, TERMINALITY} and len(records) != 1:
            issues["control_episode_shape"] += 1

        for position, record in enumerate(records):
            state_id = str(record.get("state_id", ""))
            if state_id in seen_states:
                issues["duplicate_state_id"] += 1
            seen_states.add(state_id)
            if record.get("episode_id") != episode_id:
                issues["record_episode"] += 1
            if record.get("pair_id") != pair_id:
                issues["record_pair"] += 1
            if record.get("mapping_id") != shard.get("mapping_id"):
                issues["record_mapping_id"] += 1
            if record.get("round") != position or state_id != f"{episode_id}:{position}":
                issues["round_or_state_id"] += 1

            mapping = _mapping(record)
            if (
                mapping is None
                or set(mapping) != {positive, negative}
                or set(mapping.values()) != expected_labels
            ):
                issues["mapping_schema"] += 1
                mapping = None
            if (
                record.get("positive_semantic") != positive
                or record.get("negative_semantic") != negative
                or mapping is None
                or record.get("positive_label") != mapping.get(positive)
                or record.get("negative_label") != mapping.get(negative)
            ):
                issues["mapping_record_fields"] += 1
            semantic_choice = record.get("semantic_choice")
            if (
                mapping is None
                or semantic_choice not in {positive, negative}
                or record.get("raw_label") != mapping.get(semantic_choice)
            ):
                issues["raw_label_semantics"] += 1

            is_last = position == len(records) - 1
            terminated = record.get("terminated") is True
            if terminated != is_last:
                issues["terminal_position"] += 1
            reason = record.get("termination_reason")
            if expected_task in {FORAGING, SOLVABILITY}:
                if expected_task == FORAGING:
                    persist_choice, disengage_choice = "STAY", "LEAVE"
                    disengage_reason, horizon_reason = "leave", "max_decisions"
                    probability_keys = ("p_stay", "p_leave")
                else:
                    persist_choice, disengage_choice = "TRY_AGAIN", "GIVE_UP"
                    disengage_reason, horizon_reason = "give_up", "max_attempts"
                    probability_keys = ("p_try_again", "p_give_up")
                valid_terminal = (
                    semantic_choice == disengage_choice
                    and terminated
                    and reason == disengage_reason
                ) or (
                    semantic_choice == persist_choice
                    and terminated
                    and reason == horizon_reason
                )
                valid_nonterminal = (
                    semantic_choice == persist_choice
                    and not terminated
                    and reason is None
                )
                if not (valid_terminal or valid_nonterminal):
                    issues["terminal_semantics"] += 1
                try:
                    p_stay = float(record[probability_keys[0]])
                    p_leave = float(record[probability_keys[1]])
                    if (
                        not 0.0 <= p_stay <= 1.0
                        or not 0.0 <= p_leave <= 1.0
                        or abs(p_stay + p_leave - 1.0) > 1e-6
                    ):
                        issues["probability_geometry"] += 1
                except (KeyError, TypeError, ValueError):
                    issues["probability_geometry"] += 1
            elif expected_task == CONTROL:
                if not (
                    terminated
                    and is_last
                    and reason == "single_judgment"
                ):
                    issues["terminal_semantics"] += 1
                prompt = "\n".join(
                    str(message.get("content", ""))
                    for message in _conversation(record)
                ).lower()
                if any(word in prompt for word in ("stay", "leave", "continue", "quit")):
                    issues["terminal_semantics"] += 1
            else:
                if not (
                    terminated
                    and is_last
                    and reason == "rule_determined_judgment"
                ):
                    issues["terminal_semantics"] += 1
                expected_choice = (
                    "PROCEED" if int(record.get("displayed_integer", 1)) % 2 == 0 else "END"
                )
                if record.get("correct_choice") != expected_choice:
                    issues["terminal_semantics"] += 1

    condition_keys = (
        ("seed", "action_seed", "initial_quality", "depletion", "outside_option", "stay_cost")
        if expected_task == FORAGING
        else (
            "seed",
            "action_seed",
            "progress_probability",
            "attempt_cost",
            "give_up_value",
            "max_attempts",
        )
        if expected_task == SOLVABILITY
        else ("seed", "left_integer", "right_integer")
        if expected_task == CONTROL
        else ("seed", "displayed_integer")
    )
    for selected in pairs.values():
        if len(selected) != 2:
            issues["counterbalance_pair_size"] += 1
            continue
        first_records = [shard["records"][0] for shard in selected if shard.get("records")]
        if len(first_records) != 2:
            continue
        mappings = [_mapping(record) for record in first_records]
        mapping_ids = {str(shard.get("mapping_id", "")) for shard in selected}
        reversed_mapping = bool(
            None not in mappings
            and mappings[0] != mappings[1]
            and mappings[0][positive] == mappings[1][negative]
            and mappings[0][negative] == mappings[1][positive]
            and len(mapping_ids) == 2
        )
        if not reversed_mapping:
            issues["counterbalance_mapping"] += 1
        signatures = {
            tuple(record.get(key) for key in condition_keys)
            for record in first_records
        }
        if len(signatures) != 1:
            issues["paired_condition"] += 1

    counts = {name: int(issues[name]) for name in _ISSUES}
    if expected_episodes is not None and len(shards) != int(expected_episodes):
        counts["episode_count"] = abs(len(shards) - int(expected_episodes))
    return {
        "passed": not any(counts.values()),
        "task": expected_task,
        "episodes": len(shards),
        "pairs": len(pairs),
        "states": len(seen_states),
        "response_labels": list(response_labels),
        "expected_episodes": expected_episodes,
        "issue_counts": counts,
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _high_minus_low(records: list[dict], key: str, target: str) -> float | None:
    values = sorted({record[key] for record in records})
    if len(values) < 2:
        return None
    low = [float(record[target]) for record in records if record[key] == values[0]]
    high = [float(record[target]) for record in records if record[key] == values[-1]]
    return statistics.mean(high) - statistics.mean(low)


def evaluate_behavioral_gate(
    shards: list[dict],
    split: dict[str, list[str]],
    thresholds: dict[str, Any],
) -> dict:
    """Evaluate the frozen behavioral gate on train+validation episodes only."""
    development_ids = set(split.get("train", ())) | set(split.get("validation", ()))
    selected = [shard for shard in shards if shard.get("episode_id") in development_ids]
    records = [record for shard in selected for record in shard.get("records", ())]
    if not selected or not records:
        raise ValueError("behavioral validation has no development episodes or states")
    probabilities = [float(record["p_stay"]) for record in records]
    logits = [float(record["persistence_logit"]) for record in records]
    initial = [record for record in records if int(record["round"]) == 0]
    stay_rate = statistics.mean(
        float(record["semantic_choice"] == "STAY") for record in records
    )
    mean_episode_decisions = statistics.mean(
        len(shard["records"]) for shard in selected
    )
    leave_rate = statistics.mean(
        float(shard["records"][-1]["termination_reason"] == "leave")
        for shard in selected
    )
    mapping_means: dict[str, list[float]] = defaultdict(list)
    for record in initial:
        mapping_means[str(record["mapping_id"])].append(float(record["p_stay"]))
    initial_mapping_means = {
        key: statistics.mean(values) for key, values in sorted(mapping_means.items())
    }
    mapping_gap = (
        max(initial_mapping_means.values()) - min(initial_mapping_means.values())
        if len(initial_mapping_means) >= 2
        else float("inf")
    )
    outside_effect = _high_minus_low(initial, "outside_option", "persistence_logit")
    cost_effect = _high_minus_low(initial, "stay_cost", "persistence_logit")
    economic_limit = float(thresholds["minimum_expected_economic_logit_effect"])
    stay_bounds = [float(value) for value in thresholds["semantic_stay_rate_bounds"]]
    leave_bounds = [float(value) for value in thresholds["episode_leave_rate_bounds"]]
    p10, p90 = _quantile(probabilities, 0.10), _quantile(probabilities, 0.90)
    criteria = {
        "enough_development_episodes": len(selected)
        >= int(thresholds["minimum_development_episodes"]),
        "enough_development_states": len(records)
        >= int(thresholds["minimum_development_states"]),
        "episodes_contain_repeated_decisions": mean_episode_decisions
        >= float(thresholds.get("minimum_mean_episode_decisions", 1.0)),
        "persistence_logit_varies": statistics.pstdev(logits)
        >= float(thresholds["minimum_persistence_logit_sd"]),
        "stay_probability_spans_decisions": p90 - p10
        >= float(thresholds["minimum_p_stay_interdecile_range"]),
        "semantic_choices_nondegenerate": stay_bounds[0] <= stay_rate <= stay_bounds[1],
        "episode_termination_nondegenerate": leave_bounds[0]
        <= leave_rate
        <= leave_bounds[1],
        "label_mapping_initial_gap_bounded": mapping_gap
        <= float(thresholds["maximum_initial_mapping_p_stay_gap"]),
        "outside_option_reduces_persistence": outside_effect is not None
        and outside_effect <= -economic_limit,
        "stay_cost_reduces_persistence": cost_effect is not None
        and cost_effect <= -economic_limit,
    }
    return {
        "level": "behavioral_generalization",
        "analysis_role": "confirmatory_gate",
        "data_scope": ["train", "validation"],
        "test_episodes_inspected": False,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "development_episodes": len(selected),
        "development_states": len(records),
        "semantic_stay_choice_rate": stay_rate,
        "mean_episode_decisions": mean_episode_decisions,
        "episodes_ending_by_leave_rate": leave_rate,
        "p_stay": {
            "mean": statistics.mean(probabilities),
            "standard_deviation": statistics.pstdev(probabilities),
            "p10": p10,
            "p90": p90,
            "interdecile_range": p90 - p10,
        },
        "persistence_logit_standard_deviation": statistics.pstdev(logits),
        "initial_state_p_stay_by_mapping": initial_mapping_means,
        "initial_mapping_p_stay_gap": mapping_gap,
        "economic_manipulation_logit_effects": {
            "higher_outside_option_minus_lower": outside_effect,
            "higher_stay_cost_minus_lower": cost_effect,
        },
        "thresholds": dict(thresholds),
    }


def evaluate_solvability_behavioral_gate(
    shards: list[dict],
    split: dict[str, list[str]],
    thresholds: dict[str, Any],
) -> dict:
    """Confirm meaningful TRY-AGAIN/GIVE-UP behavior without test labels."""
    development_ids = set(split.get("train", ())) | set(split.get("validation", ()))
    selected = [shard for shard in shards if shard.get("episode_id") in development_ids]
    records = [record for shard in selected for record in shard.get("records", ())]
    if not selected or not records:
        raise ValueError("solvability validation has no development episodes or states")
    probabilities = [float(record["p_try_again"]) for record in records]
    logits = [float(record["persistence_logit"]) for record in records]
    initial = [record for record in records if int(record["round"]) == 0]
    persistence_rate = statistics.mean(
        float(record["semantic_choice"] == "TRY_AGAIN") for record in records
    )
    disengagement_rate = statistics.mean(
        float(shard["records"][-1]["termination_reason"] == "give_up")
        for shard in selected
    )
    mean_decisions = statistics.mean(len(shard["records"]) for shard in selected)
    mapping_values: dict[str, list[float]] = defaultdict(list)
    for record in initial:
        mapping_values[str(record["mapping_id"])].append(
            float(record["p_try_again"])
        )
    mapping_means = {
        key: statistics.mean(values) for key, values in sorted(mapping_values.items())
    }
    mapping_gap = (
        max(mapping_means.values()) - min(mapping_means.values())
        if len(mapping_means) >= 2
        else float("inf")
    )
    progress_effect = _high_minus_low(
        initial, "progress_probability", "persistence_logit"
    )
    cost_effect = _high_minus_low(initial, "attempt_cost", "persistence_logit")
    fallback_effect = _high_minus_low(
        initial, "give_up_value", "persistence_logit"
    )
    minimum_effect = float(thresholds["minimum_expected_logit_effect"])
    persist_bounds = [
        float(value) for value in thresholds["semantic_persistence_rate_bounds"]
    ]
    disengage_bounds = [
        float(value) for value in thresholds["episode_disengagement_rate_bounds"]
    ]
    p10, p90 = _quantile(probabilities, 0.10), _quantile(probabilities, 0.90)
    criteria = {
        "enough_development_episodes": len(selected)
        >= int(thresholds["minimum_development_episodes"]),
        "enough_development_states": len(records)
        >= int(thresholds["minimum_development_states"]),
        "episodes_contain_repeated_decisions": mean_decisions
        >= float(thresholds["minimum_mean_episode_decisions"]),
        "persistence_logit_varies": statistics.pstdev(logits)
        >= float(thresholds["minimum_persistence_logit_sd"]),
        "persistence_probability_spans_decisions": p90 - p10
        >= float(thresholds["minimum_persistence_probability_interdecile_range"]),
        "semantic_choices_nondegenerate": persist_bounds[0]
        <= persistence_rate
        <= persist_bounds[1],
        "episode_termination_nondegenerate": disengage_bounds[0]
        <= disengagement_rate
        <= disengage_bounds[1],
        "label_mapping_initial_gap_bounded": mapping_gap
        <= float(thresholds["maximum_initial_mapping_probability_gap"]),
        "solvability_evidence_increases_persistence": progress_effect is not None
        and progress_effect >= minimum_effect,
        "attempt_cost_reduces_persistence": cost_effect is not None
        and cost_effect <= -minimum_effect,
        "give_up_value_reduces_persistence": fallback_effect is not None
        and fallback_effect <= -minimum_effect,
    }
    return {
        "level": "behavioral_generalization",
        "task": "solvability",
        "analysis_role": "confirmatory_gate",
        "data_scope": ["train", "validation"],
        "test_episodes_inspected": False,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "development_episodes": len(selected),
        "development_states": len(records),
        "semantic_persistence_choice_rate": persistence_rate,
        "episodes_ending_by_disengagement_rate": disengagement_rate,
        "mean_episode_decisions": mean_decisions,
        "persistence_probability": {
            "mean": statistics.mean(probabilities),
            "standard_deviation": statistics.pstdev(probabilities),
            "p10": p10,
            "p90": p90,
            "interdecile_range": p90 - p10,
        },
        "persistence_logit_standard_deviation": statistics.pstdev(logits),
        "initial_probability_by_mapping": mapping_means,
        "initial_mapping_probability_gap": mapping_gap,
        "manipulation_logit_effects": {
            "higher_progress_evidence_minus_lower": progress_effect,
            "higher_attempt_cost_minus_lower": cost_effect,
            "higher_give_up_value_minus_lower": fallback_effect,
        },
        "thresholds": dict(thresholds),
    }


def _load_summary(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        return source
    with Path(source).open(encoding="utf-8") as handle:
        return json.load(handle)


def require_behavioral_clearance(source: str | Path | dict) -> dict:
    summary = _load_summary(source)
    if summary.get("passed") is not True or summary.get("test_episodes_inspected") is not False:
        raise RuntimeError(
            "Track B behavioral validation did not pass on development episodes only"
        )
    return summary


def require_representational_clearance(source: str | Path | dict) -> dict:
    summary = _load_summary(source)
    classification = summary.get("classification")
    if classification not in {"strong_transfer", "partial_transfer"}:
        raise RuntimeError(
            "causal Track B is gated on strong or partial representational transfer; "
            f"observed {classification!r}"
        )
    return summary
