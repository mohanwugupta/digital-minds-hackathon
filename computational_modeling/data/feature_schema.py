"""Explicit information-set registry for cross-task behavioral models.

The registry is deliberately allow-list based.  Model code may request only
named features from this module; it must never discover numeric dataframe
columns automatically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


COMMON_OBSERVABLE = (
    "task_bandit",
    "task_foraging",
    "task_solvability",
    "round",
    "log_round",
    "normalized_time",
    "previous_outcome",
    "failure_streak",
    "success_streak",
    "previous_choice",
    "second_previous_choice",
    "action_lag_1",
    "action_lag_2",
    "action_lag_3",
    "action_lag_5",
    "outcome_lag_1",
    "outcome_lag_2",
    "outcome_lag_3",
    "outcome_lag_5",
    "cumulative_progress",
    "estimated_continue_value",
    "estimated_outside_value",
    "cost_pressure",
    "progress_evidence",
    "termination_advantage",
    "relative_value",
    "disengagement_evidence",
)

FEATURE_SCHEMA = {
    "bandit": {
        "observable": list(
            COMMON_OBSERVABLE
            + (
                "cumulative_score",
                "rw_a",
                "rw_b",
                "rw_best",
                "rw_gap",
                "bayes_a",
                "bayes_b",
                "bayes_best",
                "bayes_gap",
            )
        ),
        "oracle": [
            "oracle_p_a",
            "oracle_p_b",
            "oracle_continue_value",
            "oracle_outside_value",
            "oracle_termination_advantage",
            "oracle_relative_value",
        ],
    },
    "foraging": {
        "observable": list(
            COMMON_OBSERVABLE
            + (
                "outside_option",
                "stay_cost",
                "search_count",
                "cumulative_score",
                "mvt_like_advantage",
                "bayes_patch_probability",
            )
        ),
        "oracle": [
            "oracle_initial_quality",
            "oracle_depletion",
            "oracle_patch_probability",
            "oracle_continue_value",
            "oracle_outside_value",
            "oracle_termination_advantage",
            "oracle_relative_value",
        ],
    },
    "solvability": {
        "observable": list(
            COMMON_OBSERVABLE
            + (
                "attempt_cost",
                "give_up_value",
                "attempts_used",
                "max_attempts",
                "cumulative_cost",
                "progress_count",
                "displayed_progress_cue",
                "bayes_progress_probability",
            )
        ),
        "oracle": [
            "oracle_progress_probability",
            "oracle_continue_value",
            "oracle_outside_value",
            "oracle_termination_advantage",
            "oracle_relative_value",
        ],
    },
}

FEATURE_DESCRIPTIONS = {
    "oracle_p_a": "Experimenter-known Bandit arm-A success probability; absent from prompts.",
    "oracle_p_b": "Experimenter-known Bandit arm-B success probability; absent from prompts.",
    "oracle_patch_probability": "Private current Foraging food probability.",
    "oracle_initial_quality": "Private exact initial Foraging patch quality.",
    "oracle_depletion": "Private exact Foraging depletion rate.",
    "oracle_progress_probability": "Private exact Solvability progress probability; prompts expose only a categorical cue.",
    "termination_advantage": "Observable estimated continuation value minus displayed outside value.",
    "disengagement_evidence": "Negative observable termination advantage; higher favors disengagement.",
    "relative_value": "Observable estimated generic continuation value relative to the outside option.",
    "displayed_progress_cue": "Weak=-1, mixed=0, strong=1, matching the text shown in the prompt.",
}


# Conceptual families for the pre-L21/L22 flexible-model follow-up.  These are
# explicit for the same reason as FEATURE_SCHEMA: an analysis must not infer
# scientific groups from column names or dataframe dtypes.  Task indicators are
# nuisance controls retained in every ablation and group-only model.
FLEXIBLE_NUISANCE_FEATURES = (
    "task_bandit",
    "task_foraging",
    "task_solvability",
)

FLEXIBLE_FEATURE_GROUPS = {
    "history": (
        "previous_outcome",
        "failure_streak",
        "success_streak",
        "previous_choice",
        "second_previous_choice",
        "action_lag_1",
        "action_lag_2",
        "action_lag_3",
        "action_lag_5",
        "outcome_lag_1",
        "outcome_lag_2",
        "outcome_lag_3",
        "outcome_lag_5",
    ),
    "time_effort": (
        "log_round",
        "normalized_time",
    ),
    "continuation_value": ("estimated_continue_value",),
    "outside_option": ("estimated_outside_value",),
    "cost": ("cost_pressure",),
    "progress_solvability": (
        "cumulative_progress",
        "progress_evidence",
    ),
    "derived_termination": ("termination_advantage",),
}

FEATURE_GROUP_DESCRIPTIONS = {
    "history": "Past choices, past outcomes, action/outcome lags, and success/failure streaks.",
    "time_effort": "Round and normalized elapsed effort.",
    "continuation_value": "Past-derived estimate of the value of continuing.",
    "outside_option": "Displayed or otherwise observable outside option.",
    "cost": "Harmonized task-specific displayed action cost.",
    "progress_solvability": "Cumulative progress and task-appropriate progress evidence.",
    "derived_termination": "Observable termination advantage (continuation minus outside option).",
}

FEATURE_GROUP_EXCLUDED_ALIASES = {
    "relative_value": "Exact alias of termination_advantage in the exported records.",
    "disengagement_evidence": "Negative of termination_advantage; excluded to avoid duplicate predictors.",
}


def flexible_features() -> tuple[str, ...]:
    """Return the preregistered full observable flexible-linear design."""

    return FLEXIBLE_NUISANCE_FEATURES + tuple(
        feature
        for group in FLEXIBLE_FEATURE_GROUPS.values()
        for feature in group
    )


def validate_feature_groups() -> None:
    """Fail if a feature is duplicated or is not observable in every task."""

    grouped = [
        feature
        for group in FLEXIBLE_FEATURE_GROUPS.values()
        for feature in group
    ]
    all_features = list(FLEXIBLE_NUISANCE_FEATURES) + grouped
    duplicates = sorted(
        {feature for feature in all_features if all_features.count(feature) > 1}
    )
    if duplicates:
        raise ValueError(f"flexible feature groups overlap: {duplicates}")
    for task, schema in FEATURE_SCHEMA.items():
        missing = sorted(set(all_features) - set(schema["observable"]))
        if missing:
            raise ValueError(
                f"flexible features are not observable for {task}: {missing}"
            )


def features_for(task: str, information_set: str = "observable") -> list[str]:
    """Return the explicit allow-list for a task and information set."""

    task = str(task)
    if task not in FEATURE_SCHEMA:
        raise ValueError(f"unsupported computational-modeling task: {task!r}")
    if information_set == "observable":
        return list(FEATURE_SCHEMA[task]["observable"])
    if information_set == "oracle":
        return list(FEATURE_SCHEMA[task]["observable"] + FEATURE_SCHEMA[task]["oracle"])
    raise ValueError("information_set must be 'observable' or 'oracle'")


def select_features(
    records: Sequence[Mapping],
    task: str,
    information_set: str,
    requested: Sequence[str],
) -> list[list[float]]:
    """Select only explicitly permitted numeric features."""

    permitted = set(features_for(task, information_set))
    forbidden = sorted(set(requested) - permitted)
    if forbidden:
        raise ValueError(
            f"features are not permitted for {information_set} {task} models: {forbidden}"
        )
    output = []
    for index, record in enumerate(records):
        missing = [name for name in requested if name not in record]
        if missing:
            raise ValueError(f"record {index} is missing requested features: {missing}")
        output.append([float(record[name]) for name in requested])
    return output


def serialized_schema() -> dict:
    validate_feature_groups()
    return {
        "tasks": FEATURE_SCHEMA,
        "descriptions": FEATURE_DESCRIPTIONS,
        "flexible_model": {
            "nuisance_features": list(FLEXIBLE_NUISANCE_FEATURES),
            "feature_groups": {
                name: list(features)
                for name, features in FLEXIBLE_FEATURE_GROUPS.items()
            },
            "feature_group_descriptions": FEATURE_GROUP_DESCRIPTIONS,
            "excluded_aliases": FEATURE_GROUP_EXCLUDED_ALIASES,
        },
        "policy": {
            "observable": "Only prompt/history-observable fields and past-derived estimates.",
            "oracle": "Observable fields plus explicitly prefixed environment-private fields.",
        },
    }
