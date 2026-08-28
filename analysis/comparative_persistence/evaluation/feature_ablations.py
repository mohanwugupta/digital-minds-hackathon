"""Shared motivational-ingredient ablations and family-only models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..hazard_models.baselines import fit_ridge_logistic, predict_ridge_logistic
from ..hazard_models.modeling import _sharing_design, task_balanced_weights
from ..semantic_features import HISTORY_FEATURES, build_feature_matrix
from .metrics import summarize_predictions, task_metrics


FEATURE_FAMILIES = {
    "history": HISTORY_FEATURES,
    "time_effort": ("time_norm", "effort_norm", "invested_norm"),
    "cost": ("cost_norm", "remaining_effort_norm"),
    "progress": ("progress_norm",),
    "outside_option": ("outside_norm",),
    "prospective_value": (
        "success_evidence",
        "remaining_time_norm",
        "continue_payoff_norm",
        "futility_norm",
    ),
}


def _fit_features(train, validation, test, features, config):
    known = sorted(train.task.unique())
    multiplier = np.sqrt(float(config.get("hierarchical_deviation_penalty", 2.0)))

    def design(frame):
        values, _names = build_feature_matrix(frame.reset_index(drop=True), features)
        return _sharing_design(values, frame.task, "hierarchical", known, multiplier)

    x_train, penalty_weights = design(train)
    x_validation, _ = design(validation)
    candidates = []
    for penalty in config["ridge_penalties"]:
        coefficient = fit_ridge_logistic(
            x_train,
            train.hazard_event,
            penalty=float(penalty),
            weights=task_balanced_weights(train),
            penalty_weights=penalty_weights,
        )
        probability = predict_ridge_logistic(coefficient, x_validation)
        score = summarize_predictions(
            pd.DataFrame(
                {"task": validation.task, "observed": validation.hazard_event, "predicted": probability}
            )
        )["macro_log_loss"]
        candidates.append((score, float(penalty)))
    _loss, selected = min(candidates)
    refit = pd.concat((train, validation), ignore_index=True)
    x_refit_values, _ = build_feature_matrix(refit, features)
    x_refit, refit_penalties = _sharing_design(
        x_refit_values, refit.task, "hierarchical", known, multiplier
    )
    x_test, _ = design(test)
    coefficient = fit_ridge_logistic(
        x_refit,
        refit.hazard_event,
        penalty=selected,
        weights=task_balanced_weights(refit),
        penalty_weights=refit_penalties,
    )
    probability = predict_ridge_logistic(coefficient, x_test)
    scored = pd.DataFrame(
        {
            "task": test.task,
            "episode_id": test.episode_id,
            "observed": test.hazard_event,
            "predicted": probability,
        }
    )
    return summarize_predictions(scored), task_metrics(scored), selected


def run_feature_ablations(records, config, *, logger=None):
    records = records[records.is_persistence_task.astype(bool)]
    train = records[records.split == "train"]
    validation = records[records.split == "validation"]
    test = records[records.split == "test"]
    full_features = tuple(
        feature for family in FEATURE_FAMILIES.values() for feature in family
    )
    full, full_taskwise, _penalty = _fit_features(
        train, validation, test, full_features, config
    )
    ablations, only, support = [], [], []
    for family, family_features in FEATURE_FAMILIES.items():
        remaining = tuple(value for value in full_features if value not in family_features)
        ablated, taskwise, penalty = _fit_features(
            train, validation, test, remaining, config
        )
        ablations.append(
            {
                "feature_family": family,
                "full_macro_log_loss": full["macro_log_loss"],
                "ablated_macro_log_loss": ablated["macro_log_loss"],
                "delta_log_loss": ablated["macro_log_loss"] - full["macro_log_loss"],
                "selected_penalty": penalty,
            }
        )
        family_only, only_taskwise, only_penalty = _fit_features(
            train, validation, test, family_features, config
        )
        only.append(
            {
                "feature_family": family,
                "macro_log_loss": family_only["macro_log_loss"],
                "macro_brier": family_only["macro_brier"],
                "selected_penalty": only_penalty,
            }
        )
        merged = full_taskwise[["task", "log_loss"]].merge(
            taskwise[["task", "log_loss"]], on="task", suffixes=("_full", "_ablated")
        )
        for row in merged.itertuples():
            delta = row.log_loss_ablated - row.log_loss_full
            support.append(
                {
                    "feature_family": family,
                    "task": row.task,
                    "delta_log_loss": delta,
                    "supported": bool(delta > 0.002),
                }
            )
        if logger is not None:
            logger.note("feature_ablation", f"completed {family}")
    support_frame = pd.DataFrame(support)
    classification = []
    for family, part in support_frame.groupby("feature_family"):
        positive = int(part.supported.sum())
        if positive >= max(2, int(np.ceil(0.6 * len(part)))):
            label = "broadly_shared"
        elif positive >= 2:
            label = "family_specific"
        elif positive == 1:
            label = "task_specific"
        else:
            label = "unsupported"
        classification.extend(
            [{**row, "support_class": label} for row in part.to_dict("records")]
        )
    return pd.DataFrame(ablations), pd.DataFrame(only), pd.DataFrame(classification)
