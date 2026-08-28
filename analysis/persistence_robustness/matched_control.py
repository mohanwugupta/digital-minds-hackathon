"""History decomposition for exactly yoked goal-continuity states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.comparative_persistence.hazard_models.baselines import (
    binary_log_loss,
    fit_ridge_logistic,
    predict_ridge_logistic,
)
from analysis.comparative_persistence.semantic_features import build_feature_matrix
from experiments.persistence_battery.storage import read_records_frame


CURRENT_FEATURES = (
    "step_norm",
    "cost_norm",
    "outside_norm",
    "success_probability",
    "reward_norm",
    "secondary_version",
)
ACTION_FEATURES = (
    "continue_streak",
    *(f"action_lag_{lag}" for lag in (1, 2, 3, 4, 5)),
)
OUTCOME_FEATURES = (
    "success_streak",
    "failure_streak",
    *(f"outcome_lag_{lag}" for lag in (1, 2, 3, 4, 5)),
)
MODEL_FEATURES = {
    "current_state": CURRENT_FEATURES,
    "action_only": (*CURRENT_FEATURES, *ACTION_FEATURES),
    "outcome_only": (*CURRENT_FEATURES, *OUTCOME_FEATURES),
    "joint_history": (*CURRENT_FEATURES, *ACTION_FEATURES, *OUTCOME_FEATURES),
}


def load_matched_records(run_root):
    return read_records_frame(Path(run_root) / "matched_control", "paired_records")


def add_matched_history(frame):
    output = []
    for _episode, episode in frame.groupby("episode_id", sort=False):
        actions, outcomes = [], []
        continue_streak = 0
        for _, source in episode.sort_values("step").iterrows():
            row = source.to_dict()
            row["step_norm"] = float(row["step"]) / float(row["horizon"])
            row["cost_norm"] = float(row["effort_cost"]) / max(
                1.0, float(row["reward_magnitude"])
            )
            row["outside_norm"] = float(row["outside_option"]) / max(
                1.0, float(row["reward_magnitude"])
            )
            row["reward_norm"] = 1.0
            row["secondary_version"] = float(
                row["version"] == "advancing_secondary"
            )
            row["continue_streak"] = continue_streak
            for lag in (1, 2, 3, 4, 5):
                row[f"action_lag_{lag}"] = (
                    actions[-lag] if len(actions) >= lag else float("nan")
                )
                row[f"outcome_lag_{lag}"] = (
                    outcomes[-lag] if len(outcomes) >= lag else float("nan")
                )
            output.append(row)
            action = float(row["history_action"] == "ENGAGE")
            outcome = row.get("subsequent_outcome")
            outcome = 0.0 if pd.isna(outcome) else float(outcome) / max(
                1.0, float(row["reward_magnitude"])
            )
            actions.append(action)
            outcomes.append(outcome)
            continue_streak = continue_streak + 1 if action else 0
    return pd.DataFrame(output)


def _fit(train, validation, test, features, penalties):
    x_train, names = build_feature_matrix(train, features)
    x_validation, validation_names = build_feature_matrix(validation, features)
    x_test, test_names = build_feature_matrix(test, features)
    if names != validation_names or names != test_names:
        raise RuntimeError("matched-control features changed across splits")
    candidates = []
    for penalty in penalties:
        coefficient = fit_ridge_logistic(
            x_train,
            train.behavior_target_probability,
            penalty=float(penalty),
        )
        probability = predict_ridge_logistic(coefficient, x_validation)
        candidates.append(
            (
                binary_log_loss(
                    validation.behavior_target_probability, probability
                ),
                float(penalty),
            )
        )
    _validation_loss, selected = min(candidates)
    refit = pd.concat((train, validation), ignore_index=True)
    x_refit, _ = build_feature_matrix(refit, features)
    coefficient = fit_ridge_logistic(
        x_refit,
        refit.behavior_target_probability,
        penalty=selected,
    )
    prediction = predict_ridge_logistic(coefficient, x_test)
    return {
        "loss": binary_log_loss(test.behavior_target_probability, prediction),
        "prediction": prediction,
        "coefficient": coefficient,
        "feature_names": names,
        "selected_penalty": selected,
        "test": test.reset_index(drop=True),
    }


def _value_coefficients(fit, prefix):
    return {
        name: float(fit["coefficient"][index + 1])
        for index, name in enumerate(fit["feature_names"])
        if name.startswith(prefix) and not name.endswith("__present")
    }


def _cluster_interval(scored, *, samples, seed):
    rng = np.random.default_rng(seed)
    clusters = list(scored.groupby("latent_sequence_id", sort=False))
    estimates = []
    for _ in range(int(samples)):
        draw = rng.integers(0, len(clusters), size=len(clusters))
        local = pd.concat([clusters[index][1] for index in draw], ignore_index=True)
        gain_p = local[local.framing == "persistent_goal"].gain.mean()
        gain_i = local[local.framing == "independent_goals"].gain.mean()
        estimates.append(float(gain_p - gain_i))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def run_matched_control_analysis(frame, config, *, smoke=False, logger=None):
    frame = add_matched_history(frame)
    penalties = config["smoke"]["ridge_penalties"] if smoke else config["ridge_penalties"]
    rows, kernels, predictions = [], [], []
    fits = {}
    for version in sorted(frame.version.unique()):
        for framing in ("persistent_goal", "independent_goals"):
            local = frame[(frame.version == version) & (frame.framing == framing)]
            train = local[local.split == "train"]
            validation = local[local.split == "validation"]
            test = local[local.split == "test"]
            for model, features in MODEL_FEATURES.items():
                fit = _fit(train, validation, test, features, penalties)
                fits[(version, framing, model)] = fit
                rows.append(
                    {
                        "version": version,
                        "framing": framing,
                        "model": model,
                        "log_loss": fit["loss"],
                        "selected_penalty": fit["selected_penalty"],
                        "states": len(test),
                        "semantic_pairs": int(test.latent_sequence_id.nunique()),
                    }
                )
                prediction = fit["test"][[
                    "state_id",
                    "latent_sequence_id",
                    "framing",
                    "version",
                ]].copy()
                prediction["model"] = model
                prediction["observed_probability"] = fit["test"].behavior_target_probability
                prediction["predicted_probability"] = fit["prediction"]
                predictions.append(prediction)
            joint = fits[(version, framing, "joint_history")]
            action = _value_coefficients(joint, "action_lag_")
            outcome = _value_coefficients(joint, "outcome_lag_")
            kernels.append(
                {
                    "version": version,
                    "framing": framing,
                    "action_kernel": json.dumps(action, sort_keys=True),
                    "outcome_kernel": json.dumps(outcome, sort_keys=True),
                    "previous_choice_sensitivity": action.get("action_lag_1", float("nan")),
                    "previous_outcome_sensitivity": outcome.get("outcome_lag_1", float("nan")),
                    "continue_streak_sensitivity": _value_coefficients(joint, "continue_streak").get("continue_streak", float("nan")),
                }
            )
            if logger is not None:
                logger.note("matched_control", f"completed {version}/{framing}")

    losses = pd.DataFrame(rows)
    gains = []
    for (version, framing), part in losses.groupby(["version", "framing"]):
        current = float(part[part.model == "current_state"].log_loss.iloc[0])
        for model in ("action_only", "outcome_only", "joint_history"):
            loss = float(part[part.model == model].log_loss.iloc[0])
            gains.append(
                {
                    "version": version,
                    "framing": framing,
                    "model": model,
                    "current_state_log_loss": current,
                    "history_log_loss": loss,
                    "history_gain": current - loss,
                }
            )
    gain_frame = pd.DataFrame(gains)
    for (version, model), part in gain_frame.groupby(["version", "model"]):
        persistent = float(part[part.framing == "persistent_goal"].history_gain.iloc[0])
        independent = float(part[part.framing == "independent_goals"].history_gain.iloc[0])
        mask = (gain_frame.version == version) & (gain_frame.model == model)
        gain_frame.loc[mask, "delta_history_gain"] = persistent - independent

    kernel_frame = pd.DataFrame(kernels)
    for version, part in kernel_frame.groupby("version"):
        left = part[part.framing == "persistent_goal"].iloc[0]
        right = part[part.framing == "independent_goals"].iloc[0]
        for kind in ("action", "outcome"):
            a = np.asarray(list(json.loads(left[f"{kind}_kernel"]).values()), dtype=float)
            b = np.asarray(list(json.loads(right[f"{kind}_kernel"]).values()), dtype=float)
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            cosine = float(a @ b / denominator) if denominator > 0 else float("nan")
            kernel_frame.loc[kernel_frame.version == version, f"{kind}_kernel_cosine"] = cosine

    # Clustered uncertainty for the joint-history matched contrast uses fixed
    # held-out predictions and resamples complete latent sequences.
    prediction_frame = pd.concat(predictions, ignore_index=True)
    joint = prediction_frame[prediction_frame.model.isin(["current_state", "joint_history"])]
    for version in joint.version.unique():
        local = joint[joint.version == version]
        current = local[local.model == "current_state"][
            [
                "state_id",
                "latent_sequence_id",
                "framing",
                "observed_probability",
                "predicted_probability",
            ]
        ].rename(columns={"predicted_probability": "current_prediction"})
        history = local[local.model == "joint_history"][
            ["state_id", "predicted_probability"]
        ].rename(columns={"predicted_probability": "history_prediction"})
        wide = current.merge(history, on="state_id", validate="one_to_one")
        scored_rows = []
        for row in wide.itertuples(index=False):
            # Per-state soft cross entropy difference; averaging states inside
            # a resampled latent sequence preserves matched histories.
            observed = float(row.observed_probability)
            current_p = float(row.current_prediction)
            history_p = float(row.history_prediction)
            current_loss = binary_log_loss([observed], [current_p])
            history_loss = binary_log_loss([observed], [history_p])
            scored_rows.append(
                {
                    "latent_sequence_id": row.latent_sequence_id,
                    "framing": row.framing,
                    "gain": current_loss - history_loss,
                }
            )
        low, high = _cluster_interval(
            pd.DataFrame(scored_rows),
            samples=(
                config["smoke"]["bootstrap_samples"]
                if smoke
                else config["bootstrap"]["episode_samples"]
            ),
            seed=int(config["base_seed"]) + 5,
        )
        mask = (gain_frame.version == version) & (gain_frame.model == "joint_history")
        gain_frame.loc[mask, "delta_history_gain_ci_low"] = low
        gain_frame.loc[mask, "delta_history_gain_ci_high"] = high
        gain_frame.loc[mask, "bootstrap_unit"] = "latent_sequence"
    return losses, gain_frame, kernel_frame, prediction_frame
