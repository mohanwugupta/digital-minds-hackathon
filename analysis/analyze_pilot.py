"""Behavioral diagnostics that gate probe-state collection."""

import argparse
import json
import os

from bandit.environment import condition_class, expected_pull_reward


def summarize(frame) -> dict:
    import numpy as np

    choices = frame["sampled_action"].value_counts(normalize=True)
    after_loss = frame[frame["previous_outcome"] == -2]
    after_gain = frame[frame["previous_outcome"] == 3]
    result = {
        "episodes": int(frame["episode_id"].nunique()),
        "decision_states": int(len(frame)),
        "action_frequencies": {label: float(choices.get(label, 0.0)) for label in "ABC"},
        "mean_p_stop": float(frame["p_stop"].mean()),
        "mean_p_continue_after_loss": float(after_loss["p_continue"].mean()) if len(after_loss) else None,
        "mean_p_continue_after_gain": float(after_gain["p_continue"].mean()) if len(after_gain) else None,
        "arm_preference_variation": float((frame["p_A"] - frame["p_B"]).std()),
        "sampled_actions_valid": bool(frame["sampled_action"].isin(list("ABC")).all()),
        "top_token_action_rate": float(frame["top_token_is_action"].mean())
        if "top_token_is_action" in frame else None,
        "mean_raw_action_probability_mass": float(frame["p_action_mass_raw"].mean())
        if "p_action_mass_raw" in frame else None,
    }
    loss_streaks = []
    for value in frame["reward_history"]:
        history = json.loads(value) if isinstance(value, str) else list(value)
        streak = 0
        for reward in reversed(history):
            if reward != -2:
                break
            streak += 1
        loss_streaks.append(streak)
    result["loss_streak_stop_correlation"] = float(
        np.corrcoef(loss_streaks, frame["p_stop"])[0, 1]
    ) if len(set(loss_streaks)) > 1 else None
    result["repeated_losses_increase_stopping"] = bool(
        result["loss_streak_stop_correlation"] is not None
        and result["loss_streak_stop_correlation"] > 0
    )
    result["stop_not_floor_or_ceiling"] = 0.01 < result["mean_p_stop"] < 0.99
    result["all_actions_observed"] = all(result["action_frequencies"][label] > 0 for label in "ABC")
    result["favorable_history_increases_persistence"] = bool(
        result["mean_p_continue_after_loss"] is not None
        and result["mean_p_continue_after_gain"] is not None
        and result["mean_p_continue_after_gain"] > result["mean_p_continue_after_loss"]
    )
    frame = frame.copy()
    frame["condition_class"] = [
        condition_class(p_a, p_b)
        for p_a, p_b in zip(frame["p_A_true"], frame["p_B_true"])
    ]
    result["condition_diagnostics"] = {}
    for label, group in frame.groupby("condition_class"):
        result["condition_diagnostics"][label] = {
            "decision_states": int(len(group)),
            "episodes": int(group["episode_id"].nunique()),
            "mean_p_stop": float(group["p_stop"].mean()),
            "mean_p_continue": float(group["p_continue"].mean()),
            "near_boundary_fraction": float(group["p_stop"].between(0.2, 0.8).mean()),
        }
    result["arm_expected_values"] = {
        str(probability): expected_pull_reward(probability)
        for probability in sorted(set(frame["p_A_true"]) | set(frame["p_B_true"]))
    }
    return result


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/bandit_pilot.csv")
    parser.add_argument("--output", default="artifacts/bandit_pilot_analysis.json")
    args = parser.parse_args()
    result = summarize(pd.read_csv(args.input))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
