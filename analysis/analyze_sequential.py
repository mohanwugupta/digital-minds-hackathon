"""Summarize complete-episode persistence outcomes by steering condition."""

import argparse
import glob
import json
import os


def episode_summary(frame):
    import pandas as pd

    rows = []
    for episode_id, episode in frame.groupby("episode_id", sort=False):
        episode = episode.sort_values("round")
        choices = episode["sampled_action"].tolist()
        rewards = episode["subsequent_reward"].fillna(0).tolist()
        losses_before_stop = sum(reward == -2 for reward in rewards[:-1])
        switches = sum(
            left in "AB" and right in "AB" and left != right
            for left, right in zip(choices, choices[1:])
        )
        rows.append({
            "episode_id": episode_id,
            "alpha": float(episode["alpha"].iloc[0]),
            "decisions_before_stop": len(episode) - int(choices[-1] == "C"),
            "stopped": choices[-1] == "C",
            "cumulative_reward": sum(rewards),
            "losses_tolerated": losses_before_stop,
            "arm_switches": switches,
        })
    return pd.DataFrame(rows)


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="artifacts/sequential")
    parser.add_argument("--output", default="artifacts/sequential_analysis.json")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.input_dir, "alpha_*.csv")))
    if not paths:
        raise FileNotFoundError("no sequential CSV files found")
    episodes = episode_summary(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    summaries = {}
    for alpha, group in episodes.groupby("alpha"):
        summaries[str(alpha)] = {
            "episodes": int(len(group)),
            "mean_decisions_before_stop": float(group["decisions_before_stop"].mean()),
            "stop_rate": float(group["stopped"].mean()),
            "mean_cumulative_reward": float(group["cumulative_reward"].mean()),
            "mean_losses_tolerated": float(group["losses_tolerated"].mean()),
            "mean_arm_switches": float(group["arm_switches"].mean()),
        }
    result = {"conditions": summaries}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
