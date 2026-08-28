"""Clustered bootstrap intervals for within- and cross-task conclusions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..hazard_models.baselines import binary_log_loss


def add_task_bootstrap_intervals(summary, taskwise, *, group_columns, value, samples, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for keys, part in taskwise.groupby(list(group_columns)):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        tasks = sorted(part.iloc[:, part.columns.get_loc("task")].unique())
        values = part.set_index("task")[value]
        estimates = []
        for _ in range(int(samples)):
            draw = rng.choice(tasks, size=len(tasks), replace=True)
            estimates.append(float(values.loc[draw].mean()))
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                f"{value}_ci_low": float(np.quantile(estimates, 0.025)),
                f"{value}_ci_high": float(np.quantile(estimates, 0.975)),
            }
        )
    return summary.merge(pd.DataFrame(rows), on=list(group_columns), how="left")


def add_episode_bootstrap_intervals(taskwise, predictions, *, samples, seed):
    """Attach held-out log-loss intervals clustered by complete episode.

    States from one episode are dependent, so the episode—not the individual
    state—is the within-task resampling unit required by the frozen protocol.
    """

    required = {
        "model",
        "sharing",
        "task",
        "episode_id",
        "observed",
        "predicted",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"episode bootstrap is missing columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows = []
    for keys, part in predictions.groupby(["model", "sharing", "task"], sort=False):
        clusters = list(part.groupby("episode_id", sort=False))
        estimates = []
        for _ in range(int(samples)):
            drawn = rng.integers(0, len(clusters), size=len(clusters))
            observed = np.concatenate(
                [clusters[index][1].observed.to_numpy(dtype=float) for index in drawn]
            )
            predicted = np.concatenate(
                [clusters[index][1].predicted.to_numpy(dtype=float) for index in drawn]
            )
            estimates.append(binary_log_loss(observed, predicted))
        rows.append(
            {
                "model": keys[0],
                "sharing": keys[1],
                "task": keys[2],
                "bootstrap_unit": "episode",
                "log_loss_ci_low": float(np.quantile(estimates, 0.025)),
                "log_loss_ci_high": float(np.quantile(estimates, 0.975)),
            }
        )
    return taskwise.merge(
        pd.DataFrame(rows), on=["model", "sharing", "task"], how="left"
    )
