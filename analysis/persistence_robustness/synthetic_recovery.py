"""Narrow PRD 2.5 recovery checks for sharing, memory, and matched control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from analysis.comparative_persistence.hazard_models.baselines import (
    binary_log_loss,
    fit_ridge_logistic,
    predict_ridge_logistic,
)
from analysis.comparative_persistence.synthetic.generators import (
    generate_latent_data,
    generate_sharing_data,
)
from analysis.comparative_persistence.synthetic.recovery import (
    recover_latent_rho,
    recover_sharing,
)


def generate_history_specificity(*, persistent_strength, independent_strength, episodes=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for framing, strength in (
        ("persistent_goal", persistent_strength),
        ("independent_goals", independent_strength),
    ):
        for episode in range(int(episodes)):
            current = rng.normal()
            history = rng.normal()
            probability = expit(-0.2 + 0.8 * current + float(strength) * history)
            rows.append(
                {
                    "framing": framing,
                    "episode_id": f"{framing}-{episode}",
                    "split": (
                        "train"
                        if episode < 0.6 * episodes
                        else "validation"
                        if episode < 0.8 * episodes
                        else "test"
                    ),
                    "current": current,
                    "history": history,
                    "hazard_event": int(rng.random() < probability),
                }
            )
    return pd.DataFrame(rows)


def recover_delta_history_gain(frame):
    gains = {}
    for framing, local in frame.groupby("framing"):
        train = local[local.split != "test"]
        test = local[local.split == "test"]
        current = fit_ridge_logistic(
            train[["current"]], train.hazard_event, penalty=0.05
        )
        history = fit_ridge_logistic(
            train[["current", "history"]], train.hazard_event, penalty=0.05
        )
        current_loss = binary_log_loss(
            test.hazard_event,
            predict_ridge_logistic(current, test[["current"]]),
        )
        history_loss = binary_log_loss(
            test.hazard_event,
            predict_ridge_logistic(history, test[["current", "history"]]),
        )
        gains[framing] = current_loss - history_loss
    return float(gains["persistent_goal"] - gains["independent_goals"])


def run_robustness_recovery(config, *, smoke=False):
    repetitions = 4 if smoke else int(config["synthetic_recovery"]["repetitions"])
    episodes = 160 if smoke else int(config["synthetic_recovery"]["episodes_per_task"])
    rows = []
    for repetition in range(repetitions):
        seed = int(config["base_seed"]) + repetition * 31
        shared = recover_sharing(
            generate_sharing_data("shared", episodes_per_task=episodes, seed=seed)
        )
        specific = recover_sharing(
            generate_sharing_data(
                "task_specific", episodes_per_task=episodes, seed=seed + 1
            )
        )
        rho = recover_latent_rho(
            generate_latent_data(rho=0.9, episodes=episodes, seed=seed + 2),
            candidates=(0.0, 0.5, 0.9),
        )
        generic_delta = recover_delta_history_gain(
            generate_history_specificity(
                persistent_strength=1.5,
                independent_strength=1.5,
                episodes=episodes,
                seed=seed + 3,
            )
        )
        specific_delta = recover_delta_history_gain(
            generate_history_specificity(
                persistent_strength=2.2,
                independent_strength=0.0,
                episodes=episodes,
                seed=seed + 4,
            )
        )
        rows.extend(
            (
                {
                    "repetition": repetition,
                    "generator": "shared_history_task_state",
                    "recovered": shared in {"fully_shared", "hierarchical"},
                    "estimate": shared,
                },
                {
                    "repetition": repetition,
                    "generator": "task_specific_history",
                    "recovered": specific == "task_specific",
                    "estimate": specific,
                },
                {
                    "repetition": repetition,
                    "generator": "recurrent_latent_state",
                    "recovered": rho >= 0.5,
                    "estimate": rho,
                },
                {
                    "repetition": repetition,
                    "generator": "generic_sequential_history",
                    # Approximate null recovery: finite simulated samples need
                    # a tolerance, while the persistence-specific generator
                    # must still clear a positive separation threshold.
                    "recovered": abs(generic_delta) <= 0.12,
                    "estimate": generic_delta,
                },
                {
                    "repetition": repetition,
                    "generator": "persistence_specific_history",
                    "recovered": specific_delta > 0.03,
                    "estimate": specific_delta,
                },
            )
        )
    recovery = pd.DataFrame(rows)
    summary = (
        recovery.groupby("generator", as_index=False)
        .agg(recovery_rate=("recovered", "mean"), mean_estimate=("estimate", lambda values: pd.to_numeric(values, errors="coerce").mean()))
    )
    return recovery, summary
