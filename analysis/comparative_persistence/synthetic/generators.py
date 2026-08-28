"""Synthetic H1/H2/H3/H4 datasets at realistic absorbing hazards."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def _split(index, count):
    fraction = index / count
    return "train" if fraction < 0.6 else "validation" if fraction < 0.8 else "test"


def generate_sharing_data(mode, *, episodes_per_task=120, seed=0):
    if mode not in {"shared", "task_specific"}:
        raise ValueError("mode must be shared or task_specific")
    rng = np.random.default_rng(seed)
    tasks = ("task_a", "task_b", "task_c")
    shared_beta = np.asarray((2.4, -1.6))
    task_beta = (
        np.asarray((3.5, 0.0)),
        np.asarray((-3.5, 0.0)),
        np.asarray((0.0, 3.5)),
    )
    rows = []
    for task_index, task in enumerate(tasks):
        for episode in range(int(episodes_per_task)):
            x = rng.normal(size=2)
            beta = shared_beta if mode == "shared" else task_beta[task_index]
            event = int(rng.random() < expit(-0.3 + x @ beta))
            rows.append(
                {
                    "task": task,
                    "episode_id": f"{task}-{episode}",
                    "pair_id": f"{task}-{episode}",
                    "state_id": f"{task}-{episode}:0",
                    "split": _split(episode, episodes_per_task),
                    "x1": x[0],
                    "x2": x[1],
                    "hazard_event": event,
                }
            )
    return pd.DataFrame(rows)


def generate_history_data(*, lag=5, episodes=240, decisions=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(int(episodes)):
        outcomes = list(rng.choice((-1.0, 1.0), size=int(decisions) + 8))
        for step in range(int(decisions)):
            history_value = outcomes[step + 8 - int(lag)]
            probability = expit(-3.0 + 2.8 * history_value)
            event = int(rng.random() < probability)
            row = {
                "task": "history_task",
                "episode_id": f"history-{episode}",
                "pair_id": f"history-{episode}",
                "state_id": f"history-{episode}:{step}",
                "round": step,
                "split": _split(episode, episodes),
                "hazard_event": event,
            }
            for candidate in (1, 2, 3, 5, 8):
                row[f"outcome_lag_{candidate}"] = outcomes[step + 8 - candidate]
            rows.append(row)
            if event:
                break
    return pd.DataFrame(rows)


def generate_latent_data(*, rho, episodes=240, decisions=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(int(episodes)):
        latent = 0.0
        for step in range(int(decisions)):
            drive = rng.normal()
            latent = float(rho) * latent + drive
            event = int(rng.random() < expit(-3.2 + 1.8 * latent))
            rows.append(
                {
                    "task": "latent_task",
                    "episode_id": f"latent-{episode}",
                    "pair_id": f"latent-{episode}",
                    "state_id": f"latent-{episode}:{step}",
                    "round": step,
                    "split": _split(episode, episodes),
                    "drive": drive,
                    "hazard_event": event,
                }
            )
            if event:
                break
    return pd.DataFrame(rows)


def generate_hypothesis(kind, **kwargs):
    """Named PRD generators used by the full confusion-matrix pipeline."""

    if kind == "H1_latent_commitment":
        return generate_latent_data(rho=0.9, **kwargs)
    if kind == "H2_shared_rule":
        return generate_sharing_data("shared", **kwargs)
    if kind == "H3_task_specific_evaluation":
        return generate_sharing_data("task_specific", **kwargs)
    if kind == "H4_generic_sequential_choice":
        return generate_history_data(lag=3, **kwargs)
    raise ValueError(f"unknown synthetic hypothesis: {kind}")
