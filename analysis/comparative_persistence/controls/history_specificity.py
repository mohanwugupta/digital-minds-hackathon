"""Persistence-specific history effect relative to sequential choice control."""

from __future__ import annotations

import numpy as np


def persistence_specific_history_index(gains):
    persistence = gains[gains.is_persistence_task.astype(bool)]
    control = gains[~gains.is_persistence_task.astype(bool)]
    if persistence.empty or control.empty:
        raise ValueError("PSH requires persistence tasks and a sequential control")
    persistence_gain = float(persistence.history_log_loss_gain.mean())
    control_gain = float(control.history_log_loss_gain.mean())
    return {
        "persistence_gain": persistence_gain,
        "control_gain": control_gain,
        "psh": persistence_gain - control_gain,
    }


def bootstrap_persistence_specific_history(gains, *, samples, seed):
    """Bootstrap PSH over task identities within persistence/control strata."""

    persistence = gains[gains.is_persistence_task.astype(bool)]
    control = gains[~gains.is_persistence_task.astype(bool)]
    if persistence.empty or control.empty:
        raise ValueError("PSH bootstrap requires persistence and control tasks")
    persistence_values = persistence.history_log_loss_gain.to_numpy(dtype=float)
    control_values = control.history_log_loss_gain.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(samples)):
        persistence_draw = rng.choice(
            persistence_values, size=len(persistence_values), replace=True
        )
        control_draw = rng.choice(control_values, size=len(control_values), replace=True)
        estimates.append(float(persistence_draw.mean() - control_draw.mean()))
    return {
        "psh_ci_low": float(np.quantile(estimates, 0.025)),
        "psh_ci_high": float(np.quantile(estimates, 0.975)),
        "psh_bootstrap_unit": "task_identity",
    }
