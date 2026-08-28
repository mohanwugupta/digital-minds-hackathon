"""Matched history models for persistence and independent sequential choice."""

from __future__ import annotations

import pandas as pd

from ..flexible.neural import select_and_fit_neural
from ..hazard_models.baselines import MODEL_SPECS
from ..hazard_models.modeling import select_and_fit_linear_model
from ..evaluation.metrics import task_metrics
from .history_specificity import (
    bootstrap_persistence_specific_history,
    persistence_specific_history_index,
)


def run_sequential_control(records, config, *, models=None, logger=None):
    requested = set(models or MODEL_SPECS)
    history_models = [
        model
        for model in (
            "finite_history",
            "exponential_reward",
            "perseveration",
            "dual_history",
            "mlp",
            "gru",
        )
        if model in requested
    ]
    if not history_models:
        raise ValueError("sequential control requires at least one history model")
    rows = []
    for task, task_records in records.groupby("task"):
        train = task_records[task_records.split == "train"]
        validation = task_records[task_records.split == "validation"]
        test = task_records[task_records.split == "test"]
        fits = {}
        for model in ("immediate_state", *history_models):
            fits[model] = (
                select_and_fit_neural(
                    train, validation, test, model, "task_specific", config
                )
                if MODEL_SPECS[model].kind in {"mlp", "gru"}
                else select_and_fit_linear_model(
                    train, validation, test, model, "task_specific", config
                )
            )
        losses = {}
        for model, fit in fits.items():
            scored = pd.DataFrame(
                {"task": test.task, "observed": test.hazard_event, "predicted": fit.prediction}
            )
            losses[model] = float(task_metrics(scored).iloc[0].log_loss)
        best_model = min(history_models, key=losses.get)
        best_history = losses[best_model]
        rows.append(
            {
                "task": task,
                "is_persistence_task": bool(task_records.is_persistence_task.iloc[0]),
                "immediate_log_loss": losses["immediate_state"],
                **{f"{model}_log_loss": losses[model] for model in history_models},
                **{
                    f"{model}_history_gain": losses["immediate_state"]
                    - losses[model]
                    for model in history_models
                },
                "best_history_model": best_model,
                "history_log_loss_gain": losses["immediate_state"] - best_history,
            }
        )
        if logger is not None:
            logger.note("sequential_control", f"completed {task}")
    frame = pd.DataFrame(rows)
    summary = persistence_specific_history_index(frame)
    summary.update(
        bootstrap_persistence_specific_history(
            frame,
            samples=int(config["bootstrap"]["samples"]),
            seed=int(config["seed"]) + 2,
        )
    )
    for key, value in summary.items():
        frame[key] = value
    return frame
