"""Reduced PRD 2.5 model comparison over the expanded persistence battery."""

from __future__ import annotations

import pandas as pd

from analysis.comparative_persistence.evaluation.bootstrap import (
    add_episode_bootstrap_intervals,
    add_task_bootstrap_intervals,
)
from analysis.comparative_persistence.evaluation.generalization import (
    run_few_shot,
    run_loto,
)
from analysis.comparative_persistence.evaluation.within_task import run_within_task


def sharing_comparison(macro):
    pivot = macro.pivot_table(
        index="model",
        columns="sharing",
        values="macro_log_loss",
        aggfunc="first",
    ).reset_index()
    sharing_columns = [
        name
        for name in ("task_specific", "hierarchical", "fully_shared")
        if name in pivot
    ]
    pivot["best_sharing"] = pivot[sharing_columns].idxmin(axis=1)
    pivot["best_macro_log_loss"] = pivot[sharing_columns].min(axis=1)
    if {"hierarchical", "task_specific"} <= set(pivot.columns):
        pivot["hierarchical_minus_task_specific"] = (
            pivot.hierarchical - pivot.task_specific
        )
    if {"fully_shared", "hierarchical"} <= set(pivot.columns):
        pivot["fully_shared_minus_hierarchical"] = (
            pivot.fully_shared - pivot.hierarchical
        )
    return pivot.sort_values("best_macro_log_loss")


def run_reduced_model_comparison(
    records,
    config,
    model_config,
    *,
    models,
    gru_ceiling=None,
    gru_taskwise=None,
    smoke=False,
    logger=None,
):
    within = run_within_task(
        records, model_config, models=models, logger=logger
    )
    samples = int(
        config["smoke"]["bootstrap_samples"]
        if smoke
        else config["bootstrap"]["episode_samples"]
    )
    within["taskwise"] = add_episode_bootstrap_intervals(
        within["taskwise"],
        within["predictions"],
        samples=samples,
        seed=int(config["base_seed"]) + 11,
    )
    task_samples = int(
        config["smoke"]["bootstrap_samples"]
        if smoke
        else config["bootstrap"]["task_samples"]
    )
    within["macro"] = add_task_bootstrap_intervals(
        within["macro"],
        within["taskwise"],
        group_columns=("model", "sharing"),
        value="log_loss",
        samples=task_samples,
        seed=int(config["base_seed"]) + 12,
    )
    if gru_taskwise is not None and not gru_taskwise.empty:
        local = gru_taskwise.copy()
        local["information_set"] = "observable"
        local["selected_hyperparameters"] = "see gru/hyperparameter_results.csv"
        within["taskwise"] = pd.concat(
            (within["taskwise"], local), ignore_index=True, sort=False
        )
    if gru_ceiling is not None and not gru_ceiling.empty:
        gru = gru_ceiling[gru_ceiling.model == "gru"]
        if not gru.empty:
            row = gru.iloc[0]
            macro_row = {
                "model": "large_gru",
                "sharing": "fully_shared",
                "information_set": "observable",
                "selected_hyperparameters": "see gru/hyperparameter_results.csv",
                "validation_macro_log_loss": row.validation_macro_log_loss,
                "parameter_count": row.parameter_count,
                "macro_log_loss": row.macro_log_loss,
                "macro_brier": (
                    float(gru_taskwise.brier.mean())
                    if gru_taskwise is not None and not gru_taskwise.empty
                    else float("nan")
                ),
                "macro_auc": (
                    float(gru_taskwise.auc.mean())
                    if gru_taskwise is not None and not gru_taskwise.empty
                    else float("nan")
                ),
                "tasks": (
                    int(gru_taskwise.task.nunique())
                    if gru_taskwise is not None
                    else records[records.is_persistence_task].task.nunique()
                ),
                "states": (
                    int(gru_taskwise.states.sum())
                    if gru_taskwise is not None
                    else 0
                ),
            }
            within["macro"] = pd.concat(
                (within["macro"], pd.DataFrame([macro_row])),
                ignore_index=True,
                sort=False,
            ).sort_values("macro_log_loss")

    loto, loto_summary, architecture, frozen = run_loto(
        records,
        model_config,
        within,
        models=models,
        logger=logger,
    )
    loto_summary = add_task_bootstrap_intervals(
        loto_summary,
        loto.rename(columns={"heldout_task": "task"}),
        group_columns=("model", "sharing"),
        value="delta_log_loss_vs_null",
        samples=task_samples,
        seed=int(config["base_seed"]) + 13,
    )
    requested_few_models = (
        config["smoke"]["few_shot_models"]
        if smoke
        else config["evaluation"]["few_shot_models"]
    )
    few_models = [name for name in requested_few_models if name in models]
    few_shot = run_few_shot(
        records,
        model_config,
        frozen,
        models=few_models,
        logger=logger,
    )
    return {
        "within_task": within["taskwise"],
        "task_macro": within["macro"],
        "loto": loto,
        "loto_summary": loto_summary,
        "architecture_transfer": architecture,
        "sharing_comparison": sharing_comparison(within["macro"]),
        "few_shot": few_shot,
    }
