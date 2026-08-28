"""LOTO, LOFO, few-shot, and architecture-transfer analyses."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..hazard_models.baselines import MODEL_SPECS
from ..hazard_models.modeling import (
    fit_fixed_linear_model,
    select_and_fit_linear_model,
)
from ..flexible.neural import select_and_fit_neural
from .few_shot_adaptation import few_shot_partition
from .leave_one_family_out import lofo_partition
from .leave_one_task_out import loto_partition
from .metrics import summarize_predictions


def _score(fit):
    frame = pd.DataFrame(
        {
            "task": fit.application_records.task,
            "episode_id": fit.application_records.episode_id,
            "observed": fit.application_records.hazard_event,
            "predicted": fit.prediction,
        }
    )
    return summarize_predictions(frame)


def _linear_models(models):
    return [
        name
        for name in models
        if MODEL_SPECS[name].kind not in {"mlp", "gru"}
        and not MODEL_SPECS[name].oracle
    ]


def _eligible_models(models):
    return [name for name in models if not MODEL_SPECS[name].oracle]


def _fit_selected(train, selection, evaluation, model, sharing, config):
    if MODEL_SPECS[model].kind in {"mlp", "gru"}:
        return select_and_fit_neural(
            train, selection, evaluation, model, sharing, config
        )
    return select_and_fit_linear_model(
        train, selection, evaluation, model, sharing, config
    )


def run_loto(records, config, within_task, *, models=None, logger=None):
    persistence = records[records.is_persistence_task.astype(bool)].copy()
    models = _eligible_models(models or MODEL_SPECS)
    rows, frozen = [], {}
    for heldout in sorted(persistence.task.unique()):
        partition = loto_partition(persistence, heldout)
        null_loss = None
        task_results = []
        for model in models:
            for sharing in ("fully_shared", "hierarchical"):
                if logger is not None:
                    logger.note("loto", f"heldout={heldout}; fitting {model}/{sharing}")
                fit = _fit_selected(
                    partition.fit, partition.selection, partition.evaluation,
                    model, sharing, config
                )
                score = _score(fit)
                row = {
                    "heldout_task": heldout,
                    "model": model,
                    "sharing": sharing,
                    "information_set": "oracle" if MODEL_SPECS[model].oracle else "observable",
                    "selected_hyperparameters": json.dumps(
                        fit.selected_hyperparameters, sort_keys=True
                    ),
                    **score,
                }
                task_results.append(row)
                frozen[(heldout, model, sharing)] = fit
                if model == "intercept" and sharing == "fully_shared":
                    null_loss = score["macro_log_loss"]
        if null_loss is None:
            raise RuntimeError("LOTO requires the intercept benchmark")
        for row in task_results:
            row["null_log_loss"] = null_loss
            row["delta_log_loss_vs_null"] = null_loss - row["macro_log_loss"]
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["model", "sharing"], as_index=False)
        .agg(
            macro_log_loss=("macro_log_loss", "mean"),
            macro_delta_log_loss=("delta_log_loss_vs_null", "mean"),
            tasks=("heldout_task", "nunique"),
        )
        .sort_values("macro_log_loss")
    )
    summary["rank"] = range(1, len(summary) + 1)

    architecture = []
    taskwise = within_task["taskwise"]
    for row in frame.itertuples():
        ceiling = taskwise[
            (taskwise.task == row.heldout_task)
            & (taskwise.model == row.model)
            & (taskwise.sharing == "task_specific")
        ]
        if ceiling.empty:
            continue
        denominator = row.null_log_loss - float(ceiling.iloc[0].log_loss)
        architecture.append(
            {
                "task": row.heldout_task,
                "model": row.model,
                "sharing": row.sharing,
                "null_log_loss": row.null_log_loss,
                "loto_log_loss": row.macro_log_loss,
                "task_specific_log_loss": float(ceiling.iloc[0].log_loss),
                "architecture_transfer": (
                    (row.null_log_loss - row.macro_log_loss) / denominator
                    if denominator > 1e-12
                    else float("nan")
                ),
            }
        )
    return frame, summary, pd.DataFrame(architecture), frozen


def run_lofo(records, config, *, models=None, logger=None):
    persistence = records[records.is_persistence_task.astype(bool)].copy()
    models = _eligible_models(models or MODEL_SPECS)
    rows = []
    for family in sorted(persistence.family.unique()):
        # A singleton source side is not a meaningful family-transfer fit.
        if persistence[persistence.family != family].task.nunique() < 2:
            continue
        partition = lofo_partition(persistence, family)
        for model in models:
            for sharing in ("fully_shared", "hierarchical"):
                if logger is not None:
                    logger.note("lofo", f"heldout_family={family}; {model}/{sharing}")
                fit = _fit_selected(
                    partition.fit, partition.selection, partition.evaluation,
                    model, sharing, config
                )
                rows.append(
                    {
                        "heldout_family": family,
                        "heldout_tasks": ";".join(sorted(partition.evaluation.task.unique())),
                        "model": model,
                        "sharing": sharing,
                        "information_set": "oracle" if MODEL_SPECS[model].oracle else "observable",
                        "selected_hyperparameters": json.dumps(
                            fit.selected_hyperparameters, sort_keys=True
                        ),
                        **_score(fit),
                    }
                )
    return pd.DataFrame(rows)


def run_few_shot(records, config, frozen_loto, *, models, logger=None):
    persistence = records[records.is_persistence_task.astype(bool)].copy()
    rows = []
    for heldout in sorted(persistence.task.unique()):
        partition = loto_partition(persistence, heldout)
        source = pd.concat((partition.fit, partition.selection), ignore_index=True)
        for model in _linear_models(models):
            for sharing in ("fully_shared", "hierarchical"):
                zero = frozen_loto[(heldout, model, sharing)]
                for count in config["evaluation"]["few_shot_counts"]:
                    few = few_shot_partition(
                        persistence,
                        heldout,
                        pair_count=int(count),
                        seed=int(config["seed"]) + int(count),
                    )
                    if int(count) == 0:
                        fit = zero
                    else:
                        adaptation_fit = pd.concat(
                            (source, few.adaptation), ignore_index=True
                        )
                        fit = fit_fixed_linear_model(
                            adaptation_fit,
                            few.evaluation,
                            model,
                            sharing,
                            zero.selected_hyperparameters,
                            config,
                        )
                    rows.append(
                        {
                            "heldout_task": heldout,
                            "model": model,
                            "sharing": sharing,
                            "requested_pairs": int(count),
                            "adaptation_pairs": few.selected_pairs,
                            "adaptation_recorded_episodes": int(
                                few.adaptation.episode_id.nunique()
                            ),
                            **_score(fit),
                        }
                    )
                    if logger is not None:
                        logger.note(
                            "few_shot",
                            f"{heldout}/{model}/{sharing}: {few.selected_pairs} pairs",
                        )
    return pd.DataFrame(rows)
