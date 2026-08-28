"""Within-familiar-task model comparison under all sharing assumptions."""

from __future__ import annotations

import json

import pandas as pd

from ..hazard_models.baselines import MODEL_SPECS
from ..hazard_models.modeling import select_and_fit_linear_model
from ..flexible.neural import select_and_fit_neural
from .metrics import summarize_predictions, task_metrics


def score_fit(fit):
    scored = pd.DataFrame(
        {
            "task": fit.application_records.task,
            "episode_id": fit.application_records.episode_id,
            "observed": fit.application_records.hazard_event,
            "predicted": fit.prediction,
            "model_choice_logit": fit.application_records.model_choice_logit,
        }
    )
    return scored, task_metrics(scored), summarize_predictions(scored)


def run_within_task(records, config, *, models=None, logger=None):
    records = records[records.is_persistence_task.astype(bool)].copy()
    train = records[records.split == "train"]
    validation = records[records.split == "validation"]
    test = records[records.split == "test"]
    models = list(models or MODEL_SPECS)
    task_rows, macro_rows, predictions, fits = [], [], [], {}
    for model in models:
        for sharing in config["sharing"]:
            if logger is not None:
                logger.note("within_task", f"fitting {model}/{sharing}")
            fit = (
                select_and_fit_neural(train, validation, test, model, sharing, config)
                if MODEL_SPECS[model].kind in {"mlp", "gru"}
                else select_and_fit_linear_model(
                    train, validation, test, model, sharing, config
                )
            )
            scored, taskwise, macro = score_fit(fit)
            selected = json.dumps(fit.selected_hyperparameters, sort_keys=True)
            for row in taskwise.to_dict("records"):
                task_rows.append({"model": model, "sharing": sharing, "information_set": "oracle" if MODEL_SPECS[model].oracle else "observable", "selected_hyperparameters": selected, **row})
            macro_rows.append(
                {
                    "model": model,
                    "sharing": sharing,
                    "information_set": "oracle" if MODEL_SPECS[model].oracle else "observable",
                    "selected_hyperparameters": selected,
                    "validation_macro_log_loss": fit.validation_macro_log_loss,
                    "parameter_count": getattr(fit, "parameter_count", len(getattr(fit, "feature_names", ()))),
                    **macro,
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "state_id": fit.application_records.state_id,
                        "episode_id": fit.application_records.episode_id,
                        "pair_id": fit.application_records.pair_id,
                        "task": fit.application_records.task,
                        "model": model,
                        "sharing": sharing,
                        "observed": fit.application_records.hazard_event,
                        "predicted": fit.prediction,
                    }
                )
            )
            fits[(model, sharing)] = fit
    task_frame = pd.DataFrame(task_rows)
    macro_frame = pd.DataFrame(macro_rows).sort_values("macro_log_loss")
    macro_frame["rank"] = range(1, len(macro_frame) + 1)
    return {
        "taskwise": task_frame,
        "macro": macro_frame,
        "predictions": pd.concat(predictions, ignore_index=True),
        "fits": fits,
    }
