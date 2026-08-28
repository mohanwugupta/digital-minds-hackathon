"""Task-macro primary metrics and secondary probabilistic diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.special import logit

from ..hazard_models.baselines import binary_log_loss


def task_metrics(frame):
    rows = []
    for task, part in frame.groupby("task"):
        observed = part.observed.to_numpy(dtype=float)
        predicted = part.predicted.to_numpy(dtype=float)
        auc = (
            float(roc_auc_score(observed, predicted))
            if len(np.unique(observed)) == 2
            else float("nan")
        )
        clipped = np.clip(predicted, 1e-6, 1 - 1e-6)
        bins = pd.qcut(clipped, min(10, len(np.unique(clipped))), duplicates="drop")
        calibration = pd.DataFrame({"observed": observed, "predicted": clipped, "bin": bins}).groupby("bin", observed=False).mean()
        ece = float(
            sum(
                len(part[bins == index]) / len(part) * abs(row.observed - row.predicted)
                for index, row in calibration.iterrows()
            )
        ) if len(calibration) else float("nan")
        score = logit(clipped)
        if len(np.unique(observed)) == 2 and np.std(score) > 1e-12:
            design = np.column_stack((np.ones(len(score)), score))
            coefficient = np.linalg.lstsq(design, observed, rcond=None)[0]
            calibration_intercept, calibration_slope = map(float, coefficient)
        else:
            calibration_intercept = calibration_slope = float("nan")
        loss = binary_log_loss(observed, predicted)
        null_loss = binary_log_loss(
            observed, np.repeat(np.clip(observed.mean(), 1e-6, 1 - 1e-6), len(observed))
        )
        policy_correlation = float("nan")
        if "model_choice_logit" in part and np.std(part.model_choice_logit) > 0 and np.std(score) > 0:
            policy_correlation = float(np.corrcoef(score, part.model_choice_logit)[0, 1])
        rows.append(
            {
                "task": task,
                "states": len(part),
                "log_loss": loss,
                "brier": float(brier_score_loss(observed, predicted)),
                "auc": auc,
                "calibration_error": ece,
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
                "deviance_explained": 1.0 - loss / null_loss if null_loss > 0 else float("nan"),
                "policy_logit_correlation": policy_correlation,
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(frame):
    metrics = task_metrics(frame)
    episode_weighted = float("nan")
    if "episode_id" in frame:
        episode_losses = [
            binary_log_loss(part.observed, part.predicted)
            for _episode, part in frame.groupby("episode_id", sort=False)
        ]
        if episode_losses:
            episode_weighted = float(np.mean(episode_losses))
    return {
        "macro_log_loss": round(float(metrics.log_loss.mean()), 15),
        "macro_brier": round(float(metrics.brier.mean()), 15),
        "macro_auc": float(metrics.auc.mean()),
        "state_weighted_log_loss": binary_log_loss(frame.observed, frame.predicted),
        "episode_weighted_log_loss": episode_weighted,
        "tasks": len(metrics),
        "states": len(frame),
    }
