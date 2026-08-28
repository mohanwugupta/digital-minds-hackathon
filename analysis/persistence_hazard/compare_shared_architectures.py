"""History-kernel comparisons built on the common hazard fitter."""

from __future__ import annotations

import pandas as pd

from analysis.persistence_hazard.fit_hazard_models import fit_hazard_architecture
from analysis.persistence_hazard.history_kernels import (
    add_exponential_history,
    finite_history_features,
)


def compare_history_kernels(records, config):
    rows = []
    penalties = config.get("ridge_grid", (0.01, 0.1, 1.0))
    for lag in config.get("history_lags", (1, 2, 3, 5)):
        fit = fit_hazard_architecture(
            records,
            "shared_history",
            history_features=finite_history_features(int(lag)),
            penalties=penalties,
        )
        rows.append(
            {
                "kernel": "finite",
                "parameter": int(lag),
                "history_features": ";".join(finite_history_features(int(lag))),
                "validation_log_loss": fit["validation_log_loss"],
                "test_log_loss": fit["test_log_loss"],
                "test_brier": fit["test_brier"],
                "test_auc": fit["test_auc"],
                "selected_penalty": fit["selected_penalty"],
            }
        )
    for decay in config.get("exponential_decay", (0.0, 0.5, 0.85)):
        augmented = add_exponential_history(records, decay=float(decay))
        features = ("choice_kernel", "outcome_kernel")
        fit = fit_hazard_architecture(
            augmented,
            "shared_history",
            history_features=features,
            penalties=penalties,
        )
        rows.append(
            {
                "kernel": "exponential",
                "parameter": float(decay),
                "history_features": ";".join(features),
                "validation_log_loss": fit["validation_log_loss"],
                "test_log_loss": fit["test_log_loss"],
                "test_brier": fit["test_brier"],
                "test_auc": fit["test_auc"],
                "selected_penalty": fit["selected_penalty"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["validation_selected"] = False
    frame.loc[frame.validation_log_loss.idxmin(), "validation_selected"] = True
    return frame

