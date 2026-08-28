"""Validation-controlled recurrent bottleneck sweep."""

from __future__ import annotations

import pandas as pd

from analysis.persistence_gru.memory_ablations import _metric_row, fit_windowed_gru


def run_bottleneck_analysis(records, features, settings, *, logger=None):
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    test = [row for row in records if row["split"] == "test"]
    rows = []
    for hidden_size in settings["bottleneck_sizes"]:
        if logger is not None:
            logger.note("gru_bottleneck", f"fitting hidden_size={hidden_size}")
        fit = fit_windowed_gru(
            train,
            validation,
            test,
            features,
            hidden_size=int(hidden_size),
            learning_rate=float(settings["learning_rate"]),
            dropout=float(settings.get("dropout", 0.0)),
            max_epochs=int(settings["max_epochs"]),
            patience=int(settings["patience"]),
            seed=int(settings["seed"]) + int(hidden_size),
        )
        row = _metric_row(f"hidden_{hidden_size}", fit, test)
        row["hidden_size"] = int(hidden_size)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("hidden_size").reset_index(drop=True)
    best = float(frame.r_squared.max())
    frame["fraction_best_r_squared"] = frame.r_squared / best if abs(best) > 1e-12 else float("nan")
    return frame

