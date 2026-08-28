"""Causal-window GRU fits and preregistered information ablations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from computational_modeling.analysis.evaluate_models import (
    choice_metrics,
    persistence_metrics,
    sigmoid,
)
from computational_modeling.models.base import (
    TrainStandardizer,
    assert_selection_blind,
    balanced_weights,
    weighted_ridge_fit,
)
from computational_modeling.models.mlp import fit_mlp_ceiling


CHOICE_HISTORY = {
    "previous_choice",
    "second_previous_choice",
    "action_lag_1",
    "action_lag_2",
    "action_lag_3",
    "action_lag_5",
}
OUTCOME_HISTORY = {
    "previous_outcome",
    "failure_streak",
    "success_streak",
    "outcome_lag_1",
    "outcome_lag_2",
    "outcome_lag_3",
    "outcome_lag_5",
}
TIME_FEATURES = {"log_round", "normalized_time"}
TASK_EVIDENCE = {
    "estimated_continue_value",
    "estimated_outside_value",
    "cost_pressure",
    "cumulative_progress",
    "progress_evidence",
    "termination_advantage",
}


def build_causal_windows(records, features, *, max_history=None):
    """Build one past-and-current window per state without future inputs."""

    by_episode = defaultdict(list)
    for index, row in enumerate(records):
        by_episode[str(row["episode_id"])].append(index)
    windows = [None] * len(records)
    for episode, indices in by_episode.items():
        indices.sort(key=lambda index: int(records[index]["round"]))
        rounds = [int(records[index]["round"]) for index in indices]
        if rounds != list(range(len(rounds))):
            raise ValueError(f"non-contiguous GRU episode: {episode}")
        for local_index, row_index in enumerate(indices):
            start = 0 if max_history is None else max(0, local_index - int(max_history) + 1)
            source = indices[start : local_index + 1]
            windows[row_index] = {
                "values": np.asarray(
                    [[float(records[index][name]) for name in features] for index in source],
                    dtype=np.float32,
                ),
                "target": float(records[row_index]["persistence_logit"]),
                "row_index": row_index,
                "episode_id": str(records[row_index]["episode_id"]),
                "task": str(records[row_index].get("task", "single_task")),
            }
    return windows


def fit_windowed_gru(
    train_records,
    validation_records,
    application_records,
    features,
    *,
    hidden_size,
    learning_rate,
    dropout,
    max_epochs,
    patience,
    seed,
    max_history=None,
):
    import copy
    import torch

    assert_selection_blind(train_records, validation_records)
    torch.manual_seed(int(seed))
    normalizer = TrainStandardizer.fit(
        [[float(row[name]) for name in features] for row in train_records], features
    )

    def standardized(records):
        values = normalizer.transform(
            [[float(row[name]) for name in features] for row in records]
        )
        return [
            {**dict(row), **dict(zip(features, transformed))}
            for row, transformed in zip(records, values)
        ]

    train_windows = build_causal_windows(
        standardized(train_records), features, max_history=max_history
    )
    validation_windows = build_causal_windows(
        standardized(validation_records), features, max_history=max_history
    )
    application_windows = build_causal_windows(
        standardized(application_records), features, max_history=max_history
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batch(windows, records, weighted):
        maximum = max(len(window["values"]) for window in windows)
        x = torch.zeros((len(windows), maximum, len(features)), dtype=torch.float32)
        lengths = torch.empty(len(windows), dtype=torch.long)
        current = torch.empty((len(windows), len(features)), dtype=torch.float32)
        target = torch.empty(len(windows), dtype=torch.float32)
        for index, window in enumerate(windows):
            length = len(window["values"])
            x[index, :length] = torch.tensor(window["values"])
            lengths[index] = length
            current[index] = torch.tensor(window["values"][-1])
            target[index] = float(window["target"])
        weight = (
            torch.tensor(balanced_weights(records, task_balanced=True), dtype=torch.float32)
            if weighted
            else torch.ones(len(windows), dtype=torch.float32)
        )
        return tuple(value.to(device) for value in (x, lengths, current, target, weight))

    train_batch = batch(train_windows, train_records, True)
    validation_batch = batch(validation_windows, validation_records, True)
    application_batch = batch(application_windows, application_records, False)

    class WindowGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skip = torch.nn.Linear(len(features), 1)
            self.project = torch.nn.Linear(len(features), int(hidden_size))
            self.gru = torch.nn.GRU(int(hidden_size), int(hidden_size), batch_first=True)
            self.dropout = torch.nn.Dropout(float(dropout))
            self.output = torch.nn.Linear(int(hidden_size), 1)

        def forward(self, x, lengths, current):
            projected = torch.nn.functional.gelu(self.project(x))
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                projected, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            return (
                self.skip(current).squeeze(-1)
                + self.output(self.dropout(hidden[-1])).squeeze(-1)
            )

    model = WindowGRU().to(device)
    initial = weighted_ridge_fit(
        normalizer.transform(
            [[float(row[name]) for name in features] for row in train_records]
        ),
        [float(row["persistence_logit"]) for row in train_records],
        balanced_weights(train_records, task_balanced=True),
    )
    with torch.no_grad():
        model.skip.bias.copy_(torch.tensor([initial[0]], dtype=torch.float32, device=device))
        model.skip.weight.copy_(torch.tensor(initial[1:][None, :], dtype=torch.float32, device=device))
        model.output.weight.zero_()
        model.output.bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    def loss_on(values, train_mode):
        model.train(train_mode)
        x, lengths, current, target, weight = values
        prediction = model(x, lengths, current)
        return torch.sum(weight * (prediction - target) ** 2) / weight.sum()

    model.eval()
    with torch.no_grad():
        best_loss = float(loss_on(validation_batch, False))
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    trained_epochs = 0
    for epoch in range(int(max_epochs)):
        optimizer.zero_grad()
        loss_on(train_batch, True).backward()
        optimizer.step()
        trained_epochs = epoch + 1
        with torch.no_grad():
            validation_loss = float(loss_on(validation_batch, False))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(
            application_batch[0], application_batch[1], application_batch[2]
        ).cpu().numpy()
    return {
        "prediction": np.asarray(prediction, dtype=float),
        "selected_epochs": best_epoch,
        "trained_epochs": trained_epochs,
        "validation_mse": best_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "features": list(features),
        "max_history": max_history,
    }


def _metric_row(name, fit, test_records):
    weights = balanced_weights(test_records, task_balanced=True)
    policy = persistence_metrics(
        [row["persistence_logit"] for row in test_records], fit["prediction"], weights
    )
    choice = choice_metrics(
        [row["continue"] for row in test_records], sigmoid(fit["prediction"]), weights
    )
    return {
        "ablation": name,
        **policy,
        **choice,
        "validation_mse": fit["validation_mse"],
        "parameter_count": fit["parameter_count"],
        "selected_epochs": fit.get("selected_epochs", 0),
        "trained_epochs": fit.get("trained_epochs", 0),
        "features": ";".join(fit.get("features", [])),
        "max_history": fit.get("max_history"),
    }


def run_memory_ablations(records, features, settings, *, logger=None):
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    test = [row for row in records if row["split"] == "test"]
    common = {
        "hidden_size": int(settings["hidden_size"]),
        "learning_rate": float(settings["learning_rate"]),
        "dropout": float(settings.get("dropout", 0.0)),
        "max_epochs": int(settings["max_epochs"]),
        "patience": int(settings["patience"]),
        "seed": int(settings["seed"]),
    }
    variants = [("full_recurrence", tuple(features), None)]
    variants.extend(
        (f"limited_history_{lag}", tuple(features), int(lag))
        for lag in settings.get("history_windows", (1, 2, 3, 5))
    )
    variants.extend(
        [
            ("choice_history_removed", tuple(name for name in features if name not in CHOICE_HISTORY), None),
            ("outcome_history_removed", tuple(name for name in features if name not in OUTCOME_HISTORY), None),
            ("time_removed", tuple(name for name in features if name not in TIME_FEATURES), None),
            ("task_specific_evidence_removed", tuple(name for name in features if name not in TASK_EVIDENCE), None),
        ]
    )
    rows = []
    for name, local_features, window in variants:
        if logger is not None:
            logger.note("gru_memory", f"fitting {name} ({len(local_features)} features)")
        fit = fit_windowed_gru(
            train,
            validation,
            test,
            local_features,
            max_history=window,
            **common,
        )
        rows.append(_metric_row(name, fit, test))
        if name == "limited_history_1":
            rows.append(_metric_row("reset_every_decision", fit, test))
    if logger is not None:
        logger.note("gru_memory", "fitting non-recurrent MLP control")
    mlp = fit_mlp_ceiling(
        train,
        validation,
        test,
        features,
        hidden_sizes=(int(settings["hidden_size"]), max(1, int(settings["hidden_size"]) // 2)),
        learning_rate=float(settings["learning_rate"]),
        dropout=float(settings.get("dropout", 0.0)),
        max_epochs=int(settings["max_epochs"]),
        patience=int(settings["patience"]),
        seed=int(settings["seed"]),
    )
    mlp["features"] = list(features)
    mlp["max_history"] = 1
    rows.append(_metric_row("no_recurrence_mlp", mlp, test))
    frame = pd.DataFrame(rows)
    full = frame[frame.ablation == "full_recurrence"].iloc[0]
    frame["delta_r_squared_vs_full"] = frame.r_squared - float(full.r_squared)
    frame["delta_log_loss_vs_full"] = frame.log_loss - float(full.log_loss)
    return frame
