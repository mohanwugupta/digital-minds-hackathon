"""Binary-hazard MLP and causal GRU ceilings with validation early stopping."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..hazard_models.modeling import model_matrix, task_balanced_weights
from ..evaluation.metrics import summarize_predictions


def _task_columns(frame, known_tasks):
    return np.column_stack(
        [(frame.task.to_numpy() == task).astype(float) for task in known_tasks]
    ) if known_tasks else np.empty((len(frame), 0))


def _matrix(frame, model_name, config, sharing, known_tasks):
    values, names = model_matrix(frame, model_name, {}, config)
    if sharing == "hierarchical":
        values = np.column_stack((values, _task_columns(frame, known_tasks)))
        names = (*names, *(f"task::{task}" for task in known_tasks))
    return values.astype(np.float32), names


def _sequence_batch(frame, values, device):
    import torch

    frame = frame.reset_index(drop=True)
    groups = list(frame.groupby(["task", "episode_id"], sort=False).groups.values())
    maximum = max(len(indices) for indices in groups)
    x = torch.zeros((len(groups), maximum, values.shape[1]), dtype=torch.float32)
    y = torch.zeros((len(groups), maximum), dtype=torch.float32)
    weight = torch.zeros((len(groups), maximum), dtype=torch.float32)
    mask = torch.zeros((len(groups), maximum), dtype=torch.bool)
    positions = np.full((len(groups), maximum), -1, dtype=int)
    row_weights = task_balanced_weights(frame)
    for group_index, indices in enumerate(groups):
        ordered = sorted(indices, key=lambda index: int(frame.loc[index, "round"]))
        length = len(ordered)
        x[group_index, :length] = torch.tensor(values[ordered])
        y[group_index, :length] = torch.tensor(
            frame.loc[ordered, "hazard_event"].to_numpy(dtype=np.float32)
        )
        weight[group_index, :length] = torch.tensor(row_weights[ordered])
        mask[group_index, :length] = True
        positions[group_index, :length] = ordered
    return x.to(device), y.to(device), weight.to(device), mask.to(device), positions


def _fit_one(train, validation, application, model_name, config, sharing, parameters):
    import torch

    torch.manual_seed(int(config["seed"]) + int(parameters.get("hidden_size", 0)))
    known_tasks = sorted(train.task.unique())
    x_train, names = _matrix(train.reset_index(drop=True), model_name, config, sharing, known_tasks)
    x_validation, validation_names = _matrix(validation.reset_index(drop=True), model_name, config, sharing, known_tasks)
    x_application, application_names = _matrix(application.reset_index(drop=True), model_name, config, sharing, known_tasks)
    if names != validation_names or names != application_names:
        raise RuntimeError("neural feature definitions differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = int(parameters["hidden_size"])
    dropout = float(parameters.get("dropout", 0.0))

    if model_name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, max(1, hidden // 2)),
            torch.nn.GELU(),
            torch.nn.Linear(max(1, hidden // 2), 1),
        ).to(device)

        def batch(frame, values):
            return (
                torch.tensor(values, dtype=torch.float32, device=device),
                torch.tensor(frame.hazard_event.to_numpy(), dtype=torch.float32, device=device),
                torch.tensor(task_balanced_weights(frame), dtype=torch.float32, device=device),
            )

        train_batch = batch(train.reset_index(drop=True), x_train)
        validation_batch = batch(validation.reset_index(drop=True), x_validation)

        def logits(values):
            return model(values[0]).squeeze(-1)

        def loss(values):
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits(values), values[1], reduction="none"
            )
            return torch.sum(raw * values[2]) / values[2].sum()

        def apply():
            return model(
                torch.tensor(x_application, dtype=torch.float32, device=device)
            ).squeeze(-1)

    else:
        class HazardGRU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project = torch.nn.Linear(x_train.shape[1], hidden)
                self.gru = torch.nn.GRU(hidden, hidden, batch_first=True)
                self.dropout = torch.nn.Dropout(dropout)
                self.output = torch.nn.Linear(hidden, 1)

            def forward(self, values):
                projected = torch.nn.functional.gelu(self.project(values))
                sequence, _state = self.gru(projected)
                return self.output(self.dropout(sequence)).squeeze(-1)

        model = HazardGRU().to(device)
        train_batch = _sequence_batch(train, x_train, device)
        validation_batch = _sequence_batch(validation, x_validation, device)
        application_batch = _sequence_batch(application, x_application, device)

        def loss(values):
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                model(values[0]), values[1], reduction="none"
            )
            return torch.sum(raw * values[2] * values[3]) / torch.sum(
                values[2] * values[3]
            )

        def apply():
            raw = model(application_batch[0]).detach().cpu().numpy()
            prediction = np.zeros(len(application), dtype=float)
            for group in range(raw.shape[0]):
                for step in range(raw.shape[1]):
                    position = application_batch[4][group, step]
                    if position >= 0:
                        prediction[position] = raw[group, step]
            return torch.tensor(prediction, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(parameters["learning_rate"])
    )
    max_epochs = int(config[model_name]["max_epochs"])
    patience = int(config[model_name]["patience"])
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss(train_batch).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss(validation_batch))
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(apply()).cpu().numpy()
    return {
        "prediction": prediction,
        "validation_loss": best_loss,
        "selected_epochs": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_names": names,
    }


@dataclass(frozen=True)
class NeuralFit:
    model: str
    sharing: str
    prediction: np.ndarray
    application_records: pd.DataFrame
    selected_hyperparameters: dict
    validation_macro_log_loss: float
    parameter_count: int
    feature_names: tuple[str, ...]


def select_and_fit_neural(train, validation, application, model_name, sharing, config):
    if sharing == "task_specific":
        predictions = np.zeros(len(application))
        parameters, counts, losses = {}, [], []
        feature_names = tuple()
        for task in sorted(application.task.unique()):
            local_train = train[train.task == task].reset_index(drop=True)
            local_validation = validation[validation.task == task].reset_index(drop=True)
            local_application = application[application.task == task].reset_index(drop=True)
            fit = select_and_fit_neural(
                local_train,
                local_validation,
                local_application,
                model_name,
                "fully_shared",
                config,
            )
            positions = application.index.get_indexer(application[application.task == task].index)
            predictions[positions] = fit.prediction
            parameters[task] = fit.selected_hyperparameters
            counts.append(fit.parameter_count)
            losses.append(fit.validation_macro_log_loss)
            feature_names = fit.feature_names
        return NeuralFit(
            model_name,
            sharing,
            predictions,
            application.reset_index(drop=True),
            parameters,
            float(np.mean(losses)),
            int(sum(counts)),
            feature_names,
        )

    if model_name == "mlp":
        candidates = [
            {
                "hidden_size": int(hidden),
                "learning_rate": float(rate),
                "dropout": float(dropout),
            }
            for hidden in config["mlp"]["hidden_sizes"]
            for rate in config["mlp"]["learning_rates"]
            for dropout in config["mlp"]["dropout"]
        ]
    else:
        candidates = [
            {
                "hidden_size": int(hidden),
                "learning_rate": float(config["gru"]["learning_rate"]),
                "dropout": float(config["gru"]["dropout"]),
            }
            for hidden in config["gru"]["hidden_sizes"]
        ]
    selection = []
    for parameters in candidates:
        fit = _fit_one(
            train.reset_index(drop=True),
            validation.reset_index(drop=True),
            validation.reset_index(drop=True),
            model_name,
            config,
            sharing,
            parameters,
        )
        scored = pd.DataFrame(
            {
                "task": validation.task.to_numpy(),
                "episode_id": validation.episode_id.to_numpy(),
                "observed": validation.hazard_event.to_numpy(),
                "predicted": fit["prediction"],
            }
        )
        selection.append(
            (summarize_predictions(scored)["macro_log_loss"], parameters)
        )
    validation_loss, selected = min(
        selection, key=lambda item: (item[0], item[1]["hidden_size"])
    )
    final = _fit_one(
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        application.reset_index(drop=True),
        model_name,
        config,
        sharing,
        selected,
    )
    return NeuralFit(
        model_name,
        sharing,
        final["prediction"],
        application.reset_index(drop=True),
        {**selected, "epochs": final["selected_epochs"]},
        float(validation_loss),
        int(final["parameter_count"]),
        tuple(final["feature_names"]),
    )


def run_gru_bottleneck(records, config, *, logger=None):
    """Computational-complexity sweep; not a claim about Qwen state dimension."""

    records = records[records.is_persistence_task.astype(bool)]
    train = records[records.split == "train"]
    validation = records[records.split == "validation"]
    test = records[records.split == "test"]
    rows = []
    for hidden_size in config["gru"]["hidden_sizes"]:
        local = {
            **config,
            "gru": {**config["gru"], "hidden_sizes": [int(hidden_size)]},
        }
        fit = select_and_fit_neural(
            train, validation, test, "gru", "fully_shared", local
        )
        scored = pd.DataFrame(
            {
                "task": test.task,
                "episode_id": test.episode_id,
                "observed": test.hazard_event,
                "predicted": fit.prediction,
            }
        )
        summary = summarize_predictions(scored)
        rows.append(
            {
                "hidden_size": int(hidden_size),
                "parameter_count": fit.parameter_count,
                "selected_epochs": fit.selected_hyperparameters["epochs"],
                **summary,
            }
        )
        if logger is not None:
            logger.note("gru_bottleneck", f"completed hidden_size={hidden_size}")
    return pd.DataFrame(rows)
