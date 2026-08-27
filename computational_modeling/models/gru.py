"""Past-only sequence construction and a deliberately small GRU ceiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeSequence:
    episode_id: str
    task: str
    features: np.ndarray
    targets: np.ndarray
    row_indices: np.ndarray


def build_sequences(
    records: Sequence[Mapping], features: Sequence[str], *, target: str
) -> list[EpisodeSequence]:
    """Build independent causal sequences; row t contains only state-t inputs."""

    episodes: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        episodes.setdefault(str(row["episode_id"]), []).append(index)
    sequences = []
    for episode_id in sorted(episodes):
        indices = sorted(episodes[episode_id], key=lambda item: int(records[item]["round"]))
        tasks = {str(records[index].get("task", "single_task")) for index in indices}
        if len(tasks) != 1:
            raise ValueError(f"episode crosses tasks in GRU input: {episode_id}")
        rounds = [int(records[index]["round"]) for index in indices]
        if rounds != list(range(len(rounds))):
            raise ValueError(f"non-contiguous GRU sequence: {episode_id}")
        sequences.append(
            EpisodeSequence(
                episode_id=episode_id,
                task=next(iter(tasks)),
                features=np.asarray(
                    [[float(records[index][name]) for name in features] for index in indices],
                    dtype=np.float32,
                ),
                targets=np.asarray(
                    [float(records[index][target]) for index in indices], dtype=np.float32
                ),
                row_indices=np.asarray(indices, dtype=np.int64),
            )
        )
    return sequences


def fit_gru_ceiling(
    train_records,
    validation_records,
    application_records,
    features,
    *,
    hidden_size=32,
    learning_rate=1e-3,
    dropout=0.0,
    max_epochs=100,
    patience=10,
    seed=0,
    return_hidden_states=False,
):
    """Fit a one-layer GRU with validation-only early stopping."""

    import copy
    import torch

    from computational_modeling.models.base import (
        TrainStandardizer,
        assert_selection_blind,
        balanced_weights,
        weighted_ridge_fit,
    )

    assert_selection_blind(train_records, validation_records)
    torch.manual_seed(int(seed))
    raw_train = [[float(row[name]) for name in features] for row in train_records]
    normalizer = TrainStandardizer.fit(raw_train, features)

    def standardized(records):
        transformed = normalizer.transform(
            [[float(row[name]) for name in features] for row in records]
        )
        return [
            {**dict(row), **dict(zip(features, values))}
            for row, values in zip(records, transformed)
        ]

    train = build_sequences(standardized(train_records), features, target="persistence_logit")
    validation = build_sequences(
        standardized(validation_records), features, target="persistence_logit"
    )
    application = build_sequences(
        standardized(application_records), features, target="persistence_logit"
    )

    class SmallGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_skip = torch.nn.Linear(len(features), 1)
            self.project = torch.nn.Linear(len(features), hidden_size)
            self.gru = torch.nn.GRU(
                hidden_size,
                hidden_size,
                batch_first=True,
                dropout=float(dropout) if 1 > 1 else 0.0,
            )
            self.dropout = torch.nn.Dropout(float(dropout))
            self.output = torch.nn.Linear(hidden_size, 1)

        def representation(self, values):
            hidden, _ = self.gru(torch.nn.functional.gelu(self.project(values)))
            return hidden

        def forward(self, values):
            hidden = self.representation(values)
            residual = self.output(self.dropout(hidden))
            return (self.linear_skip(values) + residual).squeeze(-1)

    model = SmallGRU()
    linear_coefficient = weighted_ridge_fit(
        normalizer.transform(raw_train),
        [float(row["persistence_logit"]) for row in train_records],
        balanced_weights(train_records, task_balanced=True),
    )
    with torch.no_grad():
        model.linear_skip.bias.copy_(
            torch.tensor([linear_coefficient[0]], dtype=torch.float32)
        )
        model.linear_skip.weight.copy_(
            torch.tensor(linear_coefficient[1:], dtype=torch.float32)[None, :]
        )
        model.output.weight.zero_()
        model.output.bias.zero_()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    def padded_batch(sequences):
        maximum = max(len(sequence.targets) for sequence in sequences)
        x = torch.zeros(
            (len(sequences), maximum, len(features)), dtype=torch.float32, device=device
        )
        y = torch.zeros((len(sequences), maximum), dtype=torch.float32, device=device)
        weight = torch.zeros_like(y)
        task_counts = {}
        for sequence in sequences:
            task_counts[sequence.task] = task_counts.get(sequence.task, 0) + 1
        for index, sequence in enumerate(sequences):
            length = len(sequence.targets)
            x[index, :length] = torch.tensor(sequence.features, device=device)
            y[index, :length] = torch.tensor(sequence.targets, device=device)
            weight[index, :length] = 1.0 / (
                len(task_counts) * task_counts[sequence.task] * length
            )
        return x, y, weight

    train_batch = padded_batch(train)
    validation_batch = padded_batch(validation)
    application_batch = padded_batch(application)

    def loss_on(batch, *, train_mode):
        model.train(train_mode)
        x, y, weight = batch
        return torch.sum(weight * (model(x) - y) ** 2) / weight.sum()

    with torch.no_grad():
        best_loss = float(loss_on(validation_batch, train_mode=False))
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(int(max_epochs)):
        optimizer.zero_grad()
        train_loss = loss_on(train_batch, train_mode=True)
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss = float(loss_on(validation_batch, train_mode=False))
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
    predictions = np.empty(len(application_records), dtype=float)
    hidden_states = (
        np.empty((len(application_records), int(hidden_size)), dtype=np.float32)
        if return_hidden_states
        else None
    )
    model.eval()
    with torch.no_grad():
        values_by_sequence = model(application_batch[0]).cpu().numpy()
        hidden_by_sequence = (
            model.representation(application_batch[0]).cpu().numpy()
            if return_hidden_states
            else None
        )
        for sequence, values in zip(application, values_by_sequence):
            predictions[sequence.row_indices] = values[: len(sequence.row_indices)]
        if return_hidden_states:
            for sequence, values in zip(application, hidden_by_sequence):
                hidden_states[sequence.row_indices] = values[: len(sequence.row_indices)]
    result = {
        "prediction": predictions,
        "selected_epochs": best_epoch,
        "trained_epochs": epoch + 1,
        "validation_mse": best_loss,
        "normalizer": normalizer,
        "linear_skip_initialized": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if return_hidden_states:
        result["hidden_state"] = hidden_states
    return result
