"""Small non-recurrent nonlinear ceiling with validation-only early stopping."""

from __future__ import annotations

import copy

import numpy as np


def fit_mlp_ceiling(
    train_records,
    validation_records,
    application_records,
    features,
    *,
    hidden_sizes=(64, 32),
    learning_rate=1e-3,
    dropout=0.0,
    max_epochs=100,
    patience=10,
    seed=0,
):
    import torch

    from computational_modeling.models.base import (
        TrainStandardizer,
        assert_selection_blind,
        balanced_weights,
        weighted_ridge_fit,
    )

    assert_selection_blind(train_records, validation_records)
    torch.manual_seed(int(seed))
    normalizer = TrainStandardizer.fit(
        [[float(row[name]) for name in features] for row in train_records], features
    )

    def feature_tensor(records):
        return torch.tensor(
            normalizer.transform(
                [[float(row[name]) for name in features] for row in records]
            ),
            dtype=torch.float32,
        )

    def supervised_tensors(records):
        x = feature_tensor(records)
        y = torch.tensor(
            [float(row["persistence_logit"]) for row in records], dtype=torch.float32
        )
        weight = torch.tensor(
            balanced_weights(records, task_balanced=True), dtype=torch.float32
        )
        return x, y, weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, w_train = (
        value.to(device) for value in supervised_tensors(train_records)
    )
    x_validation, y_validation, w_validation = (
        value.to(device) for value in supervised_tensors(validation_records)
    )
    x_application = feature_tensor(application_records).to(device)

    class ResidualMLP(torch.nn.Module):
        """Nonlinear residual around an exactly representable linear solution."""

        def __init__(self):
            super().__init__()
            self.linear_skip = torch.nn.Linear(len(features), 1)
            self.residual = torch.nn.Sequential(
                torch.nn.Linear(len(features), int(hidden_sizes[0])),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_sizes[0]), int(hidden_sizes[1])),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(hidden_sizes[1]), 1),
            )

        def forward(self, values):
            return self.linear_skip(values) + self.residual(values)

    model = ResidualMLP()
    linear_coefficient = weighted_ridge_fit(
        x_train.cpu().numpy(),
        y_train.cpu().numpy(),
        w_train.cpu().numpy(),
    )
    with torch.no_grad():
        model.linear_skip.bias.copy_(
            torch.tensor([linear_coefficient[0]], dtype=torch.float32)
        )
        model.linear_skip.weight.copy_(
            torch.tensor(linear_coefficient[1:], dtype=torch.float32)[None, :]
        )
        # At initialization the network is exactly the fitted linear model.  The
        # nonlinear branch may improve it, but early stopping can always retain
        # this valid solution.
        model.residual[-1].weight.zero_()
        model.residual[-1].bias.zero_()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    model.eval()
    with torch.no_grad():
        initial_prediction = model(x_validation).squeeze(-1)
        best_loss = float(
            torch.sum(w_validation * (initial_prediction - y_validation) ** 2)
            / w_validation.sum()
        )
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_train).squeeze(-1)
        loss = torch.sum(w_train * (prediction - y_train) ** 2) / w_train.sum()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_validation).squeeze(-1)
            validation_loss = float(
                torch.sum(w_validation * (validation_prediction - y_validation) ** 2)
                / w_validation.sum()
            )
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
        prediction = model(x_application).squeeze(-1).cpu().numpy()
    return {
        "prediction": np.asarray(prediction, dtype=float),
        "selected_epochs": best_epoch,
        "trained_epochs": epoch + 1,
        "validation_mse": best_loss,
        "normalizer": normalizer,
        "linear_skip_initialized": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
