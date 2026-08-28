"""Capacity-controlled, train-normalized task-specific neural readouts."""

from __future__ import annotations

import numpy as np

from computational_modeling.analysis.evaluate_models import persistence_metrics
from computational_modeling.models.base import linear_predict, weighted_ridge_fit


def fit_task_readout(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    *,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    train_weights=None,
    validation_weights=None,
    test_weights=None,
):
    """Fit one task only; validation labels select alpha and test never sets scale."""

    x_train = np.asarray(x_train, dtype=float)
    x_validation = np.asarray(x_validation, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_validation = np.asarray(y_validation, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = (x_train - mean) / scale
    validation = (x_validation - mean) / scale
    test = (x_test - mean) / scale
    train_weights = np.ones(len(train)) if train_weights is None else np.asarray(train_weights)
    validation_weights = np.ones(len(validation)) if validation_weights is None else np.asarray(validation_weights)
    test_weights = np.ones(len(test)) if test_weights is None else np.asarray(test_weights)
    candidates = []
    for alpha in alphas:
        coefficient = weighted_ridge_fit(
            train, y_train, train_weights, penalty=float(alpha)
        )
        prediction = linear_predict(validation, coefficient)
        mse = float(np.average((y_validation - prediction) ** 2, weights=validation_weights))
        candidates.append((mse, float(alpha)))
    validation_mse, selected_alpha = min(candidates)
    fit_x = np.concatenate((train, validation), axis=0)
    fit_y = np.concatenate((y_train, y_validation), axis=0)
    fit_weights = np.concatenate((train_weights, validation_weights), axis=0)
    coefficient = weighted_ridge_fit(
        fit_x, fit_y, fit_weights, penalty=selected_alpha
    )
    prediction = linear_predict(test, coefficient)
    metrics = persistence_metrics(y_test, prediction, test_weights)
    return {
        "selected_alpha": selected_alpha,
        "validation_mse": validation_mse,
        "coefficient": coefficient,
        "direction": coefficient[1:] / scale,
        "normalizer_mean": mean,
        "normalizer_scale": scale,
        "test_prediction": prediction,
        **{f"test_{name}": value for name, value in metrics.items()},
    }


class BlockProjector:
    """Fast data-independent orthonormal block projection for laptop readouts."""

    def __init__(self, width: int, dimensions: int, seed: int):
        width, dimensions = int(width), int(dimensions)
        if dimensions <= 0 or width % dimensions:
            raise ValueError("projection dimensions must evenly divide hidden width")
        rng = np.random.default_rng(seed)
        self.width = width
        self.dimensions = dimensions
        self.permutation = rng.permutation(width)
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        self.block = width // dimensions

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        ordered = values[:, self.permutation] * self.signs
        return ordered.reshape(len(values), self.dimensions, self.block).sum(axis=2) / np.sqrt(self.block)

    def lift_direction(self, projected_direction):
        projected_direction = np.asarray(projected_direction, dtype=float)
        ordered = np.repeat(projected_direction / np.sqrt(self.block), self.block) * self.signs
        output = np.empty(self.width, dtype=float)
        output[self.permutation] = ordered
        return output

