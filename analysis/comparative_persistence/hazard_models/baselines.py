"""Registry and numerical primitives for the comparative hazard model zoo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..semantic_features import ALL_OBSERVABLE_FEATURES, HISTORY_FEATURES, IMMEDIATE_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    code: str
    features: tuple[str, ...]
    kind: str = "linear"
    oracle: bool = False


MODEL_SPECS = {
    "intercept": ModelSpec("intercept", "M0", ()),
    "time_only": ModelSpec("time_only", "M1", ("time_norm",)),
    "immediate_state": ModelSpec("immediate_state", "M2", IMMEDIATE_FEATURES),
    "finite_history": ModelSpec("finite_history", "M3", HISTORY_FEATURES, "finite_history"),
    "exponential_reward": ModelSpec("exponential_reward", "M4", (*IMMEDIATE_FEATURES, "reward_kernel"), "decay"),
    "perseveration": ModelSpec("perseveration", "M5", (*IMMEDIATE_FEATURES, "choice_kernel"), "decay"),
    "dual_history": ModelSpec("dual_history", "M6", (*IMMEDIATE_FEATURES, "reward_kernel", "choice_kernel"), "decay"),
    "dynamic_reevaluation": ModelSpec("dynamic_reevaluation", "M7", ("reevaluation_advantage",)),
    "dynamic_reevaluation_oracle": ModelSpec("dynamic_reevaluation_oracle", "M7O", ("oracle_reevaluation_advantage",), oracle=True),
    "option_termination": ModelSpec("option_termination", "M8", ("option_advantage",)),
    "competitive_time_reward": ModelSpec("competitive_time_reward", "M9", ("time_norm", "reward_kernel", "cost_norm", "outside_norm"), "decay"),
    "latent_commitment": ModelSpec("latent_commitment", "M10", IMMEDIATE_FEATURES, "latent"),
    "sunk_extension": ModelSpec("sunk_extension", "M11", (*IMMEDIATE_FEATURES, "invested_norm")),
    "flexible_linear": ModelSpec("flexible_linear", "M12", ALL_OBSERVABLE_FEATURES, "flexible"),
    "mlp": ModelSpec("mlp", "M13", ALL_OBSERVABLE_FEATURES, "mlp"),
    "gru": ModelSpec("gru", "M14", ALL_OBSERVABLE_FEATURES, "gru"),
}


def binary_log_loss(y, probability, weights=None):
    y = np.asarray(y, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    loss = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    return float(np.average(loss, weights=weights))


def fit_ridge_logistic(x, y, *, penalty=0.1, weights=None, penalty_weights=None, prior=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("invalid logistic design")
    design = np.column_stack((np.ones(len(x)), x))
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    mask = np.ones(design.shape[1], dtype=float)
    mask[0] = 0.0
    if penalty_weights is not None:
        supplied = np.asarray(penalty_weights, dtype=float)
        if len(supplied) != x.shape[1]:
            raise ValueError("penalty weight count does not match features")
        mask[1:] = supplied
    center = np.zeros(design.shape[1]) if prior is None else np.asarray(prior, dtype=float)

    def objective(coefficient):
        score = design @ coefficient
        data = np.average(np.logaddexp(0, score) - y * score, weights=weights)
        difference = mask * (coefficient - center)
        return float(data + 0.5 * float(penalty) * np.sum(difference**2))

    def gradient(coefficient):
        error = expit(design @ coefficient) - y
        data = design.T @ (weights * error) / weights.sum()
        return data + float(penalty) * mask**2 * (coefficient - center)

    result = minimize(
        objective,
        center.copy(),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 600, "ftol": 1e-10},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"logistic optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def predict_ridge_logistic(coefficient, x):
    x = np.asarray(x, dtype=float)
    return expit(np.column_stack((np.ones(len(x)), x)) @ coefficient)
