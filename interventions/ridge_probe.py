"""Episode-held-out ridge-linear probes with primal/dual closed-form fitting."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch


@dataclass
class RidgeProbe:
    weight: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    target_mean: float
    target_std: float
    alpha: float
    target: str

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        normalized = (states.float() - self.state_mean) / self.state_std
        prediction_z = normalized @ self.weight
        return prediction_z * self.target_std + self.target_mean

    def raw_activation_direction(self) -> torch.Tensor:
        """Gradient of the prediction with respect to unnormalized activations."""
        return self.weight * self.target_std / self.state_std

    def to_payload(self) -> dict:
        return {
            "weight": self.weight.detach().cpu(),
            "state_mean": self.state_mean.detach().cpu(),
            "state_std": self.state_std.detach().cpu(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "alpha": self.alpha,
            "target": self.target,
            "input_dim": int(self.weight.numel()),
            "probe_type": "ridge_linear",
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "RidgeProbe":
        if payload.get("probe_type") != "ridge_linear":
            raise ValueError("artifact is not a ridge-linear probe")
        return cls(
            weight=payload["weight"].float(),
            state_mean=payload["state_mean"].float(),
            state_std=payload["state_std"].float(),
            target_mean=float(payload["target_mean"]),
            target_std=float(payload["target_std"]),
            alpha=float(payload["alpha"]),
            target=str(payload["target"]),
        )


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    prediction, target = prediction.float().cpu(), target.float().cpu()
    residual_sum = float((prediction - target).square().sum())
    centered_sum = float((target - target.mean()).square().sum())
    correlation = 0.0
    if (
        len(target) > 1
        and float(prediction.std(unbiased=False)) > 0
        and float(target.std(unbiased=False)) > 0
    ):
        correlation = float(torch.corrcoef(torch.stack([prediction, target]))[0, 1])
    return {
        "mse": residual_sum / max(1, len(target)),
        "r_squared": 1.0 - residual_sum / centered_sum if centered_sum > 0 else 0.0,
        "correlation": correlation,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(unbiased=False)),
        "target_mean": float(target.mean()),
        "target_std": float(target.std(unbiased=False)),
    }


def fit_ridge_targets(
    train_states: torch.Tensor,
    train_targets: Mapping[str, torch.Tensor],
    validation_states: torch.Tensor,
    validation_targets: Mapping[str, torch.Tensor],
    *,
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    device: Optional[str] = None,
) -> tuple[dict[str, RidgeProbe], dict]:
    """Fit multiple targets while sharing one eigendecomposition per layer."""
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("ridge alphas must be positive")
    if set(train_targets) != set(validation_targets):
        raise ValueError("train and validation target names must match")
    if len(train_states) < 2 or len(validation_states) < 1:
        raise ValueError("ridge fitting requires train and validation states")

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_states = train_states.float()
    validation_states = validation_states.float()
    state_mean = train_states.mean(dim=0)
    state_std = train_states.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = ((train_states - state_mean) / state_std).to(target_device)
    validation_x = ((validation_states - state_mean) / state_std).to(target_device)
    sample_count, feature_count = train_x.shape

    target_stats = {}
    train_y_columns = []
    validation_y = {}
    names = list(train_targets)
    for name in names:
        train_target = train_targets[name].float()
        mean = float(train_target.mean())
        std = max(float(train_target.std(unbiased=False)), 1e-6)
        target_stats[name] = (mean, std)
        train_y_columns.append(((train_target - mean) / std).to(target_device))
        validation_y[name] = validation_targets[name].float().to(target_device)
    train_y = torch.stack(train_y_columns, dim=1)

    mode = "primal" if feature_count <= sample_count else "dual"
    if mode == "primal":
        gram = (train_x.T @ train_x) / sample_count
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0)
        projected_targets = eigenvectors.T @ ((train_x.T @ train_y) / sample_count)
    else:
        gram = train_x @ train_x.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0)
        projected_targets = eigenvectors.T @ train_y

    candidates = {name: [] for name in names}
    selected = {}
    for alpha in alphas:
        if mode == "primal":
            weights = eigenvectors @ (
                projected_targets / (eigenvalues[:, None] + alpha)
            )
        else:
            dual = eigenvectors @ (
                projected_targets
                / (eigenvalues[:, None] + sample_count * alpha)
            )
            weights = train_x.T @ dual
        prediction_z = validation_x @ weights
        for column, name in enumerate(names):
            mean, std = target_stats[name]
            prediction = prediction_z[:, column] * std + mean
            metrics = regression_metrics(prediction, validation_y[name])
            candidates[name].append({"alpha": float(alpha), **metrics})
            if name not in selected or metrics["mse"] < selected[name]["metrics"]["mse"]:
                selected[name] = {
                    "alpha": float(alpha),
                    "weight": weights[:, column].detach().cpu().clone(),
                    "metrics": metrics,
                }

    probes = {}
    for name in names:
        mean, std = target_stats[name]
        probes[name] = RidgeProbe(
            weight=selected[name]["weight"],
            state_mean=state_mean.cpu(),
            state_std=state_std.cpu(),
            target_mean=mean,
            target_std=std,
            alpha=selected[name]["alpha"],
            target=name,
        )
    return probes, {
        "solver": mode,
        "train_states": sample_count,
        "input_dim": feature_count,
        "alpha_candidates": list(alphas),
        "validation_candidates": candidates,
    }


def save_ridge_probe(path: str, probe: RidgeProbe, metadata: dict) -> None:
    from experiments.runtime import atomic_torch_save

    atomic_torch_save({**probe.to_payload(), "metadata": metadata}, path)


def load_ridge_probe(path: str, map_location: str = "cpu") -> tuple[RidgeProbe, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return RidgeProbe.from_payload(payload), payload
