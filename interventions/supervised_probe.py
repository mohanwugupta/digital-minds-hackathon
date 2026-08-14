"""Stable supervised probes for standardized Monte Carlo future return."""

from dataclasses import dataclass
from typing import Optional

import torch

from interventions.value_probe import ValueProbe


@dataclass
class SupervisedProbeResult:
    probe: ValueProbe
    best_validation_mse: float
    epochs_trained: int
    history: list
    target_mean: float
    target_std: float


def supervised_metrics(
    probe: ValueProbe,
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_mean: float,
    target_std: float,
    input_mask: Optional[torch.Tensor] = None,
) -> dict:
    probe.eval()
    device = next(probe.parameters()).device
    with torch.no_grad():
        prediction_z = probe(
            states.float().to(device),
            input_mask=None if input_mask is None else input_mask.to(device),
        ).cpu()
    prediction = prediction_z * target_std + target_mean
    targets = targets.float().cpu()
    residual_sum = float((prediction - targets).square().sum())
    centered_sum = float((targets - targets.mean()).square().sum())
    correlation = 0.0
    if len(targets) > 1 and float(prediction.std(unbiased=False)) > 0:
        correlation = float(torch.corrcoef(torch.stack([prediction, targets]))[0, 1])
    return {
        "mse": residual_sum / len(targets),
        "r_squared": 1.0 - residual_sum / centered_sum if centered_sum > 0 else 0.0,
        "correlation": correlation,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(unbiased=False)),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std(unbiased=False)),
    }


def fit_supervised_probe(
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    validation_states: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    hidden_dim: int = 1024,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
    input_mask: Optional[torch.Tensor] = None,
) -> SupervisedProbeResult:
    """Fit direct future-return regression without a moving bootstrap target."""
    torch.manual_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_states = train_states.float()
    train_targets = train_targets.float()
    validation_states = validation_states.float()
    validation_targets = validation_targets.float()
    state_mean = train_states.mean(dim=0)
    state_std = train_states.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = float(train_targets.mean())
    target_std = max(float(train_targets.std(unbiased=False)), 1e-6)
    train_targets_z = (train_targets - target_mean) / target_std
    validation_targets_z = (validation_targets - target_mean) / target_std
    probe = ValueProbe(
        train_states.shape[-1], hidden_dim, mean=state_mean, std=state_std
    ).to(target_device)
    fixed_mask = None if input_mask is None else input_mask.float().to(target_device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, stale = None, float("inf"), 0
    history = []

    for epoch in range(epochs):
        probe.train()
        permutation = torch.randperm(len(train_states), generator=generator)
        training_losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            states = train_states[indices].to(target_device)
            targets = train_targets_z[indices].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(states, input_mask=fixed_mask)
            loss = (prediction - targets).square().mean()
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))

        probe.eval()
        with torch.no_grad():
            prediction = probe(
                validation_states.to(target_device), input_mask=fixed_mask
            )
            validation_loss = float(
                (prediction - validation_targets_z.to(target_device))
                .square()
                .mean()
                .cpu()
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse_z": sum(training_losses) / max(1, len(training_losses)),
                "validation_mse_z": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    return SupervisedProbeResult(
        probe.cpu(),
        best_loss,
        len(history),
        history,
        target_mean,
        target_std,
    )
