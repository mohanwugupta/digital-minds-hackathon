"""Layer-specific two-layer value probe trained with a TD(0) objective."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class ValueProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        *,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("probe dimensions must be positive")
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.register_buffer("mean", torch.zeros(input_dim) if mean is None else mean.float().clone())
        frozen_std = torch.ones(input_dim) if std is None else std.float().clone()
        self.register_buffer("std", frozen_std.clamp_min(1e-6))

    def normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        return (hidden - self.mean) / self.std

    def forward(self, hidden: torch.Tensor, input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normalized = self.normalize(hidden)
        if input_mask is not None:
            normalized = normalized * input_mask
        return self.output(torch.relu(self.hidden(normalized))).squeeze(-1)


def td_loss(
    probe: ValueProbe,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    terminal: torch.Tensor,
    gamma: float = 1.0,
    input_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = probe(states, input_mask=input_mask)
    # Detaching the bootstrap target is standard semi-gradient TD learning.
    with torch.no_grad():
        next_values = probe(next_states, input_mask=input_mask)
        target = rewards + gamma * next_values * (~terminal.bool()).to(next_values.dtype)
    delta = target - values
    return delta.square().mean(), delta


@dataclass
class ProbeTrainingResult:
    probe: ValueProbe
    best_validation_loss: float
    epochs_trained: int
    history: list


def fit_value_probe(
    train: Dict[str, torch.Tensor],
    validation: Dict[str, torch.Tensor],
    *,
    hidden_dim: int = 1024,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
) -> ProbeTrainingResult:
    torch.manual_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_states = train["states"].float()
    mean = train_states.mean(dim=0)
    std = train_states.std(dim=0, unbiased=False).clamp_min(1e-6)
    probe = ValueProbe(train_states.shape[-1], hidden_dim, mean=mean, std=std).to(target_device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, stale = None, float("inf"), 0
    history = []

    for epoch in range(epochs):
        probe.train()
        permutation = torch.randperm(len(train_states), generator=generator)
        train_losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch = {key: value[indices].to(target_device) for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = td_loss(
                probe, batch["states"], batch["next_states"], batch["rewards"], batch["terminal"]
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        probe.eval()
        validation_device = {key: value.to(target_device) for key, value in validation.items()}
        with torch.no_grad():
            validation_loss, _ = td_loss(
                probe,
                validation_device["states"],
                validation_device["next_states"],
                validation_device["rewards"],
                validation_device["terminal"],
            )
        current = float(validation_loss.cpu())
        history.append({
            "epoch": epoch + 1,
            "train_loss": sum(train_losses) / max(1, len(train_losses)),
            "validation_loss": current,
        })
        if current < best_loss:
            best_loss = current
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    return ProbeTrainingResult(probe.cpu(), best_loss, len(history), history)
