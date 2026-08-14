"""Masked, activation-scale-normalized bidirectional value steering."""

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from .value_probe import ValueProbe


def build_value_direction(
    probe: ValueProbe,
    hidden: torch.Tensor,
    neuron_indices: torch.Tensor,
    *,
    magnitude: float = 0.1,
) -> torch.Tensor:
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    candidate = hidden.detach().to(next(probe.parameters()).device).clone().requires_grad_(True)
    value = probe(candidate).sum()
    gradient = torch.autograd.grad(value, candidate)[0]
    mask = torch.zeros_like(gradient)
    indices = neuron_indices.to(gradient.device).long()
    mask[..., indices] = 1.0
    # A natural-gradient-like scaling makes the step relative to frozen
    # train-state activation variation while retaining positive dV direction.
    scaled = gradient * probe.std.square() * mask
    whitened = scaled / probe.std
    norm = whitened.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    direction = scaled / norm * magnitude
    return direction.detach().to(hidden.device, dtype=hidden.dtype)


def steer_hidden(hidden: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    if float(alpha) == 0.0:
        return hidden
    return hidden + float(alpha) * direction


@dataclass(frozen=True)
class CalibrationResult:
    magnitude: float
    ordered_fraction: float
    relative_rms: float


def calibrate_magnitude(
    probe: ValueProbe,
    validation_states: torch.Tensor,
    neuron_indices: torch.Tensor,
    candidates: Iterable[float] = (0.01, 0.025, 0.05, 0.1, 0.2),
    required_fraction: float = 0.9,
    max_relative_rms: float = 0.25,
) -> CalibrationResult:
    best: Optional[CalibrationResult] = None
    for magnitude in candidates:
        direction = build_value_direction(probe, validation_states, neuron_indices, magnitude=magnitude)
        with torch.no_grad():
            negative = probe(validation_states - direction)
            baseline = probe(validation_states)
            positive = probe(validation_states + direction)
        ordered = float(((positive > baseline) & (baseline > negative)).float().mean().cpu())
        relative = float((direction / probe.std.to(direction.device)).square().mean().sqrt().cpu())
        result = CalibrationResult(float(magnitude), ordered, relative)
        if best is None or result.ordered_fraction > best.ordered_fraction:
            best = result
        if ordered >= required_fraction and relative <= max_relative_rms:
            return result
    raise RuntimeError(f"no steering magnitude passed calibration; best={best}")


def sample_random_neuron_sets(
    width: int,
    count: int,
    *,
    n_sets: int = 20,
    seed: int = 0,
    exclude: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    excluded = set() if exclude is None else set(exclude.detach().cpu().long().tolist())
    available = torch.tensor([index for index in range(width) if index not in excluded])
    if not 0 < count <= len(available):
        raise ValueError("count must be between one and width")
    generator = torch.Generator().manual_seed(seed)
    return [available[torch.randperm(len(available), generator=generator)[:count]] for _ in range(n_sets)]


def magnitude_match(direction: torch.Tensor, reference: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    target = (reference / std).square().sum(dim=-1, keepdim=True).sqrt()
    current = (direction / std).square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    return direction * (target / current)


def random_masked_direction(
    reference: torch.Tensor,
    neuron_indices: torch.Tensor,
    std: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Create a deterministic random-neuron direction with matched whitened norm."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate = torch.zeros_like(reference)
    indices = neuron_indices.to(reference.device).long()
    signs = torch.randint(0, 2, (indices.numel(),), generator=generator).float().mul_(2).sub_(1)
    candidate[..., indices] = signs.to(reference.device, reference.dtype) * std.to(
        reference.device, reference.dtype
    )[..., indices]
    return magnitude_match(candidate, reference, std.to(reference.device, reference.dtype))
