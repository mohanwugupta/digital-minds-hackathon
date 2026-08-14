"""Sparse input-dimension selection for trained value probes."""

import math
from typing import Dict

import torch

from .value_probe import ValueProbe, td_loss


def input_importance(probe: ValueProbe) -> torch.Tensor:
    return probe.hidden.weight.detach().abs().sum(dim=0)


def rank_input_dimensions(probe: ValueProbe) -> torch.Tensor:
    return torch.argsort(input_importance(probe), descending=True)


def select_top_fraction(probe: ValueProbe, fraction: float = 0.01) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, math.ceil(probe.hidden.in_features * fraction))
    return rank_input_dimensions(probe)[:count]


def dimension_mask(probe: ValueProbe, indices: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(probe.hidden.in_features, dtype=probe.hidden.weight.dtype)
    mask[indices.detach().cpu().long()] = 1.0
    return mask


def evaluate_full_and_pruned(
    probe: ValueProbe,
    data: Dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> Dict[str, float]:
    probe.eval()
    device = next(probe.parameters()).device
    tensors = {key: value.to(device) for key, value in data.items()}
    mask = dimension_mask(probe, indices).to(device)
    with torch.no_grad():
        full, _ = td_loss(
            probe, tensors["states"], tensors["next_states"], tensors["rewards"], tensors["terminal"]
        )
        pruned, _ = td_loss(
            probe,
            tensors["states"], tensors["next_states"], tensors["rewards"], tensors["terminal"],
            input_mask=mask,
        )
    return {"full_td_mse": float(full.cpu()), "pruned_td_mse": float(pruned.cpu())}
