"""Normalized, validation-calibrated steering for frozen ridge probes."""

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .ridge_probe import RidgeProbe


@dataclass(frozen=True)
class RidgeCalibration:
    magnitude: float
    decoded_shift: float
    decoded_sd_shift: float
    relative_rms: float
    direction_l2_norm: float
    validation_states: int

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_probe_direction(probe: RidgeProbe) -> torch.Tensor:
    """Return the unit activation-space direction that increases probe output."""
    direction = probe.raw_activation_direction().detach().float()
    norm = direction.norm()
    if float(norm) <= 1e-12:
        raise ValueError("ridge probe has a zero activation-space direction")
    return direction / norm


def calibrate_ridge_magnitude(
    probe: RidgeProbe,
    validation_states: torch.Tensor,
    *,
    decoded_sd_candidates: Iterable[float] = (1.0, 0.5, 0.25, 0.1),
    max_relative_rms: float = 0.25,
) -> RidgeCalibration:
    """Choose the largest safe validation-checked decoded-SD displacement."""
    states = validation_states.detach().float().cpu()
    if states.ndim != 2 or states.shape[1] != probe.weight.numel():
        raise ValueError("validation states must be [states, ridge input width]")
    if len(states) < 1:
        raise ValueError("at least one validation state is required")
    if max_relative_rms <= 0:
        raise ValueError("max_relative_rms must be positive")
    candidates = tuple(
        sorted((float(value) for value in decoded_sd_candidates), reverse=True)
    )
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("decoded-SD candidates must be positive")

    direction = normalized_probe_direction(probe).cpu()
    raw_gradient = probe.raw_activation_direction().detach().float().cpu()
    response_per_activation_unit = float(torch.dot(raw_gradient, direction))
    if response_per_activation_unit <= 0:
        raise RuntimeError("normalized ridge direction does not increase prediction")
    baseline = probe.predict(states)
    best = None
    for decoded_sd_shift in candidates:
        desired_shift = decoded_sd_shift * float(probe.target_std)
        magnitude = desired_shift / response_per_activation_unit
        delta = direction * magnitude
        positive = probe.predict(states + delta)
        negative = probe.predict(states - delta)
        observed_shift = float((positive - baseline).mean())
        ordered = bool(torch.all(positive > baseline) and torch.all(baseline > negative))
        relative_rms = float(
            ((delta / probe.state_std.detach().float().cpu()) ** 2).mean().sqrt()
        )
        result = RidgeCalibration(
            magnitude=float(magnitude),
            decoded_shift=observed_shift,
            decoded_sd_shift=observed_shift / float(probe.target_std),
            relative_rms=relative_rms,
            direction_l2_norm=float(direction.norm()),
            validation_states=len(states),
        )
        if best is None or result.relative_rms < best.relative_rms:
            best = result
        if ordered and relative_rms <= max_relative_rms:
            return result
    raise RuntimeError(
        "no ridge steering magnitude passed calibration; "
        f"smallest relative RMS was {best.relative_rms:.6f}"
    )


def matched_sign_random_directions(
    reference_delta: torch.Tensor,
    *,
    n_directions: int = 20,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Sign-randomize a delta while preserving every coordinate magnitude.

    Coordinatewise absolute values are unchanged, so Euclidean norm and RMS
    under any fixed per-coordinate activation scale are both matched exactly.
    """
    if n_directions < 1:
        raise ValueError("n_directions must be positive")
    reference = reference_delta.detach().clone()
    if reference.ndim != 1 or float(reference.norm()) <= 1e-12:
        raise ValueError("reference delta must be a nonzero vector")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reference_cpu = reference.float().cpu()
    excluded = {
        tuple(reference_cpu.tolist()),
        tuple((-reference_cpu).tolist()),
    }
    seen, output = set(), []
    attempts = 0
    while len(output) < n_directions:
        attempts += 1
        if attempts > n_directions * 100:
            raise RuntimeError("could not generate enough unique matched directions")
        signs = torch.randint(
            0, 2, reference_cpu.shape, generator=generator, dtype=torch.int64
        ).float().mul_(2).sub_(1)
        candidate = reference_cpu * signs
        key = tuple(candidate.tolist())
        if key in excluded or key in seen:
            continue
        seen.add(key)
        output.append(candidate.to(reference.device, reference.dtype))
    return output


def apply_ridge_steering(
    hidden: torch.Tensor, delta: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Apply the frozen delta; alpha zero returns the exact input object/value."""
    if float(alpha) == 0.0:
        return hidden
    return hidden + float(alpha) * delta.to(hidden.device, hidden.dtype)
