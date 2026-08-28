"""Project oriented interventions onto each task's own decision variable."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def functional_effect(readout, positive, negative) -> float:
    return float(np.asarray(readout, dtype=float) @ (np.asarray(positive) - np.asarray(negative)))


def _onset(values, fraction):
    values = np.abs(np.asarray(values, dtype=float))
    threshold = float(fraction) * float(values.max())
    indices = np.flatnonzero(values >= threshold)
    return int(indices[0]) if len(indices) else -1


def profile_summary(profiles, *, onset_fraction=0.5):
    normalized = {}
    for name, values in profiles.items():
        values = np.asarray(values, dtype=float)
        scale = np.linalg.norm(values)
        normalized[name] = values if scale == 0 else values / scale
    correlations = []
    for left, right in combinations(sorted(normalized), 2):
        if (
            len(normalized[left]) < 2
            or np.std(normalized[left]) == 0
            or np.std(normalized[right]) == 0
        ):
            correlations.append(float("nan"))
        else:
            correlations.append(
                float(np.corrcoef(normalized[left], normalized[right])[0, 1])
            )
    finite_correlations = [value for value in correlations if np.isfinite(value)]
    return {
        "mean_profile_correlation": (
            float(np.mean(finite_correlations))
            if finite_correlations
            else float("nan")
        ),
        "onset_layers": {
            name: _onset(values, onset_fraction) for name, values in profiles.items()
        },
        "peak_layers": {
            name: int(np.argmax(np.abs(values))) for name, values in profiles.items()
        },
        "area_under_absolute_curve": {
            name: float(np.trapezoid(np.abs(values))) for name, values in profiles.items()
        },
        "normalized_late_early_effect": {
            name: float(
                np.mean(np.abs(values[-max(1, len(values) // 4) :]))
                / (np.mean(np.abs(values[: max(1, len(values) // 4)])) + 1e-12)
            )
            for name, values in profiles.items()
        },
    }
