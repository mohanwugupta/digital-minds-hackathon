"""Environment-spec normalization that never learns target-task statistics."""

from __future__ import annotations

import math


class StaticNormalizer:
    """Normalize semantic quantities using frozen environment specifications."""

    def __init__(self, task_specs):
        self.task_specs = {str(task): dict(spec) for task, spec in task_specs.items()}

    def _spec(self, task):
        if str(task) not in self.task_specs:
            raise KeyError(f"missing static normalization specification for {task!r}")
        return self.task_specs[str(task)]

    @staticmethod
    def _divide(value, scale):
        if value is None:
            return float("nan")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if not math.isfinite(number):
            return float("nan")
        scale = float(scale)
        if not scale > 0:
            raise ValueError("normalization scales must be positive")
        return number / scale

    def transform_row(self, task, row):
        spec = self._spec(task)
        payoff = spec["payoff_scale"]
        effort = spec.get("effort_scale", payoff)
        horizon = spec["horizon"]
        return {
            "cost_norm": self._divide(row.get("current_continue_cost"), payoff),
            "outside_norm": self._divide(row.get("current_outside_option"), payoff),
            "time_norm": self._divide(row.get("elapsed_steps"), horizon),
            "effort_norm": self._divide(row.get("cumulative_effort"), effort),
            "invested_norm": self._divide(row.get("already_invested_cost"), effort),
            "progress_norm": self._divide(
                row.get("current_progress"), spec.get("progress_scale", 1.0)
            ),
            "success_evidence": self._divide(row.get("current_success_evidence"), 1.0),
            "remaining_effort_norm": self._divide(
                row.get("expected_remaining_effort"), effort
            ),
            "remaining_time_norm": self._divide(row.get("remaining_time"), horizon),
            "continue_payoff_norm": self._divide(
                row.get("expected_continue_payoff"), payoff
            ),
            "futility_norm": self._divide(row.get("futility_evidence"), payoff),
        }

