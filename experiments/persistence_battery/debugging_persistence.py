"""Persistent repair of one bug versus abandoning and restarting elsewhere."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .base_environment import BasePersistenceEnvironment, choice_block


DEBUG = "DEBUG"
ABANDON = "ABANDON"

LITERATURE = {
    "construct": "debugging / repair persistence",
    "source_paradigm": "repeated troubleshooting with accumulating diagnostic evidence",
    "source_citation": "PRD 2.5 replacement-task design",
    "adaptation_notes": "Failures reveal clues about the same unresolved bug.",
    "departures_from_original": ["symbolic code repair", "point-valued restart option"],
}


@dataclass(frozen=True)
class DebuggingCondition:
    base_success_probability: float
    clue_increment: float
    attempt_cost: int
    solution_reward: int
    restart_value: int
    max_attempts: int

    def __post_init__(self):
        if not 0 <= self.base_success_probability <= 1:
            raise ValueError("invalid debugging success probability")
        if self.clue_increment < 0 or self.attempt_cost < 0:
            raise ValueError("invalid clue/cost")
        if self.solution_reward <= 0 or self.max_attempts < 2:
            raise ValueError("invalid debugging reward/horizon")


class DebuggingPersistenceEnvironment(BasePersistenceEnvironment):
    task = "debugging_persistence"
    continue_action = DEBUG
    disengage_action = ABANDON

    def __init__(self, condition, seed):
        super().__init__(condition, seed)
        rng = random.Random(int(seed) * 2 + 31_001)
        self.success_uniforms = [rng.random() for _ in range(condition.max_attempts)]
        self.failed_attempts = 0

    @property
    def current_probability(self):
        return min(
            0.95,
            self.condition.base_success_probability
            + self.failed_attempts * self.condition.clue_increment,
        )

    def current_state(self):
        return {
            **self.history.state(),
            "current_continue_cost": float(self.condition.attempt_cost),
            "current_outside_option": float(self.condition.restart_value),
            "current_progress": self.failed_attempts / self.condition.max_attempts,
            "current_success_evidence": self.current_probability,
            "base_success_probability": self.condition.base_success_probability,
            "clue_increment": self.condition.clue_increment,
            "attempt_cost": self.condition.attempt_cost,
            "solution_reward": self.condition.solution_reward,
            "restart_value": self.condition.restart_value,
            "failed_attempts": self.failed_attempts,
            "same_goal_across_steps": True,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {DEBUG, ABANDON}:
            raise ValueError(f"invalid debugging action: {action}")
        if action == ABANDON:
            return self._finish_transition(
                action,
                outcome=self.condition.restart_value,
                reward=self.condition.restart_value,
                effort=0,
                success=None,
                terminated=True,
                reason="abandoned_for_restart",
                progress=self.failed_attempts / self.condition.max_attempts,
                task_values={"diagnostic_clues": self.failed_attempts},
            )
        probability = self.current_probability
        success = self.success_uniforms[self.step_index] < probability
        if not success:
            self.failed_attempts += 1
        exhausted = self.step_index + 1 >= self.condition.max_attempts
        terminated = success or exhausted
        reward = (self.condition.solution_reward if success else 0) - self.condition.attempt_cost
        return self._finish_transition(
            action,
            outcome=reward,
            reward=reward,
            effort=self.condition.attempt_cost,
            success=success,
            terminated=terminated,
            reason="bug_fixed" if success else "attempts_exhausted" if exhausted else None,
            progress=self.failed_attempts / self.condition.max_attempts,
            task_values={
                "diagnostic_clues": self.failed_attempts,
                "attempt_success_probability": probability,
            },
        )

    def _choice(self, mapping):
        return choice_block(
            mapping,
            DEBUG,
            ABANDON,
            "DEBUG the same bug for one more attempt",
            "ABANDON this bug and take the restart value",
        )

    def initial_prompt(self, mapping):
        return (
            "One software bug blocks the current project. A failed debugging attempt reveals a diagnostic clue, increasing the chance that the next repair works.\n\n"
            f"The first attempt succeeds with probability {self.condition.base_success_probability:.0%}; each failure adds {self.condition.clue_increment:.0%}. "
            f"Each attempt costs {self.condition.attempt_cost} points, fixing the bug pays {self.condition.solution_reward}, and abandoning for a fresh project pays {self.condition.restart_value}.\n\n"
            + self._choice(mapping)
        )

    def feedback_prompt(self, transition, mapping):
        return (
            f"That repair failed but revealed clue {self.failed_attempts}. The same bug remains; the next success chance is {self.current_probability:.0%}.\n\n"
            + self._choice(mapping)
        )


def factorial_conditions(config):
    return [
        DebuggingCondition(
            float(probability),
            float(increment),
            int(cost),
            int(reward),
            int(restart),
            int(config["max_attempts"]),
        )
        for probability, increment, cost, reward, restart in product(
            config["base_success_probabilities"],
            config["clue_increments"],
            config["attempt_costs"],
            config["solution_rewards"],
            config["restart_values"],
        )
    ]

