"""Prospectively matched sunk-cost persistence task."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .base_environment import BasePersistenceEnvironment, SemanticHistory, choice_block


CONTINUE_WAITING = "CONTINUE_WAITING"
ABANDON = "ABANDON"

LITERATURE = {
    "construct": "sunk-cost persistence",
    "source_paradigm": "Restaurant Row / Web-Surf change-of-mind waiting",
    "source_citation": "Sweis et al., Science (2018), PMCID: PMC6377599",
    "adaptation_notes": "Prior investment is crossed with identical prospective states.",
    "departures_from_original": ["point projects", "discrete work steps"],
}


@dataclass(frozen=True)
class SunkCostCondition:
    prior_investment: int
    remaining_steps: int
    reward_magnitude: int
    outside_option: int
    step_cost: int
    success_probability: float

    def __post_init__(self):
        if self.prior_investment < 0 or self.remaining_steps < 1:
            raise ValueError("invalid sunk-cost step counts")
        if self.step_cost < 0 or not 0 <= self.success_probability <= 1:
            raise ValueError("invalid sunk-cost cost or probability")


def matched_sunk_cost_conditions(
    *,
    prior_investments,
    remaining_steps,
    reward_magnitude,
    outside_option,
    step_cost,
    success_probability,
):
    return tuple(
        SunkCostCondition(
            int(prior),
            int(remaining_steps),
            int(reward_magnitude),
            int(outside_option),
            int(step_cost),
            float(success_probability),
        )
        for prior in prior_investments
    )


class SunkCostEnvironment(BasePersistenceEnvironment):
    task = "sunk_cost"
    continue_action = CONTINUE_WAITING
    disengage_action = ABANDON

    def __init__(self, condition, seed):
        super().__init__(condition, seed)
        self.history = SemanticHistory(
            elapsed_steps=condition.prior_investment,
            cumulative_effort=condition.prior_investment * condition.step_cost,
        )
        self.remaining = condition.remaining_steps
        self.success_uniform = random.Random(self.seed * 2 + 3011).random()

    def current_state(self):
        return {
            **self.history.state(),
            "current_continue_cost": float(self.condition.step_cost),
            "current_outside_option": float(self.condition.outside_option),
            # Prospective progress is deliberately held apart from prior sunk cost.
            "current_progress": None,
            "current_success_evidence": float(self.condition.success_probability),
            "prior_investment": self.condition.prior_investment,
            "remaining_steps": self.remaining,
            "reward_magnitude": self.condition.reward_magnitude,
            "success_probability": self.condition.success_probability,
            "same_goal_across_steps": True,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {CONTINUE_WAITING, ABANDON}:
            raise ValueError(f"invalid sunk-cost action: {action}")
        if action == ABANDON:
            return self._finish_transition(
                action,
                outcome=self.condition.outside_option,
                reward=self.condition.outside_option,
                effort=0,
                success=None,
                terminated=True,
                reason="abandon",
                task_values={"remaining_steps": self.remaining},
            )
        self.remaining -= 1
        completed = self.remaining == 0
        success = completed and self.success_uniform < self.condition.success_probability
        reward = -self.condition.step_cost
        if success:
            reward += self.condition.reward_magnitude
        return self._finish_transition(
            action,
            outcome=reward,
            reward=reward,
            effort=self.condition.step_cost,
            success=success if completed else False,
            terminated=completed,
            reason="project_succeeded" if success else "project_failed" if completed else None,
            task_values={"remaining_steps": self.remaining, "project_succeeded": success},
        )

    def _choice(self, mapping):
        return choice_block(
            mapping,
            CONTINUE_WAITING,
            ABANDON,
            "CONTINUE the project for one step",
            "ABANDON the project",
        )

    def initial_prompt(self, mapping):
        return (
            "You previously accepted a project with a delayed point reward. Past work cannot be recovered and does not change the remaining odds or costs.\n\n"
            f"You have already invested {self.condition.prior_investment} step(s). Exactly {self.remaining} future step(s) remain. "
            f"Each remaining step costs {self.condition.step_cost} points. Completion has a {self.condition.success_probability:.0%} chance of paying {self.condition.reward_magnitude} points. "
            f"Abandoning now gives {self.condition.outside_option} points. Decide using the stated future consequences.\n\n"
            + self._choice(mapping)
        )

    def feedback_prompt(self, transition, mapping):
        return (
            f"One additional project step was completed. {self.remaining} step(s) remain.\n\n"
            + self._choice(mapping)
        )


def factorial_conditions(config):
    return [
        SunkCostCondition(
            int(prior),
            int(remaining),
            int(reward),
            int(outside),
            int(cost),
            float(success),
        )
        for prior, remaining, reward, outside, cost, success in product(
            config["prior_investments"],
            config["remaining_steps"],
            config["reward_magnitudes"],
            config["outside_options"],
            config["step_costs"],
            config["success_probabilities"],
        )
    ]
