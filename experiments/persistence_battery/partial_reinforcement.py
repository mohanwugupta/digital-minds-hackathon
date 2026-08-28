"""Acquisition-history manipulation followed by unrewarded extinction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .base_environment import BasePersistenceEnvironment, SemanticHistory, choice_block


TRY_AGAIN = "TRY_AGAIN"
STOP = "STOP"

LITERATURE = {
    "construct": "partial-reinforcement extinction persistence",
    "source_paradigm": "partial reinforcement extinction effect (PREE)",
    "source_citation": "Capaldi sequential theory; review PMCID: PMC10842266",
    "adaptation_notes": "A visible acquisition sequence precedes an unannounced extinction phase.",
    "departures_from_original": ["symbolic point outcomes", "finite extinction horizon"],
}


@dataclass(frozen=True)
class PartialReinforcementCondition:
    reinforcement_schedule: str
    acquisition_trials: int
    partial_probability: float
    extinction_try_cost: int
    expected_reward_per_trial: int = 2

    def __post_init__(self):
        if self.reinforcement_schedule not in {"continuous", "partial"}:
            raise ValueError("unknown reinforcement schedule")
        if self.acquisition_trials < 2 or not 0 < self.partial_probability < 1:
            raise ValueError("invalid acquisition schedule")
        if self.extinction_try_cost < 0 or self.expected_reward_per_trial <= 0:
            raise ValueError("invalid extinction cost or reward")


class PartialReinforcementEnvironment(BasePersistenceEnvironment):
    task = "partial_reinforcement"
    continue_action = TRY_AGAIN
    disengage_action = STOP

    def __init__(self, condition, seed, *, max_extinction_attempts=12):
        super().__init__(condition, seed)
        self.max_extinction_attempts = int(max_extinction_attempts)
        rng = random.Random(self.seed * 2 + 5003)
        if condition.reinforcement_schedule == "continuous":
            self.acquisition_reinforced = [True] * condition.acquisition_trials
            acquisition_reward = condition.expected_reward_per_trial
        else:
            self.acquisition_reinforced = [
                rng.random() < condition.partial_probability
                for _ in range(condition.acquisition_trials)
            ]
            if all(self.acquisition_reinforced):
                self.acquisition_reinforced[-1] = False
            if not any(self.acquisition_reinforced):
                self.acquisition_reinforced[0] = True
            acquisition_reward = int(
                round(condition.expected_reward_per_trial / condition.partial_probability)
            )
        self.acquisition_rewards = [
            acquisition_reward if reinforced else 0
            for reinforced in self.acquisition_reinforced
        ]
        self.history = SemanticHistory()
        for reward in self.acquisition_rewards:
            self.history.record(
                TRY_AGAIN,
                outcome=reward,
                effort=0,
                reward=reward,
                progress=None,
            )
        self.extinction_attempts = 0

    def current_state(self):
        return {
            **self.history.state(),
            "current_continue_cost": float(self.condition.extinction_try_cost),
            "current_outside_option": 0.0,
            "current_progress": self.extinction_attempts / self.max_extinction_attempts,
            "current_success_evidence": (
                sum(self.acquisition_reinforced) / len(self.acquisition_reinforced)
            ),
            "reinforcement_schedule": self.condition.reinforcement_schedule,
            "acquisition_trials": self.condition.acquisition_trials,
            "acquisition_reinforcement_rate": sum(self.acquisition_reinforced)
            / len(self.acquisition_reinforced),
            "extinction_attempts": self.extinction_attempts,
            "phase": "extinction",
            "same_goal_across_steps": True,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {TRY_AGAIN, STOP}:
            raise ValueError(f"invalid extinction action: {action}")
        if action == STOP:
            return self._finish_transition(
                action,
                outcome=0,
                reward=0,
                effort=0,
                success=None,
                terminated=True,
                reason="stop",
                task_values={"phase": "extinction"},
            )
        self.extinction_attempts += 1
        exhausted = self.extinction_attempts >= self.max_extinction_attempts
        reward = -self.condition.extinction_try_cost
        return self._finish_transition(
            action,
            outcome=reward,
            reward=reward,
            effort=self.condition.extinction_try_cost,
            success=False,
            terminated=exhausted,
            reason="max_extinction_attempts" if exhausted else None,
            progress=self.extinction_attempts / self.max_extinction_attempts,
            task_values={
                "phase": "extinction",
                "reinforced": False,
                "extinction_attempts": self.extinction_attempts,
            },
        )

    def _choice(self, mapping):
        return choice_block(
            mapping,
            TRY_AGAIN,
            STOP,
            "TRY for the reward again",
            "STOP trying",
        )

    def initial_prompt(self, mapping):
        outcomes = ", ".join(
            f"+{reward}" if reward else "no reward"
            for reward in self.acquisition_rewards
        )
        return (
            "You have repeatedly tried the same reward opportunity. Its recent outcomes were:\n"
            f"{outcomes}\n\nYou may try the same opportunity again. Each new try costs {self.condition.extinction_try_cost} points.\n\n"
            + self._choice(mapping)
        )

    def feedback_prompt(self, transition, mapping):
        return (
            "That try produced no reward.\n\n" + self._choice(mapping)
        )


def factorial_conditions(config):
    return [
        PartialReinforcementCondition(
            str(schedule),
            int(trials),
            float(probability),
            int(cost),
            int(expected),
        )
        for schedule, trials, probability, cost, expected in product(
            config["reinforcement_schedules"],
            config["acquisition_trials"],
            config["partial_probabilities"],
            config["extinction_try_costs"],
            config["expected_rewards_per_trial"],
        )
    ]
