"""Optional yoked controllability exposure followed by identical transfer."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .base_environment import BasePersistenceEnvironment, SemanticHistory, choice_block


TRY = "TRY"
QUIT = "QUIT"

LITERATURE = {
    "construct": "controllability transfer / learned helplessness",
    "source_paradigm": "yoked controllable versus uncontrollable exposure",
    "source_citation": "Maier & Seligman tradition; review PMCID: PMC10205144",
    "adaptation_notes": "Yoked symbolic exposure precedes an identical solvable transfer task.",
    "departures_from_original": ["described causal contingency", "point-based transfer"],
}


@dataclass(frozen=True)
class ExposureTrace:
    actions: tuple[str, ...]
    outcomes: tuple[bool, ...]
    required_actions: tuple[str, ...]
    contingent: bool


def build_yoked_exposure(seed, *, trials=8):
    rng = random.Random(int(seed) * 2 + 7001)
    actions = tuple(rng.choice(("A", "B")) for _ in range(int(trials)))
    required = tuple(rng.choice(("A", "B")) for _ in range(int(trials)))
    outcomes = tuple(action == target for action, target in zip(actions, required))
    return (
        ExposureTrace(actions, outcomes, required, True),
        ExposureTrace(actions, outcomes, required, False),
    )


@dataclass(frozen=True)
class ControllabilityCondition:
    exposure_type: str
    exposure_trials: int
    transfer_success_probability: float
    transfer_cost: int
    max_transfer_attempts: int

    def __post_init__(self):
        if self.exposure_type not in {"controllable", "uncontrollable"}:
            raise ValueError("unknown exposure type")
        if not 0 <= self.transfer_success_probability <= 1:
            raise ValueError("invalid transfer probability")
        if self.transfer_cost < 0 or self.max_transfer_attempts < 1:
            raise ValueError("invalid transfer cost or horizon")


class ControllabilityEnvironment(BasePersistenceEnvironment):
    task = "controllability"
    continue_action = TRY
    disengage_action = QUIT

    def __init__(self, condition, seed):
        super().__init__(condition, seed)
        traces = build_yoked_exposure(seed, trials=condition.exposure_trials)
        self.exposure = traces[0 if condition.exposure_type == "controllable" else 1]
        self.history = SemanticHistory()
        for action, success in zip(self.exposure.actions, self.exposure.outcomes):
            self.history.record(
                action,
                outcome=1 if success else 0,
                effort=1,
                reward=1 if success else 0,
            )
        rng = random.Random(self.seed * 2 + 7013)
        self.transfer_uniforms = [
            rng.random() for _ in range(condition.max_transfer_attempts)
        ]
        self.transfer_attempts = 0

    def current_state(self):
        return {
            **self.history.state(),
            "current_continue_cost": float(self.condition.transfer_cost),
            "current_outside_option": 0.0,
            "current_progress": self.transfer_attempts
            / self.condition.max_transfer_attempts,
            "current_success_evidence": self.condition.transfer_success_probability,
            "exposure_type": self.condition.exposure_type,
            "exposure_contingent": self.exposure.contingent,
            "exposure_success_rate": sum(self.exposure.outcomes)
            / len(self.exposure.outcomes),
            "transfer_attempts": self.transfer_attempts,
            "same_goal_across_steps": True,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {TRY, QUIT}:
            raise ValueError(f"invalid transfer action: {action}")
        if action == QUIT:
            return self._finish_transition(
                action,
                outcome=0,
                reward=0,
                effort=0,
                success=None,
                terminated=True,
                reason="quit",
                task_values={"transfer_attempts": self.transfer_attempts},
            )
        success = (
            self.transfer_uniforms[self.transfer_attempts]
            < self.condition.transfer_success_probability
        )
        self.transfer_attempts += 1
        exhausted = self.transfer_attempts >= self.condition.max_transfer_attempts
        terminated = success or exhausted
        reward = (8 if success else 0) - self.condition.transfer_cost
        return self._finish_transition(
            action,
            outcome=reward,
            reward=reward,
            effort=self.condition.transfer_cost,
            success=success,
            terminated=terminated,
            reason="solved" if success else "max_attempts" if exhausted else None,
            progress=self.transfer_attempts / self.condition.max_transfer_attempts,
            task_values={"transfer_attempts": self.transfer_attempts},
        )

    def _choice(self, mapping):
        return choice_block(mapping, TRY, QUIT, "TRY the new problem", "QUIT")

    def initial_prompt(self, mapping):
        outcomes = ", ".join(
            f"{action}:{'success' if success else 'failure'}"
            for action, success in zip(self.exposure.actions, self.exposure.outcomes)
        )
        contingency = (
            "In that earlier system, choosing the matching action caused success."
            if self.exposure.contingent
            else "In that earlier system, the outcomes were preset and did not depend on your action."
        )
        return (
            f"Earlier system outcomes: {outcomes}. {contingency}\n\n"
            "You now face a new, solvable system. Each attempt can solve it and costs "
            f"{self.condition.transfer_cost} points.\n\n"
            + self._choice(mapping)
        )

    def feedback_prompt(self, transition, mapping):
        return "That attempt did not solve the new system.\n\n" + self._choice(mapping)


def factorial_conditions(config):
    return [
        ControllabilityCondition(
            str(exposure),
            int(config["exposure_trials"]),
            float(probability),
            int(cost),
            int(config["max_transfer_attempts"]),
        )
        for exposure, probability, cost in product(
            config["exposure_types"],
            config["transfer_success_probabilities"],
            config["transfer_costs"],
        )
    ]
