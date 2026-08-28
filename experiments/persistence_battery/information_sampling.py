"""Sequential evidence gathering with a separate final-answer stage."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random

from .base_environment import BasePersistenceEnvironment, choice_block


SAMPLE = "SAMPLE"
DECIDE = "DECIDE"
ANSWER_A = "ANSWER_A"
ANSWER_B = "ANSWER_B"

LITERATURE = {
    "construct": "epistemic persistence / information sampling",
    "source_paradigm": "Information Sampling Task",
    "source_citation": "Clark et al. paradigm; adaptation overview PMCID: PMC6795545",
    "adaptation_notes": "Evidence sampling and the substantive A/B answer are separate stages.",
    "departures_from_original": ["binary noisy signals", "Bayesian evidence state"],
}


@dataclass(frozen=True)
class InformationSamplingCondition:
    evidence_accuracy: float
    sample_cost: int
    error_penalty: int
    prior_a: float
    true_state: str
    max_samples: int

    def __post_init__(self):
        if not 0.5 < self.evidence_accuracy < 1:
            raise ValueError("evidence accuracy must lie in (0.5, 1)")
        if self.sample_cost < 0 or self.error_penalty < 0:
            raise ValueError("sampling cost and error penalty must be nonnegative")
        if not 0 < self.prior_a < 1 or self.true_state not in {"A", "B"}:
            raise ValueError("invalid prior or true state")
        if self.max_samples < 1:
            raise ValueError("max_samples must be positive")


class InformationSamplingEnvironment(BasePersistenceEnvironment):
    task = "information_sampling"
    continue_action = SAMPLE
    disengage_action = DECIDE

    def __init__(self, condition, seed):
        super().__init__(condition, seed)
        self.posterior_a = float(condition.prior_a)
        rng = random.Random(self.seed * 2 + 4001)
        self.observations = [
            condition.true_state
            if rng.random() < condition.evidence_accuracy
            else ("B" if condition.true_state == "A" else "A")
            for _ in range(condition.max_samples)
        ]
        self.revealed: list[str] = []

    def _update(self, observation):
        accuracy = self.condition.evidence_accuracy
        likelihood_a = accuracy if observation == "A" else 1 - accuracy
        likelihood_b = 1 - accuracy if observation == "A" else accuracy
        numerator = self.posterior_a * likelihood_a
        self.posterior_a = numerator / (
            numerator + (1 - self.posterior_a) * likelihood_b
        )

    def current_state(self):
        return {
            **self.history.state(),
            "current_continue_cost": float(self.condition.sample_cost),
            "current_outside_option": None,
            "current_progress": len(self.revealed) / self.condition.max_samples,
            "current_success_evidence": abs(self.posterior_a - 0.5) * 2,
            "posterior_a_private": self.posterior_a,
            "samples_revealed": len(self.revealed),
            "evidence_a_count": self.revealed.count("A"),
            "evidence_b_count": self.revealed.count("B"),
            "error_penalty": self.condition.error_penalty,
            "evidence_accuracy": self.condition.evidence_accuracy,
            "same_goal_across_steps": True,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {SAMPLE, DECIDE}:
            raise ValueError(f"invalid information-sampling action: {action}")
        if action == DECIDE:
            return self._finish_transition(
                action,
                outcome=0,
                reward=0,
                effort=0,
                success=None,
                terminated=True,
                reason="decide",
                progress=len(self.revealed) / self.condition.max_samples,
                task_values={"posterior_a": self.posterior_a},
            )
        observation = self.observations[len(self.revealed)]
        self.revealed.append(observation)
        self._update(observation)
        exhausted = len(self.revealed) >= self.condition.max_samples
        return self._finish_transition(
            action,
            outcome=-self.condition.sample_cost,
            reward=-self.condition.sample_cost,
            effort=self.condition.sample_cost,
            success=observation == self.condition.true_state,
            terminated=exhausted,
            reason="sample_budget_exhausted" if exhausted else None,
            progress=len(self.revealed) / self.condition.max_samples,
            task_values={
                "observation": observation,
                "posterior_a": self.posterior_a,
                "answer_required": exhausted,
            },
        )

    def _choice(self, mapping):
        return choice_block(
            mapping,
            SAMPLE,
            DECIDE,
            "reveal one more noisy signal",
            "stop sampling and choose A or B",
        )

    def initial_prompt(self, mapping):
        prior = (
            "A and B are initially equally likely."
            if math.isclose(self.condition.prior_a, 0.5)
            else f"Before sampling, A has probability {self.condition.prior_a:.0%}."
        )
        return (
            "A hidden source is either A or B. You may reveal noisy signals before giving a final answer.\n\n"
            f"{prior} Each signal costs {self.condition.sample_cost} points. An incorrect final answer loses {self.condition.error_penalty} points. "
            f"At most {self.condition.max_samples} signals can be revealed.\n\n"
            + self._choice(mapping)
        )

    def feedback_prompt(self, transition, mapping):
        observation = transition.task_values["observation"]
        return (
            f"The new signal was {observation}. Signals so far: "
            + ", ".join(self.revealed)
            + f". {self.condition.max_samples - len(self.revealed)} sample(s) remain.\n\n"
            + self._choice(mapping)
        )

    def final_answer_prompt(self):
        return (
            "Sampling is complete. Which hidden source is more likely?\n\n"
            "A = choose source A\nB = choose source B\n\nRespond with only A or B."
        )


def factorial_conditions(config):
    return [
        InformationSamplingCondition(
            float(accuracy),
            int(cost),
            int(penalty),
            float(prior),
            str(truth),
            int(config["max_samples"]),
        )
        for accuracy, cost, penalty, prior, truth in product(
            config["evidence_accuracies"],
            config["sample_costs"],
            config["error_penalties"],
            config["priors_a"],
            config["true_states"],
        )
    ]
