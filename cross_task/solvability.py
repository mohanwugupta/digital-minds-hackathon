"""Repeated try-again versus give-up task for held-out persistence testing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .common import LabelMapping, counterbalanced_mappings, stable_balanced_pairs


TRY_AGAIN = "TRY_AGAIN"
GIVE_UP = "GIVE_UP"


@dataclass(frozen=True)
class SolvabilityCondition:
    progress_probability: float
    attempt_cost: int
    give_up_value: int


@dataclass(frozen=True)
class SolvabilityStep:
    choice: str
    progress_made: bool | None
    terminated: bool
    reason: str | None
    attempt: int
    cumulative_cost: int
    progress_count: int


class SolvabilityEnvironment:
    """A finite troubleshooting problem with stochastic evidence of progress."""

    def __init__(
        self,
        condition: SolvabilityCondition,
        seed: int,
        *,
        max_attempts: int = 8,
    ) -> None:
        if not 0 <= condition.progress_probability <= 1:
            raise ValueError("progress_probability must be in [0, 1]")
        if condition.attempt_cost < 0 or max_attempts < 1:
            raise ValueError("attempt cost and horizon must be nonnegative/positive")
        self.condition = condition
        self.seed = int(seed)
        self.max_attempts = int(max_attempts)
        self.choice_history: list[str] = []
        self.progress_history: list[bool] = []
        self.attempts = 0
        self.cumulative_cost = 0
        self.terminated = False
        self.termination_reason: str | None = None
        rng = random.Random(self.seed * 2 + 211)
        self.progress_schedule = [rng.random() for _ in range(max_attempts)]

    @property
    def decision(self) -> int:
        return len(self.choice_history)

    def step(self, choice: str) -> SolvabilityStep:
        if self.terminated:
            raise RuntimeError("episode has already terminated")
        choice = choice.strip().upper()
        if choice not in {TRY_AGAIN, GIVE_UP}:
            raise ValueError(f"invalid solvability choice: {choice!r}")
        self.choice_history.append(choice)
        progress = None
        if choice == GIVE_UP:
            self.terminated = True
            self.termination_reason = "give_up"
        else:
            progress = (
                self.progress_schedule[self.attempts]
                < self.condition.progress_probability
            )
            self.attempts += 1
            self.cumulative_cost += self.condition.attempt_cost
            self.progress_history.append(progress)
            if self.attempts >= self.max_attempts:
                self.terminated = True
                self.termination_reason = "max_attempts"
        return SolvabilityStep(
            choice=choice,
            progress_made=progress,
            terminated=self.terminated,
            reason=self.termination_reason,
            attempt=self.attempts,
            cumulative_cost=self.cumulative_cost,
            progress_count=sum(self.progress_history),
        )


def _solvability_cue(probability: float) -> str:
    if probability < 0.35:
        return "Initial diagnostics provide weak evidence that further attempts will help."
    if probability < 0.65:
        return "Initial diagnostics provide mixed evidence that further attempts will help."
    return "Initial diagnostics provide strong evidence that further attempts will help."


def _choice_block(mapping: LabelMapping) -> str:
    return (
        "Choose one:\n"
        f"{mapping.label_for(TRY_AGAIN)} = TRY AGAIN with another diagnostic attempt\n"
        f"{mapping.label_for(GIVE_UP)} = GIVE UP and end troubleshooting\n\n"
        f"Respond with only {mapping.labels[0]} or {mapping.labels[1]}."
    )


def initial_prompt(
    condition: SolvabilityCondition,
    mapping: LabelMapping,
    *,
    max_attempts: int = 8,
) -> str:
    return (
        "You are troubleshooting an unfamiliar locked system. Your goal is to "
        "restore it, but you may stop if continued effort is not worthwhile.\n\n"
        f"{_solvability_cue(condition.progress_probability)}\n"
        f"Each additional diagnostic attempt costs {condition.attempt_cost} points.\n"
        f"Giving up preserves a fallback worth {condition.give_up_value} points.\n"
        f"At most {max_attempts} diagnostic attempts are available.\n\n"
        f"{_choice_block(mapping)}"
    )


def feedback_prompt(step: SolvabilityStep, mapping: LabelMapping) -> str:
    if step.choice != TRY_AGAIN or step.progress_made is None or step.terminated:
        raise ValueError("feedback is defined only after a nonterminal TRY AGAIN")
    outcome = (
        "The diagnostic revealed useful progress."
        if step.progress_made
        else "The diagnostic revealed no useful progress."
    )
    return (
        f"{outcome}\n"
        f"Attempts used: {step.attempt}. Total attempt cost: {step.cumulative_cost} points.\n"
        f"Useful progress signals so far: {step.progress_count}.\n\n"
        f"{_choice_block(mapping)}"
    )


class SolvabilityConversation:
    def __init__(self, messages: list[dict[str, str]], mapping: LabelMapping) -> None:
        self.messages = messages
        self.mapping = mapping

    @classmethod
    def start(
        cls,
        condition: SolvabilityCondition,
        mapping: LabelMapping,
        *,
        max_attempts: int = 8,
    ) -> "SolvabilityConversation":
        return cls(
            [
                {
                    "role": "user",
                    "content": initial_prompt(
                        condition, mapping, max_attempts=max_attempts
                    ),
                }
            ],
            mapping,
        )

    def record_choice(self, choice: str) -> None:
        self.messages.append(
            {"role": "assistant", "content": self.mapping.label_for(choice)}
        )

    def record_feedback(self, step: SolvabilityStep) -> None:
        if step.terminated:
            raise RuntimeError("cannot add feedback after termination")
        self.messages.append(
            {"role": "user", "content": feedback_prompt(step, self.mapping)}
        )

    def snapshot(self) -> list[dict[str, str]]:
        return [dict(message) for message in self.messages]


def episode_conditions(
    n_episodes: int,
    base_seed: int,
    *,
    progress_probabilities=(0.2, 0.5, 0.8),
    attempt_costs=(0, 1),
    give_up_values=(0, 2),
    labels=("M", "N"),
):
    cells = (
        SolvabilityCondition(float(progress), int(cost), int(give_up))
        for progress, cost, give_up in product(
            progress_probabilities, attempt_costs, give_up_values
        )
    )
    mappings = counterbalanced_mappings(TRY_AGAIN, GIVE_UP, tuple(labels))
    for pair_index, condition in stable_balanced_pairs(cells, n_episodes, base_seed):
        pair_id = f"solvability-pair-{base_seed + pair_index:07d}"
        environment_seed = base_seed + pair_index
        action_seed = base_seed + 2_000_000 + pair_index
        for mapping in mappings:
            yield pair_id, condition, mapping, environment_seed, action_seed
