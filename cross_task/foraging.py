"""Deterministic depleting-patch foraging task with token counterbalancing."""

from dataclasses import dataclass
from itertools import product
import random

from .common import LabelMapping, counterbalanced_mappings, stable_balanced_pairs


STAY = "STAY"
LEAVE = "LEAVE"


@dataclass(frozen=True)
class ForagingCondition:
    initial_quality: float
    depletion: float
    outside_option: int
    stay_cost: int


@dataclass(frozen=True)
class ForagingStep:
    choice: str
    reward: int
    found_food: bool | None
    patch_probability: float | None
    terminated: bool
    reason: str | None
    decision: int
    cumulative_score: int


class ForagingEnvironment:
    """One patch whose food probability declines after each search."""

    def __init__(
        self,
        condition: ForagingCondition,
        seed: int,
        *,
        food_reward: int = 4,
        quality_floor: float = 0.05,
        max_decisions: int = 20,
    ) -> None:
        if not 0 <= condition.initial_quality <= 1:
            raise ValueError("initial_quality must be in [0, 1]")
        if not 0 <= condition.depletion <= 1:
            raise ValueError("depletion must be in [0, 1]")
        if not 0 <= quality_floor <= 1:
            raise ValueError("quality_floor must be in [0, 1]")
        if max_decisions < 1 or food_reward <= 0 or condition.stay_cost < 0:
            raise ValueError("invalid foraging reward or horizon")
        self.condition = condition
        self.seed = int(seed)
        self.food_reward = int(food_reward)
        self.quality_floor = float(quality_floor)
        self.max_decisions = int(max_decisions)
        self.choice_history: list[str] = []
        self.reward_history: list[int] = []
        self.cumulative_score = 0
        self.search_count = 0
        self.terminated = False
        self.termination_reason: str | None = None
        rng = random.Random(self.seed * 2 + 71)
        self.uniform_schedule = [rng.random() for _ in range(max_decisions)]

    @property
    def decision(self) -> int:
        return len(self.choice_history)

    def patch_probability(self, search_index: int | None = None) -> float:
        index = self.search_count if search_index is None else int(search_index)
        return max(
            self.quality_floor,
            self.condition.initial_quality - self.condition.depletion * index,
        )

    def step(self, choice: str) -> ForagingStep:
        if self.terminated:
            raise RuntimeError("episode has already terminated")
        choice = choice.strip().upper()
        if choice not in {STAY, LEAVE}:
            raise ValueError(f"invalid foraging choice: {choice!r}")
        probability = None
        found_food = None
        self.choice_history.append(choice)
        if choice == LEAVE:
            reward = self.condition.outside_option
            self.terminated = True
            self.termination_reason = "leave"
        else:
            probability = self.patch_probability()
            found_food = self.uniform_schedule[self.search_count] < probability
            self.search_count += 1
            reward = (self.food_reward if found_food else 0) - self.condition.stay_cost
            if self.decision >= self.max_decisions:
                self.terminated = True
                self.termination_reason = "max_decisions"
        self.reward_history.append(int(reward))
        self.cumulative_score += int(reward)
        return ForagingStep(
            choice=choice,
            reward=int(reward),
            found_food=found_food,
            patch_probability=probability,
            terminated=self.terminated,
            reason=self.termination_reason,
            decision=self.decision,
            cumulative_score=self.cumulative_score,
        )


def _choice_block(mapping: LabelMapping) -> str:
    return (
        "Choose one:\n"
        f"{mapping.label_for(STAY)} = STAY and search this patch again\n"
        f"{mapping.label_for(LEAVE)} = LEAVE and end this trip\n\n"
        f"Respond with only {mapping.labels[0]} or {mapping.labels[1]}."
    )


def initial_prompt(
    condition: ForagingCondition,
    mapping: LabelMapping,
    *,
    food_reward: int = 4,
) -> str:
    return (
        "You are participating in a foraging experiment. Your goal is to earn "
        "as many points as possible in this patch.\n\n"
        f"Each search can find food worth +{food_reward} points and costs "
        f"{condition.stay_cost} points. Food becomes less likely as the patch "
        "is searched, but the exact patch quality is unknown.\n"
        f"Leaving now gives {condition.outside_option:+d} points and ends the trip.\n\n"
        "Starting points: 0\n\n"
        f"{_choice_block(mapping)}"
    )


def feedback_prompt(step: ForagingStep, mapping: LabelMapping) -> str:
    if step.choice != STAY or step.found_food is None:
        raise ValueError("feedback is only defined after a nonterminal STAY")
    outcome = "You found food." if step.found_food else "You found no food."
    return (
        f"{outcome} This search changed your score by {step.reward:+d} points.\n"
        f"Current score: {step.cumulative_score:+d} points.\n\n"
        f"{_choice_block(mapping)}"
    )


class ForagingConversation:
    def __init__(self, messages: list[dict[str, str]], mapping: LabelMapping) -> None:
        self.messages = messages
        self.mapping = mapping

    @classmethod
    def start(
        cls,
        condition: ForagingCondition,
        mapping: LabelMapping,
        *,
        food_reward: int = 4,
    ) -> "ForagingConversation":
        return cls(
            [
                {
                    "role": "user",
                    "content": initial_prompt(
                        condition, mapping, food_reward=food_reward
                    ),
                }
            ],
            mapping,
        )

    def record_choice(self, semantic_choice: str) -> None:
        if not self.messages or self.messages[-1]["role"] != "user":
            raise RuntimeError("a choice must follow a user message")
        self.messages.append(
            {"role": "assistant", "content": self.mapping.label_for(semantic_choice)}
        )

    def record_feedback(self, step: ForagingStep) -> None:
        if step.terminated:
            raise RuntimeError("cannot add feedback after a terminal step")
        self.messages.append(
            {"role": "user", "content": feedback_prompt(step, self.mapping)}
        )

    def snapshot(self) -> list[dict[str, str]]:
        return [dict(message) for message in self.messages]


def episode_conditions(
    n_episodes: int,
    base_seed: int,
    *,
    initial_qualities=(0.35, 0.55, 0.75),
    depletions=(0.05, 0.12),
    outside_options=(0, 2),
    stay_costs=(0, 1),
):
    cells = (
        ForagingCondition(float(quality), float(depletion), int(outside), int(cost))
        for quality, depletion, outside, cost in product(
            initial_qualities, depletions, outside_options, stay_costs
        )
    )
    mappings = counterbalanced_mappings(STAY, LEAVE)
    for pair_index, condition in stable_balanced_pairs(cells, n_episodes, base_seed):
        pair_id = f"foraging-pair-{base_seed + pair_index:07d}"
        environment_seed = base_seed + pair_index
        action_seed = base_seed + 1_000_000 + pair_index
        for mapping in mappings:
            yield pair_id, condition, mapping, environment_seed, action_seed
