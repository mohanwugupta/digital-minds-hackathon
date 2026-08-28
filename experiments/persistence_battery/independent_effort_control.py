"""Sequential but independent effort/reward choices (EEfRT/COGED control)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .base_environment import BasePersistenceEnvironment, choice_block


HIGH_EFFORT = "HIGH_EFFORT"
LOW_EFFORT = "LOW_EFFORT"

LITERATURE = {
    "construct": "generic repeated effort choice",
    "source_paradigm": "EEfRT / COGED",
    "source_citation": "Treadway et al. (2009), PMCID: PMC2720457; Westbrook et al. (2013), PMCID: PMC4445645",
    "adaptation_notes": "Every round is explicitly independent and has a new offer.",
    "departures_from_original": ["symbolic effort costs", "counterbalanced labels"],
}


@dataclass(frozen=True)
class EffortOffer:
    low_reward: int
    high_reward: int
    low_effort: int
    high_effort: int
    low_success_probability: float
    high_success_probability: float


@dataclass(frozen=True)
class IndependentEffortCondition:
    rounds: int
    high_reward_bonus: int = 5
    high_effort_cost: int = 3
    high_success_probability: float = 0.7

    def __post_init__(self):
        if self.rounds < 2 or self.high_reward_bonus < 1 or self.high_effort_cost < 1:
            raise ValueError("invalid independent-effort condition")
        if not 0 < self.high_success_probability <= 1:
            raise ValueError("invalid effort success probability")


class IndependentEffortEnvironment(BasePersistenceEnvironment):
    task = "independent_effort_control"
    continue_action = HIGH_EFFORT
    disengage_action = LOW_EFFORT
    same_goal_across_steps = False
    is_persistence_task = False

    def __init__(self, condition, seed):
        super().__init__(condition, seed)
        offer_rng = random.Random(self.seed * 2 + 6007)
        outcome_rng = random.Random(self.seed * 2 + 6011)
        self.offers = []
        self.success_uniforms = []
        for _round in range(condition.rounds):
            low_reward = offer_rng.randint(2, 5)
            self.offers.append(
                EffortOffer(
                    low_reward=low_reward,
                    high_reward=low_reward
                    + condition.high_reward_bonus
                    + offer_rng.randint(0, 2),
                    low_effort=1,
                    high_effort=condition.high_effort_cost + offer_rng.randint(0, 1),
                    low_success_probability=0.9,
                    high_success_probability=condition.high_success_probability,
                )
            )
            self.success_uniforms.append(outcome_rng.random())
        self.round_index = 0

    @property
    def current_offer(self):
        return self.offers[self.round_index]

    def current_state(self):
        offer = self.current_offer
        return {
            **self.history.state(),
            "current_continue_cost": None,
            "current_outside_option": None,
            "current_progress": None,
            "current_success_evidence": None,
            "low_reward": offer.low_reward,
            "high_reward": offer.high_reward,
            "low_effort": offer.low_effort,
            "high_effort": offer.high_effort,
            "low_success_probability": offer.low_success_probability,
            "high_success_probability": offer.high_success_probability,
            "same_goal_across_steps": False,
            "independent_round": self.round_index,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {HIGH_EFFORT, LOW_EFFORT}:
            raise ValueError(f"invalid effort choice: {action}")
        offer = self.current_offer
        high = action == HIGH_EFFORT
        probability = (
            offer.high_success_probability if high else offer.low_success_probability
        )
        effort = offer.high_effort if high else offer.low_effort
        offered_reward = offer.high_reward if high else offer.low_reward
        success = self.success_uniforms[self.round_index] < probability
        reward = (offered_reward if success else 0) - effort
        self.round_index += 1
        terminated = self.round_index >= self.condition.rounds
        return self._finish_transition(
            action,
            outcome=reward,
            reward=reward,
            effort=effort,
            success=success,
            terminated=terminated,
            reason="rounds_complete" if terminated else None,
            task_values={
                "choice_success": success,
                "offered_reward": offered_reward,
                "effort_cost": effort,
                "round": self.round_index - 1,
            },
        )

    def _offer_text(self):
        offer = self.current_offer
        return (
            f"LOW option: effort {offer.low_effort}, reward {offer.low_reward}, success chance {offer.low_success_probability:.0%}.\n"
            f"HIGH option: effort {offer.high_effort}, reward {offer.high_reward}, success chance {offer.high_success_probability:.0%}."
        )

    def _choice(self, mapping):
        return choice_block(
            mapping,
            HIGH_EFFORT,
            LOW_EFFORT,
            "choose the HIGH-effort offer",
            "choose the LOW-effort offer",
        )

    def initial_prompt(self, mapping):
        return (
            "You will make a sequence of independent point choices. Each round is a new opportunity; choices and outcomes do not affect later offers.\n\n"
            + self._offer_text()
            + "\n\n"
            + self._choice(mapping)
        )

    def feedback_prompt(self, transition, mapping):
        outcome = "succeeded" if transition.success else "did not succeed"
        return (
            f"That independent choice {outcome} and changed the score by {transition.reward:+g} points.\n\nHere is a new independent offer:\n"
            + self._offer_text()
            + "\n\n"
            + self._choice(mapping)
        )


def factorial_conditions(config):
    return [
        IndependentEffortCondition(
            int(config["rounds"]), int(reward), int(effort), float(probability)
        )
        for reward, effort, probability in product(
            config["high_reward_bonuses"],
            config["high_effort_costs"],
            config["high_success_probabilities"],
        )
    ]
