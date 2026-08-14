"""Deterministic counterfactual two-armed bandit."""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional


SUCCESS_REWARD = 3
FAILURE_REWARD = -2
STOP_REWARD = 0


def expected_pull_reward(success_probability: float) -> float:
    """Expected one-step points before learning/exploration value."""
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must be in [0, 1]")
    return SUCCESS_REWARD * success_probability + FAILURE_REWARD * (
        1.0 - success_probability
    )


def condition_class(p_a: float, p_b: float) -> str:
    positive = sum(expected_pull_reward(probability) > STOP_REWARD for probability in (p_a, p_b))
    return ("both_negative", "one_positive", "both_positive")[positive]


@dataclass(frozen=True)
class StepResult:
    action: str
    reward: int
    success: Optional[bool]
    terminated: bool
    reason: Optional[str]
    decision: int
    cumulative_score: int


class BanditEnvironment:
    def __init__(
        self,
        p_a: float,
        p_b: float,
        seed: int,
        max_decisions: int = 100,
    ) -> None:
        for name, probability in (("p_a", p_a), ("p_b", p_b)):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        self.p_a = float(p_a)
        self.p_b = float(p_b)
        self.seed = int(seed)
        self.max_decisions = int(max_decisions)

        # Separate streams ensure that one arm's potential outcomes never depend
        # on how many random numbers the other arm consumes.
        rng_a = random.Random(self.seed * 2 + 1)
        rng_b = random.Random(self.seed * 2 + 2)
        self.outcome_schedule: Dict[str, List[bool]] = {
            "A": [rng_a.random() < self.p_a for _ in range(self.max_decisions)],
            "B": [rng_b.random() < self.p_b for _ in range(self.max_decisions)],
        }
        self.pull_counts = {"A": 0, "B": 0}
        self.action_history: List[str] = []
        self.reward_history: List[int] = []
        self.cumulative_score = 0
        self.terminated = False
        self.termination_reason: Optional[str] = None

    @classmethod
    def from_history(
        cls,
        p_a: float,
        p_b: float,
        seed: int,
        action_history: List[str],
        reward_history: List[int],
        *,
        max_decisions: int = 100,
    ) -> "BanditEnvironment":
        """Resume a visible state with a fresh counterfactual outcome schedule.

        The supplied seed governs future potential outcomes. Past rewards are
        retained as observed history and need not agree with that new schedule.
        """
        actions = [str(action).strip().upper() for action in action_history]
        rewards = [int(reward) for reward in reward_history]
        if len(actions) != len(rewards):
            raise ValueError("action and reward histories must have equal length")
        if any(action not in {"A", "B"} for action in actions):
            raise ValueError("a resumable pre-decision history cannot contain STOP")
        if any(reward not in {SUCCESS_REWARD, FAILURE_REWARD} for reward in rewards):
            raise ValueError("pull rewards must be +3 or -2")
        if len(actions) >= max_decisions:
            raise ValueError("cannot resume a state at or beyond the decision cap")
        environment = cls(p_a, p_b, seed, max_decisions=max_decisions)
        environment.action_history = list(actions)
        environment.reward_history = list(rewards)
        environment.pull_counts = {
            "A": actions.count("A"),
            "B": actions.count("B"),
        }
        environment.cumulative_score = sum(rewards)
        return environment

    @property
    def decision(self) -> int:
        return len(self.action_history)

    def step(self, action: str) -> StepResult:
        if self.terminated:
            raise RuntimeError("episode has already terminated")
        action = action.strip().upper()
        if action not in {"A", "B", "C"}:
            raise ValueError(f"invalid action: {action!r}")

        self.action_history.append(action)
        if action == "C":
            reward, success = STOP_REWARD, None
            self.reward_history.append(reward)
            self.terminated, self.termination_reason = True, "stop"
        else:
            pull_index = self.pull_counts[action]
            success = self.outcome_schedule[action][pull_index]
            self.pull_counts[action] += 1
            reward = SUCCESS_REWARD if success else FAILURE_REWARD
            self.reward_history.append(reward)
            self.cumulative_score += reward
            if self.decision >= self.max_decisions:
                self.terminated, self.termination_reason = True, "max_decisions"

        return StepResult(
            action=action,
            reward=reward,
            success=success,
            terminated=self.terminated,
            reason=self.termination_reason,
            decision=self.decision,
            cumulative_score=self.cumulative_score,
        )

    def private_state(self) -> dict:
        return {
            "p_A_true": self.p_a,
            "p_B_true": self.p_b,
            "seed": self.seed,
            "outcome_schedule": {k: list(v) for k, v in self.outcome_schedule.items()},
            "choice_history": list(self.action_history),
            "reward_history": list(self.reward_history),
            "cumulative_score": self.cumulative_score,
            "round": self.decision,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
        }
