"""Discrete-time voluntary waiting adapted from McGuire & Kable."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .base_environment import BasePersistenceEnvironment, choice_block


WAIT = "WAIT"
QUIT = "QUIT"

ARRIVAL_HAZARDS = {
    "short_wait_optimal": (0.65, 0.55, 0.45, 0.35, 0.30, 0.25),
    "long_wait_optimal": (0.02, 0.03, 0.05, 0.08, 0.75, 0.70),
    # PRD 2.5 repair profiles avoid terminating most WAIT trajectories at the
    # very first state while retaining different temporal policies.
    "moderate_early": (0.10, 0.45, 0.60, 0.50, 0.35, 0.25, 0.20, 0.15),
    "moderate_late": (0.01, 0.03, 0.08, 0.20, 0.55, 0.70, 0.55, 0.35),
}


LITERATURE = {
    "construct": "dynamic persistence / voluntary waiting",
    "source_paradigm": "uncertain delayed-reward voluntary persistence",
    "source_citation": "McGuire & Kable, Nature Neuroscience (2015), doi:10.1038/nn.3994",
    "adaptation_notes": "Each simulated step replaces physical wall-clock waiting.",
    "departures_from_original": [
        "text-based point rewards",
        "discrete time",
        "counterbalanced single-token responses",
    ],
}


@dataclass(frozen=True)
class WaitingCondition:
    timing_environment: str
    reward_magnitude: int
    opportunity_cost: int
    quit_payoff: int

    def __post_init__(self):
        if self.timing_environment not in ARRIVAL_HAZARDS:
            raise ValueError("unknown waiting timing environment")
        if self.reward_magnitude <= 0 or self.opportunity_cost < 0:
            raise ValueError("invalid waiting reward or cost")

    def arrival_hazard(self, elapsed):
        elapsed = int(elapsed)
        profile = ARRIVAL_HAZARDS[self.timing_environment]
        return profile[min(elapsed, len(profile) - 1)]


def optimal_policy(condition, *, max_steps=8):
    """Finite-horizon expected-value policy for manipulation validation."""

    value = [float(condition.quit_payoff)] * (int(max_steps) + 1)
    policy = [QUIT] * int(max_steps)
    for step in reversed(range(int(max_steps))):
        hazard = condition.arrival_hazard(step)
        wait_value = (
            -condition.opportunity_cost
            + hazard * condition.reward_magnitude
            + (1.0 - hazard) * value[step + 1]
        )
        quit_value = float(condition.quit_payoff)
        if wait_value > quit_value:
            policy[step], value[step] = WAIT, wait_value
        else:
            policy[step], value[step] = QUIT, quit_value
    return policy


class VoluntaryWaitingEnvironment(BasePersistenceEnvironment):
    task = "voluntary_waiting"
    continue_action = WAIT
    disengage_action = QUIT

    def __init__(self, condition, seed, *, max_steps=8):
        super().__init__(condition, seed)
        self.max_steps = int(max_steps)
        rng = random.Random(self.seed * 2 + 1009)
        self.arrival_uniforms = [rng.random() for _ in range(self.max_steps)]

    def current_state(self):
        elapsed = self.step_index
        return {
            **self.history.state(),
            "current_continue_cost": float(self.condition.opportunity_cost),
            "current_outside_option": float(self.condition.quit_payoff),
            "current_progress": elapsed / self.max_steps,
            "current_success_evidence": self.condition.arrival_hazard(elapsed),
            "timing_environment": self.condition.timing_environment,
            "reward_magnitude": self.condition.reward_magnitude,
            "arrival_hazard_private": self.condition.arrival_hazard(elapsed),
            "same_goal_across_steps": True,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {WAIT, QUIT}:
            raise ValueError(f"invalid waiting action: {action}")
        if action == QUIT:
            return self._finish_transition(
                action,
                outcome=self.condition.quit_payoff,
                reward=self.condition.quit_payoff,
                effort=0,
                success=None,
                terminated=True,
                reason="quit",
                progress=self.step_index / self.max_steps,
                task_values={"reward_arrived": False},
            )
        hazard = self.condition.arrival_hazard(self.step_index)
        arrived = self.arrival_uniforms[self.step_index] < hazard
        reaches_horizon = self.step_index + 1 >= self.max_steps
        terminated = arrived or reaches_horizon
        reward = (
            self.condition.reward_magnitude - self.condition.opportunity_cost
            if arrived
            else -self.condition.opportunity_cost
            + (self.condition.quit_payoff if reaches_horizon else 0)
        )
        return self._finish_transition(
            action,
            outcome=reward,
            reward=reward,
            effort=self.condition.opportunity_cost,
            success=arrived,
            terminated=terminated,
            reason="reward_arrived" if arrived else "max_steps" if reaches_horizon else None,
            progress=(self.step_index + 1) / self.max_steps,
            task_values={"reward_arrived": arrived, "arrival_hazard": hazard},
        )

    def initial_prompt(self, mapping):
        context = (
            "Rewards in this setting usually arrive quickly."
            if self.condition.timing_environment in {"short_wait_optimal", "moderate_early"}
            else "Rewards in this setting are uncommon at first but become much more likely after several waits."
        )
        return (
            "You have an opportunity to receive a delayed point reward. Each decision represents one unit of time; no real-time waiting is required.\n\n"
            f"{context}\nThe reward is worth {self.condition.reward_magnitude} points. "
            f"Each wait costs {self.condition.opportunity_cost} points. Quitting gives {self.condition.quit_payoff} points and ends this opportunity.\n\n"
            + choice_block(mapping, WAIT, QUIT, "WAIT one more step", "QUIT now")
        )

    def feedback_prompt(self, transition, mapping):
        return (
            "The reward did not arrive during that step. "
            f"You have waited {self.step_index} step(s).\n\n"
            + choice_block(mapping, WAIT, QUIT, "WAIT one more step", "QUIT now")
        )


def factorial_conditions(config):
    return [
        WaitingCondition(str(timing), int(reward), int(cost), int(quit_payoff))
        for timing, reward, cost, quit_payoff in product(
            config["timing_environments"],
            config["reward_magnitudes"],
            config["opportunity_costs"],
            config["quit_payoffs"],
        )
    ]
