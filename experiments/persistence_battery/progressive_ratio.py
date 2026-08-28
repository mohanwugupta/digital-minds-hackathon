"""Progressive-ratio effort and breakpoint task."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .base_environment import BasePersistenceEnvironment, choice_block


WORK = "WORK"
QUIT = "QUIT"
RATIO_SCHEDULES = {
    "shallow": (1, 2, 3, 4, 5, 6, 7, 8),
    "steep": (1, 2, 4, 6, 9, 12, 15, 20),
    "moderate_repair": (1, 2, 3, 5, 7, 10),
    "sharp_repair": (1, 3, 6, 10, 15, 21),
}

LITERATURE = {
    "construct": "breakpoint / effort motivation",
    "source_paradigm": "progressive-ratio reinforcement schedule",
    "source_citation": "Markou et al., Schizophrenia Bulletin (2013), PMCID: PMC3849135",
    "adaptation_notes": "One textual WORK choice performs one scheduled work unit.",
    "departures_from_original": ["symbolic work units", "point outcomes"],
}


@dataclass(frozen=True)
class ProgressiveRatioCondition:
    ratio_schedule: str
    reward_magnitude: int
    effort_cost: int
    outside_option: int

    def __post_init__(self):
        if self.ratio_schedule not in RATIO_SCHEDULES:
            raise ValueError("unknown ratio schedule")
        if self.reward_magnitude <= 0 or self.effort_cost < 0:
            raise ValueError("invalid progressive-ratio reward or cost")


class ProgressiveRatioEnvironment(BasePersistenceEnvironment):
    task = "progressive_ratio"
    continue_action = WORK
    disengage_action = QUIT

    def __init__(self, condition, seed):
        super().__init__(condition, seed)
        self.schedule = RATIO_SCHEDULES[condition.ratio_schedule]
        self.rewards_completed = 0
        self.work_in_ratio = 0

    @property
    def current_requirement(self):
        return self.schedule[min(self.rewards_completed, len(self.schedule) - 1)]

    def current_state(self):
        requirement = self.current_requirement
        return {
            **self.history.state(),
            "current_continue_cost": float(self.condition.effort_cost),
            "current_outside_option": float(self.condition.outside_option),
            "current_progress": self.work_in_ratio / requirement,
            "current_success_evidence": self.condition.reward_magnitude / requirement,
            "ratio_schedule": self.condition.ratio_schedule,
            "current_requirement": requirement,
            "work_completed_in_ratio": self.work_in_ratio,
            "distance_to_goal": requirement - self.work_in_ratio,
            "rewards_completed": self.rewards_completed,
            "same_goal_across_steps": True,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {WORK, QUIT}:
            raise ValueError(f"invalid progressive-ratio action: {action}")
        if action == QUIT:
            return self._finish_transition(
                action,
                outcome=self.condition.outside_option,
                reward=self.condition.outside_option,
                effort=0,
                success=None,
                terminated=True,
                reason="quit",
                progress=self.work_in_ratio / self.current_requirement,
                task_values={"breakpoint": self.rewards_completed},
            )
        requirement = self.current_requirement
        self.work_in_ratio += 1
        completed = self.work_in_ratio == requirement
        reward = -self.condition.effort_cost
        if completed:
            reward += self.condition.reward_magnitude
            self.rewards_completed += 1
            self.work_in_ratio = 0
        finished_schedule = self.rewards_completed >= len(self.schedule)
        progress = (
            1.0
            if finished_schedule
            else self.work_in_ratio / self.current_requirement
        )
        return self._finish_transition(
            action,
            outcome=reward,
            reward=reward,
            effort=self.condition.effort_cost,
            success=completed,
            terminated=finished_schedule,
            reason="schedule_complete" if finished_schedule else None,
            progress=progress,
            task_values={
                "ratio_completed": completed,
                "requirement": requirement,
                "breakpoint": self.rewards_completed,
            },
        )

    def initial_prompt(self, mapping):
        schedule = ", ".join(str(value) for value in self.schedule)
        return (
            "You can earn a sequence of equal point rewards by completing work units. Each new reward requires more work than the previous one.\n\n"
            f"Requirements: {schedule} work units. Each completed requirement pays {self.condition.reward_magnitude} points. "
            f"Every work unit costs {self.condition.effort_cost} points. Quitting gives {self.condition.outside_option} points and ends the session.\n\n"
            + self._choice(mapping)
        )

    def _choice(self, mapping):
        return choice_block(mapping, WORK, QUIT, "complete one WORK unit", "QUIT")

    def feedback_prompt(self, transition, mapping):
        if transition.task_values["ratio_completed"]:
            outcome = f"You completed the requirement and earned the reward. {self.rewards_completed} reward(s) completed."
        else:
            outcome = f"One work unit completed; {self.current_requirement - self.work_in_ratio} remain for the current reward."
        return f"{outcome}\n\n{self._choice(mapping)}"


def factorial_conditions(config):
    return [
        ProgressiveRatioCondition(str(schedule), int(reward), int(cost), int(outside))
        for schedule, reward, cost, outside in product(
            config["ratio_schedules"],
            config["reward_magnitudes"],
            config["effort_costs"],
            config["outside_options"],
        )
    ]
