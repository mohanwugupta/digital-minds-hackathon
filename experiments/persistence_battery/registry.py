"""Task registry and configuration adapters for the battery."""

from __future__ import annotations

from dataclasses import dataclass

from . import controllability
from . import independent_effort_control
from . import information_sampling
from . import partial_reinforcement
from . import progressive_ratio
from . import sunk_cost
from . import voluntary_waiting


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    module: object
    environment_class: type
    positive_action: str
    negative_action: str
    persistence: bool
    construct: str
    manipulated_variables: tuple[str, ...]

    @property
    def literature(self):
        return dict(self.module.LITERATURE)

    def conditions(self, config):
        return self.module.factorial_conditions(config)

    def environment(self, condition, seed, config):
        kwargs = {}
        if self.name == "voluntary_waiting":
            kwargs["max_steps"] = int(config["max_steps"])
        elif self.name == "partial_reinforcement":
            kwargs["max_extinction_attempts"] = int(
                config["max_extinction_attempts"]
            )
        return self.environment_class(condition, seed, **kwargs)


TASKS = {
    "voluntary_waiting": TaskDefinition(
        "voluntary_waiting",
        voluntary_waiting,
        voluntary_waiting.VoluntaryWaitingEnvironment,
        voluntary_waiting.WAIT,
        voluntary_waiting.QUIT,
        True,
        "dynamic persistence / waiting",
        ("timing_environment", "reward_magnitude", "opportunity_cost", "quit_payoff"),
    ),
    "progressive_ratio": TaskDefinition(
        "progressive_ratio",
        progressive_ratio,
        progressive_ratio.ProgressiveRatioEnvironment,
        progressive_ratio.WORK,
        progressive_ratio.QUIT,
        True,
        "breakpoint / effort motivation",
        ("ratio_schedule", "reward_magnitude", "effort_cost", "outside_option"),
    ),
    "sunk_cost": TaskDefinition(
        "sunk_cost",
        sunk_cost,
        sunk_cost.SunkCostEnvironment,
        sunk_cost.CONTINUE_WAITING,
        sunk_cost.ABANDON,
        True,
        "sunk-cost persistence",
        ("prior_investment", "remaining_steps", "reward_magnitude", "outside_option", "step_cost"),
    ),
    "information_sampling": TaskDefinition(
        "information_sampling",
        information_sampling,
        information_sampling.InformationSamplingEnvironment,
        information_sampling.SAMPLE,
        information_sampling.DECIDE,
        True,
        "epistemic persistence",
        ("evidence_accuracy", "sample_cost", "error_penalty", "prior_a", "true_state"),
    ),
    "partial_reinforcement": TaskDefinition(
        "partial_reinforcement",
        partial_reinforcement,
        partial_reinforcement.PartialReinforcementEnvironment,
        partial_reinforcement.TRY_AGAIN,
        partial_reinforcement.STOP,
        True,
        "partial-reinforcement extinction",
        ("reinforcement_schedule", "acquisition_trials", "partial_probability", "extinction_try_cost"),
    ),
    "independent_effort_control": TaskDefinition(
        "independent_effort_control",
        independent_effort_control,
        independent_effort_control.IndependentEffortEnvironment,
        independent_effort_control.HIGH_EFFORT,
        independent_effort_control.LOW_EFFORT,
        False,
        "generic repeated effort choice",
        ("high_reward_bonus", "high_effort_cost", "high_success_probability"),
    ),
    "controllability": TaskDefinition(
        "controllability",
        controllability,
        controllability.ControllabilityEnvironment,
        controllability.TRY,
        controllability.QUIT,
        True,
        "controllability transfer",
        ("exposure_type", "transfer_success_probability", "transfer_cost"),
    ),
}


CORE_TASKS = tuple(name for name in TASKS if name != "controllability")


def enabled_tasks(config):
    names = list(CORE_TASKS)
    if config.get("stretch", {}).get("controllability_enabled", False):
        names.append("controllability")
    return tuple(names)
