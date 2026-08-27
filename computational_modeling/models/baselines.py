"""Predeclared interpretable model zoo (M0--M16)."""

from __future__ import annotations

from dataclasses import dataclass

from computational_modeling.data.feature_schema import flexible_features


@dataclass(frozen=True)
class ModelDefinition:
    code: str
    name: str
    features: tuple[str, ...]
    family: str = "linear"
    supported_tasks: tuple[str, ...] = ("bandit", "foraging", "solvability")
    interpretable: bool = True
    description: str = ""


MODEL_DEFINITIONS = (
    ModelDefinition("M0", "intercept", (), description="Task/intercept baseline."),
    ModelDefinition("M1", "time", ("log_round", "normalized_time")),
    ModelDefinition("M2", "previous_outcome", ("log_round", "previous_outcome")),
    ModelDefinition(
        "M3", "streak", ("log_round", "failure_streak", "success_streak")
    ),
    ModelDefinition(
        "M4",
        "choice_inertia",
        ("log_round", "previous_choice", "second_previous_choice", "previous_outcome"),
    ),
    ModelDefinition("M5", "finite_history", (), family="finite_history"),
    ModelDefinition(
        "M6",
        "rescorla_wagner",
        ("log_round", "rw_best", "rw_gap"),
        family="rw",
        supported_tasks=("bandit",),
    ),
    ModelDefinition(
        "M7",
        "bayesian",
        ("log_round", "bayes_best", "bayes_gap"),
        family="bayesian",
        supported_tasks=("bandit",),
    ),
    ModelDefinition(
        "M8",
        "value_history_hybrid",
        (
            "log_round",
            "previous_outcome",
            "failure_streak",
            "previous_choice",
            "rw_best",
            "rw_gap",
            "bayes_best",
            "bayes_gap",
        ),
        family="rw_bayesian",
        supported_tasks=("bandit",),
    ),
    ModelDefinition("M9", "termination_advantage", ("termination_advantage",)),
    ModelDefinition("M10", "sticky_termination", (), family="sticky"),
    ModelDefinition(
        "M11",
        "decomposed_meta_control",
        (
            "estimated_continue_value",
            "estimated_outside_value",
            "cost_pressure",
            "progress_evidence",
            "previous_choice",
            "failure_streak",
        ),
    ),
    ModelDefinition(
        "M12",
        "mvt_like_foraging_threshold",
        ("mvt_like_advantage", "failure_streak", "log_round"),
        supported_tasks=("foraging",),
        description="Approximate MVT-like threshold; not a literal MVT derivation.",
    ),
    ModelDefinition("M13", "disengagement_accumulator", (), family="accumulator"),
    ModelDefinition("M14", "latent_commitment", (), family="commitment"),
    ModelDefinition("M15", "generic_latent_value", (), family="generic_value"),
    ModelDefinition(
        "M16",
        "flexible_linear",
        flexible_features(),
    ),
)

FLEXIBLE_FEATURES = MODEL_DEFINITIONS[-1].features


def enabled_definitions(config: dict) -> list[ModelDefinition]:
    switches = config.get("models", {})
    return [definition for definition in MODEL_DEFINITIONS if switches.get(definition.name, True)]
