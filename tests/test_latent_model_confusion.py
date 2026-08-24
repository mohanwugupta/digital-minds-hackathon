import pytest

pytest.importorskip("torch")

from analysis.persistence_latent_state import (
    compare_behavioral_architectures,
    simulate_latent_trajectories,
)


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("immediate", "immediate_decision"),
        ("choice_inertia", "choice_history_inertia"),
        ("generic_value", "generic_latent_value"),
        ("latent_commitment", "latent_commitment"),
    ],
)
def test_synthetic_architecture_confusion_gate(architecture, expected):
    synthetic = simulate_latent_trajectories(
        architecture=architecture,
        tasks=("bandit", "foraging"),
        episodes_per_task=24,
        decisions=12,
        rho=0.7,
        seed=41,
    )
    comparison = compare_behavioral_architectures(
        synthetic["records"],
        feature_names=synthetic["feature_names"],
        generic_value_feature="relative_value",
    )
    assert comparison["selected_model"] == expected

