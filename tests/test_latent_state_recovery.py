import pytest

torch = pytest.importorskip("torch")

from analysis.persistence_latent_state import (
    fit_latent_state_model,
    simulate_latent_trajectories,
)


def test_correct_state_model_recovers_synthetic_ordering_and_rho():
    synthetic = simulate_latent_trajectories(
        architecture="latent_commitment",
        tasks=("bandit", "foraging"),
        episodes_per_task=20,
        decisions=12,
        rho=0.7,
        seed=17,
    )
    fit = fit_latent_state_model(
        synthetic["records"],
        feature_names=synthetic["feature_names"],
        rho_grid=tuple(index / 20 for index in range(20)),
    )
    truth = torch.tensor([row["true_w"] for row in synthetic["records"]])
    estimate = torch.tensor(fit["latent_state"])
    correlation = float(torch.corrcoef(torch.stack((truth, estimate)))[0, 1])
    assert correlation > 0.8
    assert abs(fit["rho"] - 0.7) <= 0.15
    assert fit["orientation"] == "higher_means_more_persistence"


def test_task_specific_emission_scales_preserve_shared_latent_ordering():
    synthetic = simulate_latent_trajectories(
        architecture="latent_commitment",
        tasks=("bandit", "foraging", "solvability"),
        episodes_per_task=12,
        decisions=10,
        rho=0.6,
        emission_scales={"bandit": 0.5, "foraging": 1.5, "solvability": 3.0},
        seed=23,
    )
    fit = fit_latent_state_model(
        synthetic["records"],
        feature_names=synthetic["feature_names"],
    )
    for task in synthetic["tasks"]:
        indices = [
            index for index, row in enumerate(synthetic["records"])
            if row["task"] == task
        ]
        truth = torch.tensor([synthetic["records"][index]["true_w"] for index in indices])
        estimate = torch.tensor([fit["latent_state"][index] for index in indices])
        assert float(torch.corrcoef(torch.stack((truth, estimate)))[0, 1]) > 0.75

