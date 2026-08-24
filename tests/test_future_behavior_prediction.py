import pytest

torch = pytest.importorskip("torch")

from analysis.persistence_future_behavior import future_behavior_validation


def test_latent_state_that_only_duplicates_current_choice_fails_future_gate():
    current = torch.linspace(-2, 2, 80)
    result = future_behavior_validation(
        current_choice=current,
        latent_state=current.clone(),
        future_outcome=2.0 * current,
        minimum_incremental_r_squared=0.01,
    )
    assert result["passed"] is False


def test_latent_state_can_add_future_prediction_beyond_current_choice():
    current = torch.tensor(([0.0, 1.0] * 40))
    latent = torch.repeat_interleave(torch.tensor([0.0, 1.0]), 40)
    future = 2.0 * current + 3.0 * latent
    result = future_behavior_validation(
        current_choice=current,
        latent_state=latent,
        future_outcome=future,
        minimum_incremental_r_squared=0.01,
    )
    assert result["passed"] is True
    assert result["incremental_r_squared"] > 0.1

