import pytest

from computational_modeling.analysis.model_recovery import (
    GENERATING_ARCHITECTURES,
    recover_architecture,
    simulate_architecture,
)


@pytest.mark.parametrize("architecture", GENERATING_ARCHITECTURES)
def test_synthetic_model_recovery_identifies_generating_family(architecture):
    frame = simulate_architecture(architecture, episodes=30, decisions=8, seed=7)
    recovered = recover_architecture(frame)
    assert recovered["selected_family"] in recovered["acceptable_families"]
    assert architecture in recovered["acceptable_families"]
