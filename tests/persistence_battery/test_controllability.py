from experiments.persistence_battery.controllability import (
    TRY,
    ControllabilityCondition,
    ControllabilityEnvironment,
    build_yoked_exposure,
)


def test_yoked_exposure_preserves_outcomes_but_changes_contingency():
    controllable, uncontrollable = build_yoked_exposure(31, trials=10)
    assert controllable.outcomes == uncontrollable.outcomes
    assert controllable.actions == uncontrollable.actions
    assert controllable.contingent is True
    assert uncontrollable.contingent is False


def test_transfer_outcomes_recover_probability_and_are_deterministic():
    condition = ControllabilityCondition("controllable", 8, 0.55, 0, 1)
    success = []
    for seed in range(1500):
        left = ControllabilityEnvironment(condition, seed)
        right = ControllabilityEnvironment(condition, seed)
        assert left.transfer_uniforms == right.transfer_uniforms
        success.append(left.step(TRY).success)
    assert abs(sum(success) / len(success) - 0.55) < 0.04
