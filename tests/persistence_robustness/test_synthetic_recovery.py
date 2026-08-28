from analysis.persistence_robustness.synthetic_recovery import (
    generate_history_specificity,
    recover_delta_history_gain,
)


def test_generic_history_recovers_near_zero_matched_difference():
    frame = generate_history_specificity(
        persistent_strength=1.5,
        independent_strength=1.5,
        episodes=800,
        seed=20,
    )
    assert abs(recover_delta_history_gain(frame)) < 0.08


def test_persistence_specific_history_recovers_positive_difference():
    frame = generate_history_specificity(
        persistent_strength=2.2,
        independent_strength=0.0,
        episodes=800,
        seed=21,
    )
    assert recover_delta_history_gain(frame) > 0.03

