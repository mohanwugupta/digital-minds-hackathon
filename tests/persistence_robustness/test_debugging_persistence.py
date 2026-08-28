from experiments.persistence_battery.debugging_persistence import (
    DEBUG,
    DebuggingCondition,
    DebuggingPersistenceEnvironment,
)


def test_debugging_failures_accumulate_clues_for_the_same_goal():
    condition = DebuggingCondition(0.0, 0.2, 1, 10, 2, 6)
    environment = DebuggingPersistenceEnvironment(condition, 4)
    before = environment.current_state()
    transition = environment.step(DEBUG)
    after = environment.current_state()
    assert not transition.success
    assert after["failed_attempts"] == 1
    assert after["current_success_evidence"] > before["current_success_evidence"]
    assert after["same_goal_across_steps"] is True

