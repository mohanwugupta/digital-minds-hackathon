from computational_modeling.models.termination import termination_advantage


def test_observable_and_oracle_termination_advantages_are_separate():
    row = {
        "estimated_continue_value": 2.0,
        "estimated_outside_value": 0.5,
        "oracle_continue_value": 4.0,
        "oracle_outside_value": 1.0,
    }
    assert termination_advantage(row, "observable") == 1.5
    assert termination_advantage(row, "oracle") == 3.0
