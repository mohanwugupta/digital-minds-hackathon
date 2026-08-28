from analysis.persistence_hazard.history_kernels import add_exponential_history


def test_exponential_history_is_past_only():
    rows = [
        {
            "episode_id": "e1",
            "round": index,
            "continue": continued,
            "outcome_after_choice": outcome,
        }
        for index, (continued, outcome) in enumerate(
            [(1, 2.0), (1, -1.0), (0, 9.0)]
        )
    ]
    first = add_exponential_history(rows, decay=0.5)
    changed = [dict(row) for row in rows]
    changed[-1]["outcome_after_choice"] = -999.0
    second = add_exponential_history(changed, decay=0.5)
    assert first[-1]["choice_kernel"] == second[-1]["choice_kernel"]
    assert first[-1]["outcome_kernel"] == second[-1]["outcome_kernel"]
    assert first[0]["choice_kernel"] == 0.0
    assert first[0]["outcome_kernel"] == 0.0

