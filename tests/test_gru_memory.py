from analysis.persistence_gru.memory_ablations import build_causal_windows


def test_limited_gru_windows_never_include_future_states():
    records = [
        {
            "episode_id": "e1",
            "round": index,
            "feature": float(index),
            "persistence_logit": float(index),
        }
        for index in range(5)
    ]
    windows = build_causal_windows(records, ("feature",), max_history=3)
    assert windows[0]["values"].ravel().tolist() == [0.0]
    assert windows[2]["values"].ravel().tolist() == [0.0, 1.0, 2.0]
    assert windows[4]["values"].ravel().tolist() == [2.0, 3.0, 4.0]
    assert windows[4]["target"] == 4.0


def test_one_step_window_resets_hidden_history_at_every_decision():
    records = [
        {
            "episode_id": "episode",
            "round": round_index,
            "feature": float(round_index),
            "persistence_logit": 0.0,
        }
        for round_index in range(3)
    ]
    windows = build_causal_windows(records, ("feature",), max_history=1)
    assert [window["values"].tolist() for window in windows] == [
        [[0.0]],
        [[1.0]],
        [[2.0]],
    ]
