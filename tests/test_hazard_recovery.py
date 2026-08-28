from analysis.persistence_hazard.fit_hazard_models import (
    fit_hazard_architectures,
    simulate_hazard_records,
)


def _scores(generating):
    records = simulate_hazard_records(generating, episodes_per_task=90, seed=17)
    result = fit_hazard_architectures(
        records,
        {
            "ridge_grid": [0.01, 0.1, 1.0],
            "history_lags": [1],
            "calibration_bins": 5,
        },
    )
    return result["comparison"].set_index("model")


def test_shared_history_architecture_recovery():
    scores = _scores("shared_history")
    assert scores.loc["shared_history", "performance_fraction"] > 0.85
    assert scores.loc["shared_history", "test_log_loss"] <= scores.loc[
        "fully_shared", "test_log_loss"
    ]


def test_separate_algorithm_recovery():
    scores = _scores("task_specific")
    assert scores.loc["task_specific", "test_log_loss"] + 0.02 < scores.loc[
        "shared_history", "test_log_loss"
    ]


def test_fully_shared_recovery():
    scores = _scores("fully_shared")
    assert scores.loc["fully_shared", "test_log_loss"] <= scores.loc[
        "task_specific", "test_log_loss"
    ] + 0.03

