import json

import pandas as pd

from computational_modeling.analysis.followup_analysis import (
    build_model_rankings,
    run_neural_linear_recovery_sanity,
)
from computational_modeling.data.feature_schema import (
    FLEXIBLE_FEATURE_GROUPS,
    FLEXIBLE_NUISANCE_FEATURES,
    flexible_features,
    validate_feature_groups,
)


def _aggregate(model, task_count, r_squared, mse):
    return {
        "model": model,
        "code": "TEST",
        "information_set": "observable",
        "sharing": "shared_architecture_task_observation",
        "parameter_count": 3,
        "task_count": task_count,
        "states": 30,
        "episodes": 9,
        "r_squared": r_squared,
        "mse": mse,
        "pearson_r": 0.8,
        "log_loss": 0.5,
        "brier": 0.2,
        "auc": 0.8,
    }


def _taskwise(model, task, r_squared, mse):
    return {
        "model": model,
        "code": "TEST",
        "information_set": "observable",
        "sharing": "shared_architecture_task_observation",
        "task": task,
        "states": 10,
        "episodes": 3,
        "r_squared": r_squared,
        "mse": mse,
        "pearson_r": 0.8,
        "log_loss": 0.5,
        "brier": 0.2,
        "auc": 0.8,
    }


def test_one_task_superstar_cannot_become_best_cross_task_model():
    tasks = ("bandit", "foraging", "solvability")
    metrics = pd.DataFrame(
        [
            _aggregate("three_task_candidate", 3, 0.70, 0.50),
            _aggregate("foraging_superstar", 1, 0.999, 0.001),
        ]
    )
    taskwise = pd.DataFrame(
        [
            *[_taskwise("three_task_candidate", task, 0.70, 0.50) for task in tasks],
            _taskwise("foraging_superstar", "foraging", 0.999, 0.001),
        ]
    )
    cross_task, within_task = build_model_rankings(metrics, taskwise, tasks)
    assert cross_task.loc[cross_task.best_cross_task_model, "model"].tolist() == [
        "three_task_candidate"
    ]
    assert "foraging_superstar" not in set(cross_task.model)
    assert "foraging_superstar" in set(
        within_task.loc[within_task.task == "foraging", "model"]
    )


def test_flexible_feature_groups_are_explicit_disjoint_and_complete():
    validate_feature_groups()
    grouped = tuple(
        feature for features in FLEXIBLE_FEATURE_GROUPS.values() for feature in features
    )
    assert len(grouped) == len(set(grouped))
    assert flexible_features() == FLEXIBLE_NUISANCE_FEATURES + grouped
    assert "relative_value" not in flexible_features()


def test_mlp_and_gru_recover_a_synthetic_linear_mapping(tmp_path):
    config = {
        "seed": 17,
        "neural_linear_recovery_tolerance_r2": 0.02,
        "neural_sanity": {
            "feature_count": 5,
            "decisions": 4,
            "episodes_per_task": {"train": 12, "validation": 4, "test": 4},
            "noise_sd": 0.02,
            "learning_rate": 0.001,
            "gru_hidden_size": 6,
            "max_epochs": 3,
            "early_stopping_patience": 2,
        },
    }
    result = run_neural_linear_recovery_sanity(config, tmp_path)
    saved = json.loads((tmp_path / "neural_ceiling_sanity.json").read_text())
    assert result["passed"]
    assert saved["checks"]["mlp_approximately_linear"]
    assert saved["checks"]["gru_not_below_linear_tolerance"]
    assert saved["linear_skip_initialized"] == {"mlp": True, "gru": True}
