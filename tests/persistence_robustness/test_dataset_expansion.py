import pandas as pd

from analysis.persistence_robustness.build_dataset import effective_model_config


def test_effective_config_preserves_frozen_base_specs_and_adds_repairs():
    import yaml

    config = yaml.safe_load(open("config/persistence_robustness_v1.yaml"))
    effective = effective_model_config(config)
    assert "bandit" in effective["task_specs"]
    assert effective["task_specs"]["voluntary_waiting"]["family"] == "temporal_waiting"
    assert effective["task_specs"]["sunk_cost"]["family"] == "investment_decision"
    assert effective["bootstrap"]["samples"] == config["bootstrap"]["task_samples"]


def test_task_macro_weights_are_equal_by_construction():
    summary = pd.DataFrame(
        {"task": ["small", "large"], "states": [10, 10_000], "task_macro_weight": [1.0, 1.0]}
    )
    assert summary.groupby("task").task_macro_weight.first().nunique() == 1

