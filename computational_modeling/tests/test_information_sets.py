import pytest

from computational_modeling.data.feature_schema import (
    FEATURE_SCHEMA,
    features_for,
    select_features,
)


def test_observable_models_cannot_access_oracle_fields():
    row = {
        "log_round": 1.0,
        "oracle_p_a": 0.9,
        "oracle_termination_advantage": 3.0,
    }
    observable = features_for("bandit", "observable")
    assert "oracle_p_a" not in observable
    with pytest.raises(ValueError, match="not permitted"):
        select_features([row], "bandit", "observable", ["oracle_p_a"])
    assert select_features([row], "bandit", "oracle", ["oracle_p_a"]) == [[0.9]]


def test_all_oracle_fields_are_explicitly_prefixed():
    for task, schema in FEATURE_SCHEMA.items():
        assert schema["oracle"]
        assert all(name.startswith("oracle_") for name in schema["oracle"])
