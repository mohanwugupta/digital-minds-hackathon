import numpy as np
import pandas as pd
import pytest

from analysis.comparative_persistence.build_modeling_dataset import (
    build_hazard_risk_set,
    validate_split_integrity,
)
from analysis.comparative_persistence.normalization import StaticNormalizer
from analysis.comparative_persistence.semantic_features import build_feature_matrix


def _row(step, continued, *, split="train", pair="pair-1"):
    return {
        "task": "demo",
        "episode_id": "episode-1",
        "pair_id": pair,
        "state_id": f"episode-1:{step}",
        "round": step,
        "split": split,
        "continued": continued,
        "is_persistence_task": True,
        "outcome_after_choice": 1.0,
    }


def test_absorbing_risk_set_rejects_post_termination_states():
    with pytest.raises(ValueError, match="post-termination"):
        build_hazard_risk_set(
            pd.DataFrame([_row(0, True), _row(1, False), _row(2, True)])
        )


def test_horizon_completion_is_right_censored_not_a_quit_event():
    risk = build_hazard_risk_set(pd.DataFrame([_row(0, True), _row(1, True)]))
    assert risk.hazard_event.tolist() == [0, 0]
    assert risk.at_risk.eq(1).all()


def test_counterbalanced_pair_cannot_cross_splits():
    records = pd.DataFrame(
        [
            _row(0, True, split="train", pair="same-pair"),
            {
                **_row(0, True, split="test", pair="same-pair"),
                "episode_id": "episode-2",
                "state_id": "episode-2:0",
            },
        ]
    )
    with pytest.raises(ValueError, match="pair crosses splits"):
        validate_split_integrity(records)


def test_environment_spec_normalization_is_data_independent_and_reproducible():
    normalizer = StaticNormalizer(
        {"demo": {"payoff_scale": 10.0, "effort_scale": 20.0, "horizon": 5}}
    )
    row = normalizer.transform_row(
        "demo",
        {
            "current_continue_cost": 2.0,
            "current_outside_option": 5.0,
            "elapsed_steps": 2.0,
            "cumulative_effort": 4.0,
            "current_progress": None,
        },
    )
    assert row["cost_norm"] == pytest.approx(0.2)
    assert row["outside_norm"] == pytest.approx(0.5)
    assert row["time_norm"] == pytest.approx(0.4)
    assert row["effort_norm"] == pytest.approx(0.2)
    assert np.isnan(row["progress_norm"])


def test_missing_construct_gets_an_explicit_mask_and_future_fields_are_forbidden():
    rows = pd.DataFrame([{"cost_norm": 0.2, "progress_norm": np.nan}])
    matrix, names = build_feature_matrix(rows, ["cost_norm", "progress_norm"])
    assert names == (
        "cost_norm",
        "cost_norm__present",
        "progress_norm",
        "progress_norm__present",
    )
    assert matrix.tolist() == [[0.2, 1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="future leakage"):
        build_feature_matrix(
            pd.DataFrame([{"subsequent_reward": 10.0}]), ["subsequent_reward"]
        )
