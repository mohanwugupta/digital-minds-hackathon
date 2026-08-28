import pandas as pd

from analysis.comparative_persistence.controls.history_specificity import (
    bootstrap_persistence_specific_history,
    persistence_specific_history_index,
)
from analysis.comparative_persistence.evaluation.bootstrap import (
    add_episode_bootstrap_intervals,
)


def test_identical_persistence_and_control_history_gains_give_zero_psh():
    gains = pd.DataFrame(
        {
            "task": ["p1", "p2", "control"],
            "is_persistence_task": [True, True, False],
            "history_log_loss_gain": [0.08, 0.08, 0.08],
        }
    )
    result = persistence_specific_history_index(gains)
    assert result["persistence_gain"] == result["control_gain"]
    assert abs(result["psh"]) < 1e-12


def test_psh_bootstrap_resamples_task_identities():
    gains = pd.DataFrame(
        {
            "task": ["p1", "p2", "control"],
            "is_persistence_task": [True, True, False],
            "history_log_loss_gain": [0.10, 0.06, 0.02],
        }
    )
    interval = bootstrap_persistence_specific_history(gains, samples=100, seed=7)
    assert interval["psh_bootstrap_unit"] == "task_identity"
    assert interval["psh_ci_low"] <= 0.06 <= interval["psh_ci_high"]


def test_within_task_bootstrap_resamples_complete_episodes():
    predictions = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "sharing": ["shared"] * 4,
            "task": ["task"] * 4,
            "pair_id": ["a", "a", "b", "b"],
            "episode_id": ["a-1", "a-1", "b-1", "b-1"],
            "observed": [0, 1, 0, 1],
            "predicted": [0.1, 0.9, 0.2, 0.8],
        }
    )
    taskwise = pd.DataFrame(
        {"model": ["m"], "sharing": ["shared"], "task": ["task"], "log_loss": [0.2]}
    )
    result = add_episode_bootstrap_intervals(
        taskwise, predictions, samples=50, seed=3
    )
    assert result.loc[0, "bootstrap_unit"] == "episode"
    assert result.loc[0, "log_loss_ci_low"] <= result.loc[0, "log_loss_ci_high"]
