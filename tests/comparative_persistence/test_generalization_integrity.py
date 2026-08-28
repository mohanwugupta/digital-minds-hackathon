import pandas as pd
import numpy as np
import pytest

from analysis.comparative_persistence.evaluation.few_shot_adaptation import (
    few_shot_partition,
)
from analysis.comparative_persistence.evaluation.leave_one_task_out import (
    loto_partition,
)
from analysis.comparative_persistence.evaluation.metrics import summarize_predictions


def _records():
    rows = []
    for task in ("a", "b", "heldout"):
        for pair_index in range(8):
            split = "train" if pair_index < 4 else "validation" if pair_index < 6 else "test"
            for mapping in ("x", "y"):
                rows.append(
                    {
                        "task": task,
                        "pair_id": f"{task}-pair-{pair_index}",
                        "episode_id": f"{task}-pair-{pair_index}-{mapping}",
                        "state_id": f"{task}-pair-{pair_index}-{mapping}:0",
                        "split": split,
                        "hazard_event": pair_index % 2,
                    }
                )
    return pd.DataFrame(rows)


def test_loto_target_never_enters_fit_or_hyperparameter_selection():
    partition = loto_partition(_records(), "heldout")
    assert set(partition.fit.task) == {"a", "b"}
    assert set(partition.selection.task) == {"a", "b"}
    assert set(partition.evaluation.task) == {"heldout"}
    assert set(partition.evaluation.split) == {"test"}
    assert set(partition.fit.state_id).isdisjoint(partition.evaluation.state_id)


def test_few_shot_adaptation_and_evaluation_pairs_are_disjoint():
    partition = few_shot_partition(_records(), "heldout", pair_count=3, seed=7)
    assert partition.adaptation.pair_id.nunique() == 3
    assert set(partition.adaptation.split) == {"train"}
    assert set(partition.evaluation.split) == {"test"}
    assert set(partition.adaptation.pair_id).isdisjoint(partition.evaluation.pair_id)


def test_macro_metric_is_invariant_to_unequal_task_row_counts():
    base = pd.DataFrame(
        {
            "task": ["small", "small", "large", "large"],
            "observed": [0, 1, 0, 1],
            "predicted": [0.2, 0.8, 0.4, 0.6],
        }
    )
    duplicated = pd.concat([base[base.task == "small"], base[base.task == "large"]] * 25)
    duplicated = pd.concat([base[base.task == "small"], duplicated[duplicated.task == "large"]])
    first = summarize_predictions(base)
    second = summarize_predictions(duplicated)
    assert first["macro_log_loss"] == second["macro_log_loss"]
    assert first["macro_brier"] == second["macro_brier"]


def test_episode_weighted_log_loss_treats_complete_episodes_equally():
    frame = pd.DataFrame(
        {
            "task": ["task"] * 4,
            "episode_id": ["short", "long", "long", "long"],
            "observed": [1, 0, 0, 0],
            "predicted": [0.9, 0.6, 0.6, 0.6],
        }
    )
    result = summarize_predictions(frame)
    short_loss = -np.log(0.9)
    long_loss = -np.log(0.4)
    assert result["episode_weighted_log_loss"] == pytest.approx(
        (short_loss + long_loss) / 2
    )
