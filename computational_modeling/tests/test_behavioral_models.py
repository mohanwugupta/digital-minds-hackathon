import numpy as np
import pandas as pd

from computational_modeling.analysis.compare_behavioral_models import (
    add_behavioral_states,
    episode_balanced_weights,
)


def test_behavioral_states_update_only_the_chosen_arm_and_track_losses():
    frame = pd.DataFrame(
        {
            "episode_id": ["e1"] * 4,
            "round": [0, 1, 2, 3],
            "sampled_action": ["A", "A", "B", "C"],
            "subsequent_reward": [3, -2, -2, 0],
        }
    )

    states = add_behavioral_states(frame, alpha=0.5)

    assert states.loc[0, "rw_A"] == 0.5
    assert states.loc[1, "rw_A"] == 1.75
    assert states.loc[1, "rw_B"] == 0.5
    assert states.loc[2, "rw_A"] == -0.125
    assert states.loc[3, "rw_B"] == -0.75
    assert states["loss_streak"].tolist() == [0, 0, 1, 2]
    assert np.isclose(states.loc[1, "bayes_A"], 2 / 3 * 5 - 2)
    assert states.loc[1, "bayes_B"] == 0.5


def test_episode_balancing_gives_each_episode_equal_total_weight():
    frame = pd.DataFrame({"episode_id": ["long"] * 3 + ["short"]})
    weights = episode_balanced_weights(frame)
    totals = pd.Series(weights).groupby(frame["episode_id"]).sum()

    assert np.isclose(totals["long"], totals["short"])
    assert np.isclose(weights.mean(), 1.0)
