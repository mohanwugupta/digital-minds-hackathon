import numpy as np

from computational_modeling.models.base import balanced_weights


def test_episode_and_task_balancing_equalizes_aggregate_contribution():
    rows = [
        {"task": "bandit", "episode_id": "b-long"},
        {"task": "bandit", "episode_id": "b-long"},
        {"task": "bandit", "episode_id": "b-short"},
        {"task": "foraging", "episode_id": "f"},
    ]
    weights = balanced_weights(rows, task_balanced=True)
    by_episode = {}
    by_task = {}
    for row, weight in zip(rows, weights):
        by_episode[row["episode_id"]] = by_episode.get(row["episode_id"], 0) + weight
        by_task[row["task"]] = by_task.get(row["task"], 0) + weight
    assert np.isclose(by_episode["b-long"], by_episode["b-short"])
    assert np.isclose(by_task["bandit"], by_task["foraging"])
    assert np.isclose(np.mean(weights), 1.0)
