import numpy as np
import pandas as pd
import pytest

from analysis.persistence_geometry import (
    FrozenPersistenceSubspace,
    compare_matched_random_subspaces,
    decode_continuous_target,
    leave_one_task_out_decode,
    stage_representation,
    task_identity_confound,
    validate_episode_splits,
)


def _synthetic_metadata(seed=0, episodes_per_split=18, states_per_episode=4):
    rng = np.random.default_rng(seed)
    rows = []
    tasks = ("bandit", "foraging", "solvability")
    for split in ("train", "validation", "test"):
        for episode in range(episodes_per_split):
            task = tasks[episode % len(tasks)]
            episode_id = f"{split}-{task}-{episode:03d}"
            for state in range(states_per_episode):
                rows.append(
                    {
                        "task": task,
                        "split": split,
                        "episode_id": episode_id,
                        "pair_id": episode_id,
                        "state_id": f"{episode_id}:{state}",
                        "round": state,
                        "noise": rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_persistence_subspace_is_frozen():
    frozen = FrozenPersistenceSubspace.from_array(np.eye(8, 4), source="synthetic")
    with pytest.raises(RuntimeError, match="persistence subspace is frozen"):
        frozen.fit(np.ones((10, 8)), np.ones(10))


def test_episode_or_pair_cannot_cross_splits():
    frame = _synthetic_metadata()
    frame.loc[frame.index[-1], "episode_id"] = frame.loc[0, "episode_id"]
    with pytest.raises(ValueError, match="episode crosses split boundaries"):
        validate_episode_splits(frame)


def test_displacement_is_exactly_h22_minus_h21():
    h21 = np.arange(24, dtype=float).reshape(3, 8)
    h22 = h21 + np.linspace(-2, 3, 24).reshape(3, 8)
    assert np.array_equal(
        stage_representation(h21, h22, "displacement"), h22 - h21
    )
    assert np.array_equal(stage_representation(h21, h22, "l21"), h21)
    assert np.array_equal(stage_representation(h21, h22, "l22"), h22)


def test_known_rank4_signal_beats_matched_random_subspaces():
    metadata = _synthetic_metadata(seed=3, episodes_per_split=30)
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(len(metadata), 32))
    basis, _ = np.linalg.qr(rng.normal(size=(32, 4)))
    coefficient = np.array([1.3, -0.8, 0.6, 0.4])
    target = hidden @ basis @ coefficient + rng.normal(scale=0.05, size=len(hidden))
    result = compare_matched_random_subspaces(
        hidden,
        FrozenPersistenceSubspace.from_array(basis, source="synthetic"),
        target,
        metadata,
        count=24,
        seed=91,
        alphas=(0.01, 0.1, 1.0),
    )
    assert result["persistence_r_squared"] > 0.95
    assert result["persistence_r_squared"] > result["random_r_squared_95th"]


def test_heldout_task_targets_never_enter_cross_task_fit():
    metadata = _synthetic_metadata(seed=5, episodes_per_split=24)
    rng = np.random.default_rng(4)
    values = rng.normal(size=(len(metadata), 4))
    target = values @ np.array([0.8, -0.3, 0.5, 0.2])
    first = leave_one_task_out_decode(
        values,
        target,
        metadata,
        heldout_task="solvability",
        alphas=(0.01, 0.1, 1.0),
    )
    changed = target.copy()
    changed[metadata.task.to_numpy() == "solvability"] += 1_000_000
    second = leave_one_task_out_decode(
        values,
        changed,
        metadata,
        heldout_task="solvability",
        alphas=(0.01, 0.1, 1.0),
    )
    assert np.allclose(first["coefficient"], second["coefficient"])
    assert np.allclose(first["test_prediction"], second["test_prediction"])
    assert "solvability" not in first["fit_tasks"]


def test_randomized_target_is_chance_level_heldout():
    metadata = _synthetic_metadata(seed=7, episodes_per_split=36)
    rng = np.random.default_rng(8)
    values = rng.normal(size=(len(metadata), 4))
    target = rng.normal(size=len(metadata))
    result = decode_continuous_target(
        values, target, metadata, alphas=(0.01, 0.1, 1.0)
    )
    assert result["test_metrics"]["r_squared"] < 0.10


def test_task_determined_target_is_flagged_as_non_generalizing():
    metadata = _synthetic_metadata(seed=9, episodes_per_split=24)
    target = metadata.task.map(
        {"bandit": -2.0, "foraging": 0.0, "solvability": 2.0}
    ).to_numpy()
    cross_task = pd.DataFrame(
        {
            "heldout_task": ["bandit", "foraging", "solvability"],
            "r_squared": [-1.0, 0.0, -1.0],
        }
    )
    result = task_identity_confound(target, metadata, cross_task)
    assert result["flagged"]
    assert result["between_task_variance_fraction"] > 0.99
