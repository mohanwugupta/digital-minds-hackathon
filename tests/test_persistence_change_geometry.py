import numpy as np
import pandas as pd
import pytest

from analysis.persistence_change_geometry import (
    audit_exact_pair,
    construct_pair_change,
    fit_change_decoder,
    oriented_difference,
    strict_group_transfer,
)
from analysis.persistence_contrasts import ContrastDefinition
from analysis.persistence_geometry import FrozenPersistenceSubspace


def _definition():
    return ContrastDefinition(
        task="synthetic",
        manipulation="evidence",
        factor="evidence",
        higher_promotes_persistence=True,
        matched_fields=("round", "choice_history", "outcome_history"),
    )


def _record(*, evidence, history=(1, 0), target=1.0):
    return {
        "task": "synthetic",
        "evidence": evidence,
        "round": 2,
        "choice_history": list(history),
        "outcome_history": [1, -1],
        "persistence_logit": target,
    }


def _metadata(tasks, splits):
    return pd.DataFrame(
        {
            "task": tasks,
            "split": splits,
            "episode_id": [f"episode-{index}" for index in range(len(tasks))],
            "pair_id": [f"pair-{index}" for index in range(len(tasks))],
        }
    )


def test_exact_pair_matching_rejects_changed_history():
    promoting = _record(evidence=1.0)
    discouraging = _record(evidence=-1.0, history=(0, 0), target=-1.0)
    with pytest.raises(ValueError, match="choice_history"):
        audit_exact_pair(promoting, discouraging, _definition())


def test_reversing_pair_reverses_neural_and_target_changes():
    positive = np.array([[2.0, 4.0], [6.0, 8.0]])
    negative = np.array([[1.0, -1.0], [2.0, 3.0]])
    basis = np.eye(2)
    forward = construct_pair_change(
        positive[0], positive[1], negative[0], negative[1], basis
    )
    reverse = construct_pair_change(
        negative[0], negative[1], positive[0], positive[1], basis
    )
    for stage in forward:
        np.testing.assert_allclose(reverse[stage], -forward[stage])
    assert oriented_difference(5.0, 2.0) == 3.0
    assert oriented_difference(2.0, 5.0) == -3.0


def test_displacement_difference_uses_exact_l21_to_l22_indexing():
    positive_l21 = np.array([1.0, 2.0])
    positive_l22 = np.array([4.0, 8.0])
    negative_l21 = np.array([-2.0, 5.0])
    negative_l22 = np.array([3.0, 7.0])
    result = construct_pair_change(
        positive_l21,
        positive_l22,
        negative_l21,
        negative_l22,
        np.eye(2),
    )
    expected = (positive_l22 - positive_l21) - (
        negative_l22 - negative_l21
    )
    np.testing.assert_allclose(result["displacement"], expected)


def test_persistence_basis_cannot_be_refit():
    frozen = FrozenPersistenceSubspace.from_array(np.eye(4), source="synthetic")
    with pytest.raises(RuntimeError, match="frozen"):
        frozen.fit(np.ones((8, 4)))
    with pytest.raises(RuntimeError, match="frozen"):
        frozen.refit(np.ones((8, 4)))


def test_loto_never_uses_heldout_task_targets_or_normalization():
    rng = np.random.default_rng(7)
    tasks = np.repeat(["bandit", "foraging", "solvability"], 60)
    splits = np.tile(np.repeat(["train", "validation", "test"], 20), 3)
    values = rng.normal(size=(len(tasks), 4))
    target = values[:, 0] - 0.5 * values[:, 1]
    metadata = _metadata(tasks, splits)
    first = strict_group_transfer(
        values,
        target,
        metadata,
        group_column="task",
        heldout="solvability",
        alphas=(0.01, 1.0),
    )
    changed = target.copy()
    changed[tasks == "solvability"] += rng.normal(1000, 100, size=60)
    second = strict_group_transfer(
        values,
        changed,
        metadata,
        group_column="task",
        heldout="solvability",
        alphas=(0.01, 1.0),
    )
    np.testing.assert_allclose(first["coefficient"], second["coefficient"])
    np.testing.assert_allclose(first["test_prediction"], second["test_prediction"])
    assert "solvability" not in first["fit_groups"]


def test_null_paired_differences_decode_at_chance():
    rng = np.random.default_rng(19)
    count = 900
    tasks = np.resize(np.array(["bandit", "foraging", "solvability"]), count)
    splits = np.resize(
        np.array(["train"] * 6 + ["validation"] * 2 + ["test"] * 2), count
    )
    values = rng.normal(size=(count, 4))
    target = rng.normal(size=count)
    fit = fit_change_decoder(
        values, target, _metadata(tasks, splits), alphas=(0.01, 1.0, 100.0)
    )
    assert fit["test_metrics"]["r_squared"] < 0.08
    assert 0.40 <= fit["test_metrics"]["sign_accuracy"] <= 0.60


def test_shared_change_recovery_succeeds_after_task_offsets_cancel():
    rng = np.random.default_rng(31)
    tasks, splits, target, absolute, change = [], [], [], [], []
    offsets = {"bandit": -25.0, "foraging": 25.0, "solvability": 100.0}
    for task in offsets:
        for split, count in (("train", 80), ("validation", 30), ("test", 40)):
            y = rng.normal(size=count)
            tasks.extend([task] * count)
            splits.extend([split] * count)
            target.extend(y)
            absolute.extend((y + offsets[task])[:, None])
            change.extend((y + rng.normal(scale=0.03, size=count))[:, None])
    metadata = _metadata(tasks, splits)
    absolute_fit = strict_group_transfer(
        np.asarray(absolute),
        np.asarray(target),
        metadata,
        group_column="task",
        heldout="solvability",
        alphas=(0.01, 1.0),
    )
    change_fit = strict_group_transfer(
        np.asarray(change),
        np.asarray(target),
        metadata,
        group_column="task",
        heldout="solvability",
        alphas=(0.01, 1.0),
    )
    assert absolute_fit["test_metrics"]["r_squared"] < 0.0
    assert change_fit["test_metrics"]["r_squared"] > 0.95

