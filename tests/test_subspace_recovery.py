import pytest

torch = pytest.importorskip("torch")

from analysis.persistence_subspace import (
    evaluate_subspace,
    fit_balanced_subspace,
    validate_initial_rank,
)


def test_initial_search_rejects_rank_above_four():
    with pytest.raises(ValueError, match="1, 2, or 4"):
        validate_initial_rank(8)


def test_rank_two_recovers_two_dimensional_shared_signal_better_than_rank_one():
    deltas = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.1, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.1, 0.0, 0.1],
        ]
    )
    rows = [
        {"task": "a", "manipulation": "m1"},
        {"task": "a", "manipulation": "m1"},
        {"task": "b", "manipulation": "m2"},
        {"task": "b", "manipulation": "m2"},
    ]
    heldout = torch.tensor([[1.0, -1.0, 0.0, 0.0]])

    rank_one = fit_balanced_subspace(deltas, rows, rank=1)
    rank_two = fit_balanced_subspace(deltas, rows, rank=2)
    score_one = evaluate_subspace(rank_one, heldout)["captured_energy_fraction"]
    score_two = evaluate_subspace(rank_two, heldout)["captured_energy_fraction"]

    assert score_two > 0.99
    assert score_two > score_one + 0.5

