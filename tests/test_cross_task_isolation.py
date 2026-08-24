import pytest

from analysis.persistence_isolation import validate_task_holdout


def test_leave_one_task_out_rejects_layer_selection_leakage():
    with pytest.raises(ValueError, match="held-out task"):
        validate_task_holdout(
            discovery_tasks=("bandit", "foraging"),
            heldout_task="solvability",
            selection_tasks=("bandit", "foraging", "solvability"),
        )


def test_leave_one_task_out_has_zero_heldout_fit_parameters():
    result = validate_task_holdout(
        discovery_tasks=("bandit", "foraging"),
        heldout_task="solvability",
        selection_tasks=("foraging", "bandit"),
    )
    assert result["heldout_parameters_fit"] == 0

