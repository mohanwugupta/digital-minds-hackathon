import pytest

from analysis.persistence_isolation import validate_manipulation_holdout


def test_leave_one_manipulation_out_rejects_discovery_leakage():
    with pytest.raises(ValueError, match="held-out manipulation"):
        validate_manipulation_holdout(
            discovery_manipulations=("continue_bonus", "outside_option"),
            heldout_manipulation="outside_option",
            selection_manipulations=("continue_bonus", "outside_option"),
        )


def test_leave_one_manipulation_out_accepts_exact_discovery_selection_set():
    result = validate_manipulation_holdout(
        discovery_manipulations=("continue_bonus", "search_cost"),
        heldout_manipulation="progress_evidence",
        selection_manipulations=("search_cost", "continue_bonus"),
    )
    assert result["heldout_parameters_fit"] == 0

