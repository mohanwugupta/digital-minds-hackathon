import pytest

from computational_modeling.data.build_cross_task_behavioral_dataset import validate_split
from computational_modeling.models.base import TrainStandardizer, assert_selection_blind


def test_counterbalanced_pair_cannot_cross_splits():
    records = [
        {"episode_id": "e-x", "pair_id": "pair", "split": "train"},
        {"episode_id": "e-y", "pair_id": "pair", "split": "test"},
    ]
    with pytest.raises(ValueError, match="pair crosses"):
        validate_split(records)


def test_standardizer_uses_training_moments_only():
    normalizer = TrainStandardizer.fit([[0.0], [2.0]], ["x"])
    assert normalizer.mean == [1.0]
    assert normalizer.transform([[101.0]])[0][0] == 100.0
    assert normalizer.fit_split == "train"


def test_hyperparameter_selection_rejects_test_rows():
    with pytest.raises(ValueError, match="test split"):
        assert_selection_blind([{"split": "train"}], [{"split": "test"}])
