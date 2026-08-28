import numpy as np
import pytest

from analysis.persistence_convergence.activation_cache import validate_activation_shard


def test_activation_layer_indexing_is_exact():
    activations = np.arange(3 * 4 * 2).reshape(3, 4, 2)
    records = [{"state_id": f"s{index}"} for index in range(3)]
    result = validate_activation_shard(records, activations, expected_layers=4)
    np.testing.assert_array_equal(result[:, 2, :], activations[:, 2, :])


def test_activation_shard_rejects_record_and_layer_mismatch():
    with pytest.raises(ValueError, match="record count"):
        validate_activation_shard([{"state_id": "s"}], np.zeros((2, 4, 3)), expected_layers=4)
    with pytest.raises(ValueError, match="layer count"):
        validate_activation_shard([{"state_id": "s"}], np.zeros((1, 3, 3)), expected_layers=4)
