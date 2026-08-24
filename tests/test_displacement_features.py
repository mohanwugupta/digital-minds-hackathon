import pytest

torch = pytest.importorskip("torch")

from analysis.persistence_displacements import displacement_features


def test_displacement_uses_next_layer_minus_current_layer():
    activations = torch.tensor(
        [
            [[1.0, 2.0], [4.0, 8.0], [9.0, 18.0]],
            [[2.0, 4.0], [7.0, 10.0], [15.0, 21.0]],
        ]
    )
    observed = displacement_features(activations)
    expected = torch.tensor(
        [
            [[3.0, 6.0], [5.0, 10.0]],
            [[5.0, 6.0], [8.0, 11.0]],
        ]
    )
    assert torch.equal(observed, expected)


def test_displacement_rejects_wrong_layer_axis():
    with pytest.raises(ValueError, match="layers"):
        displacement_features(torch.ones(4, 1, 3))

