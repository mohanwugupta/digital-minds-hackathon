import numpy as np

from analysis.persistence_convergence.layerwise_geometry import direction_similarity


def test_synthetic_convergence_detects_known_transition():
    directions = {}
    for layer in range(8):
        mixing = layer / 7
        common = np.array([1.0, 0.0, 0.0, 0.0])
        directions[layer] = {
            "bandit": mixing * common + (1 - mixing) * np.array([0, 1, 0, 0]),
            "foraging": mixing * common + (1 - mixing) * np.array([0, 0, 1, 0]),
            "solvability": mixing * common + (1 - mixing) * np.array([0, 0, 0, 1]),
        }
    result = direction_similarity(directions)
    aggregate = result[result.metric == "mean_pairwise_cosine"].sort_values("layer")
    assert aggregate.iloc[-1].value > 0.99
    assert aggregate.iloc[-1].value > aggregate.iloc[0].value + 0.9


def test_orthogonal_directions_do_not_spuriously_converge():
    stable = {
        layer: {
            "bandit": np.array([1.0, 0.0, 0.0]),
            "foraging": np.array([0.0, 1.0, 0.0]),
            "solvability": np.array([0.0, 0.0, 1.0]),
        }
        for layer in range(6)
    }
    result = direction_similarity(stable)
    aggregate = result[result.metric == "mean_pairwise_cosine"]
    assert np.max(np.abs(aggregate.value)) < 1e-12

