import pytest

torch = pytest.importorskip("torch")

from interventions.steering import (
    build_value_direction, random_masked_direction,
    sample_random_neuron_sets, steer_hidden,
)
from interventions.value_probe import ValueProbe


def test_bidirectional_steering_changes_probe_value_and_only_targets_mask():
    probe = ValueProbe(4, 3, mean=torch.zeros(4), std=torch.ones(4))
    with torch.no_grad():
        probe.hidden.weight.zero_()
        probe.hidden.bias.fill_(1.0)
        probe.hidden.weight[:, 1] = 1.0
        probe.output.weight.fill_(1.0)
        probe.output.bias.zero_()
    hidden = torch.zeros(1, 4)
    direction = build_value_direction(probe, hidden, torch.tensor([1]), magnitude=0.1)
    positive = steer_hidden(hidden, direction, alpha=1.0)
    negative = steer_hidden(hidden, direction, alpha=-1.0)
    assert probe(positive).item() > probe(hidden).item() > probe(negative).item()
    assert torch.equal(positive[:, [0, 2, 3]], hidden[:, [0, 2, 3]])
    assert torch.equal(steer_hidden(hidden, direction, alpha=0.0), hidden)


def test_twenty_random_controls_exclude_value_neurons_and_match_magnitude():
    value_indices = torch.tensor([1, 3])
    random_sets = sample_random_neuron_sets(
        20, 2, n_sets=20, seed=9, exclude=value_indices
    )
    assert len(random_sets) == 20
    assert all(not (set(indices.tolist()) & {1, 3}) for indices in random_sets)
    reference = torch.zeros(1, 20)
    reference[:, value_indices] = torch.tensor([0.1, -0.2])
    std = torch.linspace(0.5, 1.5, 20)
    random_direction = random_masked_direction(reference, random_sets[0], std, seed=4)
    reference_norm = ((reference / std) ** 2).sum().sqrt()
    random_norm = ((random_direction / std) ** 2).sum().sqrt()
    assert torch.allclose(reference_norm, random_norm)
