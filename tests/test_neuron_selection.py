import pytest

torch = pytest.importorskip("torch")

from interventions.neuron_selection import rank_input_dimensions, select_top_fraction
from interventions.value_probe import ValueProbe


def test_l1_ranking_recovers_sparse_informative_dimensions():
    probe = ValueProbe(100, 8)
    with torch.no_grad():
        probe.hidden.weight.zero_()
        probe.hidden.weight[:, 7] = 4.0
        probe.hidden.weight[:, 42] = 3.0
    ranked = rank_input_dimensions(probe)
    assert set(ranked[:2].tolist()) == {7, 42}
    assert 7 in select_top_fraction(probe, fraction=0.01).tolist()

