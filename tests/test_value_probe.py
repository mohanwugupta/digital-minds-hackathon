import pytest

torch = pytest.importorskip("torch")

from interventions.value_probe import ValueProbe, fit_value_probe, td_loss


def test_td_loss_uses_zero_next_value_for_terminal_states():
    probe = ValueProbe(input_dim=2, hidden_dim=4)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        probe.output.bias.fill_(2.0)
    states = torch.zeros(2, 2)
    next_states = torch.zeros(2, 2)
    rewards = torch.tensor([1.0, 1.0])
    terminal = torch.tensor([False, True])
    loss, delta = td_loss(probe, states, next_states, rewards, terminal)
    assert torch.allclose(delta, torch.tensor([1.0, -1.0]))
    assert torch.isclose(loss, torch.tensor(1.0))


def test_training_normalization_is_frozen_in_state_dict():
    mean = torch.tensor([2.0, 4.0])
    std = torch.tensor([0.5, 2.0])
    probe = ValueProbe(2, 4, mean=mean, std=std)
    assert torch.allclose(probe.normalize(torch.tensor([[2.5, 8.0]])), torch.tensor([[1.0, 2.0]]))
    assert "mean" in probe.state_dict() and "std" in probe.state_dict()


def test_probe_recovers_a_synthetic_terminal_value_structure():
    torch.manual_seed(5)
    states = torch.randn(160, 3)
    rewards = 2.0 * states[:, 0] - states[:, 2]
    data = {
        "states": states,
        "next_states": torch.zeros_like(states),
        "rewards": rewards,
        "terminal": torch.ones(len(states), dtype=torch.bool),
    }
    result = fit_value_probe(
        {key: value[:120] for key, value in data.items()},
        {key: value[120:] for key, value in data.items()},
        hidden_dim=16,
        learning_rate=0.01,
        epochs=150,
        batch_size=40,
        patience=25,
        device="cpu",
    )
    with torch.no_grad():
        prediction = result.probe(states[120:])
    correlation = torch.corrcoef(torch.stack([prediction, rewards[120:]]))[0, 1]
    assert correlation > 0.9
