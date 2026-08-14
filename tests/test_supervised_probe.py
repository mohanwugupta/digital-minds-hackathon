import pytest

torch = pytest.importorskip("torch")

from interventions.supervised_probe import fit_supervised_probe, supervised_metrics


def test_supervised_probe_recovers_future_return_without_bootstrapping():
    generator = torch.Generator().manual_seed(17)
    states = torch.randn(240, 5, generator=generator)
    returns = 4 * states[:, 1] - 2 * states[:, 4]
    result = fit_supervised_probe(
        states[:180],
        returns[:180],
        states[180:],
        returns[180:],
        hidden_dim=16,
        learning_rate=0.01,
        epochs=120,
        patience=20,
        batch_size=60,
        device="cpu",
    )
    metrics = supervised_metrics(
        result.probe,
        states[180:],
        returns[180:],
        target_mean=result.target_mean,
        target_std=result.target_std,
    )

    assert metrics["r_squared"] > 0.9
    assert metrics["correlation"] > 0.95


def test_masked_supervised_probe_uses_only_selected_dimensions():
    generator = torch.Generator().manual_seed(21)
    states = torch.randn(200, 4, generator=generator)
    returns = 3 * states[:, 2]
    mask = torch.tensor([0.0, 0.0, 1.0, 0.0])
    result = fit_supervised_probe(
        states[:150],
        returns[:150],
        states[150:],
        returns[150:],
        hidden_dim=8,
        learning_rate=0.01,
        epochs=100,
        patience=20,
        batch_size=50,
        device="cpu",
        input_mask=mask,
    )
    first = supervised_metrics(
        result.probe,
        states[150:],
        returns[150:],
        target_mean=result.target_mean,
        target_std=result.target_std,
        input_mask=mask,
    )
    changed = states[150:].clone()
    changed[:, [0, 1, 3]] += 100
    second = supervised_metrics(
        result.probe,
        changed,
        returns[150:],
        target_mean=result.target_mean,
        target_std=result.target_std,
        input_mask=mask,
    )

    assert first["r_squared"] > 0.9
    assert first["mse"] == pytest.approx(second["mse"], abs=1e-6)
