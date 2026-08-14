import pytest

torch = pytest.importorskip("torch")

from interventions.ridge_probe import fit_ridge_targets, regression_metrics


@pytest.mark.parametrize("samples,features,solver", [(160, 5, "primal"), (30, 50, "dual")])
def test_ridge_probe_recovers_linear_targets_in_primal_and_dual_modes(
    samples, features, solver
):
    generator = torch.Generator().manual_seed(samples + features)
    states = torch.randn(samples + 40, features, generator=generator)
    if features > samples:
        # Exercise the dual solver without making the synthetic regression
        # intrinsically underidentified on held-out examples.
        states[:, 5:] = 0
    target = 3 * states[:, 1] - 1.5 * states[:, 3]
    probes, fit = fit_ridge_targets(
        states[:samples],
        {"return": target[:samples]},
        states[samples:],
        {"return": target[samples:]},
        alphas=(1e-5, 1e-3, 0.1),
        device="cpu",
    )
    metrics = regression_metrics(probes["return"].predict(states[samples:]), target[samples:])

    assert fit["solver"] == solver
    assert metrics["r_squared"] > 0.95
    assert metrics["correlation"] > 0.98


def test_multitarget_ridge_selects_and_predicts_each_target():
    generator = torch.Generator().manual_seed(93)
    states = torch.randn(180, 6, generator=generator)
    targets = {
        "return": 2 * states[:, 0] + states[:, 2],
        "persistence": -states[:, 1] + 4 * states[:, 5],
    }
    probes, _fit = fit_ridge_targets(
        states[:140],
        {key: value[:140] for key, value in targets.items()},
        states[140:],
        {key: value[140:] for key, value in targets.items()},
        device="cpu",
    )

    for key, target in targets.items():
        metrics = regression_metrics(probes[key].predict(states[140:]), target[140:])
        assert metrics["r_squared"] > 0.95
