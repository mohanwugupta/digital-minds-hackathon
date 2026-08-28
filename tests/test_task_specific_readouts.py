import numpy as np

from analysis.persistence_convergence.task_specific_readouts import fit_task_readout


def test_readout_uses_train_only_normalization_and_no_other_task_labels():
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(80, 6))
    x_validation = rng.normal(size=(30, 6))
    x_test = rng.normal(size=(30, 6))
    y_train = x_train[:, 0] - 0.5 * x_train[:, 1]
    y_validation = x_validation[:, 0] - 0.5 * x_validation[:, 1]
    y_test = x_test[:, 0] - 0.5 * x_test[:, 1]
    fit = fit_task_readout(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        alphas=(0.01, 1.0),
    )
    shifted = fit_task_readout(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test + 10_000,
        y_test,
        alphas=(0.01, 1.0),
    )
    np.testing.assert_allclose(fit["coefficient"], shifted["coefficient"])
    np.testing.assert_allclose(fit["normalizer_mean"], x_train.mean(axis=0))
    assert fit["test_r_squared"] > 0.95


def test_permuted_target_control_is_at_chance():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(600, 8))
    y = x[:, 0] + rng.normal(scale=0.05, size=len(x))
    permutation = np.random.default_rng(1)
    fit = fit_task_readout(
        x[:350],
        permutation.permutation(y[:350]),
        x[350:450],
        permutation.permutation(y[350:450]),
        x[450:],
        y[450:],
        alphas=(0.1, 1.0),
    )
    assert fit["test_r_squared"] < 0.1
