"""Numerical tests for task-balanced shared persistence fitting."""

import unittest


try:
    import torch
except ImportError:  # The cluster TDD job has torch; lightweight local checks may not.
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class SharedRidgeProbeTest(unittest.TestCase):
    def test_recovers_common_direction_with_unequal_task_sizes(self):
        from interventions.shared_ridge_probe import fit_balanced_shared_ridge

        generator = torch.Generator().manual_seed(3)
        direction = torch.tensor([1.5, -0.7, 0.0])

        def data(count, nuisance):
            states = torch.randn(count, 3, generator=generator)
            target = states @ direction + nuisance * states[:, 2]
            return {"states": states, "target": target}

        train = {"large": data(400, 0.2), "small": data(40, -0.2)}
        validation = {"large": data(80, 0.2), "small": data(80, -0.2)}
        probe, fit = fit_balanced_shared_ridge(
            train, validation, alphas=(0.001, 0.1), device="cpu"
        )
        self.assertEqual(fit["task_weighting"], "equal_macro_weight")
        self.assertEqual(set(fit["target_moments"]), {"large", "small"})
        self.assertGreater(fit["selected"]["macro_correlation"], 0.95)
        self.assertEqual(probe.target_mean, 0.0)
        self.assertEqual(probe.target_std, 1.0)

    def test_rejects_mismatched_discovery_tasks(self):
        from interventions.shared_ridge_probe import fit_balanced_shared_ridge

        data = {"states": torch.randn(5, 2), "target": torch.randn(5)}
        with self.assertRaises(ValueError):
            fit_balanced_shared_ridge(
                {"bandit": data, "foraging": data},
                {"bandit": data, "solvability": data},
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
