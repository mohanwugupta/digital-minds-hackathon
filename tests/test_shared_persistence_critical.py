"""Dependency-free RED/GREEN gates for the shared persistence program."""

import unittest
from pathlib import Path

from analysis.shared_persistence_integrity import (
    macro_average,
    require_shared_clearance,
    validate_discovery_plan,
    validate_loto_folds,
)
from cross_task.solvability import (
    GIVE_UP,
    TRY_AGAIN,
    SolvabilityCondition,
    SolvabilityEnvironment,
    episode_conditions,
)


class SharedPersistenceCriticalTest(unittest.TestCase):
    def test_solvability_is_deterministic_and_has_valid_terminal_semantics(self):
        condition = SolvabilityCondition(0.6, 1, 2)
        left = SolvabilityEnvironment(condition, 17, max_attempts=3)
        right = SolvabilityEnvironment(condition, 17, max_attempts=3)
        self.assertEqual(left.progress_schedule, right.progress_schedule)

        first = left.step(TRY_AGAIN)
        self.assertFalse(first.terminated)
        self.assertIn(first.progress_made, {True, False})
        final = left.step(GIVE_UP)
        self.assertTrue(final.terminated)
        self.assertEqual(final.reason, "give_up")

    def test_solvability_pairs_reverse_distinct_raw_labels(self):
        episodes = list(episode_conditions(4, 101, labels=("M", "N")))
        for first, second in zip(episodes[::2], episodes[1::2]):
            pair_a, condition_a, mapping_a, seed_a, action_seed_a = first
            pair_b, condition_b, mapping_b, seed_b, action_seed_b = second
            self.assertEqual(
                (pair_a, condition_a, seed_a, action_seed_a),
                (pair_b, condition_b, seed_b, action_seed_b),
            )
            self.assertNotEqual(
                mapping_a.label_for(TRY_AGAIN), mapping_b.label_for(TRY_AGAIN)
            )

    def test_primary_discovery_plan_cannot_use_heldout_task_for_selection(self):
        plan = validate_discovery_plan(
            discovery_tasks=("bandit", "foraging"),
            heldout_task="solvability",
            layer_selection_tasks=("bandit", "foraging"),
        )
        self.assertEqual(plan["heldout_task_parameters_fit"], 0)
        with self.assertRaises(ValueError):
            validate_discovery_plan(
                discovery_tasks=("bandit", "foraging"),
                heldout_task="solvability",
                layer_selection_tasks=("bandit", "solvability"),
            )

    def test_shared_selection_uses_equal_task_weight_not_state_count(self):
        self.assertAlmostEqual(
            macro_average({"bandit": 0.2, "foraging": 0.8}), 0.5
        )

    def test_loto_requires_every_exact_task_complement(self):
        folds = [
            {"discovery": ["bandit", "foraging"], "heldout": "solvability"},
            {"discovery": ["bandit", "solvability"], "heldout": "foraging"},
            {"discovery": ["foraging", "solvability"], "heldout": "bandit"},
        ]
        self.assertEqual(
            len(validate_loto_folds(("bandit", "foraging", "solvability"), folds)),
            3,
        )
        with self.assertRaises(ValueError):
            validate_loto_folds(
                ("bandit", "foraging", "solvability"),
                [folds[0], folds[0], folds[2]],
            )

    def test_causal_work_requires_shared_heldout_clearance(self):
        require_shared_clearance(
            {
                "classification": "partial_shared_transfer",
                "heldout_task_parameters_fit": 0,
                "primary_heldout_task": "solvability",
                "primary_discovery_tasks": ["bandit", "foraging"],
            }
        )
        with self.assertRaises(RuntimeError):
            require_shared_clearance(
                {"classification": "no_convincing_shared_transfer"}
            )
        with self.assertRaises(RuntimeError):
            require_shared_clearance(
                {
                    "classification": "partial_shared_transfer",
                    "heldout_task_parameters_fit": 1,
                    "primary_heldout_task": "solvability",
                    "primary_discovery_tasks": ["bandit", "foraging"],
                }
            )

    def test_cluster_order_runs_shared_test_before_bandit_diagnostic(self):
        source = Path("scripts/submit_track_b.sh").read_text(encoding="utf-8")
        shared = source.index("PHASE=cross_task_shared_representational")
        diagnostic = source.index("PHASE=cross_task_representational")
        self.assertLess(shared, diagnostic)
        causal = Path("scripts/submit_track_b_causal.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("shared_persistence_transfer_summary.json", causal)
        self.assertNotIn("representational_transfer_summary.json", causal)


if __name__ == "__main__":
    unittest.main()
