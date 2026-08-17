"""Critical tests added from the second construct-validity review."""

import unittest
from pathlib import Path
import hashlib
import json

import yaml

from analysis.cross_task_power import fisher_correlation_power
from analysis.matched_label_integrity import audit_matched_label_shards
from analysis.analyze_shared_persistence_causal import control_specificity
from analysis.shared_persistence_integrity import source_task_gate
from cross_task.terminality import (
    END,
    PROCEED,
    episode_conditions,
    rule_prompt,
)


class NewDirectionControlTest(unittest.TestCase):
    def test_rule_terminality_has_no_worthwhile_persistence_decision(self):
        episodes = list(episode_conditions(4, 91, labels=("M", "N")))
        for first, second in zip(episodes[::2], episodes[1::2]):
            pair_a, integer_a, mapping_a, seed_a = first
            pair_b, integer_b, mapping_b, seed_b = second
            self.assertEqual((pair_a, integer_a, seed_a), (pair_b, integer_b, seed_b))
            self.assertNotEqual(
                mapping_a.label_for(PROCEED), mapping_b.label_for(PROCEED)
            )
            prompt = rule_prompt(integer_a, mapping_a).lower()
            self.assertIn("external rule", prompt)
            self.assertNotIn("worthwhile", prompt)
            expected = PROCEED if integer_a % 2 == 0 else END
            self.assertIn(mapping_a.label_for(expected), prompt.upper())

    def test_shared_source_gate_requires_both_tasks_and_random_controls(self):
        passed = source_task_gate(
            {
                "bandit": {"correlation": 0.7, "r_squared": 0.4},
                "foraging": {"correlation": 0.3, "r_squared": 0.1},
            },
            {"bandit": 0.1, "foraging": 0.1},
        )
        self.assertTrue(passed["passed"])
        failed = source_task_gate(
            {
                "bandit": {"correlation": 0.7, "r_squared": 0.4},
                "foraging": {"correlation": 0.03, "r_squared": -0.1},
            },
            {"bandit": 0.1, "foraging": 0.1},
        )
        self.assertFalse(failed["passed"])

    def test_matched_label_audit_requires_identical_semantic_histories(self):
        history = {"choice_history": ["STAY"], "round": 1}
        true_hash = hashlib.sha256(
            json.dumps(history, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        def shard(label, history_hash):
            mapping = (
                {"STAY": "X", "LEAVE": "Y"}
                if label == "X"
                else {"STAY": "Y", "LEAVE": "X"}
            )
            episode = f"matched-1-stay_{label.lower()}"
            return {
                "task": "foraging_label_replay",
                "episode_id": episode,
                "pair_id": "matched-1",
                "mapping_id": f"stay_{label.lower()}",
                "records": [
                    {
                        "episode_id": episode,
                        "pair_id": "matched-1",
                        "state_id": f"{episode}:0",
                        "mapping_id": f"stay_{label.lower()}",
                        "label_mapping": mapping,
                        "positive_semantic": "STAY",
                        "negative_semantic": "LEAVE",
                        "positive_label": mapping["STAY"],
                        "negative_label": mapping["LEAVE"],
                        "matched_history_hash": history_hash,
                        "matched_history": history,
                        "source_state_id": "source:2",
                        "persistence_logit": 0.2,
                    }
                ],
            }

        self.assertTrue(
            audit_matched_label_shards(
                [shard("X", true_hash), shard("Y", true_hash)], "foraging"
            )["passed"]
        )
        self.assertFalse(
            audit_matched_label_shards(
                [shard("X", true_hash), shard("Y", "different")], "foraging"
            )["passed"]
        )

    def test_384_pairs_power_moderate_at_conservative_effect(self):
        self.assertLess(fisher_correlation_power(0.15, 192), 0.60)
        self.assertGreater(fisher_correlation_power(0.15, 384), 0.80)
        self.assertLess(fisher_correlation_power(0.15, 59), 0.30)

    def test_terminality_must_be_weak_relative_to_causal_target(self):
        self.assertTrue(control_specificity(0.20, 0.03, 0.05, 0.50))
        self.assertFalse(control_specificity(0.20, 0.15, 0.05, 0.50))

    def test_cluster_graph_contains_every_new_direction_gate(self):
        runner = Path("run_qwen35_bandit.sh").read_text(encoding="utf-8")
        submit = Path("scripts/submit_track_b.sh").read_text(encoding="utf-8")
        causal = Path("scripts/submit_track_b_causal.sh").read_text(encoding="utf-8")
        for required in (
            "cross_task_power",
            "cross_task_collect_terminality",
            "cross_task_matched_label_foraging",
            "cross_task_matched_label_solvability",
        ):
            self.assertIn(required, runner)
            self.assertIn(required, submit)
        self.assertIn("cross_task_causal_terminality", runner)
        self.assertIn("cross_task_causal_terminality", causal)

    def test_preregistered_sample_has_at_least_300_independent_pairs(self):
        config = yaml.safe_load(
            Path("config/cross_task_experiment.yaml").read_text(encoding="utf-8")
        )
        collection = config["collection"]
        for task in ("foraging", "solvability", "control", "terminality"):
            self.assertGreaterEqual(collection[f"{task}_episodes"] // 2, 300)


if __name__ == "__main__":
    unittest.main()
