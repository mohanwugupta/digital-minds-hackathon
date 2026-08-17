"""Dependency-free integrity tests for the Track B critical-test gates."""

import copy
import json
import unittest

from analysis.analyze_cross_task_causal import analyze, summarize_task
from analysis.cross_task_integrity import (
    audit_cross_task_shards,
    evaluate_behavioral_gate,
    require_representational_clearance,
)


def _mapping(stay_label: str) -> dict[str, str]:
    leave_label = "Y" if stay_label == "X" else "X"
    return {"STAY": stay_label, "LEAVE": leave_label}


def _foraging_shard(
    pair: int,
    stay_label: str,
    *,
    p_stay: float = 0.25,
    semantic_choice: str = "LEAVE",
    outside_option: int = 0,
    stay_cost: int = 0,
) -> dict:
    mapping = _mapping(stay_label)
    mapping_id = f"stay_{stay_label.lower()}"
    episode_id = f"foraging-pair-{pair}-{mapping_id}"
    logit = -1.0 if p_stay < 0.5 else 1.0
    return {
        "task": "foraging",
        "episode_id": episode_id,
        "pair_id": f"foraging-pair-{pair}",
        "mapping_id": mapping_id,
        "records": [
            {
                "episode_id": episode_id,
                "pair_id": f"foraging-pair-{pair}",
                "state_id": f"{episode_id}:0",
                "mapping_id": mapping_id,
                "round": 0,
                "label_mapping": json.dumps(mapping),
                "positive_semantic": "STAY",
                "negative_semantic": "LEAVE",
                "positive_label": mapping["STAY"],
                "negative_label": mapping["LEAVE"],
                "raw_label": mapping[semantic_choice],
                "semantic_choice": semantic_choice,
                "p_stay": p_stay,
                "p_leave": 1.0 - p_stay,
                "persistence_logit": logit,
                "initial_quality": 0.55,
                "depletion": 0.05,
                "outside_option": outside_option,
                "stay_cost": stay_cost,
                "choice_history": [],
                "reward_history": [],
                "terminated": True,
                "termination_reason": (
                    "leave" if semantic_choice == "LEAVE" else "max_decisions"
                ),
            }
        ],
    }


def _causal_rows(task: str, *, target_effect: float) -> list[dict]:
    rows = []
    for episode in range(8):
        mapping = "stay_x" if episode % 2 == 0 else "stay_y"
        state_id = f"{task}-{episode}:0"
        for control_type, control_ids in (
            ("target", ("target",)),
            ("random", tuple(f"random_{index:02d}" for index in range(20))),
        ):
            for control_id in control_ids:
                effect = target_effect if control_type == "target" else 0.001
                for alpha in (-1.0, 0.0, 1.0):
                    rows.append(
                        {
                            "episode_id": f"{task}-{episode}",
                            "state_id": state_id,
                            "mapping_id": mapping,
                            "control_type": control_type,
                            "control_id": control_id,
                            "alpha": alpha,
                            "context_hash": f"context-{state_id}",
                            "p_positive": 0.5 + alpha * effect / 2,
                            "p_negative": 0.5 - alpha * effect / 2,
                            "choice_logit": alpha * effect * 2,
                            "probe_value_pre": 0.0,
                            "probe_value_post": alpha,
                            "direction_l2_norm": 1.0,
                            "intervention_relative_rms": 0.1,
                            "baseline_replay_absolute_difference": 0.0,
                            "logit_X": alpha * effect,
                            "logit_Y": -alpha * effect,
                        }
                    )
    return rows


class TrackBCriticalTest(unittest.TestCase):
    def test_integrity_rejects_malformed_terminal_semantics(self):
        shards = [
            _foraging_shard(0, "X"),
            _foraging_shard(0, "Y"),
        ]
        self.assertTrue(audit_cross_task_shards(shards, "foraging")["passed"])

        malformed = copy.deepcopy(shards)
        malformed[0]["records"][0]["terminated"] = False
        audit = audit_cross_task_shards(malformed, "foraging")
        self.assertFalse(audit["passed"])
        self.assertGreater(audit["issue_counts"]["terminal_position"], 0)

    def test_counterbalance_audit_requires_actual_inverse_mapping(self):
        shards = [
            _foraging_shard(0, "X"),
            _foraging_shard(0, "Y"),
        ]
        broken = copy.deepcopy(shards)
        broken[1]["records"][0]["label_mapping"] = broken[0]["records"][0][
            "label_mapping"
        ]
        audit = audit_cross_task_shards(broken, "foraging")
        self.assertFalse(audit["passed"])
        self.assertGreater(audit["issue_counts"]["counterbalance_mapping"], 0)

    def test_behavioral_gate_uses_development_episodes_only(self):
        shards = []
        for pair, probability, outside, cost in (
            (0, 0.8, 0, 0),
            (1, 0.2, 2, 1),
            (2, 0.99, 0, 0),
        ):
            choice = "STAY" if probability > 0.5 else "LEAVE"
            shards.extend(
                _foraging_shard(
                    pair,
                    label,
                    p_stay=probability,
                    semantic_choice=choice,
                    outside_option=outside,
                    stay_cost=cost,
                )
                for label in ("X", "Y")
            )
        split = {
            "train": [shards[0]["episode_id"], shards[1]["episode_id"]],
            "validation": [shards[2]["episode_id"], shards[3]["episode_id"]],
            "test": [shards[4]["episode_id"], shards[5]["episode_id"]],
        }
        result = evaluate_behavioral_gate(
            shards,
            split,
            {
                "minimum_development_episodes": 4,
                "minimum_development_states": 4,
                "minimum_persistence_logit_sd": 0.5,
                "minimum_p_stay_interdecile_range": 0.5,
                "semantic_stay_rate_bounds": [0.1, 0.9],
                "episode_leave_rate_bounds": [0.1, 0.9],
                "maximum_initial_mapping_p_stay_gap": 0.01,
                "minimum_expected_economic_logit_effect": 0.5,
            },
        )
        self.assertEqual(result["data_scope"], ["train", "validation"])
        self.assertEqual(result["development_episodes"], 4)
        self.assertEqual(result["development_states"], 4)
        self.assertTrue(result["passed"])

    def test_representational_clearance_blocks_causal_calibration(self):
        require_representational_clearance({"classification": "partial_transfer"})
        with self.assertRaises(RuntimeError):
            require_representational_clearance(
                {"classification": "no_convincing_transfer"}
            )

    def test_causal_transfer_requires_specificity_to_foraging(self):
        foraging = _causal_rows("foraging", target_effect=0.20)
        control = _causal_rows("control", target_effect=0.25)
        result = analyze(
            foraging,
            control,
            bootstrap_samples=40,
            expected_random_directions=20,
            negative_control_max_absolute_effect=0.05,
            negative_control_relative_effect_fraction=0.5,
        )
        self.assertFalse(result["criteria"]["negative_control_specificity"])
        self.assertEqual(result["classification"], "no_convincing_causal_transfer")

    def test_causal_audit_rejects_missing_random_state_coverage(self):
        rows = _causal_rows("foraging", target_effect=0.20)
        rows = [
            row
            for row in rows
            if not (
                row["state_id"] == "foraging-0:0"
                and row["control_id"] == "random_00"
            )
        ]
        with self.assertRaises(ValueError):
            summarize_task(
                rows,
                bootstrap_samples=20,
                seed=1,
                expected_random_directions=20,
            )


if __name__ == "__main__":
    unittest.main()
