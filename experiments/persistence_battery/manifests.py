"""Condition, literature, and episode-safe split manifests."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .base_environment import condition_id
from .registry import TASKS


def write_manifests(output, config, specs_by_task, *, mode, smoke):
    output = Path(output)
    directory = output / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    task_specs, condition_rows = {}, []
    split_manifest = {
        "mode": mode,
        "smoke": bool(smoke),
        "split_unit": "semantic_pair",
        "pairs": {},
        "episodes": {},
    }
    for task, specs in specs_by_task.items():
        definition = TASKS[task]
        conditions = definition.conditions(config["tasks"][task])
        task_specs[task] = {
            **definition.literature,
            "task": task,
            "persistence_task": definition.persistence,
            "same_goal_across_steps": definition.persistence,
            "positive_action": definition.positive_action,
            "negative_action": definition.negative_action,
            "manipulated_variables": list(definition.manipulated_variables),
            "factorial_condition_count": len(conditions),
            "recorded_episode_target": len(specs),
            "response_labels": list(config["response_labels"]),
            "full_activation_collection": False,
            "pilot_adjustments": config.get("pilot_adjustments", {}).get(task, []),
        }
        seen_pairs = set()
        for spec in specs:
            split_manifest["episodes"][spec.episode_id] = spec.split
            split_manifest["pairs"].setdefault(spec.pair_id, spec.split)
            if split_manifest["pairs"][spec.pair_id] != spec.split:
                raise ValueError(f"counterbalanced pair crosses split: {spec.pair_id}")
            if spec.pair_id in seen_pairs:
                continue
            seen_pairs.add(spec.pair_id)
            condition = asdict(spec.condition)
            condition_rows.append(
                {
                    "task": task,
                    "pair_id": spec.pair_id,
                    "condition_id": condition_id(task, spec.condition),
                    "split": spec.split,
                    "seed": spec.seed,
                    **condition,
                }
            )
    (directory / "task_specs.json").write_text(
        json.dumps(task_specs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(condition_rows).to_csv(
        directory / "condition_inventory.csv", index=False
    )
    (directory / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return task_specs
