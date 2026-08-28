"""Combine frozen PRD 2 behavior with approved PRD 2.5 extension tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from analysis.comparative_persistence.build_modeling_dataset import (
    _battery_rows,
    build_hazard_risk_set,
    load_modeling_dataset,
    validate_split_integrity,
)
from analysis.comparative_persistence.normalization import StaticNormalizer
from analysis.comparative_persistence.semantic_features import add_causal_history
from experiments.persistence_battery.storage import read_records_frame


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def effective_model_config(config):
    """Return the PRD 2 model config with frozen PRD 2.5 additions."""

    base = _yaml(config["inputs"]["comparative_config"])
    effective = {**base, **config}
    effective["task_specs"] = {
        **base["task_specs"],
        **config["task_specs"],
    }
    # The PRD 2.5 GRU has its own trainer.  The inherited MLP still expects
    # the original key names and uses the explicitly supplied PRD 2.5 block.
    effective["mlp"] = dict(config["mlp"])
    effective["evaluation"] = {
        **base["evaluation"],
        **config["evaluation"],
    }
    effective["bootstrap"] = {
        "samples": int(config["bootstrap"]["task_samples"]),
        **config["bootstrap"],
    }
    return effective


def _has_task_records(directory, task):
    return any(
        (directory / f"{task}{suffix}").exists()
        for suffix in (".parquet", ".csv.gz")
    )


def _extension_record_source(run_root, task, config, *, smoke):
    """Resolve each task independently so partial approved full runs are usable."""

    requested = str(config["inputs"].get("extension_dataset", "full"))
    full = run_root / "records"
    pilot = run_root / "pilot/records"
    if not smoke and requested == "full" and _has_task_records(full, task):
        return full, "full"
    if _has_task_records(pilot, task):
        return pilot, "pilot"
    if requested == "full" and _has_task_records(full, task):
        return full, "full"
    raise FileNotFoundError(
        f"extension records for {task} are missing under {run_root}; "
        "run/finalize the PRD 2.5 pilot first"
    )


def _extension_approval(run_root):
    path = run_root / "validation/pilot_approval.json"
    if not path.exists():
        raise FileNotFoundError(f"extension pilot approval is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extension_validation(run_root):
    path = run_root / "validation/behavioral_non_degeneracy.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.insert(0, "source", "prd2_5_extension")
    return frame


def build_robustness_dataset(config, run_root, *, smoke=False):
    """Build the expanded hazard dataset without outcome-based task repair."""

    run_root = Path(run_root)
    model_config = effective_model_config(config)
    base_config = _yaml(config["inputs"]["comparative_config"])
    base_records, base_inclusion = load_modeling_dataset(base_config, smoke=False)
    approval = _extension_approval(run_root)
    normalizer = StaticNormalizer(model_config["task_specs"])
    extension_rows = []
    inclusion_rows = []
    extension_stages = []
    extension_tasks = list(config["task_breadth"]["extension_tasks"])
    for task in extension_tasks:
        extension_directory, dataset_stage = _extension_record_source(
            run_root, task, config, smoke=smoke
        )
        extension_stages.append(dataset_stage)
        pilot_approved = bool(approval.get("tasks", {}).get(task, False))
        included = bool(smoke or pilot_approved)
        reason = (
            "smoke plumbing override"
            if smoke
            else "passed PRD 2.5 functional pilot gates"
            if pilot_approved
            else "failed PRD 2.5 functional pilot gates"
        )
        inclusion_rows.append(
            {
                "task": task,
                "included": included,
                "reason": reason,
                "source": "prd2_5_extension",
                "dataset_stage": dataset_stage,
            }
        )
        if not included:
            continue
        frame = read_records_frame(extension_directory, task)
        extension_rows.extend(_battery_rows(frame, task, normalizer, True))

    extension = pd.DataFrame(extension_rows)
    if not extension.empty:
        payoff_scales = {
            task: spec["payoff_scale"]
            for task, spec in model_config["task_specs"].items()
        }
        extension = add_causal_history(
            extension,
            decay=float(model_config["history_decay"][0]),
            payoff_scales=payoff_scales,
        )
        extension = build_hazard_risk_set(extension)

    # The original comparative inclusion file contains failed rows for the
    # three tasks being repaired.  Replace rather than duplicate those rows.
    base_inclusion = base_inclusion[
        ~base_inclusion.task.isin(extension_tasks)
    ].copy()
    base_inclusion["source"] = "prd2_frozen"
    base_inclusion["dataset_stage"] = "retained"
    inclusion = pd.concat(
        (base_inclusion, pd.DataFrame(inclusion_rows)), ignore_index=True
    )
    records = pd.concat((base_records, extension), ignore_index=True, sort=False)
    validate_split_integrity(records)

    specs = model_config["task_specs"]
    summaries = []
    for task, part in records.groupby("task"):
        inclusion_row = inclusion[inclusion.task == task].iloc[0]
        summaries.append(
            {
                "task": task,
                "family": specs[task]["family"],
                "source": inclusion_row["source"],
                "dataset_stage": inclusion_row["dataset_stage"],
                "is_persistence_task": bool(part.is_persistence_task.iloc[0]),
                "states": len(part),
                "episodes": int(part.episode_id.nunique()),
                "semantic_pairs": int(part.pair_id.nunique()),
                "task_macro_weight": 1.0,
                "extension_minimum_sample_met": (
                    bool(part.pair_id.nunique() >= 256)
                    if task in extension_tasks
                    else None
                ),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("task")
    validation = _extension_validation(run_root)
    unique_stages = sorted(set(extension_stages))
    dataset_stage = unique_stages[0] if len(unique_stages) == 1 else "mixed"
    return (
        records.reset_index(drop=True),
        inclusion,
        validation,
        summary,
        model_config,
        dataset_stage,
    )
