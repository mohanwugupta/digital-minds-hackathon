"""Capacity-matched readouts for existing one-shot nuisance-control banks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.persistence_convergence.activation_cache import (
    build_activation_cache,
    read_bank_records,
)
from analysis.persistence_convergence.persistence_vs_controls import (
    compare_persistence_controls,
    sequence_support,
)
from analysis.persistence_convergence.task_specific_readouts import (
    BlockProjector,
    fit_task_readout,
)


CONTROL_TASKS = {
    "arbitrary_choice": "binary_control",
    "generic_value": "generic_value_control",
    "terminality": "terminality_control",
}


def _split_lookup(inventory, task):
    local = inventory[inventory.task.astype(str) == str(task)]
    lookup = {}
    for row in local.itertuples():
        for state_id in (row.positive_state_id, row.negative_state_id):
            state_id = str(state_id)
            previous = lookup.setdefault(state_id, str(row.split))
            if previous != str(row.split):
                raise ValueError(f"control state crosses splits: {state_id}")
    return lookup


def prepare_control_records(records, inventory, task):
    """Attach inventory splits and retain only preregistered matched endpoints."""

    split = _split_lookup(inventory, task)
    selected = []
    for source in records:
        state_id = str(source["state_id"])
        if state_id not in split:
            continue
        row = dict(source)
        row["split"] = split[state_id]
        row["target_logit"] = float(row["target_logit"])
        selected.append(row)
    missing = sorted(set(split) - {str(row["state_id"]) for row in selected})
    if missing:
        raise ValueError(f"{task} bank is missing inventory states: {missing[:5]}")
    return selected


def fit_control_readouts(datasets, config, *, logger=None):
    """Fit the same all-layer projection/ridge pipeline used for persistence."""

    rows = []
    for control, dataset in sorted(datasets.items()):
        frame = dataset.metadata.reset_index(drop=True)
        support = sequence_support(frame.to_dict(orient="records"))
        projector = BlockProjector(
            dataset.shape[2],
            int(config["projection_dimensions"]),
            int(config["seed"]),
        )
        indices = {
            split: np.flatnonzero(frame.split.astype(str).to_numpy() == split)
            for split in ("train", "validation", "test")
        }
        if any(len(value) == 0 for value in indices.values()):
            raise ValueError(f"{control} requires train, validation, and test states")
        target = frame.target_logit.to_numpy(dtype=float)
        hidden = dataset.open()
        for layer in range(dataset.shape[1]):
            projected = projector.transform(hidden[:, layer, :])
            fit = fit_task_readout(
                projected[indices["train"]],
                target[indices["train"]],
                projected[indices["validation"]],
                target[indices["validation"]],
                projected[indices["test"]],
                target[indices["test"]],
                alphas=config["ridge_grid"],
            )
            random = np.random.default_rng(
                int(config["seed"]) + 10_000 + layer * 101
            )
            random_fit = fit_task_readout(
                projected[indices["train"]],
                random.permutation(target[indices["train"]]),
                projected[indices["validation"]],
                random.permutation(target[indices["validation"]]),
                projected[indices["test"]],
                target[indices["test"]],
                alphas=config["ridge_grid"],
            )
            rows.append(
                {
                    "layer": layer,
                    "control": control,
                    "test_r_squared": fit["test_r_squared"],
                    "test_mse": fit["test_mse"],
                    "test_pearson_r": fit["test_pearson_r"],
                    "selected_alpha": fit["selected_alpha"],
                    "random_target_r_squared": random_fit["test_r_squared"],
                    "history_supported": support["history"],
                    "recurrence_supported": support["recurrence"],
                    "sequence_reason": support["reason"],
                    "projection_dimensions": projector.dimensions,
                    "train_states": len(indices["train"]),
                    "validation_states": len(indices["validation"]),
                    "test_states": len(indices["test"]),
                }
            )
        if logger is not None:
            logger.note(
                "controls",
                f"{control}: fit {dataset.shape[1]} layers on {len(frame)} one-shot states",
            )
    return pd.DataFrame(rows)


def run_controls(
    inventory,
    bank_paths,
    cache_dir,
    persistence_metrics,
    config,
    *,
    maximum_states=None,
    resume=False,
    logger=None,
):
    """Build local caches, fit controls, and compare with persistence readouts."""

    datasets = {}
    for control, task in CONTROL_TASKS.items():
        path = Path(bank_paths[control])
        records = prepare_control_records(read_bank_records(path), inventory, task)
        datasets[control] = build_activation_cache(
            control,
            records,
            path,
            cache_dir,
            maximum_states=maximum_states,
            resume=resume,
            logger=logger,
        )
    control_metrics = fit_control_readouts(datasets, config, logger=logger)
    return compare_persistence_controls(persistence_metrics, control_metrics)
