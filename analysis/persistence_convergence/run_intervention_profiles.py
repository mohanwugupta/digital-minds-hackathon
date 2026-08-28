"""Layerwise functional effects of existing oriented persistence manipulations."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.persistence_convergence.functional_convergence import predict_readout
from analysis.persistence_convergence.intervention_profiles import profile_summary


def _factorial_rows(inventory, readout_result, source, config, logger):
    """Project existing Bandit factorial tensors when they are materialized."""

    import torch

    from experiments.project_factorial_layers import _activation_path, _read_factorial_rows

    pattern, activation_dir = source["rows"], source["activations"]
    activation_dir = Path(activation_dir)
    if not activation_dir.is_dir():
        if logger is not None:
            logger.note(
                "interventions",
                "bandit: factorial all-layer tensors are absent locally; skipping Bandit profiles",
            )
        return []
    local = inventory[
        (inventory.task.astype(str) == "bandit")
        & (inventory.contrast_kind.astype(str) == "persistence")
    ].copy()
    maximum = config.get("max_pairs_per_group")
    if maximum is not None:
        local = pd.concat(
            [
                part.sort_values("contrast_id").head(int(maximum))
                for _key, part in local.groupby(["manipulation", "split"], sort=False)
            ],
            ignore_index=True,
        )
    source_index = {}
    for row in _read_factorial_rows(pattern):
        state_id = (
            f"{row['state_id']}:stop={int(float(row['stop_payoff']))}:"
            f"continue={int(float(row['continue_bonus']))}"
        )
        if state_id in source_index:
            raise ValueError(f"duplicate Bandit factorial endpoint: {state_id}")
        source_index[state_id] = dict(row)
    wanted = set(local.positive_state_id.astype(str)) | set(
        local.negative_state_id.astype(str)
    )
    missing = sorted(wanted - set(source_index))
    if missing:
        raise ValueError(f"Bandit factorial rows lack intervention endpoints: {missing[:5]}")
    by_base = {}
    for state_id in wanted:
        by_base.setdefault(str(source_index[state_id]["state_id"]), []).append(state_id)
    scores = {}
    for base_index, base_state in enumerate(sorted(by_base), start=1):
        path = Path(_activation_path(str(activation_dir), base_state))
        if not path.exists():
            raise FileNotFoundError(f"missing Bandit factorial activation: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        values = artifact["activations"].detach().cpu().float().numpy()
        condition_index = {
            (int(row["stop_payoff"]), int(row["continue_bonus"])): index
            for index, row in enumerate(artifact["conditions"])
        }
        layer_scores = np.empty((len(values), values.shape[1]), dtype=float)
        for layer in range(values.shape[1]):
            layer_scores[:, layer] = predict_readout(
                readout_result["models"][layer]["bandit"],
                readout_result["projector"],
                values[:, layer, :],
            )
        for state_id in by_base[base_state]:
            row = source_index[state_id]
            condition = (
                int(float(row["stop_payoff"])),
                int(float(row["continue_bonus"])),
            )
            scores[state_id] = layer_scores[condition_index[condition]]
        if logger is not None and (base_index == len(by_base) or base_index % 100 == 0):
            logger.note(
                "interventions",
                f"bandit: {base_index}/{len(by_base)} factorial tensors",
            )
    pair_rows = []
    for row in local.itertuples():
        effect = scores[str(row.positive_state_id)] - scores[str(row.negative_state_id)]
        for layer, value in enumerate(effect):
            pair_rows.append(
                {
                    "task": "bandit",
                    "manipulation": str(row.manipulation),
                    "split": str(row.split),
                    "cluster_id": str(row.cluster_id),
                    "layer": layer,
                    "effect": float(value),
                }
            )
    rows = []
    for (manipulation, split, layer), part in pd.DataFrame(pair_rows).groupby(
        ["manipulation", "split", "layer"]
    ):
        rows.append(
            {
                "task": "bandit",
                "manipulation": manipulation,
                "split": split,
                "layer": int(layer),
                "mean_functional_effect": float(part.effect.mean()),
                "mean_absolute_effect": float(part.effect.abs().mean()),
                "effect_sd": float(part.effect.std(ddof=1)),
                "positive_fraction": float((part.effect > 0).mean()),
                "pairs": len(part),
                "readout_task": "bandit",
            }
        )
    return rows


def run_intervention_profiles(
    inventory,
    datasets,
    readout_result,
    config,
    *,
    bandit_factorial=None,
    logger=None,
):
    inventory = inventory[inventory.contrast_kind.astype(str) == "persistence"].copy()
    rows = []
    for task, dataset in datasets.items():
        if task == "bandit" and bandit_factorial is not None:
            # Organic Bandit states do not contain the exogenous factorial
            # endpoints.  They are streamed separately below when available.
            continue
        local = inventory[inventory.task.astype(str) == task]
        lookup = {
            str(state_id): index
            for index, state_id in enumerate(dataset.metadata.state_id.astype(str))
        }
        local = local[
            local.positive_state_id.astype(str).isin(lookup)
            & local.negative_state_id.astype(str).isin(lookup)
        ]
        if local.empty:
            if logger is not None:
                logger.note("interventions", f"{task}: no compatible organic intervention pairs")
            continue
        hidden = dataset.open()
        positive = np.asarray([lookup[str(value)] for value in local.positive_state_id], dtype=int)
        negative = np.asarray([lookup[str(value)] for value in local.negative_state_id], dtype=int)
        for layer in range(dataset.shape[1]):
            model = readout_result["models"][layer][task]
            score = predict_readout(
                model, readout_result["projector"], hidden[:, layer, :]
            )
            effects = score[positive] - score[negative]
            layer_frame = local.loc[:, ["manipulation", "split", "cluster_id"]].copy()
            layer_frame["effect"] = effects
            for (manipulation, split), part in layer_frame.groupby(["manipulation", "split"]):
                rows.append(
                    {
                        "task": task,
                        "manipulation": str(manipulation),
                        "split": str(split),
                        "layer": layer,
                        "mean_functional_effect": float(part.effect.mean()),
                        "mean_absolute_effect": float(part.effect.abs().mean()),
                        "effect_sd": float(part.effect.std(ddof=1)),
                        "positive_fraction": float((part.effect > 0).mean()),
                        "pairs": len(part),
                        "readout_task": task,
                    }
                )
        if logger is not None:
            logger.note("interventions", f"{task}: projected {len(local)} matched pairs")
    if bandit_factorial is not None:
        rows.extend(
            _factorial_rows(
                inventory, readout_result, bandit_factorial, config, logger
            )
        )
    profiles = pd.DataFrame(rows)
    if profiles.empty:
        return profiles, pd.DataFrame(
            columns=["kind", "profile_a", "profile_b", "value"]
        )
    test = profiles[profiles.split == "test"]
    profile_values = {
        f"{task}::{manipulation}": part.sort_values("layer").mean_functional_effect.to_numpy()
        for (task, manipulation), part in test.groupby(["task", "manipulation"])
    }
    summary = profile_summary(
        profile_values, onset_fraction=float(config.get("onset_fraction", 0.5))
    )
    similarity_rows = []
    for left, right in combinations(sorted(profile_values), 2):
        left_values, right_values = profile_values[left], profile_values[right]
        correlation = (
            float("nan")
            if len(left_values) < 2
            or np.std(left_values) == 0
            or np.std(right_values) == 0
            else float(np.corrcoef(left_values, right_values)[0, 1])
        )
        similarity_rows.append(
            {
                "kind": "profile_correlation",
                "profile_a": left,
                "profile_b": right,
                "value": correlation,
            }
        )
    for name in sorted(profile_values):
        similarity_rows.extend(
            [
                {
                    "kind": "onset_layer",
                    "profile_a": name,
                    "profile_b": "",
                    "value": summary["onset_layers"][name],
                },
                {
                    "kind": "peak_layer",
                    "profile_a": name,
                    "profile_b": "",
                    "value": summary["peak_layers"][name],
                },
                {
                    "kind": "area_under_absolute_curve",
                    "profile_a": name,
                    "profile_b": "",
                    "value": summary["area_under_absolute_curve"][name],
                },
                {
                    "kind": "normalized_late_early_effect",
                    "profile_a": name,
                    "profile_b": "",
                    "value": summary["normalized_late_early_effect"][name],
                },
            ]
        )
    similarity_rows.append(
        {
            "kind": "mean_profile_correlation",
            "profile_a": "all",
            "profile_b": "all",
            "value": summary["mean_profile_correlation"],
        }
    )
    return profiles, pd.DataFrame(similarity_rows)
