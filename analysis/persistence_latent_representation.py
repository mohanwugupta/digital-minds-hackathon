"""Conditional all-layer decoding of commitment residuals beyond current choice."""

from __future__ import annotations

from analysis.analyze_shared_persistence_transfer import (
    _association_metrics,
    _cluster_bootstrap,
)
from experiments.shared_persistence_utils import validate_compatible_tasks
from interventions.ridge_probe import fit_ridge_targets, regression_metrics
from interventions.shared_ridge_probe import fit_balanced_shared_ridge


def residualize_commitment_targets(
    records: list[dict], latent_state: list[float], split_by_task: dict[str, dict]
) -> dict[str, float]:
    """Remove train-fit immediate-logit geometry separately within each task."""

    import torch

    if len(records) != len(latent_state):
        raise ValueError("latent-state vector does not align with behavioral records")
    by_task = {}
    for index, record in enumerate(records):
        by_task.setdefault(str(record["task"]), []).append(index)
    residual_by_state = {}
    for task, indices in by_task.items():
        train_episodes = set(str(value) for value in split_by_task[task]["train"])
        train = [index for index in indices if str(records[index]["episode_id"]) in train_episodes]
        if len(train) < 3:
            raise ValueError(f"task {task!r} has too few residualization training states")
        design = torch.tensor(
            [[1.0, float(records[index]["persistence_logit"])] for index in train],
            dtype=torch.float64,
        )
        target = torch.tensor([float(latent_state[index]) for index in train], dtype=torch.float64)
        coefficient = torch.linalg.pinv(design) @ target
        for index in indices:
            prediction = coefficient[0] + coefficient[1] * float(
                records[index]["persistence_logit"]
            )
            residual_by_state[str(records[index]["state_id"])] = float(
                latent_state[index] - prediction
            )
    return residual_by_state


def _layer_data(task, shards, layer, episode_ids, target_by_state):
    import torch

    states, targets, records = [], [], []
    for shard in shards:
        if str(shard["episode_id"]) not in episode_ids:
            continue
        activation = shard["activations"]
        if not 0 <= layer < int(activation.shape[1]):
            raise IndexError(f"layer {layer} is absent from task {task}")
        states.append(activation[:, layer, :].float())
        for record in shard["records"]:
            state_id = str(record["state_id"])
            if state_id not in target_by_state:
                raise ValueError(f"latent target is absent for {state_id}")
            targets.append(float(target_by_state[state_id]))
            records.append(record)
    if not states:
        raise ValueError(f"no {task} layer data for requested episodes")
    return {
        "states": torch.cat(states),
        "target": torch.tensor(targets, dtype=torch.float32),
        "records": records,
    }


def search_latent_representation(
    *,
    shards_by_task: dict[str, list[dict]],
    splits_by_task: dict[str, dict[str, list[str]]],
    residual_target_by_state: dict[str, float],
    alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Decode commitment residuals within task and under strict LOTO transfer."""

    validate_compatible_tasks(shards_by_task)
    tasks = sorted(shards_by_task)
    layer_count = int(shards_by_task[tasks[0]][0]["activations"].shape[1])
    within, within_probes = {}, {}
    for task in tasks:
        layer_rows = []
        for layer in range(layer_count):
            train = _layer_data(
                task,
                shards_by_task[task],
                layer,
                set(splits_by_task[task]["train"]),
                residual_target_by_state,
            )
            validation = _layer_data(
                task,
                shards_by_task[task],
                layer,
                set(splits_by_task[task]["validation"]),
                residual_target_by_state,
            )
            probes, fit = fit_ridge_targets(
                train["states"],
                {"commitment_residual": train["target"]},
                validation["states"],
                {"commitment_residual": validation["target"]},
                alphas=tuple(alphas),
            )
            probe = probes["commitment_residual"]
            layer_rows.append(
                {
                    "layer": layer,
                    "alpha": probe.alpha,
                    "validation": regression_metrics(
                        probe.predict(validation["states"]), validation["target"]
                    ),
                    "fit": fit,
                }
            )
            within_probes[(task, layer)] = probe
        selected = min(layer_rows, key=lambda row: row["validation"]["mse"])
        test = _layer_data(
            task,
            shards_by_task[task],
            selected["layer"],
            set(splits_by_task[task]["test"]),
            residual_target_by_state,
        )
        probe = within_probes[(task, selected["layer"])]
        prediction = probe.predict(test["states"])
        within[task] = {
            "selected_layer": selected["layer"],
            "selected_alpha": probe.alpha,
            "selection_split": "validation",
            "test": {
                **_association_metrics(prediction, test["target"]),
                "cluster_bootstrap": _cluster_bootstrap(
                    prediction,
                    test["target"],
                    [record.get("pair_id", record["episode_id"]) for record in test["records"]],
                    samples=bootstrap_samples,
                    seed=seed + selected["layer"],
                ),
            },
            "layers": layer_rows,
        }

    loto, loto_probes = [], {}
    for heldout_index, heldout in enumerate(tasks):
        source_tasks = [task for task in tasks if task != heldout]
        candidates = []
        for layer in range(layer_count):
            train_by_task, validation_by_task = {}, {}
            for task in source_tasks:
                train_by_task[task] = _layer_data(
                    task,
                    shards_by_task[task],
                    layer,
                    set(splits_by_task[task]["train"]),
                    residual_target_by_state,
                )
                validation_by_task[task] = _layer_data(
                    task,
                    shards_by_task[task],
                    layer,
                    set(splits_by_task[task]["validation"]),
                    residual_target_by_state,
                )
            probe, fit = fit_balanced_shared_ridge(
                train_by_task,
                validation_by_task,
                alphas=tuple(alphas),
            )
            macro_mse = sum(
                regression_metrics(
                    probe.predict(validation_by_task[task]["states"]),
                    validation_by_task[task]["target"],
                )["mse"]
                for task in source_tasks
            ) / len(source_tasks)
            candidates.append(
                {"layer": layer, "probe": probe, "fit": fit, "source_macro_mse": macro_mse}
            )
        selected = min(candidates, key=lambda row: row["source_macro_mse"])
        test = _layer_data(
            heldout,
            shards_by_task[heldout],
            selected["layer"],
            set(splits_by_task[heldout]["test"]),
            residual_target_by_state,
        )
        prediction = selected["probe"].predict(test["states"])
        loto.append(
            {
                "discovery_tasks": source_tasks,
                "heldout_task": heldout,
                "selected_layer": selected["layer"],
                "selected_alpha": selected["probe"].alpha,
                "heldout_parameters_fit": 0,
                "heldout_test": {
                    **_association_metrics(prediction, test["target"]),
                    "cluster_bootstrap": _cluster_bootstrap(
                        prediction,
                        test["target"],
                        [record.get("pair_id", record["episode_id"]) for record in test["records"]],
                        samples=bootstrap_samples,
                        seed=seed + 1000 + heldout_index,
                    ),
                },
            }
        )
        loto_probes[heldout] = selected["probe"]
    all_positive = all(
        row["heldout_test"]["correlation"] > 0
        and row["heldout_test"]["cluster_bootstrap"]["correlation"]["lower_95"] > 0
        for row in loto
    )
    return (
        {
            "analysis_role": "exploratory_discovery",
            "target": "latent_commitment_residual_after_train_fit_current_choice",
            "within_task": within,
            "leave_one_task_out": loto,
            "all_loto_clustered_intervals_positive": all_positive,
        },
        {"within_task": within_probes, "loto": loto_probes},
    )

