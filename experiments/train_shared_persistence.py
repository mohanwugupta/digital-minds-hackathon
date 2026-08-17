"""Discover frozen task-balanced persistence directions without held-out leakage."""

from __future__ import annotations

import argparse
import json
import os
import time

from analysis.cross_task_integrity import require_behavioral_clearance
from analysis.shared_persistence_integrity import (
    source_task_gate,
    validate_discovery_plan,
    validate_loto_folds,
)
from experiments.runtime import run_metadata
from experiments.shared_persistence_utils import (
    load_task_shards,
    load_task_split,
    semantic_layer_dataset,
    validate_compatible_tasks,
)
from experiments.train_value_probe import parse_layers
from interventions.ridge_probe import save_ridge_probe
from interventions.ridge_steering import matched_sign_random_directions
from interventions.shared_ridge_probe import fit_balanced_shared_ridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandit-bank", default="artifacts/activation_bank")
    parser.add_argument("--bandit-split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--foraging-bank", required=True)
    parser.add_argument("--foraging-split", required=True)
    parser.add_argument("--solvability-bank", required=True)
    parser.add_argument("--solvability-split", required=True)
    parser.add_argument("--behavioral-gate", required=True)
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    import yaml

    gate = require_behavioral_clearance(args.behavioral_gate)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    specification = config["shared_persistence_transfer"]
    if specification["task_weighting"] != "equal_macro_weight":
        raise ValueError("shared implementation requires equal macro task weighting")
    if specification["target"] != "per_task_train_standardized_semantic_persistence_logit":
        raise ValueError("shared implementation requires train-standardized semantic targets")
    validate_loto_folds(
        ("bandit", "foraging", "solvability"),
        specification["leave_one_task_out_folds"],
    )
    paths = {
        "bandit": (args.bandit_bank, args.bandit_split),
        "foraging": (args.foraging_bank, args.foraging_split),
        "solvability": (args.solvability_bank, args.solvability_split),
    }
    shards = {
        task: load_task_shards(task, activation_path)
        for task, (activation_path, _split_path) in paths.items()
    }
    layer_count, hidden_width = validate_compatible_tasks(shards)
    splits = {
        task: load_task_split(
            task, shards[task], split_path, seed=int(config["split_seed"])
        )
        for task, (_activation_path, split_path) in paths.items()
    }
    configured = specification["layers"]
    configured_layers = args.layers or (
        ",".join(str(layer) for layer in configured)
        if isinstance(configured, list)
        else str(configured)
    )
    layers = parse_layers(configured_layers, layer_count)
    alphas = tuple(float(value) for value in specification["alphas"])
    os.makedirs(args.output_dir, exist_ok=True)
    provenance = run_metadata(
        {
            "experiment": "shared_semantic_persistence_discovery",
            "config": os.path.abspath(args.config),
            "hidden_width": hidden_width,
        }
    )

    fold_summaries, started = [], time.perf_counter()
    primary_heldout = str(specification["primary_heldout_task"])
    primary_probe = primary_metadata = None
    for fold_index, fold in enumerate(specification["leave_one_task_out_folds"]):
        discovery = tuple(str(task) for task in fold["discovery"])
        heldout = str(fold["heldout"])
        plan = validate_discovery_plan(
            discovery_tasks=discovery,
            heldout_task=heldout,
            layer_selection_tasks=discovery,
        )
        if heldout == primary_heldout and set(discovery) != set(
            specification["primary_discovery_tasks"]
        ):
            raise ValueError(
                "primary held-out fold does not match primary discovery tasks"
            )
        per_layer, selected = [], None
        for layer in layers:
            train_by_task = {
                task: semantic_layer_dataset(
                    task, shards[task], layer, set(splits[task]["train"])
                )
                for task in discovery
            }
            validation_by_task = {
                task: semantic_layer_dataset(
                    task, shards[task], layer, set(splits[task]["validation"])
                )
                for task in discovery
            }
            probe, fit = fit_balanced_shared_ridge(
                train_by_task,
                validation_by_task,
                alphas=alphas,
                device=args.device,
            )
            random_directions = matched_sign_random_directions(
                probe.raw_activation_direction(),
                n_directions=int(
                    specification["source_task_matched_random_directions"]
                ),
                seed=int(config["analysis_seed"]) + 10_000 * fold_index + layer,
            )
            random_95 = {}
            for task in discovery:
                states = validation_by_task[task]["states"].float()
                target_mean = fit["target_moments"][task]["mean"]
                target_std = fit["target_moments"][task]["std"]
                target = (
                    validation_by_task[task]["target"].float() - target_mean
                ) / target_std
                direction_matrix = torch.stack(
                    [direction.float().cpu() for direction in random_directions],
                    dim=1,
                )
                prediction = (states - probe.state_mean) @ direction_matrix
                centered_prediction = prediction - prediction.mean(dim=0)
                centered_target = target - target.mean()
                numerator = (centered_prediction * centered_target[:, None]).mean(
                    dim=0
                )
                denominator = centered_prediction.std(
                    dim=0, unbiased=False
                ) * centered_target.std(unbiased=False)
                correlations = (
                    numerator / denominator.clamp_min(1e-12)
                ).tolist()
                random_95[task] = sorted(correlations)[
                    max(0, int(0.95 * len(correlations) + 0.999999) - 1)
                ]
            source_gate = source_task_gate(
                fit["selected"]["per_task"], random_95
            )
            row = {
                "layer": layer,
                "alpha": probe.alpha,
                "macro_validation_mse": fit["selected"]["macro_mse"],
                "macro_validation_correlation": fit["selected"]["macro_correlation"],
                "validation_by_task": fit["selected"]["per_task"],
                "target_moments": fit["target_moments"],
                "train_states": fit["train_states"],
                "validation_states": fit["validation_states"],
                "solver": fit["solver"],
                "source_task_gate": source_gate,
            }
            per_layer.append(row)
            if source_gate["passed"] and (
                selected is None
                or row["macro_validation_mse"]
                < selected["row"]["macro_validation_mse"]
            ):
                selected = {"row": row, "probe": probe, "fit": fit}
            print(
                f"LOTO held out {heldout}, layer {layer}: macro validation "
                f"MSE={row['macro_validation_mse']:.4f}, "
                f"r={row['macro_validation_correlation']:.4f}",
                flush=True,
            )
        if selected is None:
            raise RuntimeError(
                f"no layer passed the per-source-task shared-direction gate for {discovery}"
            )
        metadata = {
            **plan,
            "analysis_role": (
                "confirmatory_primary_discovery"
                if heldout == primary_heldout
                else "secondary_leave_one_task_out_robustness_discovery"
            ),
            "selected_layer": selected["row"]["layer"],
            "selected_alpha": selected["probe"].alpha,
            "selection_metric": "equal_task_macro_validation_mse",
            "target": specification["target"],
            "fit": selected["fit"],
            "layers": per_layer,
            "split_paths": {task: os.path.abspath(paths[task][1]) for task in discovery},
            "provenance": provenance,
        }
        artifact_path = os.path.join(args.output_dir, f"heldout_{heldout}.pt")
        save_ridge_probe(artifact_path, selected["probe"], metadata)
        fold_summary = {
            "discovery_tasks": list(discovery),
            "heldout_task": heldout,
            "heldout_task_parameters_fit": 0,
            "analysis_role": metadata["analysis_role"],
            "selected_layer": selected["row"]["layer"],
            "selected_alpha": selected["probe"].alpha,
            "selection_metric": "equal_task_macro_validation_mse",
            "artifact": os.path.abspath(artifact_path),
            "layers": per_layer,
        }
        fold_summaries.append(fold_summary)
        if heldout == primary_heldout:
            primary_probe, primary_metadata = selected["probe"], metadata

    if primary_probe is None:
        raise ValueError("configured LOTO folds omit the primary held-out task")
    save_ridge_probe(
        os.path.join(args.output_dir, "frozen_primary.pt"),
        primary_probe,
        {**primary_metadata, "artifact_role": "primary_bandit_plus_foraging_to_solvability"},
    )
    result = {
        "analysis_role": "discovery_only_no_heldout_test_evaluation",
        "primary_discovery_tasks": specification["primary_discovery_tasks"],
        "primary_heldout_task": primary_heldout,
        "heldout_task_parameters_fit": 0,
        "task_weighting": specification["task_weighting"],
        "target": specification["target"],
        "folds": fold_summaries,
        "runtime_seconds": time.perf_counter() - started,
        "development_behavioral_gate": {
            "passed": gate["passed"],
            "test_episodes_inspected": gate["test_episodes_inspected"],
        },
        "provenance": provenance,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
