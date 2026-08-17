"""Exercise the shared-probe machinery on tiny real-model cross-task banks."""

import argparse
import json
import os

from analysis.shared_persistence_integrity import validate_loto_folds
from experiments.cross_task_utils import (
    layer_dataset as cross_task_layer_dataset,
    load_activation_shards,
    make_or_validate_split,
)
from experiments.shared_persistence_utils import (
    load_task_shards,
    load_task_split,
    semantic_layer_dataset,
    validate_compatible_tasks,
)
from interventions.ridge_probe import load_ridge_probe, regression_metrics, save_ridge_probe
from interventions.ridge_steering import matched_sign_random_directions
from interventions.shared_ridge_probe import fit_balanced_shared_ridge


FOLDS = (
    {"discovery": ("bandit", "foraging"), "heldout": "solvability"},
    {"discovery": ("bandit", "solvability"), "heldout": "foraging"},
    {"discovery": ("foraging", "solvability"), "heldout": "bandit"},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandit-bank", default="artifacts/activation_bank")
    parser.add_argument("--bandit-split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--foraging-bank", required=True)
    parser.add_argument("--solvability-bank", required=True)
    parser.add_argument("--control-bank", required=True)
    parser.add_argument("--terminality-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=72026)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    task_paths = {
        "bandit": (args.bandit_bank, args.bandit_split),
        "foraging": (
            args.foraging_bank,
            os.path.join(args.output_dir, "foraging_split.json"),
        ),
        "solvability": (
            args.solvability_bank,
            os.path.join(args.output_dir, "solvability_split.json"),
        ),
    }
    shards = {
        task: load_task_shards(task, bank)
        for task, (bank, _split) in task_paths.items()
    }
    layer_count, hidden_width = validate_compatible_tasks(shards)
    if not 0 <= args.layer < layer_count:
        raise ValueError("smoke layer is outside the activation banks")
    splits = {
        task: load_task_split(task, shards[task], split, seed=args.seed)
        for task, (_bank, split) in task_paths.items()
    }
    if any(not split["validation"] or not split["test"] for split in splits.values()):
        raise ValueError("smoke banks need enough counterbalanced pairs for all splits")
    validate_loto_folds(("bandit", "foraging", "solvability"), list(FOLDS))
    os.makedirs(args.output_dir, exist_ok=True)

    fold_results = []
    primary_probe = None
    for fold in FOLDS:
        discovery = tuple(fold["discovery"])
        train = {
            task: semantic_layer_dataset(
                task, shards[task], args.layer, set(splits[task]["train"])
            )
            for task in discovery
        }
        validation = {
            task: semantic_layer_dataset(
                task, shards[task], args.layer, set(splits[task]["validation"])
            )
            for task in discovery
        }
        probe, fit = fit_balanced_shared_ridge(
            train, validation, alphas=(0.01, 1.0), device=args.device
        )
        heldout = str(fold["heldout"])
        test = semantic_layer_dataset(
            heldout,
            shards[heldout],
            args.layer,
            set(splits[heldout]["test"]),
        )
        metrics = regression_metrics(probe.predict(test["states"]), test["target"])
        artifact = os.path.join(args.output_dir, f"heldout_{heldout}.pt")
        save_ridge_probe(
            artifact,
            probe,
            {
                "layer": args.layer,
                "heldout_task": heldout,
                "heldout_task_parameters_fit": 0,
                "smoke_only": True,
            },
        )
        restored, payload = load_ridge_probe(artifact)
        if not torch.equal(restored.weight, probe.weight):
            raise RuntimeError("shared smoke probe changed during save/load")
        fold_results.append(
            {
                "discovery": list(discovery),
                "heldout": heldout,
                "selected_alpha": probe.alpha,
                "macro_validation_mse": fit["selected"]["macro_mse"],
                "heldout_correlation_not_a_scientific_result": metrics["correlation"],
                "artifact_layer": payload["metadata"]["layer"],
            }
        )
        if heldout == "solvability":
            primary_probe = probe
    assert primary_probe is not None

    control_shards = load_activation_shards(args.control_bank)
    control_split = make_or_validate_split(
        control_shards,
        os.path.join(args.output_dir, "control_split.json"),
        seed=args.seed,
    )
    control = cross_task_layer_dataset(
        control_shards,
        args.layer,
        set(control_split["test"]),
        target_key="choice_logit",
    )
    control_metrics = regression_metrics(
        primary_probe.predict(control["states"]), control["target"]
    )
    terminality_shards = load_activation_shards(args.terminality_bank)
    terminality_split = make_or_validate_split(
        terminality_shards,
        os.path.join(args.output_dir, "terminality_split.json"),
        seed=args.seed,
    )
    terminality = cross_task_layer_dataset(
        terminality_shards,
        args.layer,
        set(terminality_split["test"]),
        target_key="terminality_logit",
    )
    terminality_metrics = regression_metrics(
        primary_probe.predict(terminality["states"]), terminality["target"]
    )
    random_controls = matched_sign_random_directions(
        primary_probe.raw_activation_direction(), n_directions=3, seed=args.seed
    )
    result = {
        "passed": True,
        "analysis_role": "pipeline_smoke_only_not_a_scientific_result",
        "layer": args.layer,
        "layer_count": layer_count,
        "hidden_width": hidden_width,
        "folds": fold_results,
        "matched_random_directions_constructed": len(random_controls),
        "binary_control_states": len(control["records"]),
        "binary_control_correlation_not_a_scientific_result": control_metrics[
            "correlation"
        ],
        "terminality_control_states": len(terminality["records"]),
        "terminality_control_correlation_not_a_scientific_result": (
            terminality_metrics["correlation"]
        ),
    }
    with open(
        os.path.join(args.output_dir, "shared_pipeline_smoke_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
