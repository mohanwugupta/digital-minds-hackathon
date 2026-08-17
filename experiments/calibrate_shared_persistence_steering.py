"""Calibrate the shared direction on held-out-task validation states only."""

import argparse
import json
import os

from analysis.shared_persistence_integrity import require_shared_clearance
from experiments.cross_task_utils import (
    layer_dataset,
    load_activation_shards,
    make_or_validate_split,
    probe_layer,
)
from experiments.runtime import run_metadata
from interventions.ridge_probe import load_ridge_probe
from interventions.ridge_steering import calibrate_ridge_magnitude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--representational-summary", required=True)
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import yaml

    representational = require_shared_clearance(args.representational_summary)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    shards = load_activation_shards(args.activation_dir)
    if {shard["task"] for shard in shards} != {"solvability"}:
        raise ValueError("shared steering calibration requires Solvability states")
    split = make_or_validate_split(
        shards, args.split, seed=int(config["split_seed"])
    )
    probe, payload = load_ridge_probe(args.probe)
    layer = probe_layer(payload, args.probe)
    metadata = payload.get("metadata", {})
    if metadata.get("heldout_task") != "solvability":
        raise ValueError("causal probe was not frozen for held-out Solvability")
    if int(metadata.get("heldout_task_parameters_fit", -1)) != 0:
        raise ValueError("shared probe metadata records held-out leakage")
    validation = layer_dataset(
        shards,
        layer,
        set(split["validation"]),
        target_key="persistence_logit",
    )
    specification = config["causal_transfer"]
    calibrated = calibrate_ridge_magnitude(
        probe,
        validation["states"],
        decoded_sd_candidates=tuple(
            float(value) for value in specification["decoded_sd_candidates"]
        ),
        max_relative_rms=float(specification["max_relative_rms"]),
    )
    output = {
        "status": "valid",
        "selection_data": "solvability validation states only",
        "test_states_inspected": False,
        "representational_gate": {
            "path": os.path.abspath(args.representational_summary),
            "classification": representational["classification"],
            "heldout_task_parameters_fit": representational[
                "heldout_task_parameters_fit"
            ],
        },
        "probe_path": os.path.abspath(args.probe),
        "probe_source_tasks": ["bandit", "foraging"],
        "causal_target_task": "solvability",
        "layer": layer,
        "split_path": os.path.abspath(args.split),
        "validation_episodes": len(split["validation"]),
        "validation_states": len(validation["records"]),
        "decoded_sd_candidates": specification["decoded_sd_candidates"],
        "max_relative_rms": specification["max_relative_rms"],
        "provenance": run_metadata(
            {
                "model": shards[0].get("model_id", config["model"]),
                "experiment": "shared_persistence_solvability_steering_calibration",
                "config": os.path.abspath(args.config),
            }
        ),
        **calibrated.to_dict(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
