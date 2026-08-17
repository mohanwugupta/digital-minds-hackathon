"""Calibrate the frozen bandit direction on foraging validation states only."""

import argparse
import json
import os

from experiments.cross_task_utils import (
    layer_dataset,
    load_activation_shards,
    make_or_validate_split,
    probe_layer,
)
from interventions.ridge_probe import load_ridge_probe
from interventions.ridge_steering import calibrate_ridge_magnitude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation-dir", default="artifacts/cross_task/foraging_activation_bank"
    )
    parser.add_argument(
        "--split", default="artifacts/cross_task/foraging_episode_split.json"
    )
    parser.add_argument(
        "--probe", default="artifacts/linear_probes/frozen_best_persistence.pt"
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument(
        "--output", default="artifacts/cross_task/causal/calibration.json"
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    shards = load_activation_shards(args.activation_dir)
    if {shard["task"] for shard in shards} != {"foraging"}:
        raise ValueError("cross-task calibration requires foraging states")
    split = make_or_validate_split(
        shards, args.split, seed=int(config["split_seed"])
    )
    probe, payload = load_ridge_probe(args.probe)
    layer = probe_layer(payload, args.probe)
    validation = layer_dataset(
        shards,
        layer,
        set(split["validation"]),
        target_key="persistence_logit",
    )
    specification = config["causal_transfer"]
    result = calibrate_ridge_magnitude(
        probe,
        validation["states"],
        decoded_sd_candidates=tuple(
            float(value) for value in specification["decoded_sd_candidates"]
        ),
        max_relative_rms=float(specification["max_relative_rms"]),
    )
    output = {
        "status": "valid",
        "selection_data": "foraging validation states only",
        "test_states_inspected": False,
        "probe_path": os.path.abspath(args.probe),
        "probe_source_task": "bandit",
        "layer": layer,
        "split_path": os.path.abspath(args.split),
        "validation_episodes": len(split["validation"]),
        "validation_states": len(validation["records"]),
        "decoded_sd_candidates": specification["decoded_sd_candidates"],
        "max_relative_rms": specification["max_relative_rms"],
        **result.to_dict(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
