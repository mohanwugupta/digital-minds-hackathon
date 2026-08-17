"""Train a validation-selected task-specific persistence ceiling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

from analysis.cross_task_integrity import require_behavioral_clearance
from experiments.runtime import run_metadata
from experiments.shared_persistence_utils import (
    activation_shape,
    load_task_shards,
    load_task_split,
    semantic_layer_dataset,
)
from experiments.train_value_probe import parse_layers
from interventions.ridge_probe import (
    fit_ridge_targets,
    load_ridge_probe,
    regression_metrics,
    save_ridge_probe,
)


def _write_predictions(path: str, data: dict, prediction) -> None:
    rows = [
        {
            "task": record["task"],
            "episode_id": record["episode_id"],
            "state_id": record["state_id"],
            "mapping_id": record["mapping_id"],
            "persistence_logit": record["persistence_logit"],
            "task_specific_prediction": float(prediction[index]),
        }
        for index, record in enumerate(data["records"])
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("foraging", "solvability"), required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--behavioral-gate",
        default="artifacts/cross_task/behavioral/behavioral_validation_summary.json",
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--alphas", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--defer-test",
        action="store_true",
        help="freeze validation-selected ceiling without evaluating its test split",
    )
    args = parser.parse_args()

    import yaml

    gate = require_behavioral_clearance(args.behavioral_gate)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    shards = load_task_shards(args.task, args.activation_dir)
    split = load_task_split(
        args.task, shards, args.split, seed=int(config["split_seed"])
    )
    split_sets = {name: set(values) for name, values in split.items()}
    layer_count, _ = activation_shape(shards)
    layers = parse_layers(args.layers, layer_count)
    alphas = tuple(float(value) for value in args.alphas.split(","))
    os.makedirs(args.output_dir, exist_ok=True)
    provenance = run_metadata(
        {
            "experiment": "task_specific_persistence_ceiling",
            "task": args.task,
            "activation_dir": os.path.abspath(args.activation_dir),
            "split": os.path.abspath(args.split),
        }
    )

    summaries, started = [], time.perf_counter()
    for layer in layers:
        train = semantic_layer_dataset(args.task, shards, layer, split_sets["train"])
        validation = semantic_layer_dataset(
            args.task, shards, layer, split_sets["validation"]
        )
        probes, fit = fit_ridge_targets(
            train["states"],
            {"persistence": train["target"]},
            validation["states"],
            {"persistence": validation["target"]},
            alphas=alphas,
            device=args.device,
        )
        probe = probes["persistence"]
        summary = {
            "layer": layer,
            "alpha": probe.alpha,
            "validation": regression_metrics(
                probe.predict(validation["states"]), validation["target"]
            ),
            "alpha_validation": fit["validation_candidates"]["persistence"],
        }
        summaries.append(summary)
        save_ridge_probe(
            os.path.join(args.output_dir, f"layer_{layer:02d}_persistence.pt"),
            probe,
            {
                "layer": layer,
                "task": args.task,
                "fit": fit,
                "metrics": summary,
                "split_path": os.path.abspath(args.split),
                "provenance": provenance,
            },
        )
        print(
            f"{args.task} layer {layer}: validation R2="
            f"{summary['validation']['r_squared']:.4f}",
            flush=True,
        )

    best = max(summaries, key=lambda row: row["validation"]["r_squared"])
    source = os.path.join(
        args.output_dir, f"layer_{best['layer']:02d}_persistence.pt"
    )
    probe, payload = load_ridge_probe(source)
    selected_test = None
    if not args.defer_test:
        test = semantic_layer_dataset(
            args.task, shards, best["layer"], split_sets["test"]
        )
        test_prediction = probe.predict(test["states"])
        selected_test = regression_metrics(test_prediction, test["target"])
    save_ridge_probe(
        os.path.join(args.output_dir, "frozen_best_persistence.pt"),
        probe,
        {
            **payload["metadata"],
            "selected_layer": best["layer"],
            "selection_metric": "validation_r_squared",
            "selected_test": selected_test,
            "test_evaluation_deferred": args.defer_test,
        },
    )
    if not args.defer_test:
        _write_predictions(
            os.path.join(args.output_dir, "test_predictions.csv"),
            test,
            test_prediction,
        )
    result = {
        "task": args.task,
        "target": "semantic_persistence_logit",
        "best_layer": best["layer"],
        "selection_metric": "validation_r_squared",
        "selected_test": selected_test,
        "test_evaluation_deferred": args.defer_test,
        "layers": summaries,
        "split_episode_counts": {name: len(values) for name, values in split.items()},
        "runtime_seconds": time.perf_counter() - started,
        "development_behavioral_gate": {
            "path": os.path.abspath(args.behavioral_gate),
            "passed": gate["passed"],
            "test_episodes_inspected": gate["test_episodes_inspected"],
        },
        "provenance": provenance,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
