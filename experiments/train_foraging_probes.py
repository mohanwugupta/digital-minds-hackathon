"""Train the validation-selected within-foraging persistence ceiling."""

import argparse
import csv
import json
import os
import time

from experiments.cross_task_utils import (
    layer_dataset,
    load_activation_shards,
    make_or_validate_split,
)
from experiments.train_value_probe import parse_layers
from interventions.ridge_probe import (
    fit_ridge_targets,
    load_ridge_probe,
    regression_metrics,
    save_ridge_probe,
)


def _write_test_rows(path: str, records: list[dict], prediction) -> None:
    rows = [
        {
            "episode_id": record["episode_id"],
            "pair_id": record["pair_id"],
            "state_id": record["state_id"],
            "mapping_id": record["mapping_id"],
            "round": record["round"],
            "persistence_logit": record["persistence_logit"],
            "foraging_probe_prediction": float(prediction[index]),
        }
        for index, record in enumerate(records)
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation-dir", default="artifacts/cross_task/foraging_activation_bank"
    )
    parser.add_argument(
        "--split", default="artifacts/cross_task/foraging_episode_split.json"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/cross_task/foraging_probes"
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--alphas", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--split-seed", type=int, default=72026)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    shards = load_activation_shards(args.activation_dir)
    if {shard["task"] for shard in shards} != {"foraging"}:
        raise ValueError("the foraging ceiling requires a foraging activation bank")
    split = make_or_validate_split(shards, args.split, seed=args.split_seed)
    split_sets = {name: set(episodes) for name, episodes in split.items()}
    if not split_sets["train"] or not split_sets["validation"] or not split_sets["test"]:
        raise ValueError("foraging ceiling requires nonempty train/validation/test splits")
    layer_count = int(shards[0]["activations"].shape[1])
    layers = parse_layers(args.layers, layer_count)
    alphas = tuple(float(value) for value in args.alphas.split(","))
    os.makedirs(args.output_dir, exist_ok=True)

    summaries, started = [], time.perf_counter()
    for layer in layers:
        train = layer_dataset(
            shards, layer, split_sets["train"], target_key="persistence_logit"
        )
        validation = layer_dataset(
            shards, layer, split_sets["validation"], target_key="persistence_logit"
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
            "solver": fit["solver"],
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
                "task": "foraging",
                "fit": fit,
                "metrics": summary,
                "split_path": os.path.abspath(args.split),
            },
        )
        print(
            f"foraging ridge layer {layer}: validation R2="
            f"{summary['validation']['r_squared']:.4f}",
            flush=True,
        )

    best = max(summaries, key=lambda row: row["validation"]["r_squared"])
    source = os.path.join(
        args.output_dir, f"layer_{best['layer']:02d}_persistence.pt"
    )
    probe, payload = load_ridge_probe(source)
    test = layer_dataset(
        shards, best["layer"], split_sets["test"], target_key="persistence_logit"
    )
    selected_test = regression_metrics(probe.predict(test["states"]), test["target"])
    save_ridge_probe(
        os.path.join(args.output_dir, "frozen_best_persistence.pt"),
        probe,
        {
            **payload["metadata"],
            "selected_layer": best["layer"],
            "selection_metric": "validation_r_squared",
            "selected_test": selected_test,
        },
    )
    _write_test_rows(
        os.path.join(args.output_dir, "test_predictions.csv"),
        test["records"],
        probe.predict(test["states"]),
    )
    result = {
        "task": "foraging",
        "target": "persistence_logit",
        "split_path": os.path.abspath(args.split),
        "split_episode_counts": {name: len(values) for name, values in split.items()},
        "best_layer": best["layer"],
        "selection_metric": "validation_r_squared",
        "selected_test": selected_test,
        "layers": summaries,
        "runtime_seconds": time.perf_counter() - started,
    }
    with open(
        os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
