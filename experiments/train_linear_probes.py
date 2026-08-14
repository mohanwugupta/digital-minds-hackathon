"""Train ridge-linear future-return and direct-persistence probes at every layer."""

import argparse
import csv
import json
import math
import os
import time

from experiments.train_value_probe import load_shards, parse_layers
from interventions.ridge_probe import (
    fit_ridge_targets,
    load_ridge_probe,
    regression_metrics,
    save_ridge_probe,
)


TARGET_KEYS = {
    "future_return": "future_cumulative_return",
    "persistence": "persistence_logit",
}


def layer_dataset(shards: list, layer: int, episode_ids: set[str]) -> dict:
    import torch

    states, targets, records = [], {key: [] for key in TARGET_KEYS}, []
    for shard in shards:
        if shard["episode_id"] not in episode_ids:
            continue
        states.append(shard["activations"][:, layer, :].float())
        for record in shard["records"]:
            for target, source in TARGET_KEYS.items():
                targets[target].append(float(record[source]))
        records.extend(shard["records"])
    if not states:
        raise ValueError("split contains no states")
    return {
        "states": torch.cat(states),
        "targets": {
            key: torch.tensor(values, dtype=torch.float32)
            for key, values in targets.items()
        },
        "records": records,
    }


def _loss_streak(record: dict) -> int:
    rewards = record["reward_history"]
    if isinstance(rewards, str):
        rewards = json.loads(rewards)
    streak = 0
    for reward in reversed(rewards):
        if float(reward) != -2:
            break
        streak += 1
    return streak


def _direction_overlap(left, right, top_fraction: float = 0.01) -> dict:
    import torch

    left_direction = left.raw_activation_direction().float()
    right_direction = right.raw_activation_direction().float()
    cosine = float(
        torch.dot(left_direction, right_direction)
        / (left_direction.norm() * right_direction.norm()).clamp_min(1e-12)
    )
    count = max(1, math.ceil(left_direction.numel() * top_fraction))
    left_top = set(torch.topk(left_direction.abs(), count).indices.tolist())
    right_top = set(torch.topk(right_direction.abs(), count).indices.tolist())
    intersection = len(left_top & right_top)
    union = len(left_top | right_top)
    return {
        "cosine_similarity": cosine,
        "absolute_cosine_similarity": abs(cosine),
        "top_fraction": top_fraction,
        "top_dimension_count": count,
        "top_dimension_intersection": intersection,
        "top_dimension_jaccard": intersection / union if union else 0.0,
    }


def _write_test_rows(path: str, records: list[dict], predictions: dict) -> None:
    rows = []
    for index, record in enumerate(records):
        previous = record.get("previous_outcome")
        if previous in (None, ""):
            previous = None
        else:
            previous = float(previous)
        rows.append(
            {
                "episode_id": record["episode_id"],
                "state_id": record["state_id"],
                "round": int(record["round"]),
                "previous_outcome": previous,
                "loss_streak": _loss_streak(record),
                "cumulative_score": float(record["cumulative_score"]),
                "future_return": float(record["future_cumulative_return"]),
                "persistence_logit": float(record["persistence_logit"]),
                "ridge_future_return": float(predictions["future_return"][index]),
                "ridge_persistence": float(predictions["persistence"][index]),
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", default="artifacts/activation_bank")
    parser.add_argument("--split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--output-dir", default="artifacts/linear_probes")
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--alphas", default="0.0001,0.001,0.01,0.1,1,10,100"
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    alphas = tuple(float(value) for value in args.alphas.split(","))
    shards = load_shards(args.activation_dir)
    with open(args.split, encoding="utf-8") as handle:
        split = json.load(handle)
    split_sets = {name: set(values) for name, values in split.items()}
    shard_ids = {shard["episode_id"] for shard in shards}
    if set().union(*split_sets.values()) != shard_ids:
        raise ValueError("activation shards do not exactly match the frozen split")
    layer_count = int(shards[0]["activations"].shape[1])
    layers = parse_layers(args.layers, layer_count)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "episode_split.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(split, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summaries, total_started = [], time.perf_counter()
    for layer in layers:
        started = time.perf_counter()
        train = layer_dataset(shards, layer, split_sets["train"])
        validation = layer_dataset(shards, layer, split_sets["validation"])
        test = layer_dataset(shards, layer, split_sets["test"])
        probes, fit = fit_ridge_targets(
            train["states"],
            train["targets"],
            validation["states"],
            validation["targets"],
            alphas=alphas,
            device=args.device,
        )
        target_summaries = {}
        for target, probe in probes.items():
            target_summaries[target] = {
                "alpha": probe.alpha,
                "validation": regression_metrics(
                    probe.predict(validation["states"]), validation["targets"][target]
                ),
                "test": regression_metrics(
                    probe.predict(test["states"]), test["targets"][target]
                ),
                "alpha_validation": fit["validation_candidates"][target],
            }
            save_ridge_probe(
                os.path.join(args.output_dir, f"layer_{layer:02d}_{target}.pt"),
                probe,
                {"layer": layer, "fit": fit, "metrics": target_summaries[target]},
            )
        overlap = _direction_overlap(
            probes["future_return"], probes["persistence"]
        )
        summaries.append(
            {
                "layer": layer,
                "solver": fit["solver"],
                "targets": target_summaries,
                "return_persistence_overlap": overlap,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"ridge layer {layer}: return val/test R2="
            f"{target_summaries['future_return']['validation']['r_squared']:.4f}/"
            f"{target_summaries['future_return']['test']['r_squared']:.4f}; "
            f"persistence val/test R2="
            f"{target_summaries['persistence']['validation']['r_squared']:.4f}/"
            f"{target_summaries['persistence']['test']['r_squared']:.4f}; "
            f"runtime={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    best_layers = {
        target: max(
            summaries,
            key=lambda row: row["targets"][target]["validation"]["r_squared"],
        )["layer"]
        for target in TARGET_KEYS
    }
    best_probes = {}
    for target, layer in best_layers.items():
        source = os.path.join(args.output_dir, f"layer_{layer:02d}_{target}.pt")
        probe, artifact = load_ridge_probe(source)
        save_ridge_probe(
            os.path.join(args.output_dir, f"frozen_best_{target}.pt"),
            probe,
            {
                **artifact["metadata"],
                "selection_metric": "validation_r_squared",
                "selected_layer": layer,
            },
        )
        best_probes[target] = (probe, layer)

    # Assemble predictions from each target's independently selected layer.
    test_records = None
    predictions = {}
    for target, (probe, layer) in best_probes.items():
        data = layer_dataset(shards, layer, split_sets["test"])
        if test_records is None:
            test_records = data["records"]
        elif [row["state_id"] for row in test_records] != [
            row["state_id"] for row in data["records"]
        ]:
            raise ValueError("test record order differs across layers")
        predictions[target] = probe.predict(data["states"])
    _write_test_rows(
        os.path.join(args.output_dir, "test_predictions.csv"),
        test_records,
        predictions,
    )
    payload = {
        "probe_type": "ridge_linear",
        "targets": TARGET_KEYS,
        "best_layers": best_layers,
        "episode_split": {
            "source": args.split,
            **{f"{name}_episodes": len(values) for name, values in split.items()},
        },
        "layers": summaries,
        "runtime_seconds": time.perf_counter() - total_started,
    }
    with open(
        os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        f"ridge probe runtime: {payload['runtime_seconds']:.1f}s; "
        f"best layers={best_layers}",
        flush=True,
    )


if __name__ == "__main__":
    main()
