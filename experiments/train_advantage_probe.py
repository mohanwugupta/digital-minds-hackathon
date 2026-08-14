"""Train ridge-linear continuation-advantage probes from rollout targets."""

import argparse
import csv
import glob
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


def load_targets(pattern: str) -> dict[str, dict]:
    rows = {}
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no advantage target CSVs match {pattern}")
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                state_id = row["state_id"]
                if state_id in rows:
                    raise ValueError(f"duplicate advantage target: {state_id}")
                rows[state_id] = row
    return rows


def layer_dataset(shards: list, layer: int, episode_ids: set, targets: dict) -> dict:
    import torch

    states, values, records = [], [], []
    for shard in shards:
        if shard["episode_id"] not in episode_ids:
            continue
        selected_indices = [
            index
            for index, record in enumerate(shard["records"])
            if record["state_id"] in targets
        ]
        if not selected_indices:
            continue
        states.append(shard["activations"][selected_indices, layer, :].float())
        for index in selected_indices:
            record = shard["records"][index]
            values.append(float(targets[record["state_id"]]["continuation_advantage"]))
            records.append(record)
    if not states:
        raise ValueError("split has no labeled advantage states")
    return {
        "states": torch.cat(states),
        "targets": torch.tensor(values, dtype=torch.float32),
        "records": records,
    }


def direction_overlap(left, right, top_fraction: float = 0.01) -> dict:
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
    return {
        "cosine_similarity": cosine,
        "absolute_cosine_similarity": abs(cosine),
        "top_dimension_count": count,
        "top_dimension_intersection": intersection,
        "top_dimension_jaccard": intersection / len(left_top | right_top),
    }


def loss_streak(record: dict) -> int:
    rewards = record["reward_history"]
    if isinstance(rewards, str):
        rewards = json.loads(rewards)
    streak = 0
    for reward in reversed(rewards):
        if float(reward) != -2:
            break
        streak += 1
    return streak


def write_predictions(path: str, records: list, targets: dict, prediction) -> None:
    rows = []
    for index, record in enumerate(records):
        target = targets[record["state_id"]]
        previous = record.get("previous_outcome")
        rows.append(
            {
                "episode_id": record["episode_id"],
                "state_id": record["state_id"],
                "round": int(record["round"]),
                "previous_outcome": "" if previous in (None, "") else float(previous),
                "loss_streak": loss_streak(record),
                "cumulative_score": float(record["cumulative_score"]),
                "persistence_logit": float(record["persistence_logit"]),
                "q_A": float(target["q_A"]),
                "q_B": float(target["q_B"]),
                "continuation_advantage": float(target["continuation_advantage"]),
                "advantage_standard_error": float(
                    target["q_A_standard_error"]
                    if target["best_forced_action"] == "A"
                    else target["q_B_standard_error"]
                ),
                "ridge_advantage": float(prediction[index]),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", default="artifacts/activation_bank")
    parser.add_argument("--targets", default="artifacts/advantage_targets/targets*.csv")
    parser.add_argument("--split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--linear-dir", default="artifacts/linear_probes")
    parser.add_argument("--output-dir", default="artifacts/advantage_probes")
    parser.add_argument("--minimum-states-per-split", type=int, default=128)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--alphas", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    targets = load_targets(args.targets)
    shards = load_shards(args.activation_dir)
    with open(args.split, encoding="utf-8") as handle:
        split = json.load(handle)
    split_sets = {name: set(values) for name, values in split.items()}
    target_ids_by_split = {
        name: {
            state_id
            for state_id, row in targets.items()
            if row["episode_id"] in episode_ids
        }
        for name, episode_ids in split_sets.items()
    }
    if any(not values for values in target_ids_by_split.values()):
        raise ValueError("advantage targets must include train, validation, and test episodes")
    incomplete = {
        name: len(values)
        for name, values in target_ids_by_split.items()
        if len(values) < args.minimum_states_per_split
    }
    if incomplete:
        raise ValueError(
            "advantage target collection is incomplete for the requested minimum: "
            f"{incomplete}"
        )
    layer_count = int(shards[0]["activations"].shape[1])
    layers = parse_layers(args.layers, layer_count)
    alphas = tuple(float(value) for value in args.alphas.split(","))
    os.makedirs(args.output_dir, exist_ok=True)
    summaries, started = [], time.perf_counter()

    for layer in layers:
        data = {
            name: layer_dataset(shards, layer, episode_ids, targets)
            for name, episode_ids in split_sets.items()
        }
        probes, fit = fit_ridge_targets(
            data["train"]["states"],
            {"continuation_advantage": data["train"]["targets"]},
            data["validation"]["states"],
            {"continuation_advantage": data["validation"]["targets"]},
            alphas=alphas,
            device=args.device,
        )
        probe = probes["continuation_advantage"]
        validation = regression_metrics(
            probe.predict(data["validation"]["states"]),
            data["validation"]["targets"],
        )
        test = regression_metrics(
            probe.predict(data["test"]["states"]), data["test"]["targets"]
        )
        persistence_path = os.path.join(
            args.linear_dir, f"layer_{layer:02d}_persistence.pt"
        )
        overlap = None
        if os.path.exists(persistence_path):
            persistence_probe, _ = load_ridge_probe(persistence_path)
            overlap = direction_overlap(probe, persistence_probe)
        summary = {
            "layer": layer,
            "alpha": probe.alpha,
            "solver": fit["solver"],
            "validation": validation,
            "test": test,
            "alpha_validation": fit["validation_candidates"]["continuation_advantage"],
            "advantage_persistence_overlap": overlap,
        }
        summaries.append(summary)
        save_ridge_probe(
            os.path.join(args.output_dir, f"layer_{layer:02d}_advantage.pt"),
            probe,
            {"layer": layer, "metrics": summary},
        )
        print(
            f"advantage ridge layer {layer}: validation/test R2="
            f"{validation['r_squared']:.4f}/{test['r_squared']:.4f}",
            flush=True,
        )

    best = max(summaries, key=lambda row: row["validation"]["r_squared"])
    best_probe, artifact = load_ridge_probe(
        os.path.join(args.output_dir, f"layer_{best['layer']:02d}_advantage.pt")
    )
    save_ridge_probe(
        os.path.join(args.output_dir, "frozen_best_advantage.pt"),
        best_probe,
        {
            **artifact["metadata"],
            "selection_metric": "validation_r_squared",
            "selected_layer": best["layer"],
        },
    )
    test_data = layer_dataset(
        shards, best["layer"], split_sets["test"], targets
    )
    write_predictions(
        os.path.join(args.output_dir, "test_predictions.csv"),
        test_data["records"],
        targets,
        best_probe.predict(test_data["states"]),
    )
    payload = {
        "target": "continuation_advantage=max(Q_A,Q_B)-Q_STOP; Q_STOP=0",
        "rollout_target_files": sorted(glob.glob(args.targets)),
        "labeled_states": len(targets),
        "labeled_states_by_split": {
            name: len(values) for name, values in target_ids_by_split.items()
        },
        "best_layer": best["layer"],
        "layers": summaries,
        "runtime_seconds": time.perf_counter() - started,
        "caveat": (
            "The max of finite-rollout Q estimates is upward biased; target CSVs "
            "retain per-arm standard errors and raw returns for sensitivity checks."
        ),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"best_layer": best["layer"], "test_r_squared": best["test"]["r_squared"]}, indent=2))


if __name__ == "__main__":
    main()
