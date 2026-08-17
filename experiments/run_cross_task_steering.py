"""Replay cross-task test states under bandit persistence steering."""

import argparse
import csv
import hashlib
import json
import os

from experiments.cross_task_utils import (
    layer_dataset,
    load_activation_shards,
    make_or_validate_split,
    probe_layer,
)
from experiments.replay_utils import build_matched_replays


def _stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def evaluate_binary_replays(
    model,
    *,
    record: dict,
    layer: int,
    delta,
    probe,
    baseline: dict | None = None,
) -> list[dict]:
    from interventions.ridge_steering import apply_ridge_steering

    conversation = record["conversation"]
    if isinstance(conversation, str):
        conversation = json.loads(conversation)
    labels = tuple(sorted((record["positive_label"], record["negative_label"])))
    if baseline is None:
        baseline = model.binary_decision(
            conversation,
            labels,
            positive_label=record["positive_label"],
            capture_hidden_states=True,
        )
    hidden = baseline["hidden_states"][layer].unsqueeze(0).float().cpu()
    baseline_metrics = {
        key: value for key, value in baseline.items() if key != "hidden_states"
    }
    rows = []
    for replay in build_matched_replays(record["state_id"], conversation):
        if replay.alpha == 0.0:
            metrics = dict(baseline_metrics)
        else:
            def transform(current, alpha=replay.alpha):
                return apply_ridge_steering(current, delta, alpha)

            metrics = model.binary_decision(
                replay.conversation,
                labels,
                positive_label=record["positive_label"],
                layer=layer,
                transform=transform,
            )
        post_hidden = apply_ridge_steering(hidden, delta.cpu(), replay.alpha)
        rows.append(
            {
                "state_id": record["state_id"],
                "alpha": replay.alpha,
                "context_hash": replay.context_hash,
                "probe_value_pre": float(probe.predict(hidden).item()),
                "probe_value_post": float(probe.predict(post_hidden).item()),
                **metrics,
            }
        )
    return rows


def _completed(path: str) -> set[tuple[str, str, str, str]]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row["state_id"], row["control_type"], row["control_id"], row["alpha"])
            for row in csv.DictReader(handle)
        }


def _append(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("foraging", "control"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--activation-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--calibration", default="artifacts/cross_task/causal/calibration.json"
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--random-directions", type=int, default=None)
    parser.add_argument("--maximum-states", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=92026)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num-shards)")

    import yaml

    from experiments.runtime import save_run_metadata
    from interventions.ridge_probe import load_ridge_probe
    from interventions.ridge_steering import (
        matched_sign_random_directions,
        normalized_probe_direction,
    )
    from models.hooked_qwen import HookedQwen

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("status") != "valid":
        raise ValueError("foraging validation calibration is not valid")
    probe, payload = load_ridge_probe(calibration["probe_path"])
    layer = probe_layer(payload, calibration["probe_path"])
    if layer != int(calibration["layer"]):
        raise ValueError("calibration and bandit probe layers differ")
    delta = normalized_probe_direction(probe) * float(calibration["magnitude"])
    observed_rms = float(((delta / probe.state_std) ** 2).mean().sqrt())
    if abs(observed_rms - float(calibration["relative_rms"])) > 1e-5:
        raise ValueError("reconstructed intervention does not match calibration")

    bank = args.activation_dir or (
        "artifacts/cross_task/foraging_activation_bank"
        if args.task == "foraging"
        else "artifacts/cross_task/control_activation_bank"
    )
    split_path = args.split or (
        "artifacts/cross_task/foraging_episode_split.json"
        if args.task == "foraging"
        else "artifacts/cross_task/control_episode_split.json"
    )
    output = args.output or f"artifacts/cross_task/causal/{args.task}_replays.csv"
    shards = load_activation_shards(bank)
    split = make_or_validate_split(
        shards, split_path, seed=int(config["split_seed"])
    )
    target_key = "persistence_logit" if args.task == "foraging" else "choice_logit"
    data = layer_dataset(shards, layer, set(split["test"]), target_key=target_key)
    records = sorted(data["records"], key=lambda row: row["state_id"])
    if args.maximum_states > 0:
        records = records[: args.maximum_states]
    records = [
        row
        for row in records
        if _stable_integer(row["state_id"]) % args.num_shards == args.shard_index
    ]
    if not records:
        raise ValueError("this shard contains no cross-task test states")
    n_random = (
        int(args.random_directions)
        if args.random_directions is not None
        else int(config["causal_transfer"]["matched_random_directions"])
    )
    if n_random not in (0,) and n_random < 20:
        raise ValueError("use zero random controls for a gate or at least 20")
    controls = (
        matched_sign_random_directions(delta, n_directions=n_random, seed=args.seed)
        if n_random
        else []
    )
    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    save_run_metadata(
        output + ".metadata.json",
        {
            **vars(args),
            "task": args.task,
            "activation_dir": bank,
            "split": split_path,
            "output": output,
            "selected_test_states": len(records),
            "foraging_validation_calibration_frozen": True,
            "semantic_positive": "STAY" if args.task == "foraging" else "LEFT_GREATER",
        },
        model,
    )
    completed = _completed(output)
    for index, record in enumerate(records, 1):
        conversation = record["conversation"]
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        labels = tuple(sorted((record["positive_label"], record["negative_label"])))
        baseline = model.binary_decision(
            conversation,
            labels,
            positive_label=record["positive_label"],
            capture_hidden_states=True,
        )
        baseline_difference = abs(
            float(baseline["choice_logit"]) - float(record[target_key])
        )
        if baseline_difference > args.baseline_tolerance:
            raise RuntimeError(
                f"unsteered replay drift at {record['state_id']}: "
                f"{baseline_difference:.6g}"
            )
        directions = [("target", "target", delta)] + [
            ("random", f"random_{control_index:02d}", direction)
            for control_index, direction in enumerate(controls)
        ]
        output_rows = []
        for control_type, control_id, direction in directions:
            for replay in evaluate_binary_replays(
                model,
                record=record,
                layer=layer,
                delta=direction,
                probe=probe,
                baseline=baseline,
            ):
                key = (
                    record["state_id"],
                    control_type,
                    control_id,
                    str(replay["alpha"]),
                )
                if key in completed:
                    continue
                combined = dict(record)
                combined.update(
                    {
                        "control_type": control_type,
                        "control_id": control_id,
                        "layer": layer,
                        "magnitude": float(calibration["magnitude"]),
                        "decoded_sd_shift": float(calibration["decoded_sd_shift"]),
                        "intervention_relative_rms": float(calibration["relative_rms"]),
                        "direction_l2_norm": float(direction.norm()),
                        "collection_choice_logit": float(record[target_key]),
                        "baseline_replay_absolute_difference": baseline_difference,
                        **replay,
                    }
                )
                output_rows.append(combined)
        _append(output, output_rows)
        print(
            f"cross-task causal replay {index}/{len(records)}: {record['state_id']} "
            f"({len(output_rows)} new rows)",
            flush=True,
        )


if __name__ == "__main__":
    main()
