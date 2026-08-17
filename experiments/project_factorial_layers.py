"""Project existing factorial states through every frozen persistence probe."""

import argparse
import csv
import glob
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path


def _stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _read_factorial_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"no factorial rows match {pattern}")
    keys = [
        (row["state_id"], row["stop_payoff"], row["continue_bonus"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("factorial inputs contain duplicate state/cell rows")
    return rows


def _load_layer_probes(pattern: str):
    from interventions.ridge_probe import load_ridge_probe

    probes = {}
    for path in sorted(glob.glob(pattern)):
        probe, payload = load_ridge_probe(path)
        metadata = payload.get("metadata", {})
        layer = metadata.get("layer", metadata.get("selected_layer"))
        if layer is None:
            raise ValueError(f"probe does not identify a layer: {path}")
        layer = int(layer)
        if probe.target != "persistence" or layer in probes:
            raise ValueError(f"invalid or duplicate persistence probe: {path}")
        probes[layer] = probe
    if not probes or sorted(probes) != list(range(max(probes) + 1)):
        raise ValueError("persistence-probe pattern must cover contiguous layers from zero")
    return probes


def _activation_path(directory: str, state_id: str) -> str:
    name = hashlib.sha256(state_id.encode("utf-8")).hexdigest()[:20]
    return os.path.join(directory, f"state_{name}.pt")


def _load_saved_activation(directory: str, row: dict, cache: dict):
    import torch

    path = _activation_path(directory, row["state_id"])
    if not os.path.exists(path):
        return None
    if cache.get("path") != path:
        cache.clear()
        cache.update(
            {
                "path": path,
                "artifact": torch.load(
                    path, map_location="cpu", weights_only=False
                ),
            }
        )
    artifact = cache["artifact"]
    if artifact["state_id"] != row["state_id"]:
        raise ValueError(f"factorial activation state mismatch in {path}")
    condition_index = {
        (int(item["stop_payoff"]), int(item["continue_bonus"])): index
        for index, item in enumerate(artifact["conditions"])
    }
    key = (int(float(row["stop_payoff"])), int(float(row["continue_bonus"])))
    if key not in condition_index:
        raise ValueError(f"factorial activation is missing condition {key} in {path}")
    return artifact["activations"][condition_index[key]].float()


def project_states(hidden_states, probes: dict[int, object]) -> dict[str, float]:
    if int(hidden_states.shape[0]) != len(probes):
        raise ValueError("activation layer count does not match frozen probes")
    return {
        f"layer_{layer:02d}_projection": float(
            probe.predict(hidden_states[layer].unsqueeze(0)).item()
        )
        for layer, probe in probes.items()
    }


def _completed(path: str) -> set[tuple[str, str, str]]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row["state_id"], row["stop_payoff"], row["continue_bonus"])
            for row in csv.DictReader(handle)
        }


def _append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/value_dissociation/factorial_shard_*.csv"
    )
    parser.add_argument(
        "--probe-pattern",
        default="artifacts/linear_probes/layer_*_persistence.pt",
    )
    parser.add_argument(
        "--activation-dir",
        default=None,
        help="optional saved factorial tensors; missing states are replayed",
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        help="retain condition x layer x width tensors while projecting",
    )
    parser.add_argument(
        "--activation-output-dir",
        default="artifacts/value_dissociation/activations",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--output", default="artifacts/value_dissociation/layerwise_projections.csv"
    )
    parser.add_argument("--replay-tolerance", type=float, default=1e-4)
    parser.add_argument("--maximum-states", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num-shards)")

    probes = _load_layer_probes(args.probe_pattern)
    rows = _read_factorial_rows(args.input)
    state_ids = sorted(
        {
            row["state_id"]
            for row in rows
            if _stable_integer(row["state_id"]) % args.num_shards == args.shard_index
        }
    )
    if args.maximum_states > 0:
        state_ids = state_ids[: args.maximum_states]
    selected_states = set(state_ids)
    rows = sorted(
        (row for row in rows if row["state_id"] in selected_states),
        key=lambda row: (
            row["state_id"],
            int(float(row["stop_payoff"])),
            int(float(row["continue_bonus"])),
        ),
    )
    if not rows:
        raise ValueError("no factorial states selected")

    model = None
    completed = _completed(args.output)
    replayed = saved = activation_states_written = 0
    activation_cache = {}
    processed_cells = 0
    for state_index, (state_id, grouped_rows) in enumerate(
        groupby(rows, key=lambda row: row["state_id"]), 1
    ):
        state_rows = list(grouped_rows)
        activation_path = _activation_path(args.activation_output_dir, state_id)
        activation_complete = args.save_activations and os.path.exists(activation_path)
        activation_rows, activation_conditions = [], []
        for source in state_rows:
            key = (
                source["state_id"],
                source["stop_payoff"],
                source["continue_bonus"],
            )
            row_complete = key in completed
            if row_complete and (not args.save_activations or activation_complete):
                processed_cells += 1
                continue
            hidden = None
            for directory in (
                args.activation_dir,
                args.activation_output_dir if activation_complete else None,
            ):
                if directory:
                    hidden = _load_saved_activation(
                        directory, source, activation_cache
                    )
                    if hidden is not None:
                        break
            if hidden is None:
                if model is None:
                    from models.hooked_qwen import HookedQwen

                    model = HookedQwen.from_pretrained(
                        args.model,
                        revision=args.revision,
                        local_files_only=not args.online,
                    )
                conversation = json.loads(source["conversation"])
                metrics = model.decision(conversation, capture_hidden_states=True)
                hidden = __import__("torch").stack(
                    metrics.pop("hidden_states")
                ).float()
                difference = abs(
                    float(metrics["persistence_logit"])
                    - float(source["persistence_logit"])
                )
                if difference > args.replay_tolerance:
                    raise RuntimeError(
                        f"baseline replay drift at {source['state_id']}: "
                        f"{difference:.6g}"
                    )
                replayed += 1
            else:
                saved += 1
            if args.save_activations and not activation_complete:
                activation_rows.append(hidden.to(dtype=__import__("torch").float16))
                activation_conditions.append(
                    {
                        "stop_payoff": int(float(source["stop_payoff"])),
                        "continue_bonus": int(float(source["continue_bonus"])),
                        "relative_incentive": int(
                            float(source["relative_incentive"])
                        ),
                        "context_hash": source["context_hash"],
                    }
                )
            if not row_complete:
                output = {
                    field: source[field]
                    for field in (
                        "episode_id",
                        "state_id",
                        "stop_payoff",
                        "continue_bonus",
                        "relative_incentive",
                        "common_incentive",
                        "history_hash",
                        "context_hash",
                        "persistence_logit",
                    )
                }
                output.update(project_states(hidden, probes))
                _append(args.output, output)
            processed_cells += 1
        if args.save_activations and not activation_complete:
            import torch

            from experiments.runtime import atomic_torch_save

            if len(activation_rows) != len(state_rows):
                raise RuntimeError(
                    f"activation replay for {state_id} recovered "
                    f"{len(activation_rows)}/{len(state_rows)} cells"
                )
            atomic_torch_save(
                {
                    "episode_id": state_rows[0]["episode_id"],
                    "state_id": state_id,
                    "conditions": activation_conditions,
                    "activations": torch.stack(activation_rows),
                    "shape": "factorial_conditions x layers x hidden_width",
                    "source": "existing factorial prompt replay",
                },
                activation_path,
            )
            activation_states_written += 1
        if state_index % 10 == 0 or state_index == len(selected_states):
            print(
                f"projected {processed_cells}/{len(rows)} cells across "
                f"{state_index}/{len(selected_states)} states",
                flush=True,
            )

    metadata = {
        "input": args.input,
        "probe_pattern": args.probe_pattern,
        "frozen_probe_layers": sorted(probes),
        "states": len(selected_states),
        "cells": len(rows),
        "loaded_saved_activation_cells": saved,
        "replayed_cells": replayed,
        "save_activations": args.save_activations,
        "activation_output_dir": (
            os.path.abspath(args.activation_output_dir)
            if args.save_activations
            else None
        ),
        "activation_states_written": activation_states_written,
        "replay_tolerance": args.replay_tolerance,
        "probes_retrained_on_factorial": False,
    }
    path = Path(args.output + ".metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
