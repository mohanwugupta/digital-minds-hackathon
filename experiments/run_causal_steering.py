"""Matched causal replay for persistence, return, and advantage ridge directions."""

import argparse
import csv
import glob
import hashlib
import json
import os

from experiments.run_bandit_intervention import build_matched_replays


DIRECTION_NAMES = ("persistence", "generic_return", "advantage")


def _stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def evaluate_direction_replays(
    model,
    *,
    state_id: str,
    conversation: list[dict],
    layer: int,
    delta,
    probe,
    baseline: dict | None = None,
) -> list[dict]:
    """Evaluate -1/0/+1 while reusing the exact unhooked alpha-zero result."""
    from interventions.ridge_steering import apply_ridge_steering

    if baseline is None:
        baseline = model.decision(conversation, capture_hidden_states=True)
    hidden = baseline["hidden_states"][layer].unsqueeze(0).float().cpu()
    with_hidden_removed = {
        key: value for key, value in baseline.items() if key != "hidden_states"
    }
    rows = []
    for replay in build_matched_replays(state_id, conversation):
        if replay.alpha == 0.0:
            metrics = dict(with_hidden_removed)
        else:
            def transform(current, alpha=replay.alpha):
                return apply_ridge_steering(current, delta, alpha)

            metrics = model.decision(
                replay.conversation, layer=layer, transform=transform
            )
        post_hidden = apply_ridge_steering(hidden, delta.cpu(), replay.alpha)
        rows.append(
            {
                "state_id": state_id,
                "alpha": replay.alpha,
                "context_hash": replay.context_hash,
                "probe_value_pre": float(probe.predict(hidden).item()),
                "probe_value_post": float(probe.predict(post_hidden).item()),
                **{key: value for key, value in metrics.items() if key != "hidden_states"},
            }
        )
    return rows


def _completed(path: str) -> set[tuple[str, str, str, str, str]]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (
                row["state_id"],
                row["direction_name"],
                row["control_type"],
                row["control_id"],
                row["alpha"],
            )
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


def _load_probe(path: str, expected_layer: int):
    from interventions.ridge_probe import load_ridge_probe

    probe, payload = load_ridge_probe(path)
    metadata = payload.get("metadata", {})
    layer = metadata.get("selected_layer", metadata.get("layer"))
    if int(layer) != int(expected_layer):
        raise ValueError(
            f"calibration layer {expected_layer} does not match probe layer {layer}"
        )
    return probe


def main() -> None:
    import torch

    from experiments.runtime import save_run_metadata
    from interventions.ridge_steering import (
        matched_sign_random_directions,
        normalized_probe_direction,
    )
    from models.hooked_qwen import HookedQwen

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--state-bank", default="artifacts/confirmatory_state_bank")
    parser.add_argument("--probe-split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--calibration", default="artifacts/causal_steering/calibration.json")
    parser.add_argument("--directions", default="all")
    parser.add_argument("--random-directions", type=int, default=20)
    parser.add_argument("--maximum-states", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=62026)
    parser.add_argument("--output", default="artifacts/causal_steering/replays.csv")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num-shards)")
    if args.random_directions not in (0,) and args.random_directions < 20:
        raise ValueError("use zero controls for a target-only gate or at least 20")
    selected_names = (
        DIRECTION_NAMES
        if args.directions == "all"
        else tuple(value.strip() for value in args.directions.split(",") if value.strip())
    )
    if not selected_names or any(name not in DIRECTION_NAMES for name in selected_names):
        raise ValueError(f"directions must be drawn from {DIRECTION_NAMES}")

    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    calibrated = calibration["directions"]
    probes, deltas, controls = {}, {}, {}
    for direction_index, name in enumerate(selected_names):
        spec = calibrated[name]
        probe = _load_probe(spec["probe_path"], spec["layer"])
        delta = normalized_probe_direction(probe) * float(spec["magnitude"])
        observed_rms = float(((delta / probe.state_std) ** 2).mean().sqrt())
        if abs(observed_rms - float(spec["relative_rms"])) > 1e-5:
            raise ValueError(f"reconstructed {name} delta does not match calibration")
        probes[name], deltas[name] = probe, delta
        controls[name] = matched_sign_random_directions(
            delta,
            n_directions=args.random_directions,
            seed=args.seed + 10_000 * direction_index,
        ) if args.random_directions else []

    with open(args.probe_split, encoding="utf-8") as handle:
        split = json.load(handle)
    fitted_episodes = set(split["train"] + split["validation"] + split["test"])
    records = []
    for path in sorted(glob.glob(os.path.join(args.state_bank, "episode_*.pt"))):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard["episode_id"] in fitted_episodes:
            raise ValueError(f"confirmatory state overlaps probe fitting: {shard['episode_id']}")
        records.extend(shard["records"])
    records.sort(key=lambda row: row["state_id"])
    if not records:
        raise FileNotFoundError(
            f"no held-out state records found under {args.state_bank}; "
            "collect the confirmatory state bank first"
        )
    if args.maximum_states > 0:
        records = records[: args.maximum_states]
    records = [
        row
        for row in records
        if _stable_integer(row["state_id"]) % args.num_shards == args.shard_index
    ]
    if not records:
        raise ValueError(
            "this shard contains no held-out states; reduce num-shards or "
            "choose a populated shard index"
        )

    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    save_run_metadata(
        args.output + ".metadata.json",
        {
            **vars(args),
            "selected_states_this_shard": len(records),
            "selected_directions": list(selected_names),
            "calibration_frozen_before_confirmatory_replay": True,
        },
        model,
    )
    completed = _completed(args.output)
    for index, source in enumerate(records, 1):
        conversation = source["conversation"]
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        baseline = model.decision(conversation, capture_hidden_states=True)
        output_rows = []
        for name in selected_names:
            spec = calibrated[name]
            direction_specs = [("target", "target", deltas[name])]
            direction_specs.extend(
                ("random", f"random_{control_index:02d}", delta)
                for control_index, delta in enumerate(controls[name])
            )
            for control_type, control_id, delta in direction_specs:
                rows = evaluate_direction_replays(
                    model,
                    state_id=source["state_id"],
                    conversation=conversation,
                    layer=int(spec["layer"]),
                    delta=delta,
                    probe=probes[name],
                    baseline=baseline,
                )
                for row in rows:
                    key = (
                        source["state_id"],
                        name,
                        control_type,
                        control_id,
                        str(row["alpha"]),
                    )
                    if key in completed:
                        continue
                    combined = dict(source)
                    combined.update(
                        {
                            "direction_name": name,
                            "control_type": control_type,
                            "control_id": control_id,
                            "layer": int(spec["layer"]),
                            "magnitude": float(spec["magnitude"]),
                            "decoded_sd_shift": float(spec["decoded_sd_shift"]),
                            "intervention_relative_rms": float(spec["relative_rms"]),
                            "direction_l2_norm": float(delta.norm()),
                            **row,
                        }
                    )
                    output_rows.append(combined)
        _append(args.output, output_rows)
        print(
            f"causal replay {index}/{len(records)}: {source['state_id']} "
            f"({len(output_rows)} new rows)",
            flush=True,
        )


if __name__ == "__main__":
    main()
