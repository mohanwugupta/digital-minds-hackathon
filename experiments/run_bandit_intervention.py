"""Matched-state causal replay under negative, zero, and positive steering."""

from dataclasses import dataclass
import hashlib
import json
import argparse
import csv
import glob
import os
from typing import List


@dataclass(frozen=True)
class MatchedReplay:
    state_id: str
    alpha: float
    conversation: list
    context_bytes: bytes
    context_hash: str


def canonical_context(conversation: list) -> bytes:
    return json.dumps(
        conversation, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def build_matched_replays(state_id: str, conversation: list) -> List[MatchedReplay]:
    context = canonical_context(conversation)
    digest = hashlib.sha256(context).hexdigest()
    return [
        MatchedReplay(
            state_id=state_id,
            alpha=alpha,
            conversation=json.loads(context.decode("utf-8")),
            context_bytes=context,
            context_hash=digest,
        )
        for alpha in (-1.0, 0.0, 1.0)
    ]


def replay_state(model, replay: MatchedReplay, layer: int, transform_factory) -> dict:
    transform = transform_factory(replay.alpha)
    metrics = model.decision(
        replay.conversation,
        layer=layer if transform is not None else None,
        transform=transform,
    )
    return {
        "state_id": replay.state_id,
        "alpha": replay.alpha,
        "context_hash": replay.context_hash,
        **{key: value for key, value in metrics.items() if key != "hidden_states"},
    }


def _append_rows(path: str, rows: list) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _completed_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row["state_id"], row["intervention_type"], row["neuron_set"], row["alpha"])
            for row in csv.DictReader(handle)
        }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--state-bank", required=True, help="held-out activation-bank directory")
    parser.add_argument("--probe", default="artifacts/value_probes/frozen_best.pt")
    parser.add_argument("--probe-split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--calibration", default="artifacts/value_probes/steering_calibration.json")
    parser.add_argument("--output", default="artifacts/matched_intervention.csv")
    parser.add_argument("--random-sets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=32026)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    from experiments.runtime import save_run_metadata
    from interventions.artifacts import load_frozen_probe
    from interventions.steering import (
        build_value_direction, random_masked_direction, sample_random_neuron_sets, steer_hidden,
    )
    from models.hooked_qwen import HookedQwen

    if args.random_sets < 20:
        raise ValueError("the PRD requires at least 20 random-neuron sets")
    probe, layer, value_indices, _ = load_frozen_probe(args.probe)
    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration["layer"] != layer or calibration["neuron_indices"] != value_indices.tolist():
        raise ValueError("calibration does not match the frozen probe/neuron set")
    magnitude = float(calibration["magnitude"])
    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    random_sets = sample_random_neuron_sets(
        probe.hidden.in_features, len(value_indices), n_sets=args.random_sets, seed=args.seed,
        exclude=value_indices,
    )
    completed = _completed_keys(args.output)
    save_run_metadata(args.output + ".metadata.json", vars(args), model)

    probe_episode_ids = set()
    if os.path.exists(args.probe_split):
        with open(args.probe_split, encoding="utf-8") as handle:
            split = json.load(handle)
        probe_episode_ids = set(split["train"] + split["validation"] + split["test"])

    for path in sorted(glob.glob(os.path.join(args.state_bank, "episode_*.pt"))):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard["episode_id"] in probe_episode_ids:
            raise ValueError(
                f"confirmatory state {shard['episode_id']} overlaps probe data"
            )
        for source in shard["records"]:
            conversation = json.loads(source["conversation"])
            baseline = model.decision(conversation, capture_hidden_states=True)
            hidden = baseline["hidden_states"][layer].unsqueeze(0)
            value_direction = build_value_direction(
                probe, hidden, value_indices, magnitude=magnitude
            )
            with torch.no_grad():
                probe_pre = float(probe(hidden).item())
            directions = [("value", "value", value_direction)]
            for random_id, indices in enumerate(random_sets):
                random_direction = random_masked_direction(
                    value_direction, indices, probe.std, seed=args.seed + random_id
                )
                directions.append(("random", f"random_{random_id:02d}", random_direction))

            output_rows = []
            for intervention_type, neuron_set, direction in directions:
                for replay in build_matched_replays(source["state_id"], conversation):
                    key = (source["state_id"], intervention_type, neuron_set, str(replay.alpha))
                    if key in completed:
                        continue
                    if replay.alpha == 0.0:
                        metrics = baseline
                    else:
                        def transform(current, d=direction, alpha=replay.alpha):
                            return steer_hidden(current, d.to(current.device, current.dtype), alpha)
                        metrics = model.decision(conversation, layer=layer, transform=transform)
                    post_hidden = steer_hidden(hidden, direction, replay.alpha)
                    with torch.no_grad():
                        probe_post = float(probe(post_hidden).item())
                    row = dict(source)
                    row.update({
                        "layer": layer,
                        "neuron_set": neuron_set,
                        "intervention_type": intervention_type,
                        "alpha": replay.alpha,
                        "probe_value_pre": probe_pre,
                        "probe_value_post": probe_post,
                        "context_hash": replay.context_hash,
                    })
                    row.update({key: value for key, value in metrics.items() if key != "hidden_states"})
                    output_rows.append(row)
            _append_rows(args.output, output_rows)
        print(f"replayed {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    main()
