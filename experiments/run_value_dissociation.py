"""Replay held-out histories under a STOP-payoff × CONTINUE-bonus factorial."""

import argparse
import csv
from dataclasses import dataclass
import glob
import hashlib
import json
import os
from typing import Iterable

from bandit.prompts import current_decision_prefix, factorial_decision_prompt
from experiments.replay_utils import canonical_context


STOP_PAYOFFS = (-10, 0, 10, 20)
CONTINUE_BONUSES = (-10, 0, 10)


@dataclass(frozen=True)
class FactorialReplay:
    state_id: str
    stop_payoff: int
    continue_bonus: int
    relative_incentive: int
    conversation: list[dict]
    history_hash: str
    context_hash: str


def _visible_history_bytes(conversation: list[dict]) -> bytes:
    if not conversation or conversation[-1].get("role") != "user":
        raise ValueError("factorial state must end at a user decision prompt")
    history = [dict(message) for message in conversation]
    history[-1]["content"] = current_decision_prefix(history[-1]["content"])
    return canonical_context(history)


def build_factorial_replays(
    state_id: str,
    conversation: list[dict],
    *,
    action_labels: str = "ABC",
    stop_payoffs: Iterable[int] = STOP_PAYOFFS,
    continue_bonuses: Iterable[int] = CONTINUE_BONUSES,
) -> list[FactorialReplay]:
    """Create all incentive cells while freezing the underlying visible history."""
    stop_payoffs = tuple(int(value) for value in stop_payoffs)
    continue_bonuses = tuple(int(value) for value in continue_bonuses)
    base = json.loads(canonical_context(conversation).decode("utf-8"))
    history_hash = hashlib.sha256(_visible_history_bytes(base)).hexdigest()
    output = []
    for stop_payoff in stop_payoffs:
        for continue_bonus in continue_bonuses:
            manipulated = json.loads(canonical_context(base).decode("utf-8"))
            manipulated[-1]["content"] = factorial_decision_prompt(
                manipulated[-1]["content"],
                int(stop_payoff),
                int(continue_bonus),
                action_labels,
            )
            context = canonical_context(manipulated)
            output.append(
                FactorialReplay(
                    state_id=state_id,
                    stop_payoff=int(stop_payoff),
                    continue_bonus=int(continue_bonus),
                    relative_incentive=int(continue_bonus) - int(stop_payoff),
                    conversation=manipulated,
                    history_hash=history_hash,
                    context_hash=hashlib.sha256(context).hexdigest(),
                )
            )
    expected = len(stop_payoffs) * len(continue_bonuses)
    if len(output) != expected or len({item.context_hash for item in output}) != expected:
        raise RuntimeError("factorial construction did not create unique complete cells")
    if len({item.history_hash for item in output}) != 1:
        raise RuntimeError("factorial conditions changed underlying visible history")
    return output


def _stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _completed(path: str) -> set[tuple[str, str, str]]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row["state_id"], row["stop_payoff"], row["continue_bonus"])
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


def _load_probe_spec(path: str) -> dict:
    from interventions.ridge_probe import load_ridge_probe

    probe, payload = load_ridge_probe(path)
    layer = payload.get("metadata", {}).get("selected_layer")
    if layer is None:
        layer = payload.get("metadata", {}).get("layer")
    if layer is None:
        raise ValueError(f"ridge artifact does not identify its native layer: {path}")
    return {"probe": probe, "layer": int(layer), "path": os.path.abspath(path)}


def main() -> None:
    import torch

    from experiments.runtime import save_run_metadata
    from models.hooked_qwen import HookedQwen

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--state-bank", default="artifacts/confirmatory_state_bank")
    parser.add_argument("--probe-split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--generic-probe", default="artifacts/linear_probes/frozen_best_future_return.pt")
    parser.add_argument("--advantage-probe", default="artifacts/advantage_probes/frozen_best_advantage.pt")
    parser.add_argument("--persistence-probe", default="artifacts/linear_probes/frozen_best_persistence.pt")
    parser.add_argument("--output", default="artifacts/value_dissociation/factorial.csv")
    parser.add_argument(
        "--activation-output-dir",
        default="artifacts/value_dissociation/activations",
        help="destination used only with --save-activations",
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        help="retain optional all-layer factorial tensors (about 2.2 GB for the full run)",
    )
    parser.add_argument("--maximum-states", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num-shards)")

    probes = {
        "generic_return": _load_probe_spec(args.generic_probe),
        "advantage": _load_probe_spec(args.advantage_probe),
        "persistence": _load_probe_spec(args.persistence_probe),
    }
    with open(args.probe_split, encoding="utf-8") as handle:
        split = json.load(handle)
    fitted_episodes = set(split["train"] + split["validation"] + split["test"])
    records = []
    for path in sorted(glob.glob(os.path.join(args.state_bank, "episode_*.pt"))):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard["episode_id"] in fitted_episodes:
            raise ValueError(f"held-out state overlaps probe fitting: {shard['episode_id']}")
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
            "stop_payoffs": list(STOP_PAYOFFS),
            "continue_bonuses": list(CONTINUE_BONUSES),
            "probe_layers": {name: spec["layer"] for name, spec in probes.items()},
        },
        model,
    )
    completed = _completed(args.output)
    for index, source in enumerate(records, 1):
        conversation = source["conversation"]
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        output_rows, activation_rows, activation_conditions = [], [], []
        activation_path = None
        activation_complete = not args.save_activations
        if args.save_activations:
            activation_name = hashlib.sha256(
                source["state_id"].encode("utf-8")
            ).hexdigest()[:20]
            activation_path = os.path.join(
                args.activation_output_dir, f"state_{activation_name}.pt"
            )
            activation_complete = os.path.exists(activation_path)
        for replay in build_factorial_replays(
            source["state_id"], conversation, action_labels=model.action_labels
        ):
            key = (
                replay.state_id,
                str(replay.stop_payoff),
                str(replay.continue_bonus),
            )
            row_complete = key in completed
            if row_complete and activation_complete:
                continue
            metrics = model.decision(
                replay.conversation, capture_hidden_states=True
            )
            hidden_states = metrics.pop("hidden_states")
            if args.save_activations:
                activation_rows.append(
                    torch.stack(hidden_states).to(dtype=torch.float16)
                )
                activation_conditions.append(
                    {
                        "stop_payoff": replay.stop_payoff,
                        "continue_bonus": replay.continue_bonus,
                        "relative_incentive": replay.relative_incentive,
                        "context_hash": replay.context_hash,
                    }
                )
            projections = {}
            for name, spec in probes.items():
                hidden = hidden_states[spec["layer"]].unsqueeze(0)
                projections[f"{name}_projection"] = float(
                    spec["probe"].predict(hidden).item()
                )
            if row_complete:
                continue
            row = dict(source)
            row.update(
                {
                    "conversation": json.dumps(
                        replay.conversation, separators=(",", ":")
                    ),
                    "stop_payoff": replay.stop_payoff,
                    "continue_bonus": replay.continue_bonus,
                    "relative_incentive": replay.relative_incentive,
                    "common_incentive": replay.continue_bonus + replay.stop_payoff,
                    "history_hash": replay.history_hash,
                    "context_hash": replay.context_hash,
                    **projections,
                    **metrics,
                }
            )
            output_rows.append(row)
        _append(args.output, output_rows)
        if args.save_activations and not activation_complete:
            from experiments.runtime import atomic_torch_save

            os.makedirs(args.activation_output_dir, exist_ok=True)
            atomic_torch_save(
                {
                    "episode_id": source["episode_id"],
                    "state_id": source["state_id"],
                    "conditions": activation_conditions,
                    "activations": torch.stack(activation_rows),
                    "shape": "factorial_conditions x layers x hidden_width",
                },
                activation_path,
            )
        print(
            f"factorial {index}/{len(records)}: {source['state_id']} "
            f"({len(output_rows)} new cells)",
            flush=True,
        )


if __name__ == "__main__":
    main()
