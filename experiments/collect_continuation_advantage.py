"""Estimate forced-A/B continuation advantage with paired policy rollouts."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict

from bandit.conversation import BanditConversation
from bandit.environment import BanditEnvironment
from experiments.run_bandit_baseline import sample_action


def _parse_json(value):
    return json.loads(value) if isinstance(value, str) else value


def _stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def rollout_forced_action(
    model,
    record: dict,
    forced_action: str,
    *,
    outcome_seed: int,
    action_seed: int,
    max_decisions: int = 100,
) -> float:
    """Force one arm, then sample the unmodified policy until STOP/horizon."""
    forced_action = forced_action.strip().upper()
    if forced_action not in {"A", "B"}:
        raise ValueError("forced action must be A or B")
    action_history = _parse_json(record["choice_history"])
    reward_history = _parse_json(record["reward_history"])
    environment = BanditEnvironment.from_history(
        float(record["p_A_true"]),
        float(record["p_B_true"]),
        outcome_seed,
        action_history,
        reward_history,
        max_decisions=max_decisions,
    )
    conversation = BanditConversation(
        messages=[dict(message) for message in _parse_json(record["conversation"])],
        action_labels=getattr(model, "action_labels", "ABC"),
    )
    action_rng = random.Random(action_seed)
    action, future_return = forced_action, 0.0
    while not environment.terminated:
        conversation.record_action(action)
        result = environment.step(action)
        future_return += result.reward
        if result.terminated:
            break
        conversation.record_feedback(result.reward)
        metrics = model.decision(conversation.snapshot())
        action = sample_action(metrics, action_rng)
    return future_return


def estimate_state_advantage(
    model,
    record: dict,
    *,
    rollouts: int,
    seed: int,
    max_decisions: int = 100,
) -> dict:
    if rollouts < 2:
        raise ValueError("at least two rollouts are required for uncertainty")
    state_seed = seed + _stable_integer(str(record["state_id"])) % 1_000_000_000
    returns = {"A": [], "B": []}
    for replicate in range(rollouts):
        # A and B share potential-outcome and downstream action-randomness seeds.
        outcome_seed = state_seed + 2 * replicate
        action_seed = state_seed + 1_000_000_000 + replicate
        for action in ("A", "B"):
            returns[action].append(
                rollout_forced_action(
                    model,
                    record,
                    action,
                    outcome_seed=outcome_seed,
                    action_seed=action_seed,
                    max_decisions=max_decisions,
                )
            )
    means = {action: statistics.mean(values) for action, values in returns.items()}
    standard_errors = {
        action: statistics.stdev(values) / math.sqrt(rollouts)
        for action, values in returns.items()
    }
    paired = [left - right for left, right in zip(returns["A"], returns["B"])]
    return {
        "q_A": means["A"],
        "q_B": means["B"],
        "q_STOP": 0.0,
        "continuation_advantage": max(means.values()),
        "best_forced_action": max(means, key=means.get),
        "q_A_standard_error": standard_errors["A"],
        "q_B_standard_error": standard_errors["B"],
        "paired_A_minus_B_standard_error": statistics.stdev(paired)
        / math.sqrt(rollouts),
        "rollouts_per_action": rollouts,
        "returns_A": json.dumps(returns["A"], separators=(",", ":")),
        "returns_B": json.dumps(returns["B"], separators=(",", ":")),
    }


def _loss_streak(record: dict) -> int:
    streak = 0
    for reward in reversed(_parse_json(record["reward_history"])):
        if float(reward) != -2:
            break
        streak += 1
    return streak


def select_states(
    records: list[dict], maximum: int, seed: int
) -> list[dict]:
    """Deterministically preserve recent-state strata when subsampling."""
    if maximum <= 0 or len(records) <= maximum:
        return sorted(records, key=lambda row: row["state_id"])
    grouped = defaultdict(list)
    for record in records:
        previous = record.get("previous_outcome")
        previous = None if previous in (None, "") else float(previous)
        key = (int(record["round"]), previous, _loss_streak(record))
        grouped[key].append(record)
    rng = random.Random(seed)
    for selected in grouped.values():
        rng.shuffle(selected)
    keys = sorted(grouped, key=lambda key: (-len(grouped[key]), str(key)))
    output = []
    # Reserve blocks of four from common strata so the downstream exact-match
    # analysis remains identified even in a small sprint subset.
    for key in keys:
        if len(grouped[key]) >= 4 and len(output) + 4 <= maximum:
            for _ in range(4):
                output.append(grouped[key].pop())
    while keys and len(output) < maximum:
        next_keys = []
        for key in keys:
            if grouped[key] and len(output) < maximum:
                output.append(grouped[key].pop())
            if grouped[key]:
                next_keys.append(key)
        keys = next_keys
    return output


def _completed(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["state_id"] for row in csv.DictReader(handle)}


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
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--activation-dir", default="artifacts/activation_bank")
    parser.add_argument("--split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--output", default="artifacts/advantage_targets/targets.csv")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument(
        "--states-per-split",
        type=int,
        default=0,
        help="0 uses every stored state; positive values select a stratified sprint subset",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=72026)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num_shards)")

    from experiments.runtime import save_run_metadata
    from experiments.train_value_probe import load_shards
    from models.hooked_qwen import HookedQwen

    shards = load_shards(args.activation_dir)
    with open(args.split, encoding="utf-8") as handle:
        split = json.load(handle)
    shard_by_episode = {shard["episode_id"]: shard for shard in shards}
    selected = []
    for split_index, (name, episode_ids) in enumerate(split.items()):
        records = [
            record
            for episode_id in episode_ids
            for record in shard_by_episode[episode_id]["records"]
        ]
        for record in select_states(
            records, args.states_per_split, args.seed + split_index
        ):
            selected.append((name, record))
    selected = [
        item
        for item in selected
        if _stable_integer(item[1]["state_id"]) % args.num_shards == args.shard_index
    ]
    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    save_run_metadata(
        args.output + ".metadata.json",
        {**vars(args), "selected_states_this_shard": len(selected)},
        model,
    )
    completed = _completed(args.output)
    started, new_states = time.perf_counter(), 0
    for index, (split_name, record) in enumerate(selected, 1):
        if record["state_id"] in completed:
            continue
        estimate = estimate_state_advantage(
            model,
            record,
            rollouts=args.rollouts,
            seed=args.seed,
            max_decisions=args.max_decisions,
        )
        row = {
            "episode_id": record["episode_id"],
            "state_id": record["state_id"],
            "split": split_name,
            "round": int(record["round"]),
            "previous_outcome": record.get("previous_outcome"),
            "cumulative_score": float(record["cumulative_score"]),
            "persistence_logit": float(record["persistence_logit"]),
            **estimate,
        }
        _append(args.output, row)
        new_states += 1
        print(
            f"advantage {index}/{len(selected)}: {record['state_id']} "
            f"Q_A={estimate['q_A']:.2f} Q_B={estimate['q_B']:.2f} "
            f"A_continue={estimate['continuation_advantage']:.2f}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    print(
        f"advantage runtime: {new_states} new states, {args.rollouts} paired "
        f"rollouts/state, {elapsed:.1f}s total",
        flush=True,
    )


if __name__ == "__main__":
    main()
