"""Behavioral pilot and unmodified episode collection."""

import argparse
import csv
import os
import random
import time
from itertools import product
from typing import Iterable, List

from bandit.conversation import BanditConversation
from bandit.environment import BanditEnvironment
from bandit.schemas import DecisionRecord


ARM_PROBABILITIES = (0.20, 0.35, 0.50, 0.65)


def sample_action(metrics: dict, rng: random.Random) -> str:
    draw = rng.random()
    if draw < metrics["p_A"]:
        return "A"
    if draw < metrics["p_A"] + metrics["p_B"]:
        return "B"
    return "C"


def run_episode(
    model,
    p_a: float,
    p_b: float,
    *,
    seed: int,
    action_seed: int,
    max_decisions: int = 100,
) -> List[DecisionRecord]:
    environment = BanditEnvironment(p_a, p_b, seed, max_decisions=max_decisions)
    conversation = BanditConversation.start(getattr(model, "action_labels", "ABC"))
    action_rng = random.Random(action_seed)
    records: List[DecisionRecord] = []
    previous_outcome = None
    episode_id = f"seed-{seed}-pa-{p_a:.2f}-pb-{p_b:.2f}"

    while not environment.terminated:
        state_round = environment.decision
        visible_context = conversation.snapshot()
        metrics = model.decision(visible_context)
        action = sample_action(metrics, action_rng)
        record = DecisionRecord(
            episode_id=episode_id,
            state_id=f"{episode_id}:{state_round}",
            seed=seed,
            action_seed=action_seed,
            round=state_round,
            p_A_true=p_a,
            p_B_true=p_b,
            cumulative_score=environment.cumulative_score,
            choice_history=list(environment.action_history),
            reward_history=list(environment.reward_history),
            conversation=visible_context,
            previous_outcome=previous_outcome,
            **{key: metrics[key] for key in (
                "logit_A", "logit_B", "logit_C", "p_A", "p_B", "p_stop",
                "p_continue", "persistence_logit"
            )},
            sampled_action=action,
            layer=metrics.get("layer"),
            neuron_set=metrics.get("neuron_set", "none"),
            intervention_type=metrics.get("intervention_type", "none"),
            alpha=float(metrics.get("alpha", 0.0)),
            probe_value_pre=metrics.get("probe_value_pre"),
            probe_value_post=metrics.get("probe_value_post"),
            p_action_mass_raw=metrics.get("p_action_mass_raw"),
            top_token_is_action=metrics.get("top_token_is_action"),
        )
        conversation.record_action(action)
        result = environment.step(action)
        record.subsequent_reward = result.reward
        record.terminated = result.terminated
        records.append(record)
        if not result.terminated:
            conversation.record_feedback(result.reward)
            previous_outcome = result.reward

    running_return = 0.0
    for record in reversed(records):
        running_return += float(record.subsequent_reward or 0)
        record.future_cumulative_return = running_return
    return records


def episode_conditions(n_episodes: int, base_seed: int) -> Iterable[tuple]:
    cells = list(product(ARM_PROBABILITIES, repeat=2))
    random.Random(base_seed).shuffle(cells)
    for index in range(n_episodes):
        p_a, p_b = cells[index % len(cells)]
        yield p_a, p_b, base_seed + index, base_seed + 1_000_000 + index


def append_records_csv(path: str, records: List[DecisionRecord]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = [record.to_row() for record in records]
    if not rows:
        return
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def completed_episode_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["episode_id"] for row in csv.DictReader(handle) if row.get("terminated") == "True"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="artifacts/bandit_pilot.csv")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true", help="allow Hugging Face downloads")
    args = parser.parse_args()

    from models.hooked_qwen import HookedQwen
    from experiments.runtime import save_run_metadata

    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    done = completed_episode_ids(args.output)
    started = time.perf_counter()
    episodes_run = 0
    states_run = 0
    for index, (p_a, p_b, seed, action_seed) in enumerate(episode_conditions(args.episodes, args.seed), 1):
        episode_id = f"seed-{seed}-pa-{p_a:.2f}-pb-{p_b:.2f}"
        if episode_id in done:
            continue
        records = run_episode(
            model, p_a, p_b, seed=seed, action_seed=action_seed,
            max_decisions=args.max_decisions,
        )
        append_records_csv(args.output, records)
        episodes_run += 1
        states_run += len(records)
        print(f"completed {index}/{args.episodes}: {episode_id}", flush=True)
    elapsed = time.perf_counter() - started
    print(
        f"runtime: {episodes_run} new episodes, {states_run} decision states, "
        f"{elapsed:.1f}s total, {states_run / elapsed if elapsed else 0:.2f} states/s",
        flush=True,
    )
    save_run_metadata(args.output + ".metadata.json", vars(args), model)


if __name__ == "__main__":
    main()
