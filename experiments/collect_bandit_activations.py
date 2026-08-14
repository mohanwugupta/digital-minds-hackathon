"""Collect final-prompt-token states from every layer in baseline episodes."""

import argparse
import os
import random
import time
from typing import List

from bandit.conversation import BanditConversation
from bandit.environment import BanditEnvironment
from bandit.schemas import DecisionRecord
from experiments.run_bandit_baseline import episode_conditions, sample_action
from experiments.runtime import atomic_torch_save, save_run_metadata


def collect_episode(model, p_a, p_b, *, seed: int, action_seed: int, max_decisions: int = 100):
    import torch

    environment = BanditEnvironment(p_a, p_b, seed, max_decisions=max_decisions)
    conversation = BanditConversation.start(getattr(model, "action_labels", "ABC"))
    rng = random.Random(action_seed)
    episode_id = f"seed-{seed}-pa-{p_a:.2f}-pb-{p_b:.2f}"
    records: List[DecisionRecord] = []
    activations = []
    previous_outcome = None
    while not environment.terminated:
        visible = conversation.snapshot()
        decision = model.decision(visible, capture_hidden_states=True)
        hidden = decision.pop("hidden_states")
        activations.append(torch.stack(hidden).to(dtype=torch.float16))
        action = sample_action(decision, rng)
        record = DecisionRecord(
            episode_id=episode_id,
            state_id=f"{episode_id}:{environment.decision}",
            seed=seed,
            action_seed=action_seed,
            round=environment.decision,
            p_A_true=p_a,
            p_B_true=p_b,
            cumulative_score=environment.cumulative_score,
            choice_history=list(environment.action_history),
            reward_history=list(environment.reward_history),
            conversation=visible,
            previous_outcome=previous_outcome,
            sampled_action=action,
            **decision,
        )
        conversation.record_action(action)
        result = environment.step(action)
        record.subsequent_reward = result.reward
        record.terminated = result.terminated
        records.append(record)
        if not result.terminated:
            conversation.record_feedback(result.reward)
            previous_outcome = result.reward
    running = 0.0
    for record in reversed(records):
        running += float(record.subsequent_reward or 0)
        record.future_cumulative_return = running
    return {
        "episode_id": episode_id,
        "model_id": model.model_id,
        "records": [record.to_row() for record in records],
        "activations": torch.stack(activations),  # states x layers x width
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12026)
    parser.add_argument("--output-dir", default="artifacts/activation_bank")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    from models.hooked_qwen import HookedQwen

    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_metadata(os.path.join(args.output_dir, "metadata.json"), vars(args), model)
    started = time.perf_counter()
    episodes_run = 0
    states_run = 0
    for index, condition in enumerate(episode_conditions(args.episodes, args.seed), 1):
        p_a, p_b, seed, action_seed = condition
        path = os.path.join(args.output_dir, f"episode_{index:05d}.pt")
        if os.path.exists(path):
            continue
        artifact = collect_episode(model, p_a, p_b, seed=seed, action_seed=action_seed)
        atomic_torch_save(artifact, path)
        episodes_run += 1
        states_run += len(artifact["records"])
        print(f"collected {index}/{args.episodes}: {artifact['episode_id']}", flush=True)
    elapsed = time.perf_counter() - started
    print(
        f"runtime: {episodes_run} new episodes, {states_run} activation states, "
        f"{elapsed:.1f}s total, {states_run / elapsed if elapsed else 0:.2f} states/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
