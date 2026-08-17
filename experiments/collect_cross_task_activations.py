"""Collect counterbalanced foraging and binary-control activation banks."""

import argparse
import os
import random
import time

from cross_task.common import LabelMapping
from cross_task.control import (
    LEFT_GREATER,
    RIGHT_GREATER,
    comparison_prompt,
    episode_conditions as control_conditions,
)
from cross_task.foraging import (
    LEAVE,
    STAY,
    ForagingConversation,
    ForagingEnvironment,
    episode_conditions as foraging_conditions,
)
from experiments.runtime import atomic_torch_save, save_run_metadata


def _binary_decision(model, messages, mapping: LabelMapping, **kwargs) -> dict:
    metrics = model.binary_decision(
        messages,
        mapping.labels,
        positive_label=mapping.positive_label,
        **kwargs,
    )
    required = {"p_positive", "p_negative", "choice_logit", "hidden_states"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"binary model result is missing fields: {sorted(missing)}")
    return metrics


def collect_foraging_episode(
    model,
    pair_id,
    condition,
    mapping,
    *,
    seed: int,
    action_seed: int,
    max_decisions: int = 20,
    food_reward: int = 4,
) -> dict:
    import torch

    environment = ForagingEnvironment(
        condition, seed, max_decisions=max_decisions, food_reward=food_reward
    )
    conversation = ForagingConversation.start(
        condition, mapping, food_reward=food_reward
    )
    rng = random.Random(action_seed)
    episode_id = f"{pair_id}-{mapping.mapping_id}"
    records, activations = [], []
    previous_outcome = None
    while not environment.terminated:
        visible = conversation.snapshot()
        metrics = _binary_decision(
            model, visible, mapping, capture_hidden_states=True
        )
        hidden = metrics.pop("hidden_states")
        activations.append(torch.stack(hidden).to(dtype=torch.float16))
        semantic_choice = STAY if rng.random() < metrics["p_positive"] else LEAVE
        raw_label = mapping.label_for(semantic_choice)
        state_id = f"{episode_id}:{environment.decision}"
        record = {
            "task": "foraging",
            "episode_id": episode_id,
            "pair_id": pair_id,
            "state_id": state_id,
            "seed": int(seed),
            "action_seed": int(action_seed),
            "round": environment.decision,
            "conversation": visible,
            "label_mapping": mapping.to_json(),
            "mapping_id": mapping.mapping_id,
            "positive_semantic": STAY,
            "negative_semantic": LEAVE,
            "positive_label": mapping.positive_label,
            "negative_label": mapping.negative_label,
            "raw_label": raw_label,
            "semantic_choice": semantic_choice,
            "initial_quality": condition.initial_quality,
            "depletion": condition.depletion,
            "outside_option": condition.outside_option,
            "stay_cost": condition.stay_cost,
            "patch_probability_private": environment.patch_probability(),
            "search_count": environment.search_count,
            "cumulative_score": environment.cumulative_score,
            "choice_history": list(environment.choice_history),
            "reward_history": list(environment.reward_history),
            "previous_outcome": previous_outcome,
            "p_stay": float(metrics["p_positive"]),
            "p_leave": float(metrics["p_negative"]),
            "p_continue": float(metrics["p_positive"]),
            "p_stop": float(metrics["p_negative"]),
            "persistence_logit": float(metrics["choice_logit"]),
            "target_logit": float(metrics["choice_logit"]),
            **{
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "p_positive",
                    "p_negative",
                    "choice_logit",
                    "positive_label",
                    "negative_label",
                }
            },
        }
        conversation.record_choice(semantic_choice)
        result = environment.step(semantic_choice)
        record.update(
            {
                "subsequent_reward": result.reward,
                "found_food": result.found_food,
                "terminated": result.terminated,
                "termination_reason": result.reason,
            }
        )
        records.append(record)
        if not result.terminated:
            conversation.record_feedback(result)
            previous_outcome = result.reward
    running = 0.0
    for record in reversed(records):
        running += float(record["subsequent_reward"])
        record["future_cumulative_return"] = running
    return {
        "task": "foraging",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "model_id": model.model_id,
        "mapping_id": mapping.mapping_id,
        "records": records,
        "activations": torch.stack(activations),
        "shape": "states x layers x hidden_width",
    }


def collect_control_episode(
    model,
    pair_id: str,
    left: int,
    right: int,
    mapping: LabelMapping,
    *,
    seed: int,
) -> dict:
    import torch

    episode_id = f"{pair_id}-{mapping.mapping_id}"
    conversation = [{"role": "user", "content": comparison_prompt(left, right, mapping)}]
    metrics = _binary_decision(
        model, conversation, mapping, capture_hidden_states=True
    )
    hidden = metrics.pop("hidden_states")
    semantic_choice = (
        LEFT_GREATER if metrics["p_positive"] >= metrics["p_negative"] else RIGHT_GREATER
    )
    correct_choice = LEFT_GREATER if left > right else RIGHT_GREATER
    record = {
        "task": "binary_control",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "state_id": f"{episode_id}:0",
        "seed": int(seed),
        "action_seed": int(seed),
        "round": 0,
        "conversation": conversation,
        "label_mapping": mapping.to_json(),
        "mapping_id": mapping.mapping_id,
        "positive_semantic": LEFT_GREATER,
        "negative_semantic": RIGHT_GREATER,
        "positive_label": mapping.positive_label,
        "negative_label": mapping.negative_label,
        "raw_label": mapping.label_for(semantic_choice),
        "semantic_choice": semantic_choice,
        "correct_choice": correct_choice,
        "is_correct": semantic_choice == correct_choice,
        "left_integer": int(left),
        "right_integer": int(right),
        "p_choice_one": float(metrics["p_positive"]),
        "p_choice_two": float(metrics["p_negative"]),
        "choice_logit": float(metrics["choice_logit"]),
        "target_logit": float(metrics["choice_logit"]),
        "terminated": True,
        "termination_reason": "single_judgment",
        **{
            key: value
            for key, value in metrics.items()
            if key
            not in {
                "p_positive",
                "p_negative",
                "choice_logit",
                "positive_label",
                "negative_label",
            }
        },
    }
    return {
        "task": "binary_control",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "model_id": model.model_id,
        "mapping_id": mapping.mapping_id,
        "records": [record],
        "activations": torch.stack(hidden).unsqueeze(0).to(dtype=torch.float16),
        "shape": "states x layers x hidden_width",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("foraging", "control"), required=True)
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    collection = config["collection"]
    args.model = args.model or config["model"]
    args.episodes = args.episodes or int(
        collection[
            "foraging_episodes" if args.task == "foraging" else "control_episodes"
        ]
    )
    args.max_decisions = args.max_decisions or int(collection["max_foraging_decisions"])
    args.seed = args.seed or int(config["split_seed"])
    if args.episodes < 2 or args.episodes % 2:
        raise ValueError("--episodes must be even so every condition has both mappings")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num-shards)")

    from models.hooked_qwen import HookedQwen

    output_dir = args.output_dir or (
        "artifacts/cross_task/foraging_activation_bank"
        if args.task == "foraging"
        else "artifacts/cross_task/control_activation_bank"
    )
    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    os.makedirs(output_dir, exist_ok=True)
    metadata_name = (
        "metadata.json"
        if args.num_shards == 1
        else f"metadata_shard_{args.shard_index:03d}.json"
    )
    save_run_metadata(
        os.path.join(output_dir, metadata_name),
        {**vars(args), "output_dir": output_dir, "counterbalanced_labels": ["X", "Y"]},
        model,
    )
    conditions = (
        foraging_conditions(
            args.episodes,
            args.seed,
            initial_qualities=collection["initial_qualities"],
            depletions=collection["depletions"],
            outside_options=collection["outside_options"],
            stay_costs=collection["stay_costs"],
        )
        if args.task == "foraging"
        else control_conditions(args.episodes, args.seed)
    )
    started = time.perf_counter()
    episodes_run = states_run = 0
    for index, condition in enumerate(conditions, 1):
        if (index - 1) % args.num_shards != args.shard_index:
            continue
        path = os.path.join(output_dir, f"episode_{index:05d}.pt")
        if os.path.exists(path):
            continue
        if args.task == "foraging":
            pair_id, ecology, mapping, seed, action_seed = condition
            artifact = collect_foraging_episode(
                model,
                pair_id,
                ecology,
                mapping,
                seed=seed,
                action_seed=action_seed,
                max_decisions=args.max_decisions,
                food_reward=int(collection["food_reward"]),
            )
        else:
            pair_id, left, right, mapping, seed = condition
            artifact = collect_control_episode(
                model, pair_id, left, right, mapping, seed=seed
            )
        atomic_torch_save(artifact, path)
        episodes_run += 1
        states_run += len(artifact["records"])
        print(
            f"collected {args.task} {index}/{args.episodes}: {artifact['episode_id']}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    print(
        f"runtime: {episodes_run} new episodes, {states_run} states, "
        f"{elapsed:.1f}s total, {states_run / elapsed if elapsed else 0:.2f} states/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
