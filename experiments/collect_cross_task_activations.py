"""Collect counterbalanced persistence-task and negative-control activations."""

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
from cross_task.generic_value import (
    LEFT_VOUCHER,
    RIGHT_VOUCHER,
    episode_conditions as generic_value_conditions,
    voucher_prompt,
)
from cross_task.solvability import (
    GIVE_UP,
    TRY_AGAIN,
    SolvabilityConversation,
    SolvabilityEnvironment,
    episode_conditions as solvability_conditions,
)
from cross_task.terminality import (
    END,
    PROCEED,
    correct_action as terminality_correct_action,
    episode_conditions as terminality_conditions,
    rule_prompt,
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


def collect_generic_value_episode(
    model,
    pair_id: str,
    left_value: int,
    right_value: int,
    mapping: LabelMapping,
    *,
    seed: int,
) -> dict:
    """Collect a one-shot value comparison without persistence semantics."""
    import torch

    episode_id = f"{pair_id}-{mapping.mapping_id}"
    conversation = [
        {"role": "user", "content": voucher_prompt(left_value, right_value, mapping)}
    ]
    metrics = _binary_decision(
        model, conversation, mapping, capture_hidden_states=True
    )
    hidden = metrics.pop("hidden_states")
    semantic_choice = (
        LEFT_VOUCHER
        if metrics["p_positive"] >= metrics["p_negative"]
        else RIGHT_VOUCHER
    )
    correct_choice = LEFT_VOUCHER if left_value > right_value else RIGHT_VOUCHER
    record = {
        "task": "generic_value_control",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "state_id": f"{episode_id}:0",
        "seed": int(seed),
        "action_seed": int(seed),
        "round": 0,
        "conversation": conversation,
        "label_mapping": mapping.to_json(),
        "mapping_id": mapping.mapping_id,
        "positive_semantic": LEFT_VOUCHER,
        "negative_semantic": RIGHT_VOUCHER,
        "positive_label": mapping.positive_label,
        "negative_label": mapping.negative_label,
        "raw_label": mapping.label_for(semantic_choice),
        "semantic_choice": semantic_choice,
        "correct_choice": correct_choice,
        "is_correct": semantic_choice == correct_choice,
        "left_value": int(left_value),
        "right_value": int(right_value),
        "relative_value": int(left_value) - int(right_value),
        "p_left": float(metrics["p_positive"]),
        "p_right": float(metrics["p_negative"]),
        "choice_logit": float(metrics["choice_logit"]),
        "target_logit": float(metrics["choice_logit"]),
        "terminated": True,
        "termination_reason": "one_shot_value_choice",
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
        "task": "generic_value_control",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "model_id": model.model_id,
        "mapping_id": mapping.mapping_id,
        "records": [record],
        "activations": torch.stack(hidden).unsqueeze(0).to(dtype=torch.float16),
        "shape": "states x layers x hidden_width",
    }


def collect_terminality_episode(
    model,
    pair_id: str,
    integer: int,
    mapping: LabelMapping,
    *,
    seed: int,
) -> dict:
    """Collect a rule-determined continue/end judgment with no value tradeoff."""
    import torch

    episode_id = f"{pair_id}-{mapping.mapping_id}"
    conversation = [{"role": "user", "content": rule_prompt(integer, mapping)}]
    metrics = _binary_decision(
        model, conversation, mapping, capture_hidden_states=True
    )
    hidden = metrics.pop("hidden_states")
    semantic_choice = (
        PROCEED if metrics["p_positive"] >= metrics["p_negative"] else END
    )
    correct = terminality_correct_action(integer)
    record = {
        "task": "terminality_control",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "state_id": f"{episode_id}:0",
        "seed": int(seed),
        "action_seed": int(seed),
        "round": 0,
        "conversation": conversation,
        "label_mapping": mapping.to_json(),
        "mapping_id": mapping.mapping_id,
        "positive_semantic": PROCEED,
        "negative_semantic": END,
        "positive_label": mapping.positive_label,
        "negative_label": mapping.negative_label,
        "raw_label": mapping.label_for(semantic_choice),
        "semantic_choice": semantic_choice,
        "correct_choice": correct,
        "is_correct": semantic_choice == correct,
        "displayed_integer": int(integer),
        "p_proceed": float(metrics["p_positive"]),
        "p_end": float(metrics["p_negative"]),
        "terminality_logit": float(metrics["choice_logit"]),
        "choice_logit": float(metrics["choice_logit"]),
        "target_logit": float(metrics["choice_logit"]),
        "terminated": True,
        "termination_reason": "rule_determined_judgment",
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
        "task": "terminality_control",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "model_id": model.model_id,
        "mapping_id": mapping.mapping_id,
        "records": [record],
        "activations": torch.stack(hidden).unsqueeze(0).to(dtype=torch.float16),
        "shape": "states x layers x hidden_width",
    }


def collect_solvability_episode(
    model,
    pair_id,
    condition,
    mapping,
    *,
    seed: int,
    action_seed: int,
    max_attempts: int = 8,
) -> dict:
    import torch

    environment = SolvabilityEnvironment(
        condition, seed, max_attempts=max_attempts
    )
    conversation = SolvabilityConversation.start(
        condition, mapping, max_attempts=max_attempts
    )
    rng = random.Random(action_seed)
    episode_id = f"{pair_id}-{mapping.mapping_id}"
    records, activations = [], []
    previous_progress = None
    while not environment.terminated:
        visible = conversation.snapshot()
        metrics = _binary_decision(
            model, visible, mapping, capture_hidden_states=True
        )
        hidden = metrics.pop("hidden_states")
        activations.append(torch.stack(hidden).to(dtype=torch.float16))
        semantic_choice = (
            TRY_AGAIN if rng.random() < metrics["p_positive"] else GIVE_UP
        )
        state_id = f"{episode_id}:{environment.decision}"
        record = {
            "task": "solvability",
            "episode_id": episode_id,
            "pair_id": pair_id,
            "state_id": state_id,
            "seed": int(seed),
            "action_seed": int(action_seed),
            "round": environment.decision,
            "conversation": visible,
            "label_mapping": mapping.to_json(),
            "mapping_id": mapping.mapping_id,
            "positive_semantic": TRY_AGAIN,
            "negative_semantic": GIVE_UP,
            "positive_label": mapping.positive_label,
            "negative_label": mapping.negative_label,
            "raw_label": mapping.label_for(semantic_choice),
            "semantic_choice": semantic_choice,
            "progress_probability": condition.progress_probability,
            "attempt_cost": condition.attempt_cost,
            "give_up_value": condition.give_up_value,
            "max_attempts": max_attempts,
            "attempts_used": environment.attempts,
            "cumulative_cost": environment.cumulative_cost,
            "progress_count": sum(environment.progress_history),
            "choice_history": list(environment.choice_history),
            "progress_history": list(environment.progress_history),
            "previous_progress": previous_progress,
            "p_try_again": float(metrics["p_positive"]),
            "p_give_up": float(metrics["p_negative"]),
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
                "progress_made": result.progress_made,
                "terminated": result.terminated,
                "termination_reason": result.reason,
            }
        )
        records.append(record)
        if not result.terminated:
            conversation.record_feedback(result)
            previous_progress = result.progress_made
    return {
        "task": "solvability",
        "episode_id": episode_id,
        "pair_id": pair_id,
        "model_id": model.model_id,
        "mapping_id": mapping.mapping_id,
        "records": records,
        "activations": torch.stack(activations),
        "shape": "states x layers x hidden_width",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("foraging", "solvability", "control", "terminality", "generic_value"),
        required=True,
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
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
    args.episodes = args.episodes or int(collection[f"{args.task}_episodes"])
    args.max_decisions = args.max_decisions or int(collection["max_foraging_decisions"])
    args.max_attempts = args.max_attempts or int(collection["max_solvability_attempts"])
    args.seed = args.seed or int(config["split_seed"])
    if args.episodes < 2 or args.episodes % 2:
        raise ValueError("--episodes must be even so every condition has both mappings")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num-shards)")

    from models.hooked_qwen import HookedQwen

    output_dir = args.output_dir or (
        f"artifacts/cross_task/{args.task}_activation_bank"
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
        {
            **vars(args),
            "output_dir": output_dir,
            "counterbalanced_labels": collection[f"{args.task}_response_labels"],
        },
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
            labels=collection["foraging_response_labels"],
        )
        if args.task == "foraging"
        else solvability_conditions(
            args.episodes,
            args.seed,
            progress_probabilities=collection["progress_probabilities"],
            attempt_costs=collection["attempt_costs"],
            give_up_values=collection["give_up_values"],
            labels=collection["solvability_response_labels"],
        )
        if args.task == "solvability"
        else control_conditions(
            args.episodes,
            args.seed,
            labels=collection["control_response_labels"],
        )
        if args.task == "control"
        else generic_value_conditions(
            args.episodes,
            args.seed,
            labels=collection["generic_value_response_labels"],
        )
        if args.task == "generic_value"
        else terminality_conditions(
            args.episodes,
            args.seed,
            labels=collection["terminality_response_labels"],
        )
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
        elif args.task == "solvability":
            pair_id, problem, mapping, seed, action_seed = condition
            artifact = collect_solvability_episode(
                model,
                pair_id,
                problem,
                mapping,
                seed=seed,
                action_seed=action_seed,
                max_attempts=args.max_attempts,
            )
        elif args.task == "control":
            pair_id, left, right, mapping, seed = condition
            artifact = collect_control_episode(
                model, pair_id, left, right, mapping, seed=seed
            )
        elif args.task == "generic_value":
            pair_id, left, right, mapping, seed = condition
            artifact = collect_generic_value_episode(
                model, pair_id, left, right, mapping, seed=seed
            )
        else:
            pair_id, integer, mapping, seed = condition
            artifact = collect_terminality_episode(
                model, pair_id, integer, mapping, seed=seed
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
