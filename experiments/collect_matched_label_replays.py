"""Replay identical semantic histories under both raw-label mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from cross_task.common import counterbalanced_mappings
from cross_task.foraging import (
    LEAVE,
    STAY,
    ForagingCondition,
    ForagingConversation,
    ForagingEnvironment,
)
from cross_task.solvability import (
    GIVE_UP,
    TRY_AGAIN,
    SolvabilityCondition,
    SolvabilityConversation,
    SolvabilityEnvironment,
)
from experiments.collect_cross_task_activations import _binary_decision
from experiments.cross_task_utils import load_activation_shards, make_or_validate_split
from experiments.runtime import atomic_torch_save, save_run_metadata


def _history_payload(task: str, record: dict) -> dict:
    common = {
        "task": task,
        "source_state_id": record["state_id"],
        "round": int(record["round"]),
        "choice_history": record["choice_history"],
    }
    if task == "foraging":
        common.update(
            {
                "reward_history": record["reward_history"],
                "cumulative_score": record["cumulative_score"],
                "initial_quality": record["initial_quality"],
                "depletion": record["depletion"],
                "outside_option": record["outside_option"],
                "stay_cost": record["stay_cost"],
            }
        )
    else:
        common.update(
            {
                "progress_history": record["progress_history"],
                "cumulative_cost": record["cumulative_cost"],
                "progress_probability": record["progress_probability"],
                "attempt_cost": record["attempt_cost"],
                "give_up_value": record["give_up_value"],
            }
        )
    return common


def _hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _conversation(value) -> list[dict]:
    return json.loads(value) if isinstance(value, str) else value


def _foraging_messages(shard: dict, position: int, mapping, config: dict):
    first = shard["records"][0]
    condition = ForagingCondition(
        float(first["initial_quality"]),
        float(first["depletion"]),
        int(first["outside_option"]),
        int(first["stay_cost"]),
    )
    environment = ForagingEnvironment(
        condition,
        int(first["seed"]),
        max_decisions=int(config["max_foraging_decisions"]),
        food_reward=int(config["food_reward"]),
    )
    conversation = ForagingConversation.start(
        condition, mapping, food_reward=int(config["food_reward"])
    )
    for source in shard["records"][:position]:
        choice = str(source["semantic_choice"])
        conversation.record_choice(choice)
        step = environment.step(choice)
        if step.terminated:
            raise ValueError("source foraging history terminates before replay state")
        conversation.record_feedback(step)
    return conversation.snapshot()


def _solvability_messages(shard: dict, position: int, mapping, config: dict):
    first = shard["records"][0]
    condition = SolvabilityCondition(
        float(first["progress_probability"]),
        int(first["attempt_cost"]),
        int(first["give_up_value"]),
    )
    environment = SolvabilityEnvironment(
        condition,
        int(first["seed"]),
        max_attempts=int(config["max_solvability_attempts"]),
    )
    conversation = SolvabilityConversation.start(
        condition, mapping, max_attempts=int(config["max_solvability_attempts"])
    )
    for source in shard["records"][:position]:
        choice = str(source["semantic_choice"])
        conversation.record_choice(choice)
        step = environment.step(choice)
        if step.terminated:
            raise ValueError("source solvability history terminates before replay state")
        conversation.record_feedback(step)
    return conversation.snapshot()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("foraging", "solvability"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-states", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must fall within [0, num-shards)")

    import torch
    import yaml
    from models.hooked_qwen import HookedQwen

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    collection = config["collection"]
    shards = load_activation_shards(args.activation_dir)
    split = make_or_validate_split(shards, args.split, seed=int(config["split_seed"]))
    test_ids = set(split["test"])
    by_pair = {}
    for shard in shards:
        if shard["episode_id"] in test_ids:
            by_pair.setdefault(shard["pair_id"], []).append(shard)
    canonical = []
    for pair_id, paired in sorted(by_pair.items()):
        if len(paired) != 2:
            raise ValueError(f"test pair {pair_id} does not have both mappings")
        selected = min(paired, key=lambda shard: str(shard["mapping_id"]))
        canonical.extend((selected, position) for position in range(len(selected["records"])))
    if args.maximum_states > 0:
        canonical = canonical[: args.maximum_states]
    if not canonical:
        raise ValueError("no held-out semantic histories selected for label replay")

    if args.task == "foraging":
        positive, negative = STAY, LEAVE
        messages = _foraging_messages
    else:
        positive, negative = TRY_AGAIN, GIVE_UP
        messages = _solvability_messages
    mappings = counterbalanced_mappings(
        positive,
        negative,
        tuple(collection[f"{args.task}_response_labels"]),
    )
    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_metadata(
        os.path.join(args.output_dir, f"metadata_shard_{args.shard_index:03d}.json"),
        {
            **vars(args),
            "source_scope": "one canonical organic trajectory per held-out pair",
            "selected_histories": len(canonical),
        },
        model,
    )
    written = 0
    for history_index, (source_shard, position) in enumerate(canonical):
        if history_index % args.num_shards != args.shard_index:
            continue
        source = source_shard["records"][position]
        payload = _history_payload(args.task, source)
        history_hash = _hash(payload)
        matched_pair = f"{args.task}-matched-{history_hash[:20]}"
        source_mapping = next(
            mapping
            for mapping in mappings
            if mapping.mapping_id == source_shard["mapping_id"]
        )
        reconstructed_source = messages(
            source_shard, position, source_mapping, collection
        )
        if reconstructed_source != _conversation(source["conversation"]):
            raise ValueError(
                f"exact semantic replay drifted from source state {source['state_id']}"
            )
        for mapping_index, mapping in enumerate(mappings):
            episode_id = f"{matched_pair}-{mapping.mapping_id}"
            path = os.path.join(
                args.output_dir,
                f"episode_{history_index:06d}_{mapping_index}.pt",
            )
            if os.path.exists(path):
                continue
            visible = messages(source_shard, position, mapping, collection)
            metrics = _binary_decision(
                model, visible, mapping, capture_hidden_states=True
            )
            hidden = metrics.pop("hidden_states")
            record = {
                "task": f"{args.task}_label_replay",
                "episode_id": episode_id,
                "pair_id": matched_pair,
                "state_id": f"{episode_id}:0",
                "mapping_id": mapping.mapping_id,
                "label_mapping": mapping.to_json(),
                "positive_semantic": positive,
                "negative_semantic": negative,
                "positive_label": mapping.positive_label,
                "negative_label": mapping.negative_label,
                "conversation": visible,
                "source_state_id": source["state_id"],
                "source_episode_id": source["episode_id"],
                "source_pair_id": source["pair_id"],
                "source_round": int(source["round"]),
                "matched_history_hash": history_hash,
                "matched_history": payload,
                "persistence_logit": float(metrics["choice_logit"]),
                "p_continue": float(metrics["p_positive"]),
                "p_disengage": float(metrics["p_negative"]),
            }
            artifact = {
                "task": f"{args.task}_label_replay",
                "episode_id": episode_id,
                "pair_id": matched_pair,
                "mapping_id": mapping.mapping_id,
                "model_id": model.model_id,
                "records": [record],
                "activations": torch.stack(hidden)
                .unsqueeze(0)
                .to(dtype=torch.float16),
                "shape": "states x layers x hidden_width",
            }
            atomic_torch_save(artifact, path)
            written += 1
        print(
            f"matched {args.task} label replay {history_index + 1}/{len(canonical)}",
            flush=True,
        )
    print(f"wrote {written} matched-label variants", flush=True)


if __name__ == "__main__":
    main()
