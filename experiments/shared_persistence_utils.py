"""Data adapters for semantic persistence probes across three tasks."""

from __future__ import annotations

import json

from experiments.cross_task_utils import (
    layer_dataset as cross_task_layer_dataset,
    load_activation_shards,
    make_or_validate_split,
)
from experiments.train_linear_probes import layer_dataset as bandit_layer_dataset
from experiments.train_value_probe import load_shards


TASKS = ("bandit", "foraging", "solvability")


def load_task_shards(task: str, directory: str) -> list[dict]:
    if task == "bandit":
        return load_shards(directory)
    if task not in TASKS:
        raise ValueError(f"unknown persistence task: {task!r}")
    shards = load_activation_shards(directory)
    observed = {str(shard["task"]) for shard in shards}
    if observed != {task}:
        raise ValueError(f"{task} bank contains tasks {sorted(observed)}")
    return shards


def load_task_split(
    task: str, shards: list[dict], path: str, *, seed: int
) -> dict[str, list[str]]:
    if task != "bandit":
        return make_or_validate_split(shards, path, seed=seed)
    with open(path, encoding="utf-8") as handle:
        split = json.load(handle)
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("bandit split must contain train, validation, and test")
    shard_ids = {str(shard["episode_id"]) for shard in shards}
    assigned = [str(episode) for values in split.values() for episode in values]
    if len(assigned) != len(set(assigned)) or set(assigned) != shard_ids:
        raise ValueError("bandit split does not exactly partition activation episodes")
    return {name: [str(value) for value in values] for name, values in split.items()}


def semantic_layer_dataset(
    task: str, shards: list[dict], layer: int, episode_ids: set[str]
) -> dict:
    if task == "bandit":
        data = bandit_layer_dataset(shards, layer, episode_ids)
        return {
            "states": data["states"],
            "target": data["targets"]["persistence"],
            "records": data["records"],
        }
    return cross_task_layer_dataset(
        shards, layer, episode_ids, target_key="persistence_logit"
    )


def activation_shape(shards: list[dict]) -> tuple[int, int]:
    first = shards[0]["activations"]
    return int(first.shape[1]), int(first.shape[2])


def validate_compatible_tasks(shards_by_task: dict[str, list[dict]]) -> tuple[int, int]:
    shapes = {task: activation_shape(shards) for task, shards in shards_by_task.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"task activation shapes are incompatible: {shapes}")
    return next(iter(shapes.values()))
