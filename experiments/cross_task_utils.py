"""Activation-bank and split utilities shared by cross-task programs."""

import glob
import json
import os

from cross_task.common import grouped_episode_split


def load_activation_shards(directory: str) -> list[dict]:
    import torch

    paths = sorted(glob.glob(os.path.join(directory, "episode_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no activation shards found under {directory}")
    shards, episode_ids = [], set()
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        required = {"task", "episode_id", "pair_id", "records", "activations"}
        missing = required - set(shard)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        if shard["episode_id"] in episode_ids:
            raise ValueError(f"duplicate episode shard: {shard['episode_id']}")
        episode_ids.add(shard["episode_id"])
        if len(shard["records"]) != int(shard["activations"].shape[0]):
            raise ValueError(f"record/activation mismatch in {path}")
        if any(record["episode_id"] != shard["episode_id"] for record in shard["records"]):
            raise ValueError(f"record episode mismatch in {path}")
        shards.append(shard)
    tasks = {shard["task"] for shard in shards}
    if len(tasks) != 1:
        raise ValueError(f"activation directory mixes tasks: {sorted(tasks)}")
    return shards


def make_or_validate_split(
    shards: list[dict], path: str, *, seed: int
) -> dict[str, list[str]]:
    episode_to_group = {
        shard["episode_id"]: shard["pair_id"] for shard in shards
    }
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            split = json.load(handle)
    else:
        split = grouped_episode_split(episode_to_group, seed=seed)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(split, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("split must contain train, validation, and test")
    assigned = [episode for name in split for episode in split[name]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(episode_to_group):
        raise ValueError("split does not exactly partition activation episodes")
    for name, episodes in split.items():
        groups = {episode_to_group[episode] for episode in episodes}
        other = {
            episode_to_group[episode]
            for other_name, selected in split.items()
            if other_name != name
            for episode in selected
        }
        if groups & other:
            raise ValueError("a counterbalanced pair crosses split boundaries")
    return split


def layer_dataset(
    shards: list[dict], layer: int, episode_ids: set[str], *, target_key: str
) -> dict:
    import torch

    states, targets, records = [], [], []
    for shard in shards:
        if shard["episode_id"] not in episode_ids:
            continue
        if not 0 <= layer < int(shard["activations"].shape[1]):
            raise IndexError(f"layer {layer} is absent from {shard['episode_id']}")
        states.append(shard["activations"][:, layer, :].float())
        for record in shard["records"]:
            if target_key not in record or record[target_key] is None:
                raise ValueError(
                    f"record {record['state_id']} has no target {target_key!r}"
                )
            targets.append(float(record[target_key]))
            records.append(record)
    if not states:
        raise ValueError("requested split contains no activation states")
    return {
        "states": torch.cat(states),
        "target": torch.tensor(targets, dtype=torch.float32),
        "records": records,
    }


def probe_layer(payload: dict, path: str) -> int:
    metadata = payload.get("metadata", {})
    layer = metadata.get("selected_layer", metadata.get("layer"))
    if layer is None:
        raise ValueError(f"probe artifact does not identify its layer: {path}")
    return int(layer)
