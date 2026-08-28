"""Stream existing all-layer banks into local-only aligned float16 memmaps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def validate_activation_shard(records, activations, *, expected_layers):
    shape = tuple(activations.shape)
    if len(shape) != 3:
        raise ValueError("activation shard must be states x layers x width")
    if shape[0] != len(records):
        raise ValueError("activation shard record count does not match tensor")
    if shape[1] != int(expected_layers):
        raise ValueError(
            f"activation shard layer count {shape[1]} != {expected_layers}"
        )
    return activations


def _state_order_hash(records):
    payload = "\n".join(str(row["state_id"]) for row in records).encode()
    return hashlib.sha256(payload).hexdigest()


def _smoke_records(records, maximum):
    if maximum is None or len(records) <= int(maximum):
        return list(records)
    frame = pd.DataFrame(records)
    parts = []
    per_split = max(1, int(maximum) // max(1, frame.split.nunique()))
    for _split, part in frame.groupby("split", sort=False):
        parts.append(part.sort_values(["episode_id", "round"]).head(per_split))
    selected = pd.concat(parts, ignore_index=True).head(int(maximum))
    wanted = set(selected.state_id.astype(str))
    return [row for row in records if str(row["state_id"]) in wanted]


@dataclass
class ActivationDataset:
    task: str
    metadata: pd.DataFrame
    mmap_path: Path
    shape: tuple[int, int, int]

    def open(self):
        return np.memmap(
            self.mmap_path, dtype=np.float16, mode="r", shape=self.shape
        )


def build_activation_cache(
    task,
    records,
    activation_dir,
    cache_dir,
    *,
    expected_layers=32,
    maximum_states=None,
    resume=False,
    logger=None,
):
    """Align records by state ID while scanning every source shard exactly once."""

    import torch

    task = str(task)
    records = _smoke_records([dict(row) for row in records], maximum_states)
    state_ids = [str(row["state_id"]) for row in records]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError(f"duplicate requested state IDs for {task}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / f"{task}_manifest.json"
    metadata_path = cache_dir / f"{task}_metadata.csv.gz"
    mmap_path = cache_dir / f"{task}_hidden.float16.mmap"
    order_hash = _state_order_hash(records)
    if resume and manifest_path.exists() and metadata_path.exists() and mmap_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("state_order_sha256") == order_hash
            and int(manifest.get("layers", -1)) == int(expected_layers)
        ):
            shape = tuple(int(value) for value in manifest["shape"])
            if logger is not None:
                logger.note("activation_cache", f"reusing {task} cache {shape}")
            return ActivationDataset(
                task, pd.read_csv(metadata_path), mmap_path, shape
            )

    files = sorted(Path(activation_dir).glob("episode_*.pt"))
    if not files:
        raise FileNotFoundError(f"no activation shards found for {task}: {activation_dir}")
    first = torch.load(files[0], map_location="cpu", weights_only=False)
    first_tensor = validate_activation_shard(
        first["records"], first["activations"], expected_layers=expected_layers
    )
    width = int(first_tensor.shape[2])
    shape = (len(records), int(expected_layers), width)
    hidden = np.memmap(mmap_path, dtype=np.float16, mode="w+", shape=shape)
    lookup = {state_id: index for index, state_id in enumerate(state_ids)}
    found = np.zeros(len(records), dtype=bool)
    for file_index, path in enumerate(files, start=1):
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(
                f"could not load {path}; ensure Git LFS objects are materialized"
            ) from error
        tensor = validate_activation_shard(
            artifact["records"], artifact["activations"], expected_layers=expected_layers
        )
        source_indices, destination_indices = [], []
        for source_index, source in enumerate(artifact["records"]):
            destination = lookup.get(str(source["state_id"]))
            if destination is None:
                continue
            if found[destination]:
                raise ValueError(f"duplicate activation state: {source['state_id']}")
            source_indices.append(source_index)
            destination_indices.append(destination)
        if source_indices:
            values = tensor[source_indices].detach().cpu().float().numpy()
            destination_indices = np.asarray(destination_indices, dtype=int)
            hidden[destination_indices] = values.astype(np.float16)
            found[destination_indices] = True
        if logger is not None and (
            file_index == len(files) or file_index % 100 == 0
        ):
            logger.note(
                "activation_cache",
                f"{task}: {file_index}/{len(files)} shards; {int(found.sum())}/{len(found)} states",
            )
        del artifact, tensor
    missing = [state_ids[index] for index in np.flatnonzero(~found)]
    if missing:
        raise ValueError(f"{task} activation bank is missing states: {missing[:5]}")
    hidden.flush()
    pd.DataFrame(records).to_csv(metadata_path, index=False, compression="gzip")
    manifest = {
        "task": task,
        "shape": list(shape),
        "states": len(records),
        "layers": int(expected_layers),
        "width": width,
        "dtype": "float16",
        "state_order_sha256": order_hash,
        "source": str(activation_dir),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ActivationDataset(task, pd.DataFrame(records), mmap_path, shape)


def read_bank_records(directory):
    """Read record metadata from an existing bank without retaining tensors."""

    import torch

    records = []
    files = sorted(Path(directory).glob("episode_*.pt"))
    if not files:
        raise FileNotFoundError(directory)
    for path in files:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(dict(row) for row in artifact["records"])
    return records

