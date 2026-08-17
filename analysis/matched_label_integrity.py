"""Integrity checks for exact semantic-history label-counterfactual replays."""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict


SEMANTICS = {
    "foraging": ("STAY", "LEAVE"),
    "solvability": ("TRY_AGAIN", "GIVE_UP"),
}


def _mapping(record: dict) -> dict[str, str]:
    value = record.get("label_mapping")
    try:
        return json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _history_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def audit_matched_label_shards(shards: list[dict], task: str) -> dict:
    if task not in SEMANTICS:
        raise ValueError(f"matched label replay does not support {task!r}")
    positive, negative = SEMANTICS[task]
    expected_task = f"{task}_label_replay"
    groups: dict[str, list[dict]] = defaultdict(list)
    issues: Counter[str] = Counter()
    for shard in shards:
        if shard.get("task") != expected_task:
            issues["task"] += 1
        records = shard.get("records", ())
        if len(records) != 1:
            issues["one_state_per_variant"] += 1
            continue
        record = records[0]
        groups[str(shard.get("pair_id", ""))].append(record)
        if record.get("episode_id") != shard.get("episode_id"):
            issues["episode_id"] += 1
        if record.get("pair_id") != shard.get("pair_id"):
            issues["pair_id"] += 1
    for records in groups.values():
        if len(records) != 2:
            issues["pair_size"] += 1
            continue
        left, right = records
        left_mapping, right_mapping = _mapping(left), _mapping(right)
        if (
            set(left_mapping) != {positive, negative}
            or set(right_mapping) != {positive, negative}
            or left_mapping.get(positive) != right_mapping.get(negative)
            or left_mapping.get(negative) != right_mapping.get(positive)
        ):
            issues["mapping_reversal"] += 1
        if len({row.get("matched_history_hash") for row in records}) != 1:
            issues["semantic_history"] += 1
        histories = [row.get("matched_history") for row in records]
        if (
            not all(isinstance(history, dict) for history in histories)
            or histories[0] != histories[1]
            or any(
                _history_hash(history) != row.get("matched_history_hash")
                for history, row in zip(histories, records)
                if isinstance(history, dict)
            )
        ):
            issues["semantic_history_payload"] += 1
        if len({row.get("source_state_id") for row in records}) != 1:
            issues["source_state"] += 1
    return {
        "passed": not issues,
        "task": task,
        "matched_histories": len(groups),
        "variants": len(shards),
        "issue_counts": dict(sorted(issues.items())),
    }
