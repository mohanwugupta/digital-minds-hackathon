"""Matched persistence-contrast construction and behavioral gates.

The module is dependency-lazy: validation and inventory work can run without
PyTorch, while activation-bank construction imports it only when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import statistics
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ContrastDefinition:
    """Declare one causal reason for continued pursuit to become warranted."""

    task: str
    manipulation: str
    factor: str
    higher_promotes_persistence: bool
    matched_fields: tuple[str, ...]
    target_field: str = "persistence_logit"
    contrast_kind: str = "persistence"

    def __post_init__(self) -> None:
        if not self.task or not self.manipulation or not self.factor:
            raise ValueError("contrast task, manipulation, and factor are required")
        if self.factor in self.matched_fields:
            raise ValueError("the manipulated factor cannot also be a matched field")
        if len(set(self.matched_fields)) != len(self.matched_fields):
            raise ValueError("matched fields must be unique")
        if self.contrast_kind not in {"persistence", "nuisance"}:
            raise ValueError("contrast kind must be persistence or nuisance")


def _canonical(value):
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    return value


def classify_pair(left: Mapping, right: Mapping) -> str:
    """Recognize a semantic-state-preserving response-label swap."""

    label_fields = {
        "label_mapping",
        "mapping_id",
        "positive_label",
        "negative_label",
        "raw_label",
    }
    identity_fields = {"state_id", "episode_id", "pair_id", "seed", "action_seed"}
    shared = set(left) & set(right)
    semantic_fields = shared - label_fields - identity_fields
    semantic_equal = all(
        _canonical(left[field]) == _canonical(right[field]) for field in semantic_fields
    )
    label_changed = any(
        field in shared and _canonical(left[field]) != _canonical(right[field])
        for field in label_fields
    )
    if semantic_equal and label_changed:
        return "label_identity_nuisance"
    return "persistence_candidate_pair"


def validate_contrast_pair(
    promoting: Mapping,
    discouraging: Mapping,
    definition: ContrastDefinition,
) -> dict:
    """Validate a pair whose first member must promote persistence."""

    if str(promoting.get("task")) != definition.task or str(
        discouraging.get("task")
    ) != definition.task:
        raise ValueError("contrast records do not match the declared task")
    missing = [
        field
        for field in (definition.factor, *definition.matched_fields, definition.target_field)
        if field not in promoting or field not in discouraging
    ]
    if missing:
        raise ValueError(f"contrast records are missing fields: {sorted(set(missing))}")
    for field in definition.matched_fields:
        if _canonical(promoting[field]) != _canonical(discouraging[field]):
            raise ValueError(f"matched field {field!r} differs across contrast pair")
    positive_value = float(promoting[definition.factor])
    negative_value = float(discouraging[definition.factor])
    correctly_oriented = (
        positive_value > negative_value
        if definition.higher_promotes_persistence
        else positive_value < negative_value
    )
    if not correctly_oriented:
        raise ValueError(
            "contrast orientation is reversed: first record must be more "
            "persistence-promoting"
        )
    effect = float(promoting[definition.target_field]) - float(
        discouraging[definition.target_field]
    )
    return {
        "task": definition.task,
        "manipulation": definition.manipulation,
        "orientation": "positive_is_more_persistence_promoting",
        "behavior_effect": effect,
    }


def equal_task_manipulation_weights(rows: Sequence[Mapping]) -> list[float]:
    """Give each task and then each family within task equal aggregate mass."""

    if not rows:
        raise ValueError("cannot weight an empty contrast collection")
    groups: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(rows):
        task = str(row["task"])
        manipulation = str(row["manipulation"])
        groups.setdefault(task, {}).setdefault(manipulation, []).append(index)
    task_mass = 1.0 / len(groups)
    weights = [0.0] * len(rows)
    for families in groups.values():
        family_mass = task_mass / len(families)
        for indices in families.values():
            item_mass = family_mass / len(indices)
            for index in indices:
                weights[index] = item_mass
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
        raise RuntimeError("balanced contrast weights do not sum to one")
    return weights


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return float(ordered[index])


def behavioral_validity_gate(
    rows: Sequence[Mapping], *, bootstrap_samples: int = 2000, seed: int = 0
) -> dict:
    """Cluster-bootstrap the mean behavioral contrast and require a positive CI."""

    if not rows or bootstrap_samples < 20:
        raise ValueError("behavioral gate requires rows and at least 20 bootstrap draws")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        effect = float(row["behavior_effect"])
        if not math.isfinite(effect):
            raise ValueError("behavioral effects must be finite")
        grouped.setdefault(str(row["cluster_id"]), []).append(effect)
    cluster_ids = sorted(grouped)
    cluster_means = {
        cluster: statistics.mean(grouped[cluster]) for cluster in cluster_ids
    }
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        draws.append(statistics.mean(cluster_means[cluster] for cluster in selected))
    mean_effect = statistics.mean(float(row["behavior_effect"]) for row in rows)
    lower, upper = _percentile(draws, 0.025), _percentile(draws, 0.975)
    return {
        "passed": mean_effect > 0 and lower > 0,
        "mean_effect": mean_effect,
        "clustered_95_ci": [lower, upper],
        "clusters": len(cluster_ids),
        "pairs": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "orientation": "positive_is_more_persistence_promoting",
    }


def flatten_activation_shards(shards: Iterable[Mapping]) -> list[dict]:
    """Attach every record to its corresponding all-layer activation tensor."""

    rows = []
    for shard in shards:
        activations = shard["activations"]
        records = shard["records"]
        if len(records) != int(activations.shape[0]):
            raise ValueError("record/activation count mismatch while flattening shards")
        for index, source in enumerate(records):
            row = dict(source)
            row["activation"] = activations[index].float()
            rows.append(row)
    return rows


def _stable_pair_id(definition: ContrastDefinition, left: Mapping, right: Mapping) -> str:
    payload = {
        "task": definition.task,
        "manipulation": definition.manipulation,
        "positive": left["state_id"],
        "negative": right["state_id"],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"{definition.task}-{definition.manipulation}-{digest}"


def build_matched_record_contrasts(
    records: Sequence[Mapping],
    definition: ContrastDefinition,
    *,
    episode_to_split: Mapping[str, str] | None = None,
) -> list[dict]:
    """Create deterministic endpoint contrasts within exact matched strata.

    Repeated observations at each factor endpoint are paired in stable state-ID
    order. Including split in the stratum guarantees that no contrast crosses a
    train/validation/test boundary.
    """

    import torch

    strata: dict[tuple, dict[float, list[Mapping]]] = {}
    for record in records:
        if str(record.get("task")) != definition.task:
            continue
        episode = str(record["episode_id"])
        split = episode_to_split.get(episode) if episode_to_split else "unsplit"
        if split is None:
            raise ValueError(f"episode {episode!r} is absent from the split")
        key = (split,) + tuple(_canonical(record[field]) for field in definition.matched_fields)
        strata.setdefault(key, {}).setdefault(float(record[definition.factor]), []).append(record)

    result = []
    for key, levels in sorted(strata.items(), key=lambda item: repr(item[0])):
        if len(levels) < 2:
            continue
        low, high = min(levels), max(levels)
        promoting_level, discouraging_level = (
            (high, low) if definition.higher_promotes_persistence else (low, high)
        )
        positive_rows = sorted(levels[promoting_level], key=lambda row: str(row["state_id"]))
        negative_rows = sorted(levels[discouraging_level], key=lambda row: str(row["state_id"]))
        for promoting, discouraging in zip(positive_rows, negative_rows):
            validation = validate_contrast_pair(promoting, discouraging, definition)
            positive_activation = promoting["activation"].float()
            negative_activation = discouraging["activation"].float()
            if positive_activation.shape != negative_activation.shape or positive_activation.ndim != 2:
                raise ValueError("matched activations must have identical layers x width shape")
            pair_id = _stable_pair_id(definition, promoting, discouraging)
            clusters = sorted(
                {str(promoting.get("pair_id", promoting["episode_id"])), str(discouraging.get("pair_id", discouraging["episode_id"]))}
            )
            result.append(
                {
                    **validation,
                    "contrast_id": pair_id,
                    "positive_state_id": str(promoting["state_id"]),
                    "negative_state_id": str(discouraging["state_id"]),
                    "positive_episode_id": str(promoting["episode_id"]),
                    "negative_episode_id": str(discouraging["episode_id"]),
                    "cluster_id": "|".join(clusters),
                    "split": key[0],
                    "factor": definition.factor,
                    "positive_factor_value": promoting_level,
                    "negative_factor_value": discouraging_level,
                    "mapping_id": promoting.get("mapping_id"),
                    "matched_variables": {
                        field: promoting[field] for field in definition.matched_fields
                    },
                    "activation_delta": torch.sub(positive_activation, negative_activation),
                    "layers": int(positive_activation.shape[0]),
                    "hidden_width": int(positive_activation.shape[1]),
                }
            )
    return result


def build_exact_label_contrasts(
    records: Sequence[Mapping],
    *,
    semantic_fields: Sequence[str],
    episode_to_split: Mapping[str, str] | None = None,
) -> list[dict]:
    """Build exact semantic-state-preserving response-mapping contrasts."""

    import torch

    strata: dict[tuple, list[Mapping]] = {}
    for record in records:
        episode = str(record["episode_id"])
        split = episode_to_split.get(episode) if episode_to_split else "unsplit"
        key = (split,) + tuple(_canonical(record[field]) for field in semantic_fields)
        strata.setdefault(key, []).append(record)
    output = []
    for key, candidates in sorted(strata.items(), key=lambda item: repr(item[0])):
        by_mapping: dict[str, list[Mapping]] = {}
        for record in candidates:
            by_mapping.setdefault(str(record["mapping_id"]), []).append(record)
        if len(by_mapping) != 2:
            continue
        mappings = sorted(by_mapping)
        left_rows = sorted(by_mapping[mappings[0]], key=lambda row: str(row["state_id"]))
        right_rows = sorted(by_mapping[mappings[1]], key=lambda row: str(row["state_id"]))
        for left, right in zip(left_rows, right_rows):
            if any(_canonical(left[field]) != _canonical(right[field]) for field in semantic_fields):
                raise ValueError("label nuisance pair changed semantic state")
            if left["activation"].shape != right["activation"].shape:
                raise ValueError("label nuisance activation shapes differ")
            digest = hashlib.sha256(
                f"{left['state_id']}|{right['state_id']}".encode()
            ).hexdigest()[:20]
            output.append(
                {
                    "task": str(left["task"]),
                    "manipulation": "label_identity",
                    "contrast_kind": "nuisance",
                    "nuisance_type": "label",
                    "contrast_id": f"label-{digest}",
                    "positive_state_id": str(left["state_id"]),
                    "negative_state_id": str(right["state_id"]),
                    "cluster_id": str(left.get("pair_id", left["episode_id"])),
                    "split": key[0],
                    "activation_delta": torch.sub(
                        left["activation"].float(), right["activation"].float()
                    ),
                    "orientation": "arbitrary_mapping_order_nuisance_magnitude_only",
                    "behavior_effect": float(left.get("target_logit", left.get("persistence_logit", 0.0)))
                    - float(right.get("target_logit", right.get("persistence_logit", 0.0))),
                    "mapping_pair": mappings,
                    "matched_variables": {field: left[field] for field in semantic_fields},
                }
            )
    return output


def build_balanced_semantic_nuisance_contrasts(
    records: Sequence[Mapping],
    *,
    nuisance_type: str,
    positive_semantic: str,
    negative_semantic: str,
    sort_key,
    episode_to_split: Mapping[str, str] | None = None,
) -> list[dict]:
    """Pair balanced one-shot semantic classes within split and label mapping.

    These controls are intentionally classified as nuisance contrasts and are
    never allowed into persistence discovery or layer selection.
    """

    import torch

    grouped: dict[tuple[str, str], dict[str, list[Mapping]]] = {}
    for record in records:
        episode = str(record["episode_id"])
        split = episode_to_split.get(episode) if episode_to_split else "unsplit"
        semantic = str(record.get("correct_choice", record.get("semantic_choice")))
        if semantic not in {positive_semantic, negative_semantic}:
            continue
        key = (str(split), str(record["mapping_id"]))
        grouped.setdefault(key, {}).setdefault(semantic, []).append(record)
    output = []
    for (split, mapping_id), classes in sorted(grouped.items()):
        if positive_semantic not in classes or negative_semantic not in classes:
            continue
        positive_rows = sorted(classes[positive_semantic], key=sort_key)
        negative_rows = sorted(classes[negative_semantic], key=sort_key)
        for positive, negative in zip(positive_rows, negative_rows):
            if positive["activation"].shape != negative["activation"].shape:
                raise ValueError("nuisance activation shapes differ")
            digest = hashlib.sha256(
                f"{positive['state_id']}|{negative['state_id']}".encode()
            ).hexdigest()[:20]
            output.append(
                {
                    "task": str(positive["task"]),
                    "manipulation": nuisance_type,
                    "contrast_kind": "nuisance",
                    "nuisance_type": nuisance_type,
                    "contrast_id": f"{nuisance_type}-{digest}",
                    "positive_state_id": str(positive["state_id"]),
                    "negative_state_id": str(negative["state_id"]),
                    "cluster_id": "|".join(
                        sorted(
                            {
                                str(positive.get("pair_id", positive["episode_id"])),
                                str(negative.get("pair_id", negative["episode_id"])),
                            }
                        )
                    ),
                    "split": split,
                    "mapping_id": mapping_id,
                    "activation_delta": torch.sub(
                        positive["activation"].float(), negative["activation"].float()
                    ),
                    "orientation": f"positive_semantic_is_{positive_semantic}",
                    "behavior_effect": float(
                        positive.get("target_logit", positive.get("choice_logit", 0.0))
                    )
                    - float(
                        negative.get("target_logit", negative.get("choice_logit", 0.0))
                    ),
                    "unmatched_variables": "surface stimulus; balanced semantic-class control",
                }
            )
    return output
