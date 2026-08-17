"""Shared counterbalancing, serialization, and split helpers."""

from dataclasses import dataclass
import json
import random
from typing import Iterable


@dataclass(frozen=True)
class LabelMapping:
    """A reversible mapping from two semantic choices to raw response tokens."""

    positive_semantic: str
    negative_semantic: str
    positive_label: str
    negative_label: str

    def __post_init__(self) -> None:
        if self.positive_semantic == self.negative_semantic:
            raise ValueError("semantic choices must be distinct")
        if self.positive_label == self.negative_label:
            raise ValueError("response labels must be distinct")

    @property
    def labels(self) -> tuple[str, str]:
        # Keep the raw-token order stable across mapping reversals.
        return tuple(sorted((self.positive_label, self.negative_label)))

    @property
    def mapping_id(self) -> str:
        return f"{self.positive_semantic.lower()}_{self.positive_label.lower()}"

    def label_for(self, semantic_choice: str) -> str:
        if semantic_choice == self.positive_semantic:
            return self.positive_label
        if semantic_choice == self.negative_semantic:
            return self.negative_label
        raise ValueError(f"unknown semantic choice: {semantic_choice!r}")

    def semantic_for(self, label: str) -> str:
        if label == self.positive_label:
            return self.positive_semantic
        if label == self.negative_label:
            return self.negative_semantic
        raise ValueError(f"unknown response label: {label!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            self.positive_semantic: self.positive_label,
            self.negative_semantic: self.negative_label,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def counterbalanced_mappings(
    positive_semantic: str,
    negative_semantic: str,
    labels: tuple[str, str] = ("X", "Y"),
) -> tuple[LabelMapping, LabelMapping]:
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError("counterbalancing requires two distinct labels")
    return (
        LabelMapping(positive_semantic, negative_semantic, labels[0], labels[1]),
        LabelMapping(positive_semantic, negative_semantic, labels[1], labels[0]),
    )


def grouped_episode_split(
    episode_to_group: dict[str, str], seed: int = 0
) -> dict[str, list[str]]:
    """Split episodes while keeping counterbalanced pairs in one partition."""
    if not episode_to_group:
        raise ValueError("cannot split an empty episode mapping")
    groups: dict[str, list[str]] = {}
    for episode_id, group_id in episode_to_group.items():
        groups.setdefault(group_id, []).append(episode_id)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    n_train = int(len(group_ids) * 0.70)
    n_validation = int(len(group_ids) * 0.15)
    group_split = {
        "train": group_ids[:n_train],
        "validation": group_ids[n_train : n_train + n_validation],
        "test": group_ids[n_train + n_validation :],
    }
    result = {
        name: sorted(
            episode_id
            for group_id in selected
            for episode_id in groups[group_id]
        )
        for name, selected in group_split.items()
    }
    assigned = [episode for selected in result.values() for episode in selected]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(episode_to_group):
        raise RuntimeError("grouped split lost or duplicated episodes")
    return result


def parse_json_mapping(value: str | dict) -> dict[str, str]:
    mapping = json.loads(value) if isinstance(value, str) else dict(value)
    if len(mapping) != 2 or len(set(mapping.values())) != 2:
        raise ValueError("serialized mapping must contain two distinct labels")
    return {str(key): str(label) for key, label in mapping.items()}


def stable_balanced_pairs(
    conditions: Iterable[tuple], n_episodes: int, seed: int
) -> list[tuple[int, tuple]]:
    """Return paired condition assignments for an even episode budget."""
    if n_episodes < 2 or n_episodes % 2:
        raise ValueError("counterbalanced collection requires a positive even episode count")
    cells = list(conditions)
    if not cells:
        raise ValueError("at least one condition is required")
    random.Random(seed).shuffle(cells)
    return [(index, cells[index % len(cells)]) for index in range(n_episodes // 2)]
