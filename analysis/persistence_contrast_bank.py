"""Adapters from existing Track A/B artifacts to the Track C contrast bank."""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Mapping

from analysis.persistence_contrasts import (
    ContrastDefinition,
    behavioral_validity_gate,
    build_balanced_semantic_nuisance_contrasts,
    build_exact_label_contrasts,
    build_matched_record_contrasts,
    flatten_activation_shards,
)
from cross_task.common import grouped_episode_split
from experiments.cross_task_utils import load_activation_shards, make_or_validate_split
from experiments.project_factorial_layers import _activation_path, _read_factorial_rows


PERSISTENCE_DEFINITIONS = {
    "bandit_continue_incentive": ContrastDefinition(
        task="bandit",
        manipulation="continue_incentive",
        factor="continue_bonus",
        higher_promotes_persistence=True,
        matched_fields=("base_state_id", "stop_payoff"),
    ),
    "bandit_stop_outside_option": ContrastDefinition(
        task="bandit",
        manipulation="stop_outside_option",
        factor="stop_payoff",
        higher_promotes_persistence=False,
        matched_fields=("base_state_id", "continue_bonus"),
    ),
    "foraging_search_cost": ContrastDefinition(
        task="foraging",
        manipulation="search_cost",
        factor="stay_cost",
        higher_promotes_persistence=False,
        matched_fields=(
            "initial_quality",
            "depletion",
            "outside_option",
            "round",
            "mapping_id",
            "choice_history",
            "reward_history",
            "previous_outcome",
        ),
    ),
    "foraging_outside_option": ContrastDefinition(
        task="foraging",
        manipulation="outside_option",
        factor="outside_option",
        higher_promotes_persistence=False,
        matched_fields=(
            "initial_quality",
            "depletion",
            "stay_cost",
            "round",
            "mapping_id",
            "choice_history",
            "reward_history",
            "previous_outcome",
        ),
    ),
    "solvability_progress_evidence": ContrastDefinition(
        task="solvability",
        manipulation="progress_evidence",
        factor="progress_probability",
        higher_promotes_persistence=True,
        matched_fields=(
            "attempt_cost",
            "give_up_value",
            "max_attempts",
            "round",
            "mapping_id",
            "choice_history",
            "progress_history",
            "previous_progress",
        ),
    ),
}


def _inverse_split(split: Mapping[str, list[str]]) -> dict[str, str]:
    inverse = {
        str(episode): str(name)
        for name, episodes in split.items()
        for episode in episodes
    }
    if len(inverse) != sum(len(episodes) for episodes in split.values()):
        raise ValueError("split contains duplicate episode assignments")
    return inverse


def _write_split(path: str, episode_ids: set[str], *, seed: int) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            split = json.load(handle)
    else:
        split = grouped_episode_split(
            {episode: episode for episode in episode_ids}, seed=seed
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(split, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    if set(_inverse_split(split)) != episode_ids:
        raise ValueError("factorial split does not exactly partition episodes")
    return split


def load_bandit_factorial_records(
    *, factorial_pattern: str, activation_dir: str, split_path: str, split_seed: int
) -> tuple[list[dict], dict]:
    """Recover cell-level all-layer tensors from the existing factorial replay."""

    import torch

    source_rows = _read_factorial_rows(factorial_pattern)
    row_index = {
        (
            str(row["state_id"]),
            int(float(row["stop_payoff"])),
            int(float(row["continue_bonus"])),
        ): row
        for row in source_rows
    }
    episode_ids = {str(row["episode_id"]) for row in source_rows}
    split = _write_split(split_path, episode_ids, seed=split_seed)
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(
            f"Bandit factorial activations are absent from {activation_dir}; "
            "rerun experiments.project_factorial_layers with --save-activations"
        )
    records = []
    observed_cells = set()
    for state_id in sorted({str(row["state_id"]) for row in source_rows}):
        path = _activation_path(activation_dir, state_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing Bandit factorial activation {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if str(artifact.get("state_id")) != state_id:
            raise ValueError(f"Bandit factorial state mismatch in {path}")
        activations = artifact["activations"]
        conditions = artifact["conditions"]
        if len(conditions) != int(activations.shape[0]) or activations.ndim != 3:
            raise ValueError(f"malformed Bandit factorial tensor in {path}")
        for index, condition in enumerate(conditions):
            key = (
                state_id,
                int(condition["stop_payoff"]),
                int(condition["continue_bonus"]),
            )
            if key not in row_index or key in observed_cells:
                raise ValueError(f"unexpected or duplicate Bandit factorial cell {key}")
            source = row_index[key]
            observed_cells.add(key)
            records.append(
                {
                    "task": "bandit",
                    "episode_id": str(source["episode_id"]),
                    "pair_id": str(source["episode_id"]),
                    "state_id": f"{state_id}:stop={key[1]}:continue={key[2]}",
                    "base_state_id": state_id,
                    "stop_payoff": key[1],
                    "continue_bonus": key[2],
                    "persistence_logit": float(source["persistence_logit"]),
                    "activation": activations[index].float(),
                    "history_hash": source.get("history_hash"),
                    "context_hash": source.get("context_hash"),
                }
            )
    if observed_cells != set(row_index):
        raise ValueError(
            f"Bandit activation coverage is {len(observed_cells)}/{len(row_index)} cells"
        )
    return records, split


def _task_records(directory: str, split_path: str, *, seed: int):
    shards = load_activation_shards(directory)
    split = make_or_validate_split(shards, split_path, seed=seed)
    return flatten_activation_shards(shards), split


def _label_records(directory: str, source_episode_to_split: Mapping[str, str]):
    shards = load_activation_shards(directory)
    records = flatten_activation_shards(shards)
    replay_split = {}
    for record in records:
        source_episode = str(record["source_episode_id"])
        if source_episode not in source_episode_to_split:
            raise ValueError("matched-label replay references an unknown source episode")
        replay_split[str(record["episode_id"])] = source_episode_to_split[source_episode]
    return records, replay_split


def _inventory_row(row: Mapping) -> dict:
    delta = row["activation_delta"]
    return {
        "contrast_id": row["contrast_id"],
        "task": row["task"],
        "manipulation": row["manipulation"],
        "contrast_kind": row.get("contrast_kind", "persistence"),
        "nuisance_type": row.get("nuisance_type", ""),
        "split": row["split"],
        "cluster_id": row["cluster_id"],
        "positive_state_id": row["positive_state_id"],
        "negative_state_id": row["negative_state_id"],
        "orientation": row["orientation"],
        "behavior_effect": row.get("behavior_effect"),
        "mapping_id": row.get("mapping_id", ""),
        "layers": int(delta.shape[0]),
        "hidden_width": int(delta.shape[1]),
        "matched_variables": json.dumps(
            row.get("matched_variables", {}), sort_keys=True, separators=(",", ":")
        ),
        "unmatched_variables": row.get("unmatched_variables", ""),
    }


def save_contrast_bank(
    contrasts: list[dict], *, bank_path: str, inventory_path: str, audit_path: str,
    audit: dict,
) -> None:
    from experiments.runtime import atomic_torch_save

    Path(bank_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "contrasts": contrasts,
            "shape": "contrast activation_delta is layers x hidden_width",
            "analysis_role": "exploratory_discovery",
            "audit": audit,
        },
        bank_path,
    )
    rows = [_inventory_row(row) for row in contrasts]
    Path(inventory_path).parent.mkdir(parents=True, exist_ok=True)
    temporary = inventory_path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["contrast_id"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, inventory_path)
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_path).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def build_contrast_bank(config: dict, *, allow_missing_generic_value=False) -> tuple[list[dict], dict]:
    """Build every persistence and nuisance family using existing artifacts."""

    paths = config["paths"]
    split_seed = int(config["split_seed"])
    bandit_records, bandit_split = load_bandit_factorial_records(
        factorial_pattern=paths["bandit_factorial"],
        activation_dir=paths["bandit_factorial_activations"],
        split_path=paths["bandit_contrast_split"],
        split_seed=split_seed,
    )
    foraging_records, foraging_split = _task_records(
        paths["foraging_activations"], paths["foraging_split"], seed=split_seed
    )
    solvability_records, solvability_split = _task_records(
        paths["solvability_activations"], paths["solvability_split"], seed=split_seed
    )
    persistence = []
    for records, split, keys in (
        (bandit_records, bandit_split, ("bandit_continue_incentive", "bandit_stop_outside_option")),
        (foraging_records, foraging_split, ("foraging_search_cost", "foraging_outside_option")),
        (solvability_records, solvability_split, ("solvability_progress_evidence",)),
    ):
        inverse = _inverse_split(split)
        for key in keys:
            persistence.extend(
                build_matched_record_contrasts(
                    records, PERSISTENCE_DEFINITIONS[key], episode_to_split=inverse
                )
            )
    for row in persistence:
        row["contrast_kind"] = "persistence"

    nuisance = []
    for task, directory, source_split in (
        ("foraging", paths["foraging_label_replays"], foraging_split),
        ("solvability", paths["solvability_label_replays"], solvability_split),
    ):
        records, replay_split = _label_records(directory, _inverse_split(source_split))
        nuisance.extend(
            build_exact_label_contrasts(
                records,
                semantic_fields=("matched_history_hash", "source_state_id"),
                episode_to_split=replay_split,
            )
        )

    from cross_task.control import LEFT_GREATER, RIGHT_GREATER
    from cross_task.generic_value import LEFT_VOUCHER, RIGHT_VOUCHER
    from cross_task.terminality import END, PROCEED

    control_records, control_split = _task_records(
        paths["arbitrary_choice_activations"], paths["arbitrary_choice_split"], seed=split_seed
    )
    nuisance.extend(
        build_balanced_semantic_nuisance_contrasts(
            control_records,
            nuisance_type="arbitrary_choice",
            positive_semantic=LEFT_GREATER,
            negative_semantic=RIGHT_GREATER,
            sort_key=lambda row: (abs(int(row["left_integer"]) - int(row["right_integer"])), str(row["state_id"])),
            episode_to_split=_inverse_split(control_split),
        )
    )
    terminal_records, terminal_split = _task_records(
        paths["terminality_activations"], paths["terminality_split"], seed=split_seed
    )
    nuisance.extend(
        build_balanced_semantic_nuisance_contrasts(
            terminal_records,
            nuisance_type="terminality",
            positive_semantic=PROCEED,
            negative_semantic=END,
            sort_key=lambda row: (abs(int(row["displayed_integer"])), str(row["state_id"])),
            episode_to_split=_inverse_split(terminal_split),
        )
    )
    generic_available = os.path.isdir(paths["generic_value_activations"])
    if generic_available:
        generic_records, generic_split = _task_records(
            paths["generic_value_activations"], paths["generic_value_split"], seed=split_seed
        )
        nuisance.extend(
            build_balanced_semantic_nuisance_contrasts(
                generic_records,
                nuisance_type="generic_value",
                positive_semantic=LEFT_VOUCHER,
                negative_semantic=RIGHT_VOUCHER,
                sort_key=lambda row: (abs(int(row["relative_value"])), str(row["state_id"])),
                episode_to_split=_inverse_split(generic_split),
            )
        )
    elif not allow_missing_generic_value:
        raise FileNotFoundError(
            "generic-value activation bank is required; collect it with "
            "experiments.collect_cross_task_activations --task generic_value"
        )

    gates = {}
    for manipulation in sorted({row["manipulation"] for row in persistence}):
        rows = [row for row in persistence if row["manipulation"] == manipulation and row["split"] in {"train", "validation"}]
        gates[manipulation] = behavioral_validity_gate(
            rows,
            bootstrap_samples=int(config["search"]["bootstrap_samples"]),
            seed=int(config["analysis_seed"]),
        )
    shapes = {
        (int(row["activation_delta"].shape[0]), int(row["activation_delta"].shape[1]))
        for row in (*persistence, *nuisance)
    }
    from experiments.runtime import run_metadata

    audit = {
        "analysis_role": "exploratory_discovery",
        "persistence_contrasts": len(persistence),
        "nuisance_contrasts": len(nuisance),
        "counts_by_task_manipulation_split": {},
        "behavioral_gates": gates,
        "all_behavioral_gates_passed": bool(gates) and all(row["passed"] for row in gates.values()),
        "activation_shapes": [list(shape) for shape in sorted(shapes)],
        "activation_coverage_passed": len(shapes) == 1,
        "generic_value_control_available": generic_available,
        "solvability_contrast": config["solvability_audit"],
        "requires_counterfactual_solvability_replay": not any(
            row["task"] == "solvability" for row in persistence
        ),
        "provenance": run_metadata(
            {
                "analysis": "persistence_contrast_bank",
                "protocol_version": config["protocol_version"],
                "model": config["model"],
            }
        ),
    }
    for row in (*persistence, *nuisance):
        key = "|".join((str(row["task"]), str(row["manipulation"]), str(row["split"])))
        audit["counts_by_task_manipulation_split"][key] = audit["counts_by_task_manipulation_split"].get(key, 0) + 1
    if len(shapes) != 1:
        raise ValueError(f"contrast activation shapes are incompatible: {sorted(shapes)}")
    return [*persistence, *nuisance], audit
