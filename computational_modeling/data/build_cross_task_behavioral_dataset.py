"""Export records-only Bandit/Foraging/Solvability behavioral datasets.

Activation tensors are loaded one episode at a time, integrity-checked, and
immediately discarded.  Saved CSVs contain scalar behavioral records only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

from computational_modeling.data.feature_schema import FEATURE_SCHEMA, serialized_schema


CONTINUE_ACTIONS = {
    "bandit": {"A", "B"},
    "foraging": {"STAY"},
    "solvability": {"TRY_AGAIN"},
}
DISENGAGE_ACTIONS = {
    "bandit": {"C"},
    "foraging": {"LEAVE"},
    "solvability": {"GIVE_UP"},
}


def _semantic_choice(task: str, record: Mapping) -> str:
    return str(record.get("semantic_choice", record.get("sampled_action", ""))).upper()


def harmonize_record(task: str, record: Mapping, *, pair_id: str) -> dict:
    task = str(task)
    if task not in CONTINUE_ACTIONS:
        raise ValueError(f"unsupported behavioral task: {task!r}")
    choice = _semantic_choice(task, record)
    if choice not in CONTINUE_ACTIONS[task] | DISENGAGE_ACTIONS[task]:
        raise ValueError(f"unknown semantic choice for {task}: {choice!r}")
    p_continue = float(record["p_continue"])
    p_disengage = float(
        record.get("p_stop", record.get("p_leave", record.get("p_give_up", 1 - p_continue)))
    )
    if not 0 <= p_continue <= 1 or not 0 <= p_disengage <= 1:
        raise ValueError("semantic choice probabilities must fall in [0, 1]")
    return {
        "task": task,
        "episode_id": str(record["episode_id"]),
        "pair_id": str(pair_id),
        "state_id": str(record["state_id"]),
        "round": int(record["round"]),
        "semantic_choice": choice,
        "continue": int(choice in CONTINUE_ACTIONS[task]),
        "p_continue": p_continue,
        "p_disengage": p_disengage,
        "persistence_logit": float(record["persistence_logit"]),
    }


def validate_split(records: Sequence[Mapping]) -> None:
    episode_split, pair_split = {}, {}
    for row in records:
        split = str(row["split"])
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {split!r}")
        episode = str(row["episode_id"])
        pair = str(row["pair_id"])
        if episode in episode_split and episode_split[episode] != split:
            raise ValueError("episode crosses split boundaries")
        if pair in pair_split and pair_split[pair] != split:
            raise ValueError("counterbalanced pair crosses split boundaries")
        episode_split[episode] = split
        pair_split[pair] = split


def validate_behavioral_records(
    records: Sequence[Mapping], *, require_targets: bool = True
) -> None:
    if not records:
        raise ValueError("behavioral dataset is empty")
    required = {"task", "episode_id", "pair_id", "state_id", "round"}
    if require_targets:
        required |= {"continue", "p_continue", "p_disengage", "persistence_logit", "split"}
    episodes: dict[str, list[Mapping]] = {}
    state_ids = set()
    for index, row in enumerate(records):
        missing = required - set(row)
        if missing:
            raise ValueError(f"behavioral record {index} is missing {sorted(missing)}")
        if row["state_id"] in state_ids:
            raise ValueError(f"duplicate behavioral state: {row['state_id']}")
        state_ids.add(row["state_id"])
        episodes.setdefault(str(row["episode_id"]), []).append(row)
        if require_targets:
            values = (float(row["p_continue"]), float(row["p_disengage"]), float(row["persistence_logit"]))
            if any(not math.isfinite(value) for value in values):
                raise ValueError("behavioral targets must be finite")
    for episode, rows in episodes.items():
        rounds = sorted(int(row["round"]) for row in rows)
        if rounds != list(range(len(rounds))):
            raise ValueError(f"state order is not contiguous in episode {episode}")
    if require_targets:
        validate_split(records)


def _displayed_progress_cue(probability: float) -> float:
    if probability < 0.35:
        return -1.0
    if probability < 0.65:
        return 0.0
    return 1.0


def _outcome_after_choice(task: str, source: Mapping) -> float:
    if task in {"bandit", "foraging"}:
        return float(source.get("subsequent_reward", 0.0) or 0.0)
    progress = source.get("progress_made")
    if progress is None:
        return 0.0
    return 1.0 if bool(progress) else -1.0


def _episode_features(task: str, source_rows: Sequence[Mapping], *, pair_id: str, split: str):
    rows = []
    actions: list[float] = []
    outcomes: list[float] = []
    success_streak = failure_streak = 0
    bandit_success = {"A": 1.0, "B": 1.0}
    bandit_failure = {"A": 1.0, "B": 1.0}
    bandit_q = {"A": 0.5, "B": 0.5}
    patch_success = patch_failure = 1.0
    progress_success = progress_failure = 1.0
    for source in sorted(source_rows, key=lambda row: int(row["round"])):
        row = harmonize_record(task, source, pair_id=pair_id)
        round_index = int(row["round"])
        row["split"] = split
        row["mapping_id"] = str(source.get("mapping_id", "bandit_abc"))
        for task_name in ("bandit", "foraging", "solvability"):
            row[f"task_{task_name}"] = float(task == task_name)
        row["log_round"] = math.log1p(round_index)
        if task == "solvability":
            horizon = max(1, int(source.get("max_attempts", round_index + 1)))
            row["normalized_time"] = round_index / horizon
        else:
            row["normalized_time"] = round_index / max(1, round_index + 1)
        row["previous_outcome"] = outcomes[-1] if outcomes else 0.0
        row["failure_streak"] = float(failure_streak)
        row["success_streak"] = float(success_streak)
        row["previous_choice"] = actions[-1] if actions else 0.0
        row["second_previous_choice"] = actions[-2] if len(actions) > 1 else 0.0
        for lag in (1, 2, 3, 5):
            row[f"action_lag_{lag}"] = actions[-lag] if len(actions) >= lag else 0.0
            row[f"outcome_lag_{lag}"] = outcomes[-lag] if len(outcomes) >= lag else 0.0
        row["cumulative_progress"] = float(sum(outcomes))

        if task == "bandit":
            estimates = {
                arm: 5.0 * bandit_success[arm] / (bandit_success[arm] + bandit_failure[arm]) - 2.0
                for arm in ("A", "B")
            }
            row.update(
                {
                    "cumulative_score": float(source.get("cumulative_score", 0.0)),
                    "rw_a": bandit_q["A"],
                    "rw_b": bandit_q["B"],
                    "rw_best": max(bandit_q.values()),
                    "rw_gap": abs(bandit_q["A"] - bandit_q["B"]),
                    "bayes_a": estimates["A"],
                    "bayes_b": estimates["B"],
                    "bayes_best": max(estimates.values()),
                    "bayes_gap": abs(estimates["A"] - estimates["B"]),
                }
            )
            estimated_continue = max(estimates.values())
            outside, cost, progress = 0.0, 0.0, row["previous_outcome"]
            p_a, p_b = float(source["p_A_true"]), float(source["p_B_true"])
            oracle_continue = max(5.0 * p_a - 2.0, 5.0 * p_b - 2.0)
            row.update({"oracle_p_a": p_a, "oracle_p_b": p_b})
        elif task == "foraging":
            patch_probability = patch_success / (patch_success + patch_failure)
            outside = float(source["outside_option"])
            cost = float(source["stay_cost"])
            estimated_continue = 4.0 * patch_probability - cost
            progress = row["previous_outcome"]
            private_probability = float(source["patch_probability_private"])
            oracle_continue = 4.0 * private_probability - cost
            row.update(
                {
                    "outside_option": outside,
                    "stay_cost": cost,
                    "search_count": float(source.get("search_count", round_index)),
                    "cumulative_score": float(source.get("cumulative_score", 0.0)),
                    "bayes_patch_probability": patch_probability,
                    "mvt_like_advantage": estimated_continue - outside,
                    "oracle_initial_quality": float(source["initial_quality"]),
                    "oracle_depletion": float(source["depletion"]),
                    "oracle_patch_probability": private_probability,
                }
            )
        else:
            exact_probability = float(source["progress_probability"])
            cue = _displayed_progress_cue(exact_probability)
            prior_mean = {-1.0: 0.25, 0.0: 0.5, 1.0: 0.75}[cue]
            # Two pseudo-observations preserve the prompt's weak/mixed/strong cue.
            estimated_probability = (2.0 * prior_mean + progress_success - 1.0) / (
                2.0 + progress_success + progress_failure - 2.0
            )
            outside = float(source["give_up_value"])
            cost = float(source["attempt_cost"])
            estimated_continue = estimated_probability - cost
            progress = float(source.get("progress_count", 0.0))
            oracle_continue = exact_probability - cost
            row.update(
                {
                    "attempt_cost": cost,
                    "give_up_value": outside,
                    "attempts_used": float(source.get("attempts_used", round_index)),
                    "max_attempts": float(source.get("max_attempts", 0.0)),
                    "cumulative_cost": float(source.get("cumulative_cost", 0.0)),
                    "progress_count": float(source.get("progress_count", 0.0)),
                    "displayed_progress_cue": cue,
                    "bayes_progress_probability": estimated_probability,
                    "oracle_progress_probability": exact_probability,
                }
            )

        advantage = estimated_continue - outside
        oracle_advantage = oracle_continue - outside
        row.update(
            {
                "estimated_continue_value": estimated_continue,
                "estimated_outside_value": outside,
                "cost_pressure": cost,
                "progress_evidence": progress,
                "termination_advantage": advantage,
                "relative_value": advantage,
                "disengagement_evidence": -advantage,
                "oracle_continue_value": oracle_continue,
                "oracle_outside_value": outside,
                "oracle_termination_advantage": oracle_advantage,
                "oracle_relative_value": oracle_advantage,
                "outcome_after_choice": _outcome_after_choice(task, source),
            }
        )
        rows.append(row)

        action = float(row["continue"])
        outcome = float(row["outcome_after_choice"])
        actions.append(action)
        outcomes.append(outcome)
        if outcome > 0:
            success_streak += 1
            failure_streak = 0
        elif outcome < 0:
            failure_streak += 1
            success_streak = 0
        choice = row["semantic_choice"]
        if task == "bandit" and choice in {"A", "B"}:
            bandit_q[choice] += 0.5 * (outcome - bandit_q[choice])
            if outcome > 0:
                bandit_success[choice] += 1
            else:
                bandit_failure[choice] += 1
        elif task == "foraging" and choice == "STAY":
            if source.get("found_food"):
                patch_success += 1
            else:
                patch_failure += 1
        elif task == "solvability" and choice == "TRY_AGAIN":
            if source.get("progress_made"):
                progress_success += 1
            else:
                progress_failure += 1
    return rows


def _split_lookup(split: Mapping[str, Sequence[str]]) -> dict[str, str]:
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("persisted split must contain train, validation, and test")
    lookup = {}
    for name, episodes in split.items():
        for episode in episodes:
            if str(episode) in lookup:
                raise ValueError("persisted split duplicates an episode")
            lookup[str(episode)] = name
    return lookup


def load_task_records(
    task: str, bank_dir: str, split_path: str, *, progress=None
) -> tuple[list[dict], dict]:
    """Stream an activation bank and return scalar, engineered behavioral rows."""

    import torch

    split_bytes = Path(split_path).read_bytes()
    split = json.loads(split_bytes)
    lookup = _split_lookup(split)
    files = sorted(glob.glob(str(Path(bank_dir) / "episode_*.pt")))
    if not files:
        raise FileNotFoundError(f"no activation shards found under {bank_dir}")
    records, observed_episodes, inventory = [], set(), hashlib.sha256()
    started = time.perf_counter()
    for file_index, filename in enumerate(files, start=1):
        shard = torch.load(filename, map_location="cpu", weights_only=False)
        episode = str(shard["episode_id"])
        if episode in observed_episodes:
            raise ValueError(f"duplicate episode shard: {episode}")
        observed_episodes.add(episode)
        source_rows = shard.get("records")
        activations = shard.get("activations")
        if not isinstance(source_rows, list) or activations is None:
            raise ValueError(f"malformed activation shard: {filename}")
        if len(source_rows) != int(activations.shape[0]):
            raise ValueError(f"record/activation mismatch in {filename}")
        if episode not in lookup:
            raise ValueError(f"episode {episode!r} is absent from persisted split")
        pair_id = str(shard.get("pair_id", episode))
        records.extend(
            _episode_features(task, source_rows, pair_id=pair_id, split=lookup[episode])
        )
        payload = {
            "file": Path(filename).name,
            "episode_id": episode,
            "pair_id": pair_id,
            "activation_shape": list(activations.shape),
            "records": source_rows,
        }
        inventory.update(json.dumps(payload, sort_keys=True, default=str).encode())
        del shard, activations
        if progress is not None and (
            file_index == len(files) or file_index % 100 == 0
        ):
            progress(
                f"{task}: loaded {file_index}/{len(files)} episode shards; "
                f"{len(records)} states; {time.perf_counter() - started:.1f}s elapsed"
            )
    if observed_episodes != set(lookup):
        missing = sorted(set(lookup) - observed_episodes)
        raise ValueError(f"activation bank is missing persisted episodes: {missing[:5]}")
    validate_behavioral_records(records)
    return records, {
        "task": task,
        "bank_dir": str(bank_dir),
        "split_path": str(split_path),
        "behavioral_payload_sha256": inventory.hexdigest(),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "episodes": len(observed_episodes),
        "states": len(records),
        "split_counts": {name: len(value) for name, value in split.items()},
    }


def _write_csv(records: Sequence[Mapping], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def export_behavioral_dataset(
    config: Mapping, output_dir: str | Path, *, progress=None
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifests, all_records = {}, []
    export_started = time.perf_counter()
    for task in config["tasks"]:
        task_started = time.perf_counter()
        if progress is not None:
            progress(f"{task}: starting records-only export")
        task_config = config["data"][task]
        records, manifest = load_task_records(
            task,
            task_config["activation_bank"],
            task_config["split"],
            progress=progress,
        )
        _write_csv(records, output / f"{task}_records.csv")
        manifests[task] = manifest
        all_records.extend(records)
        if progress is not None:
            progress(
                f"{task}: wrote {len(records)} states from {manifest['episodes']} "
                f"episodes in {time.perf_counter() - task_started:.1f}s"
            )
    validate_split(all_records)
    schema = serialized_schema()
    (output / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    feature_information = {
        name: information
        for task in config["tasks"]
        for information, names in FEATURE_SCHEMA[task].items()
        for name in names
    }
    variables = {}
    for name in sorted({key for row in all_records for key in row}):
        if name.startswith("oracle_"):
            role, information = "model_feature", "oracle-state"
        elif name in feature_information:
            role, information = "model_feature", "observable"
        elif name in {"persistence_logit", "continue", "p_continue", "p_disengage"}:
            role, information = "target", "recorded_policy_or_behavior"
        elif name in {"task", "episode_id", "pair_id", "state_id", "round", "split", "mapping_id"}:
            role, information = "identifier", "not_a_model_feature"
        elif name == "outcome_after_choice":
            role, information = "transition_event", "available_only_to_later_states"
        else:
            role, information = "audit_field", "not_selected_without_schema_entry"
        variables[name] = {
            "role": role,
            "information_set": information,
            "description": schema["descriptions"].get(name, name.replace("_", " ")),
        }
    manifest = {
        "format": "records-only CSV; activation tensors and conversations excluded",
        "tasks": manifests,
        "total_episodes": sum(row["episodes"] for row in manifests.values()),
        "total_states": sum(row["states"] for row in manifests.values()),
        "targets": ["persistence_logit", "continue", "p_continue", "p_disengage"],
        "oracle_field_policy": "all environment-private fields use the oracle_ prefix",
        "variables": variables,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if progress is not None:
        progress(
            f"records export complete: {manifest['total_states']} states, "
            f"{manifest['total_episodes']} episodes in "
            f"{time.perf_counter() - export_started:.1f}s"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/computational_model_zoo.yaml")
    parser.add_argument("--output-dir", default="artifacts/computational_modeling/records")
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = export_behavioral_dataset(
        config, args.output_dir, progress=lambda message: print(message, flush=True)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
