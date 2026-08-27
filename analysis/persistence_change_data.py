"""Streaming adapters for the matched persistence-change follow-up.

Only layers 21 and 22 are retained from each referenced endpoint.  This keeps
the analysis practical without recreating the multi-gigabyte all-layer
``contrast_bank.pt`` artifact.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from analysis.persistence_change_geometry import audit_exact_pair
from analysis.persistence_contrast_bank import PERSISTENCE_DEFINITIONS
from analysis.persistence_geometry import FrozenPersistenceSubspace
from computational_modeling.analysis.model_fitting import _fit_apply
from computational_modeling.models.baselines import MODEL_DEFINITIONS
from computational_modeling.models.gru import fit_gru_ceiling
from experiments.project_factorial_layers import _activation_path, _read_factorial_rows


PERSISTENCE_TARGETS = (
    "persistence_policy_change",
    "gru_prediction_change",
    "history_prediction_change",
)

COMPONENT_TARGET_BY_MANIPULATION = {
    "continue_incentive": "continuation_value_change",
    "stop_outside_option": "outside_option_relief",
    "search_cost": "cost_relief",
    "outside_option": "outside_option_relief",
    "progress_evidence": "progress_evidence_change",
}

BANK_BY_INVENTORY_TASK = {
    "foraging": "foraging",
    "solvability": "solvability",
    "foraging_label_replay": "foraging_label_replay",
    "solvability_label_replay": "solvability_label_replay",
    "binary_control": "arbitrary_choice",
    "terminality_control": "terminality",
    "generic_value_control": "generic_value",
}


def _load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _compact_inventory(frame: pd.DataFrame, maximum_per_group: int | None):
    if maximum_per_group is None:
        return frame.reset_index(drop=True)
    parts = []
    for _key, part in frame.groupby(
        ["task", "manipulation", "contrast_kind", "split"],
        dropna=False,
        sort=False,
    ):
        parts.append(part.sort_values("contrast_id").head(int(maximum_per_group)))
    return pd.concat(parts, ignore_index=True)


def _stream_bank_endpoints(
    directory,
    wanted,
    *,
    layer21,
    layer22,
    logger,
    label,
):
    import torch

    directory = Path(directory)
    files = sorted(directory.glob("episode_*.pt"))
    if not files:
        raise FileNotFoundError(f"no activation shards found for {label}: {directory}")
    output = {}
    for file_index, path in enumerate(files, start=1):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        records = shard.get("records", [])
        activations = shard.get("activations")
        if activations is None or len(records) != int(activations.shape[0]):
            raise ValueError(f"malformed activation shard: {path}")
        for index, record in enumerate(records):
            state_id = str(record["state_id"])
            if state_id in wanted:
                if state_id in output:
                    raise ValueError(f"duplicate endpoint state {state_id}")
                output[state_id] = {
                    "record": dict(record),
                    "h21": activations[index, layer21].cpu().numpy().astype(np.float16),
                    "h22": activations[index, layer22].cpu().numpy().astype(np.float16),
                }
        if logger is not None and (
            file_index == len(files) or file_index % 200 == 0
        ):
            logger.note(
                "endpoint_stream",
                f"{label}: {file_index}/{len(files)} shards; {len(output)}/{len(wanted)} endpoints",
            )
        del shard, activations
    missing = sorted(set(wanted) - set(output))
    if missing:
        raise ValueError(f"{label} bank is missing referenced endpoints: {missing[:5]}")
    return output


def _factorial_index(pattern):
    rows = _read_factorial_rows(pattern)
    output = {}
    for source in rows:
        stop = int(float(source["stop_payoff"]))
        continuation = int(float(source["continue_bonus"]))
        state_id = f"{source['state_id']}:stop={stop}:continue={continuation}"
        if state_id in output:
            raise ValueError(f"duplicate Bandit factorial endpoint: {state_id}")
        output[state_id] = dict(source)
    return output


def _stream_bandit_endpoints(
    factorial_pattern,
    activation_dir,
    wanted,
    *,
    layer21,
    layer22,
    logger,
):
    import torch

    activation_dir = Path(activation_dir)
    if not activation_dir.is_dir():
        raise FileNotFoundError(
            "Bandit factorial all-layer tensors are required but absent from "
            f"{activation_dir}. Sync artifacts/value_dissociation/activations from "
            "the cluster, or regenerate them there with PHASE=factorial_layerwise_project."
        )
    source_index = _factorial_index(factorial_pattern)
    missing_rows = sorted(set(wanted) - set(source_index))
    if missing_rows:
        raise ValueError(f"factorial CSVs are missing contrast endpoints: {missing_rows[:5]}")
    by_base = {}
    for state_id in wanted:
        source = source_index[state_id]
        by_base.setdefault(str(source["state_id"]), []).append(state_id)
    output = {}
    for base_index, base_state in enumerate(sorted(by_base), start=1):
        path = Path(_activation_path(str(activation_dir), base_state))
        if not path.exists():
            raise FileNotFoundError(f"missing Bandit factorial tensor: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if str(artifact.get("state_id")) != base_state:
            raise ValueError(f"Bandit factorial state mismatch in {path}")
        condition_index = {
            (int(row["stop_payoff"]), int(row["continue_bonus"])): index
            for index, row in enumerate(artifact["conditions"])
        }
        for state_id in by_base[base_state]:
            source = source_index[state_id]
            key = (
                int(float(source["stop_payoff"])),
                int(float(source["continue_bonus"])),
            )
            if key not in condition_index:
                raise ValueError(f"Bandit tensor lacks condition {key}: {path}")
            values = artifact["activations"][condition_index[key]]
            record = {
                **source,
                "task": "bandit",
                "pair_id": str(source["episode_id"]),
                "state_id": state_id,
                "base_state_id": base_state,
                "stop_payoff": key[0],
                "continue_bonus": key[1],
                "persistence_logit": float(source["persistence_logit"]),
            }
            output[state_id] = {
                "record": record,
                "h21": values[layer21].cpu().numpy().astype(np.float16),
                "h22": values[layer22].cpu().numpy().astype(np.float16),
            }
        if logger is not None and (
            base_index == len(by_base) or base_index % 100 == 0
        ):
            logger.note(
                "endpoint_stream",
                f"bandit: {base_index}/{len(by_base)} tensors; {len(output)}/{len(wanted)} endpoints",
            )
    return output, source_index


def _definition_for(row):
    key = f"{row.task}_{row.manipulation}"
    if key not in PERSISTENCE_DEFINITIONS:
        raise KeyError(f"no persistence contrast definition for {key}")
    return PERSISTENCE_DEFINITIONS[key]


def _nuisance_target(record):
    return float(
        record.get(
            "target_logit",
            record.get("choice_logit", record.get("persistence_logit", 0.0)),
        )
    )


def _component_change(manipulation, positive, negative):
    if manipulation == "continue_incentive":
        return float(positive["continue_bonus"]) - float(negative["continue_bonus"])
    if manipulation == "stop_outside_option":
        return float(negative["stop_payoff"]) - float(positive["stop_payoff"])
    if manipulation == "search_cost":
        return float(negative["stay_cost"]) - float(positive["stay_cost"])
    if manipulation == "outside_option":
        return float(negative["outside_option"]) - float(positive["outside_option"])
    if manipulation == "progress_evidence":
        return float(positive["progress_probability"]) - float(
            negative["progress_probability"]
        )
    raise KeyError(f"unknown component target for {manipulation}")


def _audit_nuisance(row, positive, negative):
    unmatched = ""
    if row.manipulation == "label_identity":
        fields = ("matched_history_hash", "source_state_id")
        mismatched = [name for name in fields if positive.get(name) != negative.get(name)]
        mapping_changed = str(positive.get("mapping_id")) != str(negative.get("mapping_id"))
        matched = not mismatched and mapping_changed
        unmatched = ";".join(mismatched)
    else:
        # These retained controls are balanced semantic-class contrasts rather
        # than exact same-state replays.  Their known unmatched stimulus is
        # exposed explicitly in the audit instead of being hidden.
        matched = str(positive.get("mapping_id")) == str(negative.get("mapping_id"))
        unmatched = "surface_stimulus_balanced_by_semantic_class"
    return matched, unmatched, True


def build_compact_change_dataset(config, frozen: FrozenPersistenceSubspace, *, logger=None, smoke=False):
    """Load referenced endpoints and construct compact full/change arrays."""

    inventory = pd.read_csv(config["paths"]["contrast_inventory"])
    maximum = int(config["smoke"]["max_pairs_per_group"]) if smoke else None
    inventory = _compact_inventory(inventory, maximum)
    required_columns = {
        "contrast_id",
        "task",
        "manipulation",
        "contrast_kind",
        "split",
        "cluster_id",
        "positive_state_id",
        "negative_state_id",
        "orientation",
        "behavior_effect",
    }
    missing = sorted(required_columns - set(inventory))
    if missing:
        raise ValueError(f"contrast inventory lacks required columns: {missing}")
    layer21, layer22 = int(config["layers"]["l21"]), int(config["layers"]["l22"])
    endpoints = {}
    bandit_rows = inventory[inventory.task == "bandit"]
    bandit_wanted = set(bandit_rows.positive_state_id.astype(str)) | set(
        bandit_rows.negative_state_id.astype(str)
    )
    bandit_endpoints, factorial_index = _stream_bandit_endpoints(
        config["paths"]["bandit_factorial"],
        config["paths"]["bandit_factorial_activations"],
        bandit_wanted,
        layer21=layer21,
        layer22=layer22,
        logger=logger,
    )
    endpoints.update(bandit_endpoints)
    for inventory_task, bank_key in BANK_BY_INVENTORY_TASK.items():
        part = inventory[inventory.task == inventory_task]
        if part.empty:
            continue
        wanted = set(part.positive_state_id.astype(str)) | set(
            part.negative_state_id.astype(str)
        )
        local = _stream_bank_endpoints(
            config["paths"]["activation_banks"][bank_key],
            wanted,
            layer21=layer21,
            layer22=layer22,
            logger=logger,
            label=inventory_task,
        )
        overlap = set(endpoints) & set(local)
        if overlap:
            raise ValueError(f"endpoint IDs collide across banks: {sorted(overlap)[:5]}")
        endpoints.update(local)

    audit_rows, metadata_rows, target_rows = [], [], []
    hidden = {stage: [] for stage in ("l21", "displacement", "l22")}
    absolute_rows, absolute_h21, absolute_h22 = [], [], []
    retained_endpoint_records = {}
    for source in inventory.itertuples(index=False):
        pair_id = str(source.contrast_id)
        positive_id, negative_id = str(source.positive_state_id), str(source.negative_state_id)
        positive_entry, negative_entry = endpoints[positive_id], endpoints[negative_id]
        positive, negative = positive_entry["record"], negative_entry["record"]
        matched, unmatched_fields, orientation_valid = True, "", True
        if str(source.contrast_kind) == "persistence":
            try:
                result = audit_exact_pair(positive, negative, _definition_for(source))
                matched = bool(result["matched"])
                orientation_valid = bool(result["orientation_valid"])
            except ValueError as error:
                matched, orientation_valid = False, False
                unmatched_fields = str(error)
        else:
            matched, unmatched_fields, orientation_valid = _audit_nuisance(
                source, positive, negative
            )
        audit_rows.append(
            {
                "task": str(source.task),
                "contrast_family": str(source.manipulation),
                "pair_id": pair_id,
                "matched": matched,
                "unmatched_fields": unmatched_fields,
                "orientation_valid": orientation_valid,
            }
        )
        if not matched or not orientation_valid:
            continue
        p21, p22 = positive_entry["h21"].astype(np.float32), positive_entry["h22"].astype(np.float32)
        n21, n22 = negative_entry["h21"].astype(np.float32), negative_entry["h22"].astype(np.float32)
        changes = {
            "l21": p21 - n21,
            "displacement": (p22 - p21) - (n22 - n21),
            "l22": p22 - n22,
        }
        for stage, values in changes.items():
            hidden[stage].append(values.astype(np.float16))
        is_persistence = str(source.contrast_kind) == "persistence"
        endpoint_target = (
            lambda record: float(record["persistence_logit"])
            if is_persistence
            else _nuisance_target(record)
        )
        observed_effect = endpoint_target(positive) - endpoint_target(negative)
        inventory_effect = float(source.behavior_effect)
        if not np.isclose(observed_effect, inventory_effect, atol=2e-4):
            raise ValueError(f"behavior-effect mismatch for contrast {pair_id}")
        metadata_rows.append(
            {
                "task": str(source.task),
                "manipulation": str(source.manipulation),
                "contrast_kind": str(source.contrast_kind),
                "nuisance_type": "" if pd.isna(source.nuisance_type) else str(source.nuisance_type),
                "contrast_id": pair_id,
                "episode_id": str(source.cluster_id),
                "pair_id": str(source.cluster_id),
                "split": str(source.split),
                "positive_state_id": positive_id,
                "negative_state_id": negative_id,
            }
        )
        target_row = {
            "persistence_policy_change": observed_effect if is_persistence else np.nan,
            "nuisance_policy_change": observed_effect if not is_persistence else np.nan,
            "gru_prediction_change": np.nan,
            "history_prediction_change": np.nan,
            **{name: np.nan for name in set(COMPONENT_TARGET_BY_MANIPULATION.values())},
        }
        if is_persistence:
            component = COMPONENT_TARGET_BY_MANIPULATION[str(source.manipulation)]
            target_row[component] = _component_change(
                str(source.manipulation), positive, negative
            )
        target_rows.append(target_row)
        for polarity, state_id, record, h21, h22 in (
            ("positive", positive_id, positive, p21, p22),
            ("negative", negative_id, negative, n21, n22),
        ):
            absolute_rows.append(
                {
                    "task": str(source.task),
                    "manipulation": str(source.manipulation),
                    "contrast_kind": str(source.contrast_kind),
                    "contrast_id": pair_id,
                    "episode_id": str(source.cluster_id),
                    "pair_id": str(source.cluster_id),
                    "split": str(source.split),
                    "state_id": state_id,
                    "polarity": polarity,
                    "persistence_policy": endpoint_target(record),
                    "gru_prediction": np.nan,
                    "history_prediction": np.nan,
                }
            )
            absolute_h21.append(h21.astype(np.float16))
            absolute_h22.append(h22.astype(np.float16))
            retained_endpoint_records[state_id] = record
    metadata = pd.DataFrame(metadata_rows)
    targets = pd.DataFrame(target_rows)
    absolute_metadata = pd.DataFrame(absolute_rows)
    if metadata.empty:
        raise ValueError("pair audit retained no valid contrasts")
    arrays = {stage: np.stack(values) for stage, values in hidden.items()}
    absolute = {
        "l21": np.stack(absolute_h21),
        "l22": np.stack(absolute_h22),
    }
    projected = {
        stage: values.astype(np.float32) @ frozen.basis for stage, values in arrays.items()
    }
    return {
        "audit": pd.DataFrame(audit_rows),
        "metadata": metadata,
        "targets": targets,
        "hidden": arrays,
        "projected": projected,
        "absolute_metadata": absolute_metadata,
        "absolute_hidden": absolute,
        "factorial_index": factorial_index,
        "endpoint_records": retained_endpoint_records,
    }


def _tail_streak(outcomes):
    if not outcomes or outcomes[-1] == 0:
        return 0.0, 0.0
    positive = outcomes[-1] > 0
    length = 0
    for value in reversed(outcomes):
        if (value > 0) != positive or value == 0:
            break
        length += 1
    return (0.0, float(length)) if positive else (float(length), 0.0)


def _bandit_factorial_model_record(source, split):
    choices = json.loads(source["choice_history"]) if isinstance(source["choice_history"], str) else list(source["choice_history"])
    outcomes = json.loads(source["reward_history"]) if isinstance(source["reward_history"], str) else list(source["reward_history"])
    action_values = [1.0 if str(value).upper() in {"A", "B"} else 0.0 for value in choices]
    outcome_values = [float(value) for value in outcomes]
    failure_streak, success_streak = _tail_streak(outcome_values)
    successes = {"A": 1.0, "B": 1.0}
    failures = {"A": 1.0, "B": 1.0}
    for action, outcome in zip(choices, outcome_values):
        action = str(action).upper()
        if action not in successes:
            continue
        if outcome > 0:
            successes[action] += 1
        else:
            failures[action] += 1
    estimates = {
        arm: 5.0 * successes[arm] / (successes[arm] + failures[arm]) - 2.0
        for arm in ("A", "B")
    }
    stop = int(float(source["stop_payoff"]))
    continuation = int(float(source["continue_bonus"]))
    base_state = str(source["state_id"])
    state_id = f"{base_state}:stop={stop}:continue={continuation}"
    round_index = int(source["round"])
    estimated_continue = max(estimates.values()) + continuation
    row = {
        "task": "bandit",
        "episode_id": f"{source['episode_id']}|stop={stop}|continue={continuation}",
        "pair_id": str(source["episode_id"]),
        "state_id": state_id,
        "round": round_index,
        "split": str(split),
        "persistence_logit": float(source["persistence_logit"]),
        "task_bandit": 1.0,
        "task_foraging": 0.0,
        "task_solvability": 0.0,
        "log_round": math.log1p(round_index),
        "normalized_time": round_index / max(1, round_index + 1),
        "previous_outcome": outcome_values[-1] if outcome_values else 0.0,
        "failure_streak": failure_streak,
        "success_streak": success_streak,
        "previous_choice": action_values[-1] if action_values else 0.0,
        "second_previous_choice": action_values[-2] if len(action_values) > 1 else 0.0,
        "cumulative_progress": float(sum(outcome_values)),
        "estimated_continue_value": estimated_continue,
        "estimated_outside_value": float(stop),
        "cost_pressure": float(-continuation),
        "progress_evidence": outcome_values[-1] if outcome_values else 0.0,
        "termination_advantage": estimated_continue - stop,
    }
    for lag in (1, 2, 3, 5):
        row[f"action_lag_{lag}"] = action_values[-lag] if len(action_values) >= lag else 0.0
        row[f"outcome_lag_{lag}"] = outcome_values[-lag] if len(outcome_values) >= lag else 0.0
    return row


def add_behavioral_model_targets(dataset, config, *, logger=None):
    """Apply the validation-selected GRU and finite-history models to endpoints."""

    records_dir = Path(config["paths"]["behavior_records"])
    organic = {
        task: pd.read_csv(records_dir / f"{task}_records.csv").to_dict(orient="records")
        for task in ("bandit", "foraging", "solvability")
    }
    train = [row for task in organic.values() for row in task if row["split"] == "train"]
    validation = [row for task in organic.values() for row in task if row["split"] == "validation"]
    factorial_split = json.loads(Path(config["paths"]["bandit_split"]).read_text(encoding="utf-8"))
    split_lookup = {
        str(episode): str(name)
        for name, episodes in factorial_split.items()
        for episode in episodes
    }
    factorial_application = [
        _bandit_factorial_model_record(source, split_lookup[str(source["episode_id"])])
        for source in dataset["factorial_index"].values()
    ]
    application_by_task = {
        "bandit": sorted(factorial_application, key=lambda row: (row["episode_id"], row["round"])),
        "foraging": organic["foraging"],
        "solvability": organic["solvability"],
    }
    application = [row for task in application_by_task.values() for row in task]
    selected = json.loads(
        (Path(config["paths"]["model_zoo_output"]) / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    gru_key = "gru::observable::shared_architecture_task_observation"
    gru_settings = selected[gru_key]
    zoo_config = _load_yaml(config["paths"]["model_zoo_config"])
    if logger is not None:
        logger.note("behavioral_targets", f"refitting frozen validated GRU for {len(application)} endpoint-context states")
    gru = fit_gru_ceiling(
        train,
        validation,
        application,
        gru_settings["feature_names"],
        hidden_size=int(gru_settings["selected"]["hidden_size"]),
        learning_rate=float(gru_settings["selected"]["learning_rate"]),
        dropout=float(gru_settings["selected"]["dropout"]),
        max_epochs=int(zoo_config["gru"]["max_epochs"]),
        patience=int(zoo_config["gru"]["early_stopping_patience"]),
        seed=int(zoo_config["seed"]),
    )
    gru_lookup = {
        str(row["state_id"]): float(value)
        for row, value in zip(application, gru["prediction"])
    }
    definition = next(row for row in MODEL_DEFINITIONS if row.name == "finite_history")
    history_settings = selected["finite_history::observable::task_specific"]["selected"]
    history_lookup = {}
    for task, task_application in application_by_task.items():
        fit_records = [row for row in organic[task] if row["split"] in {"train", "validation"}]
        prediction, _parameters, _states, _features = _fit_apply(
            fit_records,
            task_application,
            definition,
            "observable",
            history_settings[task],
            "shared_architecture_task_observation",
        )
        history_lookup.update(
            {str(row["state_id"]): float(value) for row, value in zip(task_application, prediction)}
        )
    metadata, targets = dataset["metadata"], dataset["targets"]
    for index, row in metadata.iterrows():
        positive, negative = str(row.positive_state_id), str(row.negative_state_id)
        targets.loc[index, "gru_prediction_change"] = gru_lookup[positive] - gru_lookup[negative]
        targets.loc[index, "history_prediction_change"] = history_lookup[positive] - history_lookup[negative]
    absolute = dataset["absolute_metadata"]
    for index, row in absolute.iterrows():
        state_id = str(row.state_id)
        if state_id in gru_lookup:
            absolute.loc[index, "gru_prediction"] = gru_lookup[state_id]
            absolute.loc[index, "history_prediction"] = history_lookup[state_id]
    persistence = metadata.contrast_kind.astype(str) == "persistence"
    if targets.loc[persistence, list(PERSISTENCE_TARGETS)].isna().any().any():
        raise ValueError("behavioral model predictions are missing persistence endpoints")
    dataset["targets"] = targets
    dataset["absolute_metadata"] = absolute
    return dataset

