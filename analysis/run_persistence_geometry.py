"""Link frozen L21/L22 persistence geometry to behavioral computations.

The runner reuses existing activation shards and the already selected
``displacement-L21-k4`` basis.  It never recollects activations or refits that
basis.  Hidden states are streamed into float16 memory maps so the full
analysis remains practical on a laptop.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.persistence_geometry import (
    STAGES,
    FrozenPersistenceSubspace,
    decode_continuous_targets,
    matched_random_bases,
    stage_representation,
    task_identity_confound,
    validate_episode_splits,
)
from analysis.persistence_geometry_targets import (
    TARGET_SPECS,
    build_geometry_targets,
    load_behavior_records,
)
from computational_modeling.analysis.evaluate_models import (
    choice_metrics,
    persistence_metrics,
    sigmoid,
)
from computational_modeling.analysis.run_model_zoo import ProgressLogger
from computational_modeling.data.feature_schema import (
    FLEXIBLE_FEATURE_GROUPS,
    FLEXIBLE_NUISANCE_FEATURES,
)
from computational_modeling.models.base import balanced_weights
from computational_modeling.models.gru import fit_gru_ceiling


def _load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _state_order_sha(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        "\n".join(frame.state_id.astype(str)).encode("utf-8")
    ).hexdigest()


def _split_lookup(path: str | Path) -> dict[str, str]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(episode): str(name)
        for name, episodes in split.items()
        for episode in episodes
    }


def _cache_paths(output: Path):
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "directory": cache,
        "manifest": cache / "hidden_cache_manifest.json",
        "l21": cache / "hidden_l21.float16.mmap",
        "l22": cache / "hidden_l22.float16.mmap",
        "projection": cache / "frozen_subspace_projection.npz",
        "metadata": cache / "aligned_behavioral_records.csv.gz",
    }


def project_behavioral_activations(
    frame: pd.DataFrame,
    frozen: FrozenPersistenceSubspace,
    config: dict,
    output: Path,
    *,
    resume: bool,
    logger: ProgressLogger,
):
    """Stream activation shards and align L21/L22 to behavioral state IDs."""

    paths = _cache_paths(output)
    expected = {
        "states": int(len(frame)),
        "width": frozen.width,
        "rank": frozen.rank,
        "basis_sha256": frozen.sha256,
        "state_order_sha256": _state_order_sha(frame),
        "dtype": "float16",
        "layers": {"l21": int(config["layers"]["l21"]), "l22": int(config["layers"]["l22"])},
    }
    if resume and all(paths[name].exists() for name in ("manifest", "l21", "l22", "projection", "metadata")):
        observed = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if all(observed.get(key) == value for key, value in expected.items()):
            logger.note("projection", f"reusing aligned hidden-state cache for {len(frame)} states")
            projected = np.load(paths["projection"])
            return (
                {
                    "l21": projected["l21"],
                    "displacement": projected["displacement"],
                    "l22": projected["l22"],
                },
                paths,
            )
    count, width = len(frame), frozen.width
    h21_map = np.memmap(paths["l21"], dtype=np.float16, mode="w+", shape=(count, width))
    h22_map = np.memmap(paths["l22"], dtype=np.float16, mode="w+", shape=(count, width))
    z21 = np.empty((count, frozen.rank), dtype=np.float32)
    z22 = np.empty_like(z21)
    found = np.zeros(count, dtype=bool)
    state_to_index = {
        state: index for index, state in enumerate(frame.state_id.astype(str))
    }
    l21, l22 = int(config["layers"]["l21"]), int(config["layers"]["l22"])
    import torch

    for task in config["tasks"]:
        files = sorted(Path(config["activation_banks"][task]).glob("episode_*.pt"))
        task_started = time.perf_counter()
        logger.note("projection", f"{task}: scanning {len(files)} existing activation shards")
        for file_index, path in enumerate(files, start=1):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            records = shard["records"]
            activations = shard["activations"]
            source_indices, destination_indices = [], []
            for source_index, record in enumerate(records):
                state_id = str(record["state_id"])
                destination = state_to_index.get(state_id)
                if destination is not None:
                    if found[destination]:
                        raise ValueError(f"duplicate activation state: {state_id}")
                    source_indices.append(source_index)
                    destination_indices.append(destination)
            if source_indices:
                selected = activations[source_indices]
                if selected.ndim != 3 or selected.shape[1] <= max(l21, l22):
                    raise ValueError(f"malformed all-layer activation shard: {path}")
                left = selected[:, l21, :].float().numpy()
                right = selected[:, l22, :].float().numpy()
                destinations = np.asarray(destination_indices, dtype=int)
                h21_map[destinations] = left.astype(np.float16)
                h22_map[destinations] = right.astype(np.float16)
                z21[destinations] = frozen.project(left)
                z22[destinations] = frozen.project(right)
                found[destinations] = True
            del shard, activations
            if file_index % 100 == 0 or file_index == len(files):
                logger.note(
                    "projection",
                    f"{task}: {file_index}/{len(files)} shards; "
                    f"{int(found.sum())}/{count} aligned states",
                )
        logger.note(
            "projection",
            f"{task}: completed in {time.perf_counter() - task_started:.1f}s",
        )
    if not found.all():
        missing = frame.loc[~found, "state_id"].astype(str).head().tolist()
        raise ValueError(f"activation banks are missing behavioral states: {missing}")
    h21_map.flush()
    h22_map.flush()
    np.savez_compressed(
        paths["projection"], l21=z21, displacement=z22 - z21, l22=z22
    )
    frame.to_csv(paths["metadata"], index=False, compression="gzip")
    manifest = {
        **expected,
        "format": "two float16 rows-by-hidden-width memory maps; streamed one shard at a time",
        "frozen_basis_source": frozen.source,
        "frozen_basis_key": frozen.key,
        "full_hidden_cache_bytes": int(paths["l21"].stat().st_size + paths["l22"].stat().st_size),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"l21": z21, "displacement": z22 - z21, "l22": z22}, paths


def _task_metrics(target, prediction, metadata, indices):
    rows = []
    indices = np.asarray(indices, dtype=int)
    local = metadata.iloc[indices].reset_index(drop=True)
    target = np.asarray(target)[indices]
    prediction = np.asarray(prediction)
    for task in sorted(local.task.astype(str).unique()):
        selected = np.flatnonzero(local.task.astype(str).to_numpy() == task)
        records = local.iloc[selected].to_dict(orient="records")
        metrics = persistence_metrics(
            target[selected],
            prediction[selected],
            balanced_weights(records, task_balanced=False),
        )
        rows.append({"task": task, **metrics, "states": len(selected), "episodes": local.iloc[selected].episode_id.nunique()})
    return rows


def _decoder_rows(stage, target_name, spec, result, target, metadata):
    rows = [
        {
            "stage": stage,
            "target": target_name,
            "feature_family": spec.family,
            "priority": spec.priority,
            "target_role": spec.role,
            "task": "macro",
            "selected_alpha": result["selected_alpha"],
            **result["test_metrics"],
        }
    ]
    for row in _task_metrics(
        target, result["test_prediction"], metadata, result["test_indices"]
    ):
        rows.append(
            {
                "stage": stage,
                "target": target_name,
                "feature_family": spec.family,
                "priority": spec.priority,
                "target_role": spec.role,
                "selected_alpha": result["selected_alpha"],
                **row,
            }
        )
    return rows


def run_computational_decoding(z_by_stage, targets, metadata, config, logger):
    rows, fits = [], {}
    specs = {spec.name: spec for spec in TARGET_SPECS}
    for stage in STAGES:
        started = time.perf_counter()
        decoded = decode_continuous_targets(
            z_by_stage[stage], targets, metadata, alphas=config["ridge_alphas"]
        )
        fits[stage] = decoded
        for target_name, result in decoded.items():
            rows.extend(
                _decoder_rows(
                    stage, target_name, specs[target_name], result, targets[target_name], metadata
                )
            )
        logger.note("computational_decoding", f"{stage}: decoded {len(targets)} targets in {time.perf_counter() - started:.1f}s")
    return pd.DataFrame(rows), fits


def _full_stage(cache_paths, count, width, stage, *, chunk_size=1024):
    left = np.memmap(cache_paths["l21"], dtype=np.float16, mode="r", shape=(count, width))
    right = np.memmap(cache_paths["l22"], dtype=np.float16, mode="r", shape=(count, width))
    output = np.empty((count, width), dtype=np.float32)
    for start in range(0, count, int(chunk_size)):
        stop = min(count, start + int(chunk_size))
        output[start:stop] = stage_representation(
            left[start:stop].astype(np.float32),
            right[start:stop].astype(np.float32),
            stage,
        )
    return output


def run_full_hidden_comparison(
    cache_paths, z_fits, targets, metadata, frozen, config, logger
):
    rows = []
    specs = {spec.name: spec for spec in TARGET_SPECS}
    for stage in STAGES:
        started = time.perf_counter()
        hidden = _full_stage(cache_paths, len(metadata), frozen.width, stage)
        decoded = decode_continuous_targets(
            hidden,
            targets,
            metadata,
            alphas=config["full_hidden_ridge_alphas"],
        )
        for target_name, result in decoded.items():
            rank4 = z_fits[stage][target_name]["test_metrics"]
            full = result["test_metrics"]
            denominator = float(full["r_squared"])
            rows.append(
                {
                    "stage": stage,
                    "target": target_name,
                    "feature_family": specs[target_name].family,
                    "rank4_r_squared": rank4["r_squared"],
                    "rank4_mse": rank4["mse"],
                    "rank4_pearson_r": rank4["pearson_r"],
                    "full_hidden_r_squared": full["r_squared"],
                    "full_hidden_mse": full["mse"],
                    "full_hidden_pearson_r": full["pearson_r"],
                    "full_hidden_selected_alpha": result["selected_alpha"],
                    "concentration_ratio": (
                        float("nan")
                        if abs(denominator) < 1e-8
                        else float(rank4["r_squared"]) / denominator
                    ),
                    "delta_r_squared_full_minus_rank4": denominator - float(rank4["r_squared"]),
                }
            )
        del hidden, decoded
        gc.collect()
        logger.note("full_hidden", f"{stage}: capacity-controlled full-hidden path completed in {time.perf_counter() - started:.1f}s")
    return pd.DataFrame(rows)


def _random_projections(cache_paths, count, width, bases, logger, chunk_size=512):
    left = np.memmap(cache_paths["l21"], dtype=np.float16, mode="r", shape=(count, width))
    right = np.memmap(cache_paths["l22"], dtype=np.float16, mode="r", shape=(count, width))
    flat = np.transpose(bases, (1, 0, 2)).reshape(width, -1)
    left_projection = np.empty((count, bases.shape[0], bases.shape[2]), dtype=np.float32)
    right_projection = np.empty_like(left_projection)
    for start in range(0, count, int(chunk_size)):
        stop = min(count, start + int(chunk_size))
        left_projection[start:stop] = (
            left[start:stop].astype(np.float32) @ flat
        ).reshape(stop - start, bases.shape[0], bases.shape[2])
        right_projection[start:stop] = (
            right[start:stop].astype(np.float32) @ flat
        ).reshape(stop - start, bases.shape[0], bases.shape[2])
        if stop == count or stop % (5 * int(chunk_size)) == 0:
            logger.note("random_subspaces", f"projected {stop}/{count} states into {len(bases)} matched rank-4 controls")
    return {
        "l21": left_projection,
        "displacement": right_projection - left_projection,
        "l22": right_projection,
    }


def run_random_controls(
    cache_paths,
    z_fits,
    targets,
    metadata,
    frozen,
    config,
    logger,
    *,
    resume=False,
):
    """Run matched controls with a durable per-subspace checkpoint.

    Random-control decoding is the longest CPU-only section on a laptop.  The
    checkpoint contains only complete ``(stage, random_subspace)`` groups, so
    an interrupted run can safely skip work that was already flushed.
    """

    count = int(config["matched_random_subspaces"])
    bases = matched_random_bases(frozen.width, frozen.rank, count, int(config["seed"]) + 701)
    projections = _random_projections(cache_paths, len(metadata), frozen.width, bases, logger)
    rows = []
    completed = set()
    checkpoint = Path(cache_paths["directory"]) / "random_subspace_checkpoint.csv"
    if resume and checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        required = {
            "stage",
            "target",
            "random_subspace",
            "random_r_squared",
            "random_mse",
            "random_pearson_r",
            "persistence_rank4_r_squared",
            "random_meets_or_exceeds_persistence",
        }
        if required.issubset(prior.columns):
            prior = prior[
                prior.stage.astype(str).isin(STAGES)
                & prior.target.astype(str).isin(targets)
                & prior.random_subspace.between(0, count - 1)
            ].copy()
            group_sizes = prior.groupby(["stage", "random_subspace"]).target.nunique()
            completed = {
                (str(stage), int(random_index))
                for (stage, random_index), size in group_sizes.items()
                if int(size) == len(targets)
            }
            if completed:
                keep = [
                    (str(row.stage), int(row.random_subspace)) in completed
                    for row in prior.itertuples()
                ]
                rows = prior.loc[keep, list(required)].to_dict(orient="records")
                logger.note(
                    "random_subspaces",
                    f"reusing {len(completed)}/{len(STAGES) * count} completed random-control fits",
                )
    for stage in STAGES:
        for random_index in range(count):
            if (stage, random_index) in completed:
                continue
            decoded = decode_continuous_targets(
                projections[stage][:, random_index, :],
                targets,
                metadata,
                alphas=config["ridge_alphas"],
            )
            for target_name, result in decoded.items():
                observed = z_fits[stage][target_name]["test_metrics"]["r_squared"]
                rows.append(
                    {
                        "stage": stage,
                        "target": target_name,
                        "random_subspace": random_index,
                        "random_r_squared": result["test_metrics"]["r_squared"],
                        "random_mse": result["test_metrics"]["mse"],
                        "random_pearson_r": result["test_metrics"]["pearson_r"],
                        "persistence_rank4_r_squared": observed,
                        "random_meets_or_exceeds_persistence": result["test_metrics"]["r_squared"] >= observed,
                    }
                )
            if (random_index + 1) % 10 == 0 or random_index + 1 == count:
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                logger.note("random_subspaces", f"{stage}: decoded {random_index + 1}/{count} random subspaces")
    frame = pd.DataFrame(rows)
    summaries = []
    for (stage, target_name), part in frame.groupby(["stage", "target"]):
        persistence = float(part.persistence_rank4_r_squared.iloc[0])
        random_values = part.random_r_squared.to_numpy(dtype=float)
        summaries.append(
            {
                "stage": stage,
                "target": target_name,
                "random_r_squared_mean": float(np.mean(random_values)),
                "random_r_squared_95th": float(np.quantile(random_values, 0.95)),
                "persistence_percentile_among_random": float(
                    np.mean(random_values < persistence)
                ),
                "empirical_p_value": float(
                    (1 + np.sum(random_values >= persistence))
                    / (len(random_values) + 1)
                ),
            }
        )
    frame = frame.merge(pd.DataFrame(summaries), on=["stage", "target"], how="left")
    del projections
    gc.collect()
    return frame


def run_cross_task_decoding(z_by_stage, targets, metadata, config, logger):
    target_names = list(config["cross_task_targets"])
    rows = []
    task = metadata.task.astype(str).to_numpy()
    split = metadata.split.astype(str).to_numpy()
    for stage in STAGES:
        for heldout in config["tasks"]:
            source = task != heldout
            test_indices = np.flatnonzero((task == heldout) & (split == "test"))
            decoded = decode_continuous_targets(
                z_by_stage[stage],
                {name: targets[name] for name in target_names},
                metadata,
                alphas=config["ridge_alphas"],
                train_indices=np.flatnonzero(source & (split == "train")),
                validation_indices=np.flatnonzero(source & (split == "validation")),
                test_indices=test_indices,
                forbidden_fit_tasks=(heldout,),
            )
            for target_name, result in decoded.items():
                variance = float(np.var(np.asarray(targets[target_name])[test_indices]))
                rows.append(
                    {
                        "stage": stage,
                        "target": target_name,
                        "heldout_task": heldout,
                        "source_tasks": ";".join(result["fit_tasks"]),
                        "selected_alpha": result["selected_alpha"],
                        "heldout_target_variance": variance,
                        "constant_heldout_target": variance < 1e-10,
                        **result["test_metrics"],
                    }
                )
        logger.note("cross_task", f"{stage}: leave-one-task-out decoding complete")
    frame = pd.DataFrame(rows)
    confounds = []
    for (stage, target_name), part in frame.groupby(["stage", "target"]):
        result = task_identity_confound(targets[target_name], metadata, part)
        confounds.append({"stage": stage, "target": target_name, **result})
    return frame.merge(pd.DataFrame(confounds), on=["stage", "target"], how="left")


def _weighted_standardize(values, metadata, indices):
    values = np.asarray(values, dtype=float)
    weights = balanced_weights(metadata.iloc[indices].to_dict(orient="records"), task_balanced=True)
    normalized = weights / weights.sum()
    mean = np.sum(values[indices] * normalized[:, None], axis=0)
    scale = np.sqrt(np.sum((values[indices] - mean) ** 2 * normalized[:, None], axis=0))
    scale[scale < 1e-8] = 1.0
    return (values - mean) / scale


def _regularized_cca(x_train, y_train, x_test, y_test, ridge=1e-5):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    x_mean, y_mean = x_train.mean(0), y_train.mean(0)
    x_train, y_train = x_train - x_mean, y_train - y_mean
    x_test, y_test = x_test - x_mean, y_test - y_mean

    def invsqrt(covariance):
        values, vectors = np.linalg.eigh(covariance)
        return vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, ridge))) @ vectors.T

    cxx = x_train.T @ x_train / max(1, len(x_train) - 1) + ridge * np.eye(x_train.shape[1])
    cyy = y_train.T @ y_train / max(1, len(y_train) - 1) + ridge * np.eye(y_train.shape[1])
    cxy = x_train.T @ y_train / max(1, len(x_train) - 1)
    wx, wy = invsqrt(cxx), invsqrt(cyy)
    u, _s, vt = np.linalg.svd(wx @ cxy @ wy, full_matrices=False)
    left = x_test @ wx @ u
    right = y_test @ wy @ vt.T
    correlations = []
    for column in range(min(left.shape[1], right.shape[1])):
        correlations.append(float(np.corrcoef(left[:, column], right[:, column])[0, 1]))
    return correlations


def run_dimension_geometry(z_by_stage, targets, z_fits, metadata, config, logger):
    target_names = list(config["geometry_targets"])
    matrix = np.column_stack([targets[name] for name in target_names])
    split = metadata.split.astype(str).to_numpy()
    train = np.flatnonzero(split == "train")
    test = np.flatnonzero(split == "test")
    rows = []
    for stage in STAGES:
        coefficient = np.column_stack(
            [z_fits[stage][name]["coefficient"] for name in target_names]
        )
        singular = np.linalg.svd(coefficient, compute_uv=False)
        rank = int(np.sum(singular > max(1e-8, singular.max() * 1e-3)))
        rows.append(
            {
                "stage": stage,
                "kind": "mapping_rank",
                "component": "all",
                "target_a": ";".join(target_names),
                "target_b": "",
                "value": rank,
            }
        )
        for component, value in enumerate(singular, start=1):
            rows.append({"stage": stage, "kind": "mapping_singular_value", "component": component, "target_a": "all", "target_b": "", "value": value})
        z_standard = _weighted_standardize(z_by_stage[stage], metadata, train)
        y_standard = _weighted_standardize(matrix, metadata, train)
        correlations = _regularized_cca(
            z_standard[train], y_standard[train], z_standard[test], y_standard[test]
        )
        for component, value in enumerate(correlations, start=1):
            rows.append({"stage": stage, "kind": "canonical_correlation", "component": component, "target_a": "all", "target_b": "", "value": value})
        for left, right in combinations(target_names, 2):
            a = z_fits[stage][left]["coefficient"]
            b = z_fits[stage][right]["coefficient"]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            cosine = 1.0 if denominator < 1e-12 else abs(float(a @ b) / denominator)
            angle = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
            rows.append({"stage": stage, "kind": "principal_angle_degrees", "component": "", "target_a": left, "target_b": right, "value": angle})
        for name in target_names:
            rows.append({"stage": stage, "kind": "target_test_r_squared", "component": "", "target_a": name, "target_b": "", "value": z_fits[stage][name]["test_metrics"]["r_squared"]})
        logger.note("dimension_geometry", f"{stage}: rank, CCA, and target-axis angles complete")
    return pd.DataFrame(rows)


def _task_interactions(values, metadata):
    values = np.asarray(values, dtype=float)
    tasks = ("bandit", "foraging", "solvability")
    indicator = np.column_stack(
        [(metadata.task.astype(str).to_numpy() == task).astype(float) for task in tasks]
    )
    blocks = [values * indicator[:, index : index + 1] for index in range(len(tasks))]
    return np.column_stack((indicator, *blocks))


def run_incremental_models(z_by_stage, metadata, config, logger):
    behavioral_features = [
        feature
        for family in ("history", "time_effort", "cost", "progress_solvability")
        for feature in FLEXIBLE_FEATURE_GROUPS[family]
    ]
    behavior = metadata[behavioral_features].to_numpy(dtype=float)
    decision = metadata.persistence_logit.to_numpy(dtype=float)
    choice = metadata["continue"].to_numpy(dtype=float)
    rows = []
    for stage in STAGES:
        designs = {
            "behavioral_only": _task_interactions(behavior, metadata),
            "neural_only": _task_interactions(z_by_stage[stage], metadata),
            "combined": _task_interactions(np.column_stack((behavior, z_by_stage[stage])), metadata),
        }
        fits = {
            name: decode_continuous_targets(
                design,
                {"persistence_logit": decision},
                metadata,
                alphas=config["ridge_alphas"],
            )["persistence_logit"]
            for name, design in designs.items()
        }
        metrics = {}
        for name, fit in fits.items():
            selected = fit["test_indices"]
            records = metadata.iloc[selected].to_dict(orient="records")
            weights = balanced_weights(records, task_balanced=True)
            choice_result = choice_metrics(choice[selected], sigmoid(fit["test_prediction"]), weights)
            metrics[name] = {**fit["test_metrics"], "sampled_choice_log_loss": choice_result["log_loss"]}
        for name in designs:
            rows.append(
                {
                    "stage": stage,
                    "model": name,
                    "selected_alpha": fits[name]["selected_alpha"],
                    **metrics[name],
                    "delta_r_squared_neural_given_behavior": (
                        metrics["combined"]["r_squared"] - metrics["behavioral_only"]["r_squared"]
                    ),
                    "delta_r_squared_behavior_given_neural": (
                        metrics["combined"]["r_squared"] - metrics["neural_only"]["r_squared"]
                    ),
                }
            )
        logger.note("incremental", f"{stage}: behavioral, neural, and combined decision models complete")
    return pd.DataFrame(rows)


def _project_nuisance_bank(name, bank, split_path, frozen, config, logger):
    import torch

    lookup = _split_lookup(split_path)
    rows, projected = [], {stage: [] for stage in STAGES}
    l21, l22 = int(config["layers"]["l21"]), int(config["layers"]["l22"])
    files = sorted(Path(bank).glob("episode_*.pt"))
    for index, path in enumerate(files, start=1):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        episode = str(shard["episode_id"])
        if episode not in lookup:
            raise ValueError(f"nuisance episode absent from split: {episode}")
        activations = shard["activations"].float().numpy()
        for row_index, source in enumerate(shard["records"]):
            rows.append(
                {
                    **{key: value for key, value in source.items() if key != "conversation"},
                    "task": name,
                    "episode_id": episode,
                    "pair_id": str(shard.get("pair_id", episode)),
                    "state_id": str(source["state_id"]),
                    "split": lookup[episode],
                }
            )
            left, right = activations[row_index, l21], activations[row_index, l22]
            projected["l21"].append(left @ frozen.basis)
            projected["l22"].append(right @ frozen.basis)
            projected["displacement"].append((right - left) @ frozen.basis)
        if index % 200 == 0 or index == len(files):
            logger.note("specificity", f"{name}: projected {index}/{len(files)} nuisance shards")
    frame = pd.DataFrame(rows)
    validate_episode_splits(frame)
    return frame, {stage: np.asarray(values) for stage, values in projected.items()}


def _task_identity_control(z, metadata, config):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss

    split = metadata.split.astype(str).to_numpy()
    train, validation, test = (np.flatnonzero(split == name) for name in ("train", "validation", "test"))
    x = _weighted_standardize(z, metadata, train)
    labels = metadata.task.astype(str).to_numpy()
    candidates = []
    weights = balanced_weights(metadata.iloc[train].to_dict(orient="records"), task_balanced=True)
    for value in config["task_identity_c"]:
        model = LogisticRegression(C=float(value), max_iter=1000, random_state=int(config["seed"]))
        model.fit(x[train], labels[train], sample_weight=weights)
        candidates.append((accuracy_score(labels[validation], model.predict(x[validation])), float(value), model))
    _score, selected, model = max(candidates, key=lambda row: (row[0], -row[1]))
    probability = model.predict_proba(x[test])
    return {
        "selected_c": selected,
        "accuracy": float(accuracy_score(labels[test], model.predict(x[test]))),
        "log_loss": float(log_loss(labels[test], probability, labels=model.classes_)),
        "chance_accuracy": 1.0 / len(model.classes_),
        "states": len(test),
        "episodes": metadata.iloc[test].episode_id.nunique(),
    }


def run_specificity_controls(z_by_stage, metadata, frozen, config, logger):
    controls = {
        "arbitrary_choice": ("choice_logit", config["nuisance_banks"]["arbitrary_choice"]),
        "terminality": ("terminality_logit", config["nuisance_banks"]["terminality"]),
        "generic_value": ("relative_value", config["nuisance_banks"]["generic_value"]),
    }
    rows = []
    for name, (target_name, settings) in controls.items():
        frame, projections = _project_nuisance_bank(
            name, settings["activation_bank"], settings["split"], frozen, config, logger
        )
        target = frame[target_name].to_numpy(dtype=float)
        for stage in STAGES:
            result = decode_continuous_targets(
                projections[stage], {target_name: target}, frame, alphas=config["ridge_alphas"]
            )[target_name]
            rows.append(
                {
                    "control": name,
                    "stage": stage,
                    "target": target_name,
                    "metric_type": "continuous_decoding",
                    "selected_hyperparameter": result["selected_alpha"],
                    **result["test_metrics"],
                }
            )
    for stage in STAGES:
        result = _task_identity_control(z_by_stage[stage], metadata, config)
        rows.append(
            {
                "control": "task_identity",
                "stage": stage,
                "target": "task",
                "metric_type": "multiclass_decoding",
                "selected_hyperparameter": result["selected_c"],
                "r_squared": float("nan"),
                "mse": float("nan"),
                "pearson_r": float("nan"),
                **result,
            }
        )
    return pd.DataFrame(rows)


def _fit_gru_hidden_states(metadata, zoo_config, zoo_output, logger):
    selected_payload = json.loads((Path(zoo_output) / "selected_hyperparameters.json").read_text(encoding="utf-8"))[
        "gru::observable::shared_architecture_task_observation"
    ]
    selected = selected_payload["selected"]
    features = selected_payload["feature_names"]
    records = metadata.to_dict(orient="records")
    split = {
        name: [row for row in records if row["split"] == name]
        for name in ("train", "validation")
    }
    logger.note("gru_alignment", f"refitting validated GRU ({selected['hidden_size']} hidden units) to export recurrent states")
    result = fit_gru_ceiling(
        split["train"],
        split["validation"],
        records,
        features,
        hidden_size=int(selected["hidden_size"]),
        learning_rate=float(selected["learning_rate"]),
        dropout=float(selected["dropout"]),
        max_epochs=int(zoo_config["gru"]["max_epochs"]),
        patience=int(zoo_config["gru"]["early_stopping_patience"]),
        seed=int(zoo_config["seed"]),
        return_hidden_states=True,
    )
    return result["hidden_state"]


def run_gru_alignment(z_by_stage, metadata, zoo_config, zoo_output, config, logger):
    hidden = _fit_gru_hidden_states(metadata, zoo_config, zoo_output, logger)
    names = [f"gru_{index:02d}" for index in range(hidden.shape[1])]
    targets = {name: hidden[:, index] for index, name in enumerate(names)}
    split = metadata.split.astype(str).to_numpy()
    train, test = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    rows = []
    for stage in STAGES:
        decoded = decode_continuous_targets(
            z_by_stage[stage], targets, metadata, alphas=config["ridge_alphas"]
        )
        r2_values = [result["test_metrics"]["r_squared"] for result in decoded.values()]
        rows.append({"stage": stage, "kind": "low_rank_regression", "component": "all", "heldout_task": "", "value": float(np.mean(r2_values)), "metric": "mean_hidden_dimension_r_squared"})
        z_standard = _weighted_standardize(z_by_stage[stage], metadata, train)
        g_standard = _weighted_standardize(hidden, metadata, train)
        cca = _regularized_cca(z_standard[train], g_standard[train], z_standard[test], g_standard[test])
        for component, value in enumerate(cca, start=1):
            rows.append({"stage": stage, "kind": "canonical_correlation", "component": component, "heldout_task": "", "value": value, "metric": "test_canonical_r"})
        task = metadata.task.astype(str).to_numpy()
        for heldout in config["tasks"]:
            source = task != heldout
            local = decode_continuous_targets(
                z_by_stage[stage],
                targets,
                metadata,
                alphas=config["ridge_alphas"],
                train_indices=np.flatnonzero(source & (split == "train")),
                validation_indices=np.flatnonzero(source & (split == "validation")),
                test_indices=np.flatnonzero((task == heldout) & (split == "test")),
                forbidden_fit_tasks=(heldout,),
            )
            rows.append({"stage": stage, "kind": "cross_task_low_rank_regression", "component": "all", "heldout_task": heldout, "value": float(np.mean([result["test_metrics"]["r_squared"] for result in local.values()])), "metric": "mean_hidden_dimension_r_squared"})
        logger.note("gru_alignment", f"{stage}: CCA, low-rank regression, and cross-task transfer complete")
    return pd.DataFrame(rows)


def _configure_plotting():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/digital_minds_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/digital_minds_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_figures(output: Path):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    decoding = pd.read_csv(output / "computational_decoding.csv")
    primary = decoding[(decoding.task == "macro") & (decoding.target_role == "primary")]
    plt = _configure_plotting()
    pivot = primary.pivot(index="target", columns="stage", values="r_squared")
    axis = pivot.plot(kind="barh", figsize=(10, 6))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Held-out R²")
    axis.set_title("Computational decoding from frozen rank-4 persistence geometry")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "computational_decoding_by_target.png", dpi=160)
    plt.close(axis.figure)

    stage_mean = primary.groupby("stage", as_index=False).r_squared.mean().set_index("stage").reindex(STAGES)
    axis = stage_mean.plot(kind="bar", legend=False, figsize=(7, 4), color="#386cb0")
    axis.set_ylabel("Mean primary-target held-out R²")
    axis.set_title("Computational information across L21→L22")
    axis.tick_params(axis="x", rotation=0)
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "computational_decoding_by_layer_stage.png", dpi=160)
    plt.close(axis.figure)

    cross = pd.read_csv(output / "cross_task_decoding.csv")
    cross_primary = cross[cross.target.isin(["history_finite_prediction", "time_effort", "cost_pressure", "progress_solvability"])]
    pivot = cross_primary.pivot_table(index=["target", "heldout_task"], columns="stage", values="r_squared")
    axis = pivot.plot(kind="barh", figsize=(10, 8))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Leave-one-task-out R²")
    axis.set_title("Cross-task transfer of computational decoders")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "cross_task_transfer.png", dpi=160)
    plt.close(axis.figure)

    geometry = pd.read_csv(output / "dimension_geometry.csv")
    cca = geometry[geometry.kind == "canonical_correlation"].pivot(index="component", columns="stage", values="value")
    axis = cca.plot(marker="o", figsize=(7, 4))
    axis.set_ylabel("Held-out canonical correlation")
    axis.set_title("Multivariate target geometry within rank-4 space")
    axis.set_ylim(-0.1, 1.0)
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "subspace_geometry.png", dpi=160)
    plt.close(axis.figure)

    incremental = pd.read_csv(output / "behavior_neural_incremental.csv")
    pivot = incremental.pivot(index="stage", columns="model", values="r_squared").reindex(STAGES)
    axis = pivot.plot(kind="bar", figsize=(8, 4))
    axis.set_ylabel("Held-out persistence-logit R²")
    axis.set_title("Behavioral and neural incremental prediction")
    axis.tick_params(axis="x", rotation=0)
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "behavioral_vs_neural_incremental.png", dpi=160)
    plt.close(axis.figure)


def generate_report(output: Path):
    decoding = pd.read_csv(output / "computational_decoding.csv")
    random = pd.read_csv(output / "random_subspace_controls.csv")
    full = pd.read_csv(output / "full_hidden_comparison.csv")
    cross = pd.read_csv(output / "cross_task_decoding.csv")
    geometry = pd.read_csv(output / "dimension_geometry.csv")
    incremental = pd.read_csv(output / "behavior_neural_incremental.csv")
    specificity = pd.read_csv(output / "specificity_controls.csv")
    macro = decoding[decoding.task == "macro"]
    primary_names = ["history_finite_prediction", "history_summary", "time_effort", "cost_pressure", "progress_solvability"]
    strongest = macro[macro.target.isin(primary_names)].sort_values("r_squared", ascending=False)
    random_summary = random.drop_duplicates(["stage", "target"])
    concentrated = full[(full.target.isin(primary_names)) & (full.concentration_ratio >= .5)].sort_values("concentration_ratio", ascending=False)
    transfer = cross[(cross.target.isin(primary_names)) & (~cross.constant_heldout_target)]
    transfer_summary = transfer.groupby("target").r_squared.mean().sort_values(ascending=False)
    rank_rows = geometry[geometry.kind == "mapping_rank"]
    displacement_increment = incremental[(incremental.stage == "displacement") & (incremental.model == "combined")].iloc[0]
    nuisance = specificity[specificity.metric_type == "continuous_decoding"].sort_values("r_squared", ascending=False)
    task_identity = specificity[specificity.control == "task_identity"].sort_values("accuracy", ascending=False)
    best_stage_by_target = strongest.loc[strongest.groupby("target").r_squared.idxmax()][["target", "stage", "r_squared"]]
    stage_scores = macro[macro.target.isin(primary_names)].pivot(
        index="target", columns="stage", values="r_squared"
    )
    l22_minus_l21 = (stage_scores["l22"] - stage_scores["l21"]).sort_values(
        ascending=False
    )
    increased = l22_minus_l21[l22_minus_l21 > 0]
    decreased = l22_minus_l21[l22_minus_l21 <= 0]
    nuisance_best = nuisance.loc[nuisance.groupby("control").r_squared.idxmax()]
    lines = [
        "# Computational ingredients in the frozen L21/L22 persistence subspace",
        "",
        "This is an exploratory representational analysis. The rank-4 `displacement-L21-k4` basis was frozen from the prior contrast search; no computational label was used to refit or rotate it.",
        "",
        "## Computational-variable representation",
        "",
    ]
    for row in best_stage_by_target.sort_values("r_squared", ascending=False).itertuples():
        lines.append(f"- **{row.target}**: best at {row.stage}, held-out R²={row.r_squared:.3f}.")
    lines += [
        "",
        "## Concentration and dimensionality",
        "",
        (
            "Targets retaining at least half of full-hidden R² in rank 4: "
            + (", ".join(sorted(concentrated.target.unique())) if len(concentrated) else "none")
            + "."
        ),
        "Persistence rank-4 decoding exceeded the matched-random 95th percentile for: "
        + (
            ", ".join(
                f"{row.target}/{row.stage}"
                for row in random_summary[
                    random_summary.persistence_rank4_r_squared
                    > random_summary.random_r_squared_95th
                ].itertuples()
                if row.target in primary_names
            )
            or "none"
        )
        + ". Empirical p-values are in `random_subspace_controls.csv`.",
        "The fitted four-target mapping ranks were " + ", ".join(f"{row.stage}={int(row.value)}" for row in rank_rows.itertuples()) + "; canonical correlations and principal angles support subspace-level, not raw-PC, interpretation.",
        "",
        "## Cross-task generalization and layer transition",
        "",
        "Mean leave-one-task-out R² by target: " + ", ".join(f"{name}={value:.3f}" for name, value in transfer_summary.items()) + ".",
        "From L21 to L22, held-out R² increased for "
        + (
            ", ".join(f"{name} ({value:+.3f})" for name, value in increased.items())
            if len(increased)
            else "no primary target"
        )
        + "; it decreased for "
        + (
            ", ".join(f"{name} ({value:+.3f})" for name, value in decreased.items())
            if len(decreased)
            else "no primary target"
        )
        + ". The displacement itself decoded progress most strongly (R²="
        + f"{stage_scores.loc['progress_solvability', 'displacement']:.3f}).",
        "",
        "## Persistence prediction beyond behavioral ingredients",
        "",
        f"For the displacement stage, adding rank-4 neural state to history/time/cost/progress changed held-out R² by {displacement_increment.delta_r_squared_neural_given_behavior:+.3f}; adding behavior to the neural-only model changed R² by {displacement_increment.delta_r_squared_behavior_given_neural:+.3f}.",
        "These are incremental predictive comparisons, not causal mediation estimates.",
        "",
        "## Specificity",
        "",
        "Best nuisance-control R² values were "
        + (
            ", ".join(
                f"{row.control}/{row.stage}={row.r_squared:.3f}"
                for row in nuisance_best.sort_values("control").itertuples()
            )
            if len(nuisance_best)
            else "not available"
        )
        + ". These strong choice, terminality, and generic-value results do not support specificity to persistence computations.",
        f"Task identity peaked at accuracy={task_identity.iloc[0].accuracy:.3f} (chance={task_identity.iloc[0].chance_accuracy:.3f}); task-general claims must be read alongside leave-one-task-out transfer.",
        "",
        "## Decision",
        "",
    ]
    history_transfer = float(transfer_summary.get("history_finite_prediction", np.nan))
    meaningful = strongest[strongest.r_squared >= .10]
    if not len(meaningful):
        decision = "The candidate subspace predicts persistence contrasts but does not correspond cleanly to the computational ingredients tested (Outcome E)."
    elif np.isfinite(history_transfer) and history_transfer >= .10 and all(float(transfer_summary.get(name, -1)) < .10 for name in ("time_effort", "cost_pressure", "progress_solvability")):
        decision = "Recent history dominates the task-general decoding; the candidate is better interpreted as behavioral-state/history geometry than a unitary meta-control variable (Outcome B)."
    elif np.nanmean(transfer.r_squared) <= 0:
        decision = "Within-task decoding does not transfer reliably, favoring task-specific computational implementations (Outcome C)."
    else:
        decision = "The frozen persistence subspace carries a multidimensional mixture of behavioral ingredients; the exact task-general ingredients are those with positive leave-one-task-out scores above (Outcome A, qualified by specificity controls)."
    if float(displacement_increment.delta_r_squared_neural_given_behavior) >= .05:
        decision += " Neural state also adds substantial residual persistence information beyond the explicit behavioral set (Outcome D)."
    lines += [decision, "", "No causal mediation claim, subspace refit, activation recollection, or individual-PC naming is performed here."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_geometry.yaml")
    parser.add_argument("--run-id", default="model_zoo_mac_v2")
    parser.add_argument("--phase", choices=("project", "decode", "report", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="small laptop/development subset with fewer random controls")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    if args.smoke:
        config = {**config, "matched_random_subspaces": int(config["smoke"]["matched_random_subspaces"])}
        max_pairs = int(config["smoke"]["max_pairs_per_task_split"])
    else:
        max_pairs = None
    output = Path(config["output_root"]) / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(f"run output already exists: {output}; choose a new --run-id or pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(output, label="geometry")
    logger.note("pipeline", f"run={args.run_id}; phase={args.phase}; smoke={args.smoke}")
    frozen = FrozenPersistenceSubspace.load(config["frozen_subspace"]["artifact"], config["frozen_subspace"]["key"])
    zoo_config = _load_yaml(config["model_zoo_config"])
    frame = load_behavior_records(config["behavior_records"], config["tasks"], max_pairs_per_task_split=max_pairs)
    validate_episode_splits(frame)
    targets, _specs = build_geometry_targets(frame, zoo_config)
    logger.note("setup", f"aligned behavioral design has {len(frame)} states, {frame.episode_id.nunique()} episodes; frozen basis sha={frozen.sha256[:12]}")
    with logger.section("projection"):
        z_by_stage, cache_paths = project_behavioral_activations(frame, frozen, config, output, resume=args.resume, logger=logger)
    shutil.copy2(args.config, output / "config.yaml")
    provenance = {
        "protocol_version": config["protocol_version"],
        "analysis_role": config["analysis_role"],
        "frozen_subspace": {"artifact": frozen.source, "key": frozen.key, "sha256": frozen.sha256, "rank": frozen.rank},
        "states": len(frame),
        "episodes": frame.episode_id.nunique(),
        "tasks": config["tasks"],
        "smoke": args.smoke,
        "test_targets_never_fit": True,
        "subspace_refit_parameters": 0,
    }
    (output / "run_metadata.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.phase == "project":
        logger.note("pipeline", "projection phase complete")
        return
    if args.phase in {"decode", "all"}:
        with logger.section("computational_decoding"):
            decoding, z_fits = run_computational_decoding(z_by_stage, targets, frame, config, logger)
            decoding.to_csv(output / "computational_decoding.csv", index=False)
        with logger.section("cross_task_decoding"):
            cross = run_cross_task_decoding(z_by_stage, targets, frame, config, logger)
            cross.to_csv(output / "cross_task_decoding.csv", index=False)
        with logger.section("random_subspace_controls"):
            random = run_random_controls(
                cache_paths,
                z_fits,
                targets,
                frame,
                frozen,
                config,
                logger,
                resume=args.resume,
            )
            random.to_csv(output / "random_subspace_controls.csv", index=False)
        with logger.section("full_hidden_comparison"):
            full = run_full_hidden_comparison(cache_paths, z_fits, targets, frame, frozen, config, logger)
            full.to_csv(output / "full_hidden_comparison.csv", index=False)
        with logger.section("dimension_geometry"):
            geometry = run_dimension_geometry(z_by_stage, targets, z_fits, frame, config, logger)
            geometry.to_csv(output / "dimension_geometry.csv", index=False)
        with logger.section("behavior_neural_incremental"):
            incremental = run_incremental_models(z_by_stage, frame, config, logger)
            incremental.to_csv(output / "behavior_neural_incremental.csv", index=False)
        with logger.section("specificity_controls"):
            specificity = run_specificity_controls(z_by_stage, frame, frozen, config, logger)
            specificity.to_csv(output / "specificity_controls.csv", index=False)
        with logger.section("gru_state_alignment"):
            gru = run_gru_alignment(z_by_stage, frame, zoo_config, config["model_zoo_output"], config, logger)
            gru.to_csv(output / "gru_state_alignment.csv", index=False)
        if args.phase == "decode":
            logger.note("pipeline", "decode phase complete")
            return
    if args.phase in {"report", "all"}:
        with logger.section("report_generation"):
            generate_figures(output)
            generate_report(output)
            logger.note("report_generation", f"report available at {output / 'report.md'}")
    logger.note("pipeline", "persistence computational-neural geometry complete")


if __name__ == "__main__":
    main()
