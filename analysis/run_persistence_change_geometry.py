"""Analyze cross-task persistence computations in matched change space."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.persistence_change_data import (
    COMPONENT_TARGET_BY_MANIPULATION,
    PERSISTENCE_TARGETS,
    add_behavioral_model_targets,
    build_compact_change_dataset,
)
from analysis.persistence_change_geometry import (
    clustered_bootstrap_predictions,
    direction_alignment_rows,
    fit_change_decoder,
    strict_group_transfer,
)
from analysis.persistence_geometry import (
    STAGES,
    FrozenPersistenceSubspace,
    matched_random_bases,
    validate_episode_splits,
)
from computational_modeling.analysis.evaluate_models import persistence_metrics
from computational_modeling.analysis.run_model_zoo import ProgressLogger
from computational_modeling.models.base import balanced_weights


def _load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _cache_paths(output):
    cache = Path(output) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "directory": cache,
        "arrays": cache / "change_arrays.npz",
        "metadata": cache / "change_metadata.csv.gz",
        "targets": cache / "change_targets.csv.gz",
        "absolute_metadata": cache / "absolute_metadata.csv.gz",
        "manifest": cache / "manifest.json",
        "random_checkpoint": cache / "random_controls_checkpoint.csv",
    }


def _save_dataset(dataset, output, frozen, config, smoke):
    paths = _cache_paths(output)
    np.savez(
        paths["arrays"],
        hidden_l21=dataset["hidden"]["l21"],
        hidden_displacement=dataset["hidden"]["displacement"],
        hidden_l22=dataset["hidden"]["l22"],
        projected_l21=dataset["projected"]["l21"],
        projected_displacement=dataset["projected"]["displacement"],
        projected_l22=dataset["projected"]["l22"],
        absolute_l21=dataset["absolute_hidden"]["l21"],
        absolute_l22=dataset["absolute_hidden"]["l22"],
    )
    dataset["metadata"].to_csv(paths["metadata"], index=False, compression="gzip")
    dataset["targets"].to_csv(paths["targets"], index=False, compression="gzip")
    dataset["absolute_metadata"].to_csv(
        paths["absolute_metadata"], index=False, compression="gzip"
    )
    dataset["audit"].to_csv(Path(output) / "contrast_pair_audit.csv", index=False)
    manifest = {
        "protocol_version": config["protocol_version"],
        "pairs": len(dataset["metadata"]),
        "absolute_rows": len(dataset["absolute_metadata"]),
        "frozen_basis_sha256": frozen.sha256,
        "frozen_basis_key": frozen.key,
        "rank": frozen.rank,
        "width": frozen.width,
        "smoke": bool(smoke),
        "format": "compact float16 L21/L22 matched endpoints and changes",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _load_dataset(output, frozen, smoke):
    paths = _cache_paths(output)
    required = ("arrays", "metadata", "targets", "absolute_metadata", "manifest")
    if not all(paths[name].exists() for name in required):
        return None
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if (
        manifest.get("frozen_basis_sha256") != frozen.sha256
        or bool(manifest.get("smoke")) != bool(smoke)
    ):
        return None
    arrays = np.load(paths["arrays"])
    return {
        "metadata": pd.read_csv(paths["metadata"]),
        "targets": pd.read_csv(paths["targets"]),
        "absolute_metadata": pd.read_csv(paths["absolute_metadata"]),
        "hidden": {stage: arrays[f"hidden_{stage}"] for stage in STAGES},
        "projected": {stage: arrays[f"projected_{stage}"] for stage in STAGES},
        "absolute_hidden": {
            "l21": arrays["absolute_l21"],
            "l22": arrays["absolute_l22"],
        },
        "audit": pd.read_csv(Path(output) / "contrast_pair_audit.csv"),
    }


def _subset(values, target, metadata, mask):
    mask = np.asarray(mask, dtype=bool)
    return (
        np.asarray(values)[mask],
        np.asarray(target, dtype=float)[mask],
        metadata.loc[mask].reset_index(drop=True),
    )


def _fit_with_ci(values, target, metadata, config, seed):
    fit = fit_change_decoder(
        values, target, metadata, alphas=config["ridge_alphas"]
    )
    ci = clustered_bootstrap_predictions(
        target,
        fit["test_prediction"],
        metadata,
        fit["test_indices"],
        samples=int(config["bootstrap_samples"]),
        seed=int(seed),
    )
    return fit, ci


def _result_row(stage, target_name, fit, ci, **extra):
    metrics = fit["test_metrics"]
    row = {
        "stage": stage,
        "target": target_name,
        "status": "fit",
        "selected_alpha": fit["selected_alpha"],
        **metrics,
        **extra,
    }
    for metric, values in ci.items():
        row[f"{metric}_ci_low"] = values["ci_low"]
        row[f"{metric}_ci_high"] = values["ci_high"]
        row["bootstrap_samples"] = values["samples"]
        row["bootstrap_clusters"] = values["clusters"]
    return row


def _constant_row(stage, target_name, metadata, **extra):
    return {
        "stage": stage,
        "target": target_name,
        "status": "constant_or_incomplete_target",
        "selected_alpha": np.nan,
        "r_squared": np.nan,
        "mse": np.nan,
        "pearson_r": np.nan,
        "sign_accuracy": np.nan,
        "states": len(metadata),
        "episodes": metadata.episode_id.nunique(),
        **extra,
    }


def run_change_decoding(dataset, config, logger):
    metadata, targets = dataset["metadata"], dataset["targets"]
    persistence = metadata.contrast_kind.astype(str).to_numpy() == "persistence"
    target_names = list(PERSISTENCE_TARGETS) + sorted(
        set(COMPONENT_TARGET_BY_MANIPULATION.values())
    )
    rows, fits = [], {}
    seed = int(config["seed"])
    for stage_index, stage in enumerate(STAGES):
        for target_index, target_name in enumerate(target_names):
            values = targets[target_name].to_numpy(dtype=float)
            mask = persistence & np.isfinite(values)
            x, y, local = _subset(dataset["projected"][stage], values, metadata, mask)
            if (
                len(local) == 0
                or set(local.split.astype(str)) != {"train", "validation", "test"}
                or np.std(y[local.split.astype(str).to_numpy() == "train"]) < 1e-10
            ):
                rows.append(_constant_row(stage, target_name, local))
                continue
            fit, ci = _fit_with_ci(
                x, y, local, config, seed + stage_index * 101 + target_index
            )
            fits[(stage, target_name)] = (fit, x, y, local)
            rows.append(_result_row(stage, target_name, fit, ci))
        logger.note("change_decoding", f"{stage}: decoded {len(target_names)} computational changes")
    return pd.DataFrame(rows), fits


def run_group_transfer(dataset, config, *, group_column, target_names, logger):
    metadata, targets = dataset["metadata"], dataset["targets"]
    persistence = metadata.contrast_kind.astype(str).to_numpy() == "persistence"
    groups = sorted(metadata.loc[persistence, group_column].astype(str).unique())
    rows, fits = [], {}
    seed = int(config["seed"]) + (1000 if group_column == "task" else 2000)
    for stage_index, stage in enumerate(STAGES):
        for target_index, target_name in enumerate(target_names):
            values = targets[target_name].to_numpy(dtype=float)
            mask = persistence & np.isfinite(values)
            x, y, local = _subset(dataset["projected"][stage], values, metadata, mask)
            for group_index, heldout in enumerate(groups):
                try:
                    fit = strict_group_transfer(
                        x,
                        y,
                        local,
                        group_column=group_column,
                        heldout=heldout,
                        alphas=config["ridge_alphas"],
                    )
                except ValueError:
                    rows.append(
                        _constant_row(
                            stage,
                            target_name,
                            local[local[group_column].astype(str) == heldout],
                            **{f"heldout_{group_column}": heldout},
                        )
                    )
                    continue
                ci = clustered_bootstrap_predictions(
                    y,
                    fit["test_prediction"],
                    local,
                    fit["test_indices"],
                    samples=int(config["bootstrap_samples"]),
                    seed=seed + stage_index * 1009 + target_index * 101 + group_index,
                )
                fits[(stage, target_name, heldout)] = (fit, x, y, local)
                rows.append(
                    _result_row(
                        stage,
                        target_name,
                        fit,
                        ci,
                        **{
                            f"heldout_{group_column}": heldout,
                            "source_groups": ";".join(fit["fit_groups"]),
                        },
                    )
                )
        logger.note(
            f"cross_{group_column}",
            f"{stage}: completed {len(groups)} strict {group_column} holdouts",
        )
    return pd.DataFrame(rows), fits


def _absolute_projected(dataset, frozen):
    h21 = dataset["absolute_hidden"]["l21"].astype(np.float32)
    h22 = dataset["absolute_hidden"]["l22"].astype(np.float32)
    return {
        "l21": h21 @ frozen.basis,
        "displacement": (h22 - h21) @ frozen.basis,
        "l22": h22 @ frozen.basis,
    }


def _metric_on_rows(observed, predicted, metadata, selected):
    selected = np.asarray(selected, dtype=int)
    records = metadata.iloc[selected].to_dict(orient="records")
    return persistence_metrics(
        np.asarray(observed)[selected],
        np.asarray(predicted)[selected],
        balanced_weights(records, task_balanced=True),
    )


def _paired_improvement_ci(change_bundle, absolute_bundle, samples, seed):
    change_fit, _x, change_y, change_meta = change_bundle
    absolute_fit, _ax, absolute_y, absolute_meta = absolute_bundle
    change_local = change_meta.iloc[change_fit["test_indices"]].reset_index(drop=True)
    absolute_local = absolute_meta.iloc[absolute_fit["test_indices"]].reset_index(drop=True)
    change_observed = change_y[change_fit["test_indices"]]
    absolute_observed = absolute_y[absolute_fit["test_indices"]]
    change_prediction = change_fit["test_prediction"]
    absolute_prediction = absolute_fit["test_prediction"]
    common = sorted(set(change_local.pair_id.astype(str)) & set(absolute_local.pair_id.astype(str)))
    if not common:
        return (np.nan, np.nan)
    change_groups = {
        key: np.flatnonzero(change_local.pair_id.astype(str).to_numpy() == key)
        for key in common
    }
    absolute_groups = {
        key: np.flatnonzero(absolute_local.pair_id.astype(str).to_numpy() == key)
        for key in common
    }
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(samples)):
        sampled = rng.choice(common, size=len(common), replace=True)
        c_observed, c_predicted, c_frames = [], [], []
        a_observed, a_predicted, a_frames = [], [], []
        for draw_index, key in enumerate(sampled):
            cidx, aidx = change_groups[key], absolute_groups[key]
            cpart, apart = change_local.iloc[cidx].copy(), absolute_local.iloc[aidx].copy()
            for part in (cpart, apart):
                part["episode_id"] = part.episode_id.astype(str) + f"#boot-{draw_index}"
                part["pair_id"] = part.pair_id.astype(str) + f"#boot-{draw_index}"
            c_frames.append(cpart)
            a_frames.append(apart)
            c_observed.append(change_observed[cidx])
            c_predicted.append(change_prediction[cidx])
            a_observed.append(absolute_observed[aidx])
            a_predicted.append(absolute_prediction[aidx])
        cframe, aframe = pd.concat(c_frames, ignore_index=True), pd.concat(a_frames, ignore_index=True)
        change_r2 = persistence_metrics(
            np.concatenate(c_observed),
            np.concatenate(c_predicted),
            balanced_weights(cframe.to_dict(orient="records"), task_balanced=True),
        )["r_squared"]
        absolute_r2 = persistence_metrics(
            np.concatenate(a_observed),
            np.concatenate(a_predicted),
            balanced_weights(aframe.to_dict(orient="records"), task_balanced=True),
        )["r_squared"]
        draws.append(change_r2 - absolute_r2)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def run_absolute_comparison(dataset, frozen, change_fits, config, logger):
    absolute = dataset["absolute_metadata"].copy()
    keep = absolute.contrast_kind.astype(str).to_numpy() == "persistence"
    absolute = absolute.loc[keep].reset_index(drop=True)
    absolute_values = {stage: values[keep] for stage, values in _absolute_projected(dataset, frozen).items()}
    target_columns = {
        "persistence_policy_change": "persistence_policy",
        "gru_prediction_change": "gru_prediction",
    }
    rows = []
    for stage_index, stage in enumerate(STAGES):
        for target_index, (change_target, absolute_target) in enumerate(target_columns.items()):
            y = absolute[absolute_target].to_numpy(dtype=float)
            for task_index, heldout in enumerate(sorted(absolute.task.astype(str).unique())):
                fit = strict_group_transfer(
                    absolute_values[stage],
                    y,
                    absolute,
                    group_column="task",
                    heldout=heldout,
                    alphas=config["ridge_alphas"],
                )
                absolute_bundle = (fit, absolute_values[stage], y, absolute)
                change_bundle = change_fits[(stage, change_target, heldout)]
                change_fit = change_bundle[0]
                low, high = _paired_improvement_ci(
                    change_bundle,
                    absolute_bundle,
                    int(config["bootstrap_samples"]),
                    int(config["seed"]) + 3000 + stage_index * 101 + target_index * 11 + task_index,
                )
                rows.append(
                    {
                        "stage": stage,
                        "target": change_target,
                        "heldout_task": heldout,
                        "absolute_r_squared": fit["test_metrics"]["r_squared"],
                        "change_r_squared": change_fit["test_metrics"]["r_squared"],
                        "contrast_benefit_r_squared": change_fit["test_metrics"]["r_squared"] - fit["test_metrics"]["r_squared"],
                        "contrast_benefit_ci_low": low,
                        "contrast_benefit_ci_high": high,
                        "bootstrap_samples": int(config["bootstrap_samples"]),
                        "absolute_selected_alpha": fit["selected_alpha"],
                        "change_selected_alpha": change_fit["selected_alpha"],
                    }
                )
        logger.note("absolute_vs_change", f"{stage}: strict absolute/change comparison complete")
    return pd.DataFrame(rows)


def _mean_fit_score(values, target, metadata, group_column, groups, alphas):
    scores = []
    for heldout in groups:
        try:
            fit = strict_group_transfer(
                values,
                target,
                metadata,
                group_column=group_column,
                heldout=heldout,
                alphas=alphas,
            )
        except ValueError:
            continue
        scores.append(float(fit["test_metrics"]["r_squared"]))
    return float(np.mean(scores)) if scores else np.nan


def run_random_controls(dataset, change, cross_task, cross_manipulation, frozen, config, output, resume, logger):
    metadata = dataset["metadata"]
    persistence = metadata.contrast_kind.astype(str).to_numpy() == "persistence"
    local = metadata.loc[persistence].reset_index(drop=True)
    target_names = ("persistence_policy_change", "gru_prediction_change")
    targets = {
        name: dataset["targets"].loc[persistence, name].to_numpy(dtype=float)
        for name in target_names
    }
    count = int(config["matched_random_subspaces"])
    bases = matched_random_bases(frozen.width, frozen.rank, count, int(config["seed"]) + 7001)
    flat = np.transpose(bases, (1, 0, 2)).reshape(frozen.width, -1)
    projections = {
        stage: (dataset["hidden"][stage][persistence].astype(np.float32) @ flat).reshape(len(local), count, frozen.rank)
        for stage in STAGES
    }
    observed = {}
    for row in change.itertuples():
        if row.target in target_names and row.status == "fit":
            observed[(row.stage, row.target, "heldout")] = float(row.r_squared)
    for frame, analysis in ((cross_task, "cross_task"), (cross_manipulation, "cross_manipulation")):
        selected = frame[(frame.target.isin(target_names)) & (frame.status == "fit")]
        for (stage, target), part in selected.groupby(["stage", "target"]):
            observed[(stage, target, analysis)] = float(part.r_squared.mean())
    checkpoint = _cache_paths(output)["random_checkpoint"]
    rows, completed = [], set()
    if resume and checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        rows = prior.to_dict(orient="records")
        completed = {
            (str(row.stage), str(row.target), int(row.random_subspace), str(row.analysis))
            for row in prior.itertuples()
        }
        logger.note("random_controls", f"reusing {len(completed)} checkpointed random scores")
    tasks = sorted(local.task.astype(str).unique())
    manipulations = sorted(local.manipulation.astype(str).unique())
    for random_index in range(count):
        for stage in STAGES:
            x = projections[stage][:, random_index, :]
            for target_name in target_names:
                y = targets[target_name]
                calculations = {
                    "heldout": lambda: fit_change_decoder(x, y, local, alphas=config["ridge_alphas"])["test_metrics"]["r_squared"],
                    "cross_task": lambda: _mean_fit_score(x, y, local, "task", tasks, config["ridge_alphas"]),
                    "cross_manipulation": lambda: _mean_fit_score(x, y, local, "manipulation", manipulations, config["ridge_alphas"]),
                }
                for analysis, calculate in calculations.items():
                    key = (stage, target_name, random_index, analysis)
                    if key in completed:
                        continue
                    rows.append(
                        {
                            "stage": stage,
                            "target": target_name,
                            "random_subspace": random_index,
                            "analysis": analysis,
                            "random_r_squared": float(calculate()),
                            "persistence_rank4_r_squared": observed[(stage, target_name, analysis)],
                        }
                    )
        if (random_index + 1) % 5 == 0 or random_index + 1 == count:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            logger.note("random_controls", f"decoded {random_index + 1}/{count} matched rank-4 controls")
    frame = pd.DataFrame(rows).drop_duplicates(
        ["stage", "target", "random_subspace", "analysis"], keep="last"
    )
    summaries = []
    for keys, part in frame.groupby(["stage", "target", "analysis"]):
        candidate = float(part.persistence_rank4_r_squared.iloc[0])
        values = part.random_r_squared.to_numpy(dtype=float)
        summaries.append(
            {
                "stage": keys[0],
                "target": keys[1],
                "analysis": keys[2],
                "random_r_squared_mean": float(np.nanmean(values)),
                "random_r_squared_95th": float(np.nanquantile(values, 0.95)),
                "persistence_percentile_among_random": float(np.nanmean(values < candidate)),
                "empirical_p_value": float((1 + np.sum(values >= candidate)) / (1 + np.sum(np.isfinite(values)))),
            }
        )
    return frame.merge(pd.DataFrame(summaries), on=["stage", "target", "analysis"], how="left")


def _cluster_bootstrap_mean(values, clusters, samples, seed):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    grouped = {
        cluster: values[clusters == cluster]
        for cluster in sorted(set(clusters))
    }
    rng = np.random.default_rng(int(seed))
    draws = []
    keys = sorted(grouped)
    for _ in range(int(samples)):
        selected = rng.choice(keys, size=len(keys), replace=True)
        draws.append(float(np.mean([np.mean(grouped[key]) for key in selected])))
    return {
        "estimate": float(np.mean([np.mean(grouped[key]) for key in keys])),
        "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)),
        "draws": np.asarray(draws),
        "clusters": len(keys),
    }


def run_nuisance_controls(dataset, change, config, logger):
    metadata, targets = dataset["metadata"], dataset["targets"]
    nuisance = metadata.contrast_kind.astype(str).to_numpy() == "nuisance"
    rows = []
    persistence_reference = {
        row.stage: row
        for row in change[
            (change.target == "persistence_policy_change") & (change.status == "fit")
        ].itertuples()
    }
    for stage_index, stage in enumerate(STAGES):
        persistence_test = (
            (metadata.contrast_kind.astype(str).to_numpy() == "persistence")
            & (metadata.split.astype(str).to_numpy() == "test")
        )
        persistence_norm = _cluster_bootstrap_mean(
            np.linalg.norm(dataset["projected"][stage][persistence_test], axis=1),
            metadata.loc[persistence_test, "pair_id"],
            int(config["bootstrap_samples"]),
            int(config["seed"]) + 3900 + stage_index,
        )
        for control_index, control in enumerate(sorted(metadata.loc[nuisance, "nuisance_type"].astype(str).unique())):
            mask = nuisance & (metadata.nuisance_type.astype(str).to_numpy() == control)
            x, y, local = _subset(dataset["projected"][stage], targets.nuisance_policy_change, metadata, mask)
            reference = persistence_reference[stage]
            nuisance_test = local.split.astype(str).to_numpy() == "test"
            nuisance_norm = _cluster_bootstrap_mean(
                np.linalg.norm(x[nuisance_test], axis=1),
                local.loc[nuisance_test, "pair_id"],
                int(config["bootstrap_samples"]),
                int(config["seed"]) + 3950 + stage_index * 101 + control_index,
            )
            norm_difference = persistence_norm["draws"] - nuisance_norm["draws"]
            magnitude = {
                "mean_projection_norm": nuisance_norm["estimate"],
                "mean_projection_norm_ci_low": nuisance_norm["ci_low"],
                "mean_projection_norm_ci_high": nuisance_norm["ci_high"],
                "persistence_reference_projection_norm": persistence_norm["estimate"],
                "persistence_minus_nuisance_projection_norm": persistence_norm["estimate"] - nuisance_norm["estimate"],
                "persistence_minus_nuisance_projection_norm_ci_low": float(np.quantile(norm_difference, .025)),
                "persistence_minus_nuisance_projection_norm_ci_high": float(np.quantile(norm_difference, .975)),
            }
            if set(local.split.astype(str)) != {"train", "validation", "test"} or np.std(y) < 1e-10:
                rows.append(
                    {
                        **_constant_row(stage, "nuisance_policy_change", local),
                        "control": control,
                        **magnitude,
                        "persistence_reference_r_squared": float(reference.r_squared),
                    }
                )
                continue
            fit, ci = _fit_with_ci(
                x,
                y,
                local,
                config,
                int(config["seed"]) + 4000 + stage_index * 101 + control_index,
            )
            row = _result_row(stage, "nuisance_policy_change", fit, ci, control=control)
            row.update(
                {
                    **magnitude,
                    "persistence_reference_r_squared": float(reference.r_squared),
                    "persistence_minus_nuisance_r_squared": float(reference.r_squared) - float(fit["test_metrics"]["r_squared"]),
                    "persistence_minus_nuisance_ci_low": float(reference.r_squared_ci_low) - ci["r_squared"]["ci_high"],
                    "persistence_minus_nuisance_ci_high": float(reference.r_squared_ci_high) - ci["r_squared"]["ci_low"],
                }
            )
            rows.append(row)
        logger.note("nuisance_controls", f"{stage}: persistence/nuisance change comparison complete")
    return pd.DataFrame(rows)


def run_stage_transition(direction, change, cross_task, cross_manipulation):
    rows = []
    for stage in STAGES:
        local_direction = direction[direction.stage == stage]
        cross_cosine = local_direction[
            (local_direction.kind == "direction_cosine")
            & (local_direction.task_a != local_direction.task_b)
        ].value.mean()
        overlap = local_direction[local_direction.kind == "task_subspace_overlap"].value.mean()
        for target in ("persistence_policy_change", "gru_prediction_change"):
            primary = change[(change.stage == stage) & (change.target == target)]
            task = cross_task[(cross_task.stage == stage) & (cross_task.target == target) & (cross_task.status == "fit")]
            manipulation = cross_manipulation[(cross_manipulation.stage == stage) & (cross_manipulation.target == target) & (cross_manipulation.status == "fit")]
            rows.append(
                {
                    "stage": stage,
                    "target": target,
                    "heldout_r_squared": float(primary.r_squared.iloc[0]),
                    "mean_cross_task_r_squared": float(task.r_squared.mean()),
                    "mean_cross_manipulation_r_squared": float(manipulation.r_squared.mean()),
                    "mean_cross_task_direction_cosine": float(cross_cosine),
                    "mean_task_subspace_overlap": float(overlap),
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame[frame.stage == "l21"].set_index("target")
    for metric in (
        "heldout_r_squared",
        "mean_cross_task_r_squared",
        "mean_cross_manipulation_r_squared",
        "mean_cross_task_direction_cosine",
        "mean_task_subspace_overlap",
    ):
        frame[f"delta_{metric}_vs_l21"] = [
            float(row[metric]) - float(baseline.loc[row["target"], metric])
            for row in frame.to_dict(orient="records")
        ]
    return frame


def _configure_plotting():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/digital_minds_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/digital_minds_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_figures(output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt = _configure_plotting()
    change = pd.read_csv(output / "change_decoding.csv")
    fitted = change[change.status == "fit"]
    pivot = fitted.pivot(index="target", columns="stage", values="r_squared")
    axis = pivot.plot(kind="barh", figsize=(10, 6))
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_xlabel("Held-out R²")
    axis.set_title("Computational-change decoding from matched neural differences")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="stage")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "change_decoding.png", dpi=160)
    plt.close(axis.figure)

    absolute = pd.read_csv(output / "absolute_vs_change.csv")
    grouped = absolute.groupby(["stage", "target"])[["absolute_r_squared", "change_r_squared"]].mean().reset_index()
    grouped["comparison"] = grouped.stage + "\n" + grouped.target.str.replace("_change", "", regex=False)
    axis = grouped.set_index("comparison")[["absolute_r_squared", "change_r_squared"]].plot(kind="bar", figsize=(11, 5))
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_ylabel("Mean strict LOTO R²")
    axis.set_xlabel("")
    axis.set_title("Absolute-state versus matched-difference transfer")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "absolute_vs_change_transfer.png", dpi=160)
    plt.close(axis.figure)

    for filename, source, heldout, title in (
        ("cross_task_change_transfer.png", "cross_task_change_transfer.csv", "heldout_task", "Strict cross-task change transfer"),
        ("cross_manipulation_transfer.png", "cross_manipulation_transfer.csv", "heldout_manipulation", "Strict cross-manipulation change transfer"),
    ):
        frame = pd.read_csv(output / source)
        frame = frame[(frame.status == "fit") & (frame.target.isin(("persistence_policy_change", "gru_prediction_change")))]
        pivot = frame.pivot_table(index=["target", heldout], columns="stage", values="r_squared")
        axis = pivot.plot(kind="barh", figsize=(11, max(5, .45 * len(pivot))))
        axis.axvline(0, color="black", linewidth=.8)
        axis.set_xlabel("Held-out R²")
        axis.set_title(title)
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="stage")
        axis.figure.tight_layout()
        axis.figure.savefig(figures / filename, dpi=160)
        plt.close(axis.figure)

    nuisance = pd.read_csv(output / "nuisance_change_controls.csv")
    nuisance = nuisance[nuisance.status == "fit"]
    axis = nuisance.pivot(index="control", columns="stage", values="r_squared").plot(kind="bar", figsize=(9, 5))
    axis.set_ylabel("Held-out R²")
    axis.set_title("Persistence candidate on nuisance changes")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="stage")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "persistence_vs_nuisance_change.png", dpi=160)
    plt.close(axis.figure)

    direction = pd.read_csv(output / "direction_alignment.csv")
    cosine = direction[(direction.kind == "direction_cosine") & (direction.task_a != direction.task_b)]
    axis = cosine.groupby("stage").value.mean().reindex(STAGES).plot(kind="bar", figsize=(7, 4))
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_ylabel("Mean cross-task family-direction cosine")
    axis.set_title("Direction alignment across L21→L22")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "direction_alignment.png", dpi=160)
    plt.close(axis.figure)


def generate_report(output):
    output = Path(output)
    change = pd.read_csv(output / "change_decoding.csv")
    task = pd.read_csv(output / "cross_task_change_transfer.csv")
    manipulation = pd.read_csv(output / "cross_manipulation_transfer.csv")
    absolute = pd.read_csv(output / "absolute_vs_change.csv")
    random = pd.read_csv(output / "random_subspace_controls.csv").drop_duplicates(["stage", "target", "analysis"])
    nuisance = pd.read_csv(output / "nuisance_change_controls.csv")
    direction = pd.read_csv(output / "direction_alignment.csv")
    primary = change[(change.target == "persistence_policy_change") & (change.status == "fit")]
    task_primary = task[(task.target == "persistence_policy_change") & (task.status == "fit")]
    manipulation_primary = manipulation[(manipulation.target == "persistence_policy_change") & (manipulation.status == "fit")]
    best = primary.sort_values("r_squared", ascending=False).iloc[0]
    mean_loto = float(task_primary.r_squared.mean())
    mean_lomo = float(manipulation_primary.r_squared.mean())
    mean_benefit = float(absolute[absolute.target == "persistence_policy_change"].contrast_benefit_r_squared.mean())
    random_primary = random[(random.target == "persistence_policy_change") & (random.analysis == "cross_task")]
    best_random_p = float(random_primary.empirical_p_value.min())
    fitted_nuisance = nuisance[nuisance.status == "fit"]
    nuisance_best = float(fitted_nuisance.r_squared.max()) if len(fitted_nuisance) else np.nan
    direction_mean = direction[(direction.kind == "direction_cosine") & (direction.task_a != direction.task_b)].groupby("stage").value.mean()
    if mean_loto > 0 and mean_lomo > 0 and mean_benefit > 0 and best_random_p <= .05 and (not np.isfinite(nuisance_best) or float(best.r_squared) > nuisance_best):
        outcome = "Outcome A — strong shared difference-space representation"
        conclusion = "Different tasks occupy distinct baselines, while persistence manipulations induce a shared neural transformation."
    elif mean_lomo > 0 and mean_loto <= 0:
        outcome = "Outcome B — cross-manipulation but not cross-task"
        conclusion = "Persistence changes are internally consistent across manipulations but remain task-coordinate specific."
    elif np.isfinite(nuisance_best) and nuisance_best >= float(best.r_squared):
        outcome = "Outcome D — generic decision/value changes are at least as strong"
        conclusion = "The transformation is better characterized as generic value/control geometry than persistence-specific computation."
    else:
        outcome = "Outcome C — difference space does not clear random/control geometry"
        conclusion = "The frozen candidate does not support a specific universal persistence-change code."
    lines = [
        "# Cross-task persistence computation in matched change space",
        "",
        "The `displacement-L21-k4` basis was loaded read-only. No activation recollection, subspace refit, target-task calibration, or target-task normalization was performed.",
        "",
        "## Computational-change decoding",
        "",
    ]
    for row in primary.sort_values("r_squared", ascending=False).itertuples():
        lines.append(f"- {row.stage}: persistence-policy Δ held-out R²={row.r_squared:.3f} (95% CI {row.r_squared_ci_low:.3f} to {row.r_squared_ci_high:.3f}), sign accuracy={row.sign_accuracy:.3f}.")
    lines += [
        "",
        "## Strict transfer",
        "",
        f"- Mean leave-one-task-out R²: {mean_loto:.3f}.",
        f"- Mean leave-one-manipulation-out R²: {mean_lomo:.3f}.",
        f"- Mean contrast-space benefit over absolute-state LOTO: {mean_benefit:+.3f} R².",
        "",
        "## Concentration, nuisance specificity, and alignment",
        "",
        f"- Best empirical matched-random cross-task p-value: {best_random_p:.3f}.",
        f"- Strongest fitted nuisance-change R²: {nuisance_best:.3f}." if np.isfinite(nuisance_best) else "- Nuisance controls lacked a complete train/validation/test fit.",
        "- Mean cross-task direction cosines: " + ", ".join(f"{stage}={value:.3f}" for stage, value in direction_mean.items()) + ".",
        "",
        "## Decision",
        "",
        f"**{outcome}.** {conclusion}",
        "",
        "This is an exploratory representational falsification test, not a causal mediation result.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_change_geometry.yaml")
    parser.add_argument("--run-id", default="model_zoo_mac_v2")
    parser.add_argument("--phase", choices=("prepare", "decode", "report", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    if args.smoke:
        config = {
            **config,
            "bootstrap_samples": int(config["smoke"]["bootstrap_samples"]),
            "matched_random_subspaces": int(config["smoke"]["matched_random_subspaces"]),
        }
    output = Path(config["output_root"]) / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(f"run output already exists: {output}; choose a new --run-id or pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(output, label="change-geometry")
    logger.note("pipeline", f"run={args.run_id}; phase={args.phase}; smoke={args.smoke}")
    frozen = FrozenPersistenceSubspace.load(
        config["frozen_subspace"]["artifact"], config["frozen_subspace"]["key"]
    )
    dataset = _load_dataset(output, frozen, args.smoke) if args.resume else None
    with logger.section("prepare_pairs"):
        if dataset is None:
            dataset = build_compact_change_dataset(config, frozen, logger=logger, smoke=args.smoke)
            dataset = add_behavioral_model_targets(dataset, config, logger=logger)
            _save_dataset(dataset, output, frozen, config, args.smoke)
            logger.note("prepare_pairs", f"retained {len(dataset['metadata'])} valid matched pairs")
        else:
            logger.note("prepare_pairs", f"reusing compact cache with {len(dataset['metadata'])} pairs")
    shutil.copy2(args.config, output / "config.yaml")
    metadata = dataset["metadata"]
    validate_episode_splits(metadata)
    provenance = {
        "protocol_version": config["protocol_version"],
        "analysis_role": config["analysis_role"],
        "frozen_subspace": {"artifact": frozen.source, "key": frozen.key, "sha256": frozen.sha256, "rank": frozen.rank},
        "pairs": len(metadata),
        "tasks": sorted(metadata[metadata.contrast_kind == "persistence"].task.unique()),
        "manipulations": sorted(metadata[metadata.contrast_kind == "persistence"].manipulation.unique()),
        "subspace_refit_parameters": 0,
        "test_targets_never_fit": True,
        "bandit_behavioral_model_application": (
            "validation-selected organic-task models applied without refitting to "
            "displayed factorial continuation/STOP values; counterfactual application is extrapolative"
        ),
        "smoke": args.smoke,
    }
    (output / "run_metadata.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.phase == "prepare":
        return
    if args.phase in {"decode", "all"}:
        with logger.section("change_decoding"):
            change, change_fits = run_change_decoding(dataset, config, logger)
            change.to_csv(output / "change_decoding.csv", index=False)
        with logger.section("cross_task_change_transfer"):
            cross_task, task_fits = run_group_transfer(
                dataset,
                config,
                group_column="task",
                target_names=("persistence_policy_change", "gru_prediction_change"),
                logger=logger,
            )
            cross_task.rename(columns={"heldout_task": "heldout_task"}).to_csv(output / "cross_task_change_transfer.csv", index=False)
        with logger.section("cross_manipulation_transfer"):
            cross_manipulation, _manipulation_fits = run_group_transfer(
                dataset,
                config,
                group_column="manipulation",
                target_names=("persistence_policy_change", "gru_prediction_change"),
                logger=logger,
            )
            cross_manipulation.to_csv(output / "cross_manipulation_transfer.csv", index=False)
        with logger.section("absolute_vs_change"):
            absolute = run_absolute_comparison(dataset, frozen, task_fits, config, logger)
            absolute.to_csv(output / "absolute_vs_change.csv", index=False)
        with logger.section("random_subspace_controls"):
            random = run_random_controls(dataset, change, cross_task, cross_manipulation, frozen, config, output, args.resume, logger)
            random.to_csv(output / "random_subspace_controls.csv", index=False)
        with logger.section("nuisance_change_controls"):
            nuisance = run_nuisance_controls(dataset, change, config, logger)
            nuisance.to_csv(output / "nuisance_change_controls.csv", index=False)
        with logger.section("direction_alignment"):
            persistence = metadata.contrast_kind.astype(str).to_numpy() == "persistence"
            direction = pd.DataFrame(
                direction_alignment_rows(
                    {stage: dataset["projected"][stage][persistence] for stage in STAGES},
                    metadata.loc[persistence].reset_index(drop=True),
                )
            )
            direction.to_csv(output / "direction_alignment.csv", index=False)
            transition = run_stage_transition(direction, change, cross_task, cross_manipulation)
            transition.to_csv(output / "stage_transition.csv", index=False)
        if args.phase == "decode":
            return
    if args.phase in {"report", "all"}:
        with logger.section("report_generation"):
            generate_figures(output)
            generate_report(output)
            logger.note("report_generation", f"report available at {output / 'report.md'}")
    logger.note("pipeline", "persistence change-geometry analysis complete")


if __name__ == "__main__":
    main()
