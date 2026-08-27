"""Eligibility, neural sanity, and feature-family analyses for the mini PRD."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from computational_modeling.analysis.evaluate_models import (
    evaluate_predictions,
    persistence_metrics,
    sigmoid,
)
from computational_modeling.analysis.model_fitting import fit_interpretable_model
from computational_modeling.data.feature_schema import (
    FEATURE_GROUP_DESCRIPTIONS,
    FLEXIBLE_FEATURE_GROUPS,
    FLEXIBLE_NUISANCE_FEATURES,
    flexible_features,
    validate_feature_groups,
)
from computational_modeling.models.base import (
    TrainStandardizer,
    balanced_weights,
    linear_predict,
    weighted_ridge_fit,
)
from computational_modeling.models.baselines import ModelDefinition
from computational_modeling.models.gru import fit_gru_ceiling
from computational_modeling.models.mlp import fit_mlp_ceiling


FLEXIBLE_MODELS = ("flexible_linear", "mlp", "gru")
NON_SCIENTIFIC_MODELS = {"intercept", "oracle_policy"}


def model_category(model: str) -> str:
    if model == "oracle_policy":
        return "oracle"
    if model == "intercept":
        return "baseline"
    if model in FLEXIBLE_MODELS:
        return "flexible"
    return "interpretable"


def _fit_key(row) -> tuple[str, str, str]:
    return (str(row["model"]), str(row["information_set"]), str(row["sharing"]))


def build_model_rankings(
    metrics: pd.DataFrame,
    taskwise: pd.DataFrame,
    required_tasks: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build coverage-safe cross-task and within-task held-out rankings."""

    required = frozenset(map(str, required_tasks))
    coverage = {
        (str(model), str(information_set), str(sharing)): frozenset(map(str, part.task))
        for (model, information_set, sharing), part in taskwise.groupby(
            ["model", "information_set", "sharing"], dropna=False
        )
    }
    candidates = metrics[
        (metrics["information_set"] == "observable")
        & (~metrics["model"].isin(NON_SCIENTIFIC_MODELS))
    ].copy()
    candidates["model_category"] = candidates["model"].map(model_category)
    candidates["evaluated_tasks"] = [
        ";".join(sorted(coverage.get(_fit_key(row), frozenset())))
        for _, row in candidates.iterrows()
    ]
    candidates["eligible_cross_task"] = [
        coverage.get(_fit_key(row), frozenset()) == required
        for _, row in candidates.iterrows()
    ]
    cross_task = candidates[candidates["eligible_cross_task"]].copy()
    cross_task = cross_task.sort_values(
        ["r_squared", "mse", "parameter_count", "model", "sharing"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    cross_task.insert(0, "cross_task_rank", np.arange(1, len(cross_task) + 1))
    cross_task["best_cross_task_model"] = cross_task["cross_task_rank"] == 1
    cross_task["best_cross_task_interpretable"] = False
    interpretable_indices = cross_task.index[
        cross_task["model_category"] == "interpretable"
    ]
    if len(interpretable_indices):
        cross_task.loc[interpretable_indices[0], "best_cross_task_interpretable"] = True

    aggregate_columns = [
        "model",
        "information_set",
        "sharing",
        "parameter_count",
        "task_count",
    ]
    aggregate = candidates[aggregate_columns].drop_duplicates()
    within = taskwise[
        (taskwise["information_set"] == "observable")
        & (~taskwise["model"].isin(NON_SCIENTIFIC_MODELS))
    ].merge(
        aggregate,
        on=["model", "information_set", "sharing"],
        how="left",
        suffixes=("", "_aggregate"),
    )
    within["model_category"] = within["model"].map(model_category)
    # A model can appear under several sharing regimes.  Retain its best
    # taskwise realization so the within-task table has one row per model/task.
    within = within.sort_values(
        ["task", "model", "r_squared", "mse", "parameter_count"],
        ascending=[True, True, False, True, True],
    ).drop_duplicates(["task", "model"], keep="first")
    within = within.sort_values(
        ["task", "r_squared", "mse", "parameter_count", "model"],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    within["task_rank"] = within.groupby("task").cumcount() + 1
    within["best_task_specific_model"] = within["task_rank"] == 1
    within["best_task_specific_interpretable"] = False
    for _task, part in within[within.model_category == "interpretable"].groupby("task"):
        if len(part):
            within.loc[part.index[0], "best_task_specific_interpretable"] = True
    return cross_task, within


def _synthetic_linear_records(config: Mapping, seed: int):
    settings = config.get("neural_sanity", {})
    feature_count = int(settings.get("feature_count", 8))
    decisions = int(settings.get("decisions", 8))
    split_episodes = settings.get(
        "episodes_per_task", {"train": 48, "validation": 16, "test": 16}
    )
    noise_sd = float(settings.get("noise_sd", 0.03))
    features = [f"synthetic_x_{index}" for index in range(feature_count)]
    beta = np.linspace(0.25, 1.0, feature_count)
    beta *= 1.5 / np.linalg.norm(beta)
    rng = np.random.default_rng(int(seed))
    records = []
    tasks = ("bandit", "foraging", "solvability")
    for split in ("train", "validation", "test"):
        for task_index, task in enumerate(tasks):
            for episode in range(int(split_episodes[split])):
                episode_id = f"linear-{split}-{task}-{episode:04d}"
                episode_offset = rng.normal(scale=0.05)
                for decision in range(decisions):
                    values = rng.normal(size=feature_count)
                    target = (
                        0.3
                        + float(values @ beta)
                        + episode_offset
                        + rng.normal(scale=noise_sd)
                    )
                    records.append(
                        {
                            **dict(zip(features, values)),
                            "task": task,
                            "episode_id": episode_id,
                            "pair_id": episode_id,
                            "state_id": f"{episode_id}:{decision}",
                            "round": decision,
                            "split": split,
                            "persistence_logit": target,
                            "continue": int(rng.random() < sigmoid([target])[0]),
                            "synthetic_task_index": task_index,
                        }
                    )
    return records, features, beta


def _linear_reference(train, application, features):
    normalizer = TrainStandardizer.fit(
        [[float(row[name]) for name in features] for row in train], features
    )
    x_train = normalizer.transform(
        [[float(row[name]) for name in features] for row in train]
    )
    coefficient = weighted_ridge_fit(
        x_train,
        [row["persistence_logit"] for row in train],
        balanced_weights(train, task_balanced=True),
    )
    x_application = normalizer.transform(
        [[float(row[name]) for name in features] for row in application]
    )
    return linear_predict(x_application, coefficient)


def run_neural_linear_recovery_sanity(
    config: Mapping,
    output_dir: str | Path,
    *,
    logger=None,
) -> dict:
    """Require MLP/GRU estimators to retain an explicit linear solution."""

    output = Path(output_dir)
    records, features, beta = _synthetic_linear_records(config, int(config["seed"]) + 4049)
    splits = {
        name: [row for row in records if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    settings = config.get("neural_sanity", {})
    epochs = int(settings.get("max_epochs", 80))
    patience = int(settings.get("early_stopping_patience", 12))
    learning_rate = float(settings.get("learning_rate", 1e-3))
    hidden_size = int(settings.get("gru_hidden_size", 16))
    if logger is not None:
        logger.note(
            "neural_linear_sanity",
            f"fitting linear, MLP, and GRU on {len(records)} synthetic y=Xbeta+noise states",
        )
    linear_prediction = _linear_reference(splits["train"], splits["test"], features)
    mlp = fit_mlp_ceiling(
        splits["train"],
        splits["validation"],
        splits["test"],
        features,
        hidden_sizes=(max(16, 2 * len(features)), max(8, len(features))),
        learning_rate=learning_rate,
        dropout=0.0,
        max_epochs=epochs,
        patience=patience,
        seed=int(config["seed"]) + 1,
    )
    gru = fit_gru_ceiling(
        splits["train"],
        splits["validation"],
        splits["test"],
        features,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        dropout=0.0,
        max_epochs=epochs,
        patience=patience,
        seed=int(config["seed"]) + 2,
    )
    observed = [row["persistence_logit"] for row in splits["test"]]
    weights = balanced_weights(splits["test"], task_balanced=True)
    scores = {
        "linear": persistence_metrics(observed, linear_prediction, weights),
        "mlp": persistence_metrics(observed, mlp["prediction"], weights),
        "gru": persistence_metrics(observed, gru["prediction"], weights),
    }
    tolerance = float(config.get("neural_linear_recovery_tolerance_r2", 0.02))
    mlp_gap = float(scores["mlp"]["r_squared"] - scores["linear"]["r_squared"])
    gru_gap = float(scores["gru"]["r_squared"] - scores["linear"]["r_squared"])
    checks = {
        "mlp_approximately_linear": abs(mlp_gap) <= tolerance,
        "gru_not_below_linear_tolerance": gru_gap >= -tolerance,
    }
    result = {
        "passed": bool(all(checks.values())),
        "tolerance_r2": tolerance,
        "checks": checks,
        "r_squared_gap_vs_linear": {"mlp": mlp_gap, "gru": gru_gap},
        "metrics": scores,
        "selected_epochs": {
            "mlp": int(mlp["selected_epochs"]),
            "gru": int(gru["selected_epochs"]),
        },
        "linear_skip_initialized": {
            "mlp": bool(mlp.get("linear_skip_initialized")),
            "gru": bool(gru.get("linear_skip_initialized")),
        },
        "dataset": {
            "generating_equation": "y = X beta + episode_offset + epsilon",
            "features": features,
            "beta": beta.tolist(),
            "noise_sd": float(settings.get("noise_sd", 0.03)),
            "states_by_split": {name: len(rows) for name, rows in splits.items()},
            "episodes_by_split": {
                name: len({row["episode_id"] for row in rows})
                for name, rows in splits.items()
            },
        },
        "pipeline_diagnostics": {
            "feature_scaling": "train-split standardization applied to all estimators",
            "target_scaling": "raw persistence logit; linear skip initialized by weighted ridge",
            "weighting": "equal episode mass and equal task mass",
            "sequence_masking": "zero-weight padding; one independent hidden state per episode",
            "selection": "early stopping uses validation only; test is application only",
        },
    }
    (output / "neural_ceiling_sanity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if logger is not None:
        logger.note(
            "neural_linear_sanity",
            f"linear R2={scores['linear']['r_squared']:.4f}; "
            f"MLP R2={scores['mlp']['r_squared']:.4f}; "
            f"GRU R2={scores['gru']['r_squared']:.4f}; passed={result['passed']}",
        )
    if not result["passed"]:
        raise RuntimeError(
            "neural estimators failed the synthetic linear-recovery gate; "
            "see neural_ceiling_sanity.json"
        )
    return result


def _state_ids(fit: Mapping) -> tuple[str, ...]:
    return tuple(str(row["state_id"]) for row in fit["test_records"])


def _select_flexible_linear_fit(fits: Sequence[Mapping], required_tasks: Sequence[str]):
    required = set(required_tasks)
    candidates = [
        fit
        for fit in fits
        if fit["model"] == "flexible_linear"
        and fit["information_set"] == "observable"
        and {str(row["task"]) for row in fit["test_records"]} == required
    ]
    if not candidates:
        raise ValueError("no observable three-task flexible-linear fit is available")
    best_score = min(float(fit["validation_macro_mse"]) for fit in candidates)
    tolerance = max(1e-8, 0.01 * best_score)
    sharing_order = {
        "fully_shared": 0,
        "shared_architecture_task_observation": 1,
        "task_specific": 2,
    }
    return min(
        (
            fit
            for fit in candidates
            if float(fit["validation_macro_mse"]) <= best_score + tolerance
        ),
        key=lambda fit: (
            int(fit["parameter_count"]),
            sharing_order.get(str(fit["sharing"]), 99),
            str(fit["sharing"]),
        ),
    )


def build_flexible_comparison(
    fits: Sequence[Mapping], required_tasks: Sequence[str]
) -> tuple[pd.DataFrame, Mapping]:
    """Compare full linear, MLP, and GRU on exactly the same test states."""

    linear = _select_flexible_linear_fit(fits, required_tasks)
    candidates = [linear]
    for model in ("mlp", "gru"):
        matches = [
            fit
            for fit in fits
            if fit["model"] == model and fit["information_set"] == "observable"
        ]
        if matches:
            candidates.append(matches[0])
    reference_states = _state_ids(linear)
    for fit in candidates[1:]:
        if _state_ids(fit) != reference_states:
            raise ValueError(
                f"{fit['model']} and flexible_linear do not use identical held-out states"
            )
    rows = []
    for fit in candidates:
        aggregate, _ = evaluate_predictions(fit)
        rows.append(
            {
                "model": fit["model"],
                "information_set": fit["information_set"],
                "sharing": fit["sharing"],
                "r_squared": aggregate["r_squared"],
                "mse": aggregate["mse"],
                "pearson_r": aggregate["pearson_r"],
                "sampled_choice_log_loss": aggregate["log_loss"],
                "states": aggregate["states"],
                "episodes": aggregate["episodes"],
                "parameter_count": aggregate["parameter_count"],
                "validation_macro_mse": fit["validation_macro_mse"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["r_squared", "mse", "parameter_count"], ascending=[False, True, True]
    ).reset_index(drop=True)
    comparison.insert(0, "flexible_rank", np.arange(1, len(comparison) + 1))
    comparison["best_flexible_model"] = comparison.flexible_rank == 1
    return comparison, linear


def _definition(name: str, features: Sequence[str]) -> ModelDefinition:
    return ModelDefinition(
        code="FOLLOWUP",
        name=name,
        features=tuple(features),
        family="linear",
        description="Preregistered flexible-linear feature-family follow-up.",
    )


def _metrics_by_task(fit: Mapping) -> dict[str, dict]:
    aggregate, taskwise = evaluate_predictions(dict(fit))
    return {"macro": aggregate, **{str(row["task"]): row for row in taskwise}}


def _paired_feature_bootstrap(
    full_fit: Mapping,
    ablated_fits: Mapping[str, Mapping],
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    if int(samples) < 20:
        raise ValueError("feature-group bootstrap requires at least 20 samples")
    full_lookup = {
        str(row["state_id"]): (row, float(prediction))
        for row, prediction in zip(full_fit["test_records"], full_fit["prediction"])
    }
    rows = []
    for group_index, (group, ablated) in enumerate(ablated_fits.items()):
        ablated_lookup = {
            str(row["state_id"]): float(prediction)
            for row, prediction in zip(ablated["test_records"], ablated["prediction"])
        }
        if set(ablated_lookup) != set(full_lookup):
            raise ValueError(f"feature ablation {group!r} changed the held-out states")
        episodes: dict[tuple[str, str, str], list[list[float]]] = {}
        for state_id, (record, full_prediction) in full_lookup.items():
            ablated_prediction = ablated_lookup[state_id]
            observed = float(record["persistence_logit"])
            choice = float(record["continue"])
            full_probability = float(sigmoid([full_prediction])[0])
            ablated_probability = float(sigmoid([ablated_prediction])[0])
            key = (
                str(record["task"]),
                str(record.get("pair_id", record["episode_id"])),
                str(record["episode_id"]),
            )
            episodes.setdefault(key, []).append(
                [
                    observed,
                    observed**2,
                    (observed - full_prediction) ** 2,
                    (observed - ablated_prediction) ** 2,
                    -(
                        choice * np.log(np.clip(full_probability, 1e-9, 1.0))
                        + (1 - choice)
                        * np.log(np.clip(1 - full_probability, 1e-9, 1.0))
                    ),
                    -(
                        choice * np.log(np.clip(ablated_probability, 1e-9, 1.0))
                        + (1 - choice)
                        * np.log(np.clip(1 - ablated_probability, 1e-9, 1.0))
                    ),
                ]
            )
        pairs: dict[str, dict[str, list[np.ndarray]]] = {}
        for (task, pair, _episode), values in episodes.items():
            pairs.setdefault(task, {}).setdefault(pair, []).append(
                np.asarray(values, dtype=float).mean(axis=0)
            )
        rng = np.random.default_rng(int(seed) + group_index * 104729)
        task_draws = {}
        for task, task_pairs in sorted(pairs.items()):
            values = np.stack(
                [np.mean(episode_values, axis=0) for episode_values in task_pairs.values()]
            )
            indices = rng.integers(0, len(values), size=(int(samples), len(values)))
            means = values[indices].mean(axis=1)
            variance = np.maximum(means[:, 1] - means[:, 0] ** 2, 1e-12)
            draws = {
                "delta_r_squared": (means[:, 3] - means[:, 2]) / variance,
                "delta_mse": means[:, 3] - means[:, 2],
                "delta_log_loss": means[:, 5] - means[:, 4],
            }
            task_draws[task] = draws
            for metric, values_drawn in draws.items():
                rows.append(
                    {
                        "feature_group": group,
                        "task": task,
                        "metric": metric,
                        "estimate": float(np.mean(values_drawn)),
                        "ci_low": float(np.quantile(values_drawn, 0.025)),
                        "ci_high": float(np.quantile(values_drawn, 0.975)),
                        "samples": int(samples),
                        "clusters": len(values),
                    }
                )
        for metric in ("delta_r_squared", "delta_mse", "delta_log_loss"):
            macro = np.stack(
                [task_draws[task][metric] for task in sorted(task_draws)]
            ).mean(axis=0)
            rows.append(
                {
                    "feature_group": group,
                    "task": "macro",
                    "metric": metric,
                    "estimate": float(np.mean(macro)),
                    "ci_low": float(np.quantile(macro, 0.025)),
                    "ci_high": float(np.quantile(macro, 0.975)),
                    "samples": int(samples),
                    "clusters": sum(len(value) for value in pairs.values()),
                }
            )
    return pd.DataFrame(rows)


def run_feature_group_analysis(
    config: Mapping,
    splits: Mapping[str, Sequence[Mapping]],
    full_fit: Mapping,
    *,
    samples: int,
    seed: int,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit leave-one-family-out and family-only flexible linear models."""

    validate_feature_groups()
    full = tuple(flexible_features())
    if tuple(full_fit["feature_names"]) != full:
        raise ValueError("selected full flexible fit does not match the feature-group schema")
    sharing = str(full_fit["sharing"])
    full_metrics = _metrics_by_task(full_fit)
    ablation_rows, only_rows, ablated_fits, all_fits = [], [], {}, []
    for index, (group, group_features) in enumerate(FLEXIBLE_FEATURE_GROUPS.items(), start=1):
        started = time.perf_counter()
        if logger is not None:
            logger.note(
                "feature_groups",
                f"{index}/{len(FLEXIBLE_FEATURE_GROUPS)} {group}: fitting ablated and group-only models",
            )
        ablated_features = [name for name in full if name not in set(group_features)]
        only_features = [*FLEXIBLE_NUISANCE_FEATURES, *group_features]
        ablated = fit_interpretable_model(
            splits["train"],
            splits["validation"],
            splits["test"],
            _definition(f"flexible_without_{group}", ablated_features),
            information_set="observable",
            sharing=sharing,
            config=config,
        )
        only = fit_interpretable_model(
            splits["train"],
            splits["validation"],
            splits["test"],
            _definition(f"flexible_only_{group}", only_features),
            information_set="observable",
            sharing=sharing,
            config=config,
        )
        ablated_fits[group] = ablated
        all_fits.extend((ablated, only))
        ablated_metrics = _metrics_by_task(ablated)
        only_metrics = _metrics_by_task(only)
        for task in ("bandit", "foraging", "solvability", "macro"):
            full_row = full_metrics[task]
            without_row = ablated_metrics[task]
            ablation_rows.append(
                {
                    "feature_group": group,
                    "description": FEATURE_GROUP_DESCRIPTIONS[group],
                    "features": ";".join(group_features),
                    "task": task,
                    "full_r_squared": full_row["r_squared"],
                    "ablated_r_squared": without_row["r_squared"],
                    "delta_r_squared": full_row["r_squared"] - without_row["r_squared"],
                    "full_mse": full_row["mse"],
                    "ablated_mse": without_row["mse"],
                    "delta_mse": without_row["mse"] - full_row["mse"],
                    "full_log_loss": full_row["log_loss"],
                    "ablated_log_loss": without_row["log_loss"],
                    "delta_log_loss": without_row["log_loss"] - full_row["log_loss"],
                }
            )
        row = {
            "feature_group": group,
            "description": FEATURE_GROUP_DESCRIPTIONS[group],
            "features": ";".join(group_features),
        }
        for task in ("bandit", "foraging", "solvability", "macro"):
            prefix = "cross_task_macro" if task == "macro" else task
            row[f"{prefix}_r_squared"] = only_metrics[task]["r_squared"]
            row[f"{prefix}_mse"] = only_metrics[task]["mse"]
            row[f"{prefix}_log_loss"] = only_metrics[task]["log_loss"]
        only_rows.append(row)
        if logger is not None:
            logger.note(
                "feature_groups",
                f"{group}: macro delta R2={ablation_rows[-1]['delta_r_squared']:.4f}; "
                f"completed in {time.perf_counter() - started:.1f}s",
            )
    if logger is not None:
        logger.note(
            "feature_group_bootstrap",
            f"starting {samples} paired pair/episode-cluster samples for "
            f"{len(ablated_fits)} feature families",
        )
    bootstrap = _paired_feature_bootstrap(
        full_fit, ablated_fits, samples=samples, seed=seed
    )
    return (
        pd.DataFrame(ablation_rows),
        pd.DataFrame(only_rows),
        bootstrap,
        all_fits,
    )


def run_followup_analysis(
    config: Mapping,
    splits: Mapping[str, Sequence[Mapping]],
    fits: Sequence[Mapping],
    metrics: pd.DataFrame,
    taskwise: pd.DataFrame,
    output_dir: str | Path,
    *,
    bootstrap_samples: int,
    logger=None,
) -> dict:
    """Run and save every non-neural deliverable in the mini PRD."""

    output = Path(output_dir)
    cross_task, task_specific = build_model_rankings(
        metrics, taskwise, config["tasks"]
    )
    cross_task.to_csv(output / "cross_task_model_ranking.csv", index=False)
    task_specific.to_csv(output / "task_specific_model_ranking.csv", index=False)
    flexible, full_fit = build_flexible_comparison(fits, config["tasks"])

    oracle_matches = [
        fit
        for fit in fits
        if fit["model"] == "flexible_linear"
        and fit["information_set"] == "oracle"
        and fit["sharing"] == full_fit["sharing"]
    ]
    if len(oracle_matches) != 1:
        raise ValueError("expected exactly one matching oracle flexible-linear fit")
    oracle_aggregate, _ = evaluate_predictions(oracle_matches[0])
    observable = flexible[flexible.model == "flexible_linear"].iloc[0]
    oracle_row = {
        "flexible_rank": np.nan,
        "model": "flexible_linear_oracle",
        "information_set": "oracle",
        "sharing": oracle_matches[0]["sharing"],
        "r_squared": oracle_aggregate["r_squared"],
        "mse": oracle_aggregate["mse"],
        "pearson_r": oracle_aggregate["pearson_r"],
        "sampled_choice_log_loss": oracle_aggregate["log_loss"],
        "states": oracle_aggregate["states"],
        "episodes": oracle_aggregate["episodes"],
        "parameter_count": oracle_aggregate["parameter_count"],
        "validation_macro_mse": oracle_matches[0]["validation_macro_mse"],
        "best_flexible_model": False,
        "delta_r_squared_oracle_minus_observable": (
            oracle_aggregate["r_squared"] - float(observable.r_squared)
        ),
        "delta_mse_observable_minus_oracle": (
            float(observable.mse) - oracle_aggregate["mse"]
        ),
        "delta_log_loss_observable_minus_oracle": (
            float(observable.sampled_choice_log_loss) - oracle_aggregate["log_loss"]
        ),
    }
    flexible["delta_r_squared_oracle_minus_observable"] = np.nan
    flexible["delta_mse_observable_minus_oracle"] = np.nan
    flexible["delta_log_loss_observable_minus_oracle"] = np.nan
    flexible = pd.concat((flexible, pd.DataFrame([oracle_row])), ignore_index=True)
    flexible.to_csv(output / "flexible_model_comparison.csv", index=False)

    ablation, group_only, bootstrap, feature_fits = run_feature_group_analysis(
        config,
        splits,
        full_fit,
        samples=int(bootstrap_samples),
        seed=int(config["seed"]) + 1907,
        logger=logger,
    )
    ablation.to_csv(output / "feature_group_ablation.csv", index=False)
    group_only.to_csv(output / "feature_group_only.csv", index=False)
    bootstrap.to_csv(output / "feature_group_bootstrap.csv", index=False)
    best_cross = cross_task.iloc[0]
    best_interpretable = cross_task[cross_task.best_cross_task_interpretable].iloc[0]
    best_flexible = flexible[flexible.best_flexible_model].iloc[0]
    summary = {
        "best_cross_task_model": str(best_cross.model),
        "best_cross_task_model_sharing": str(best_cross.sharing),
        "best_cross_task_interpretable_model": str(best_interpretable.model),
        "best_flexible_model": str(best_flexible.model),
        "selected_flexible_linear_sharing": str(full_fit["sharing"]),
        "oracle_delta_r_squared": float(
            oracle_row["delta_r_squared_oracle_minus_observable"]
        ),
        "feature_fits": len(feature_fits),
    }
    (output / "followup_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
