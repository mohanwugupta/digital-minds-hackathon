"""All-layer readouts and functional mappings in a shared capacity-control space."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from analysis.persistence_convergence.layerwise_geometry import direction_similarity
from analysis.persistence_convergence.task_specific_readouts import (
    BlockProjector,
    fit_task_readout,
)
from computational_modeling.analysis.evaluate_models import persistence_metrics
from computational_modeling.models.base import (
    balanced_weights,
    linear_predict,
    weighted_ridge_fit,
)


HISTORY_COLUMNS = (
    "previous_choice",
    "previous_outcome",
    "failure_streak",
    "success_streak",
    "action_lag_2",
    "outcome_lag_2",
    "action_lag_3",
    "outcome_lag_3",
    "action_lag_5",
    "outcome_lag_5",
)


def _split_indices(frame):
    return {
        split: np.flatnonzero(frame.split.astype(str).to_numpy() == split)
        for split in ("train", "validation", "test")
    }


def _weights(frame, indices):
    return balanced_weights(
        frame.iloc[indices].to_dict(orient="records"), task_balanced=False
    )


def _predict(model, projected):
    standardized = (
        projected - model["normalizer_mean"]
    ) / model["normalizer_scale"]
    return linear_predict(standardized, model["coefficient"])


def _semantic_targets(frame):
    train = frame.split.astype(str).eq("train")
    history_values = frame.loc[:, HISTORY_COLUMNS].to_numpy(dtype=float)
    mean = history_values[train].mean(axis=0)
    scale = history_values[train].std(axis=0)
    scale[scale == 0] = 1.0
    history = ((history_values - mean) / scale).mean(axis=1)
    evidence = frame.termination_advantage.to_numpy(dtype=float)
    evidence = (evidence - evidence[train].mean()) / max(evidence[train].std(), 1e-12)
    finite_coefficient = weighted_ridge_fit(
        ((history_values - mean) / scale)[train],
        frame.loc[train, "persistence_logit"].to_numpy(dtype=float),
        np.ones(int(train.sum())),
        penalty=1.0,
    )
    finite_prediction = linear_predict(
        (history_values - mean) / scale, finite_coefficient
    )
    choice_kernel = (
        frame.previous_choice.to_numpy(dtype=float)
        + 0.7 * frame.action_lag_2.to_numpy(dtype=float)
        + 0.7**2 * frame.action_lag_3.to_numpy(dtype=float)
        + 0.7**4 * frame.action_lag_5.to_numpy(dtype=float)
    )
    outcome_kernel = (
        frame.previous_outcome.to_numpy(dtype=float)
        + 0.7 * frame.outcome_lag_2.to_numpy(dtype=float)
        + 0.7**2 * frame.outcome_lag_3.to_numpy(dtype=float)
        + 0.7**4 * frame.outcome_lag_5.to_numpy(dtype=float)
    )
    return {
        "previous_semantic_choice": frame.previous_choice.to_numpy(dtype=float),
        "previous_outcome": frame.previous_outcome.to_numpy(dtype=float),
        "choice_history_kernel": choice_kernel,
        "outcome_history_kernel": outcome_kernel,
        "finite_history_prediction": finite_prediction,
        "history_summary": history,
        "current_evidence": evidence,
        "history_x_current_evidence": history * evidence,
    }


def run_task_specific_readouts(datasets, config, *, logger=None):
    tasks = sorted(datasets)
    width = {dataset.shape[2] for dataset in datasets.values()}
    layers = {dataset.shape[1] for dataset in datasets.values()}
    if len(width) != 1 or len(layers) != 1:
        raise ValueError("task activation caches must share layer and width shapes")
    width, layer_count = width.pop(), layers.pop()
    projector = BlockProjector(
        width,
        int(config["projection_dimensions"]),
        int(config["seed"]),
    )
    alphas = config["ridge_grid"]
    metrics_rows, functional_rows = [], []
    models_by_layer, directions_by_layer = {}, {}
    for layer in range(layer_count):
        models_by_layer[layer] = {}
        directions_by_layer[layer] = {}
        projected_by_task = {}
        for task in tasks:
            dataset = datasets[task]
            frame = dataset.metadata.reset_index(drop=True)
            values = projector.transform(dataset.open()[:, layer, :])
            projected_by_task[task] = values
            indices = _split_indices(frame)
            target = frame.persistence_logit.to_numpy(dtype=float)
            fit = fit_task_readout(
                values[indices["train"]],
                target[indices["train"]],
                values[indices["validation"]],
                target[indices["validation"]],
                values[indices["test"]],
                target[indices["test"]],
                alphas=alphas,
                train_weights=_weights(frame, indices["train"]),
                validation_weights=_weights(frame, indices["validation"]),
                test_weights=_weights(frame, indices["test"]),
            )
            random = np.random.default_rng(
                int(config["seed"]) + layer * 101 + tasks.index(task)
            )
            random_fit = fit_task_readout(
                values[indices["train"]],
                random.permutation(target[indices["train"]]),
                values[indices["validation"]],
                random.permutation(target[indices["validation"]]),
                values[indices["test"]],
                target[indices["test"]],
                alphas=alphas,
            )
            mapping_r_squared = float("nan")
            mapping_status = "constant"
            mapping = frame.mapping_id.astype(str)
            if mapping.nunique() == 2:
                mapping_target = (mapping == sorted(mapping.unique())[1]).astype(float).to_numpy()
                mapping_fit = fit_task_readout(
                    values[indices["train"]], mapping_target[indices["train"]],
                    values[indices["validation"]], mapping_target[indices["validation"]],
                    values[indices["test"]], mapping_target[indices["test"]],
                    alphas=alphas,
                )
                mapping_r_squared = mapping_fit["test_r_squared"]
                mapping_status = "fit"
            metrics_rows.append(
                {
                    "layer": layer,
                    "task": task,
                    "status": "fit",
                    "test_r_squared": fit["test_r_squared"],
                    "test_mse": fit["test_mse"],
                    "test_pearson_r": fit["test_pearson_r"],
                    "selected_alpha": fit["selected_alpha"],
                    "random_target_r_squared": random_fit["test_r_squared"],
                    "mapping_control_r_squared": mapping_r_squared,
                    "mapping_control_status": mapping_status,
                    "projection_dimensions": projector.dimensions,
                    "hidden_width": width,
                    "train_states": len(indices["train"]),
                    "validation_states": len(indices["validation"]),
                    "test_states": len(indices["test"]),
                }
            )
            models_by_layer[layer][task] = fit
            directions_by_layer[layer][task] = projector.lift_direction(fit["direction"])
            for target_name, semantic_target in _semantic_targets(frame).items():
                semantic_fit = fit_task_readout(
                    values[indices["train"]], semantic_target[indices["train"]],
                    values[indices["validation"]], semantic_target[indices["validation"]],
                    values[indices["test"]], semantic_target[indices["test"]],
                    alphas=alphas,
                )
                functional_rows.append(
                    {
                        "layer": layer,
                        "task": task,
                        "analysis": "semantic_variable_decoding",
                        "source_task": task,
                        "target_task": task,
                        "variable": target_name,
                        "r_squared": semantic_fit["test_r_squared"],
                        "pearson_r": semantic_fit["test_pearson_r"],
                    }
                )
        for source_task, target_task in permutations(tasks, 2):
            source_model = models_by_layer[layer][source_task]
            target_frame = datasets[target_task].metadata.reset_index(drop=True)
            target_indices = _split_indices(target_frame)["test"]
            prediction = _predict(
                source_model, projected_by_task[target_task][target_indices]
            )
            observed = target_frame.persistence_logit.to_numpy(dtype=float)[target_indices]
            transfer = persistence_metrics(observed, prediction, np.ones(len(observed)))
            functional_rows.append(
                {
                    "layer": layer,
                    "task": target_task,
                    "analysis": "cross_task_readout",
                    "source_task": source_task,
                    "target_task": target_task,
                    "variable": "persistence_logit",
                    "r_squared": transfer["r_squared"],
                    "pearson_r": transfer["pearson_r"],
                }
            )
        if logger is not None:
            logger.note("neural_readouts", f"completed layer {layer + 1}/{layer_count}")
    geometry = direction_similarity(directions_by_layer)
    literal = geometry[geometry.metric.isin(("pairwise_cosine", "mean_pairwise_cosine"))].reset_index(drop=True)
    subspace = geometry[~geometry.metric.isin(("pairwise_cosine", "mean_pairwise_cosine"))].reset_index(drop=True)
    return {
        "metrics": pd.DataFrame(metrics_rows),
        "direction_similarity": literal,
        "subspace_similarity": subspace,
        "functional_convergence": pd.DataFrame(functional_rows),
        "models": models_by_layer,
        "projector": projector,
    }


def predict_readout(model, projector, hidden):
    return _predict(model, projector.transform(hidden))
