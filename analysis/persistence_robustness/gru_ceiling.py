"""Capacity-oriented, balanced, multi-seed GRU ceiling for PRD 2.5."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import itertools
import math
import random

import numpy as np
import pandas as pd

from analysis.comparative_persistence.evaluation.metrics import (
    summarize_predictions,
    task_metrics,
)
from analysis.comparative_persistence.flexible.neural import select_and_fit_neural
from analysis.comparative_persistence.hazard_models.modeling import (
    select_and_fit_linear_model,
)
from analysis.comparative_persistence.semantic_features import (
    IMMEDIATE_FEATURES,
    build_feature_matrix,
)


CURRENT_RECURRENT_FEATURES = (
    "time_norm",
    "effort_norm",
    "invested_norm",
    *IMMEDIATE_FEATURES,
)
SHORT_HISTORY_FEATURES = (
    *CURRENT_RECURRENT_FEATURES,
    "continue_streak",
    "success_streak",
    "failure_streak",
    *(f"action_lag_{lag}" for lag in (1, 2, 3, 5)),
    *(f"outcome_lag_{lag}" for lag in (1, 2, 3, 5)),
)


@dataclass(frozen=True)
class SequenceExample:
    task: str
    episode_id: str
    features: np.ndarray
    targets: np.ndarray
    state_ids: tuple[str, ...]


def build_sequence_examples(frame, variant):
    features = (
        CURRENT_RECURRENT_FEATURES
        if variant == "current_state"
        else SHORT_HISTORY_FEATURES
        if variant == "short_history"
        else None
    )
    if features is None:
        raise ValueError(f"unknown GRU input variant: {variant}")
    frame = frame.reset_index(drop=True)
    matrix, names = build_feature_matrix(frame, features)
    examples = []
    for (task, episode_id), indices in frame.groupby(
        ["task", "episode_id"], sort=False
    ).groups.items():
        ordered = sorted(indices, key=lambda index: int(frame.loc[index, "round"]))
        examples.append(
            SequenceExample(
                str(task),
                str(episode_id),
                matrix[ordered].astype(np.float32),
                frame.loc[ordered, "hazard_event"].to_numpy(dtype=np.float32),
                tuple(frame.loc[ordered, "state_id"].astype(str)),
            )
        )
    return examples, names


def balanced_episode_batches(examples, episodes_per_task, seed):
    """Yield batches with exactly equal episode counts from every task."""

    by_task = {}
    for example in examples:
        by_task.setdefault(example.task, []).append(example)
    if not by_task:
        return []
    rng = random.Random(int(seed))
    for values in by_task.values():
        rng.shuffle(values)
    per_task = int(episodes_per_task)
    if per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    batch_count = max(math.ceil(len(values) / per_task) for values in by_task.values())
    batches = []
    for batch_index in range(batch_count):
        batch = []
        for task in sorted(by_task):
            values = by_task[task]
            for offset in range(per_task):
                index = (batch_index * per_task + offset) % len(values)
                batch.append(values[index])
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def collate_sequences(examples, device):
    import torch

    lengths = torch.tensor([len(example.targets) for example in examples], dtype=torch.long)
    maximum = int(lengths.max())
    feature_count = examples[0].features.shape[1]
    x = torch.zeros((len(examples), maximum, feature_count), dtype=torch.float32)
    y = torch.zeros((len(examples), maximum), dtype=torch.float32)
    mask = torch.zeros((len(examples), maximum), dtype=torch.bool)
    positions = []
    tasks = []
    for index, example in enumerate(examples):
        length = len(example.targets)
        x[index, :length] = torch.from_numpy(example.features)
        y[index, :length] = torch.from_numpy(example.targets)
        mask[index, :length] = True
        positions.append(example.state_ids)
        tasks.append(example.task)
    return (
        x.to(device),
        y.to(device),
        mask.to(device),
        lengths,
        positions,
        tasks,
    )


def _torch_module():
    import torch

    class CausalHazardGRU(torch.nn.Module):
        def __init__(self, input_size, hidden_size, layers):
            super().__init__()
            self.project = torch.nn.Linear(input_size, hidden_size)
            self.gru = torch.nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=int(layers),
                batch_first=True,
            )
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, values, lengths):
            projected = torch.nn.functional.gelu(self.project(values))
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                projected,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            sequence, _state = self.gru(packed)
            padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
                sequence,
                batch_first=True,
                total_length=values.shape[1],
            )
            return self.output(padded).squeeze(-1)

    return CausalHazardGRU


def make_gru(input_size, hidden_size, layers):
    return _torch_module()(input_size, hidden_size, layers)


def _masked_loss(logits, targets, mask):
    import torch

    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return raw[mask].mean()


def _predict(model, examples, device, batch_size=128):
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), int(batch_size)):
            batch_examples = examples[start : start + int(batch_size)]
            x, y, mask, lengths, positions, tasks = collate_sequences(
                batch_examples, device
            )
            probability = torch.sigmoid(model(x, lengths)).cpu().numpy()
            observed = y.cpu().numpy()
            valid = mask.cpu().numpy()
            for index, example in enumerate(batch_examples):
                for step in range(int(valid[index].sum())):
                    rows.append(
                        {
                            "state_id": positions[index][step],
                            "episode_id": example.episode_id,
                            "task": tasks[index],
                            "observed": float(observed[index, step]),
                            "predicted": float(probability[index, step]),
                        }
                    )
    return pd.DataFrame(rows)


def _fit_configuration(
    train_examples,
    validation_examples,
    test_examples,
    *,
    input_size,
    hidden_size,
    layers,
    learning_rate,
    weight_decay,
    seed,
    maximum_epochs,
    patience,
    gradient_clip,
    episodes_per_task,
):
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_gru(input_size, hidden_size, layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    curves = []
    for epoch in range(1, int(maximum_epochs) + 1):
        model.train()
        train_losses = []
        batches = balanced_episode_batches(
            train_examples,
            episodes_per_task,
            int(seed) * 100_000 + epoch,
        )
        for examples in batches:
            x, y, mask, lengths, _positions, _tasks = collate_sequences(
                examples, device
            )
            optimizer.zero_grad()
            loss = _masked_loss(model(x, lengths), y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_predictions = _predict(model, validation_examples, device)
        validation_loss = summarize_predictions(validation_predictions)["macro_log_loss"]
        curves.append(
            {
                "epoch": epoch,
                "training_log_loss": float(np.mean(train_losses)),
                "validation_macro_log_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("GRU validation loss was never finite")
    model.load_state_dict(best_state)
    validation_predictions = _predict(model, validation_examples, device)
    test_predictions = _predict(model, test_examples, device)
    return {
        "validation": summarize_predictions(validation_predictions),
        "test": summarize_predictions(test_predictions),
        "curves": curves,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_predictions": test_predictions,
    }


def _grid(config, smoke):
    block = config["gru"]
    if smoke:
        return {
            "hidden_sizes": config["smoke"]["gru_hidden_sizes"],
            "learning_rates": config["smoke"]["gru_learning_rates"],
            "weight_decay": config["smoke"]["gru_weight_decay"],
            "seeds": config["smoke"]["gru_seeds"],
            "maximum_epochs": config["smoke"]["gru_maximum_epochs"],
            "patience": config["smoke"]["gru_patience"],
        }
    return block


def _baseline_rows(records, model_config):
    train = records[records.split == "train"]
    validation = records[records.split == "validation"]
    test = records[records.split == "test"]
    finite = select_and_fit_linear_model(
        train,
        validation,
        test,
        "finite_history",
        "fully_shared",
        model_config,
    )
    finite_validation = select_and_fit_linear_model(
        train,
        validation,
        validation,
        "finite_history",
        "fully_shared",
        model_config,
    )
    mlp = select_and_fit_neural(
        train, validation, test, "mlp", "fully_shared", model_config
    )
    finite_scored = pd.DataFrame(
        {
            "task": finite.application_records.task,
            "episode_id": finite.application_records.episode_id,
            "observed": finite.application_records.hazard_event,
            "predicted": finite.prediction,
        }
    )
    mlp_scored = pd.DataFrame(
        {
            "task": mlp.application_records.task,
            "episode_id": mlp.application_records.episode_id,
            "observed": mlp.application_records.hazard_event,
            "predicted": mlp.prediction,
        }
    )
    return {
        "finite_history": {
            "validation_macro_log_loss": finite_validation.validation_macro_log_loss,
            **summarize_predictions(finite_scored),
        },
        "mlp": {
            "validation_macro_log_loss": mlp.validation_macro_log_loss,
            **summarize_predictions(mlp_scored),
            "parameter_count": mlp.parameter_count,
        },
    }


def run_gru_ceiling(records, config, model_config, *, smoke=False, logger=None):
    records = records[records.is_persistence_task.astype(bool)].copy()
    grid = _grid(config, smoke)
    variants = config["gru"]["input_variants"]
    examples = {}
    names = {}
    for variant in variants:
        examples[variant] = {}
        for split in ("train", "validation", "test"):
            examples[variant][split], names[variant] = build_sequence_examples(
                records[records.split == split], variant
            )
    baselines = _baseline_rows(records, model_config)
    results, curves, fitted = [], [], {}
    tuning_seed = int(grid["seeds"][0])
    common = {
        "maximum_epochs": int(grid["maximum_epochs"]),
        "patience": int(grid["patience"]),
        "gradient_clip": float(config["gru"]["gradient_clip"]),
        "episodes_per_task": int(config["gru"]["episodes_per_task_per_batch"]),
    }

    def run_one(variant, hidden, layers, learning_rate, weight_decay, seed):
        key = (variant, int(hidden), int(layers), float(learning_rate), float(weight_decay), int(seed))
        if key not in fitted:
            fit = _fit_configuration(
                examples[variant]["train"],
                examples[variant]["validation"],
                examples[variant]["test"],
                input_size=len(names[variant]),
                hidden_size=hidden,
                layers=layers,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=seed,
                **common,
            )
            fitted[key] = fit
            row = {
                "input_variant": variant,
                "hidden_size": int(hidden),
                "layers": int(layers),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "seed": int(seed),
                "selected_epoch": fit["best_epoch"],
                "parameter_count": fit["parameter_count"],
                "validation_macro_log_loss": fit["validation"]["macro_log_loss"],
                "test_macro_log_loss": fit["test"]["macro_log_loss"],
                "test_episode_weighted_log_loss": fit["test"]["episode_weighted_log_loss"],
            }
            results.append(row)
            curves.extend(
                [{**row, **curve} for curve in fit["curves"]]
            )
            if logger is not None:
                logger.note(
                    "gru_ceiling",
                    f"{variant}; h={hidden}; layers={layers}; lr={learning_rate}; wd={weight_decay}; seed={seed}; val={row['validation_macro_log_loss']:.4f}",
                )
        return fitted[key]

    selected_hyper = {}
    for variant, hidden in itertools.product(variants, grid["hidden_sizes"]):
        candidates = []
        for learning_rate, weight_decay in itertools.product(
            grid["learning_rates"], grid["weight_decay"]
        ):
            fit = run_one(
                variant,
                hidden,
                1,
                learning_rate,
                weight_decay,
                tuning_seed,
            )
            candidates.append(
                (
                    fit["validation"]["macro_log_loss"],
                    float(learning_rate),
                    float(weight_decay),
                )
            )
        _loss, learning_rate, weight_decay = min(candidates)
        selected_hyper[(variant, int(hidden), 1)] = (learning_rate, weight_decay)
        for seed in grid["seeds"][1:]:
            run_one(variant, hidden, 1, learning_rate, weight_decay, seed)

    result_frame = pd.DataFrame(results)
    selected_rows = []
    for key, (learning_rate, weight_decay) in selected_hyper.items():
        variant, hidden, layers = key
        selected_rows.append(
            result_frame[
                (result_frame.input_variant == variant)
                & (result_frame.hidden_size == hidden)
                & (result_frame.layers == layers)
                & (result_frame.learning_rate == learning_rate)
                & (result_frame.weight_decay == weight_decay)
                & (result_frame.seed.isin(grid["seeds"]))
            ]
        )
    selected_frame = pd.concat(selected_rows, ignore_index=True)
    one_layer = (
        selected_frame.groupby(
            ["input_variant", "hidden_size", "layers", "learning_rate", "weight_decay"],
            as_index=False,
        )
        .agg(
            validation_mean=("validation_macro_log_loss", "mean"),
            validation_sd=("validation_macro_log_loss", "std"),
            test_mean=("test_macro_log_loss", "mean"),
            test_sd=("test_macro_log_loss", "std"),
            seeds=("seed", "nunique"),
            parameter_count=("parameter_count", "first"),
        )
        .fillna({"validation_sd": 0.0, "test_sd": 0.0})
        .sort_values("validation_mean")
    )

    epsilon = float(config["gru"]["ceiling_epsilon"])
    if not smoke and one_layer.iloc[0].validation_mean > baselines["mlp"]["validation_macro_log_loss"] + epsilon:
        top_sizes = (
            one_layer.sort_values("validation_mean").hidden_size.drop_duplicates().head(2).tolist()
        )
        for hidden in top_sizes:
            candidates = one_layer[one_layer.hidden_size == hidden].sort_values("validation_mean")
            source = candidates.iloc[0]
            key = (source.input_variant, int(hidden), 2)
            selected_hyper[key] = (source.learning_rate, source.weight_decay)
            for seed in grid["seeds"]:
                run_one(
                    source.input_variant,
                    hidden,
                    2,
                    source.learning_rate,
                    source.weight_decay,
                    seed,
                )

    result_frame = pd.DataFrame(results)
    result_frame["hyperparameter_selected_on_validation"] = False
    for (variant, hidden, layers), (learning_rate, weight_decay) in selected_hyper.items():
        mask = (
            (result_frame.input_variant == variant)
            & (result_frame.hidden_size == hidden)
            & (result_frame.layers == layers)
            & (result_frame.learning_rate == learning_rate)
            & (result_frame.weight_decay == weight_decay)
        )
        result_frame.loc[mask, "hyperparameter_selected_on_validation"] = True
    selected = result_frame[result_frame.hyperparameter_selected_on_validation]
    stability = (
        selected.groupby(
            ["input_variant", "hidden_size", "layers", "learning_rate", "weight_decay"],
            as_index=False,
        )
        .agg(
            validation_mean=("validation_macro_log_loss", "mean"),
            validation_sd=("validation_macro_log_loss", "std"),
            test_mean=("test_macro_log_loss", "mean"),
            test_sd=("test_macro_log_loss", "std"),
            seeds=("seed", "nunique"),
            parameter_count=("parameter_count", "first"),
        )
        .fillna({"validation_sd": 0.0, "test_sd": 0.0})
        .sort_values("validation_mean")
    )
    best = stability.iloc[0]
    by_size = (
        stability[stability.layers == 1]
        .groupby("hidden_size", as_index=False)
        .validation_mean.min()
        .sort_values("hidden_size")
    )
    plateau = bool(
        len(by_size) >= 2
        and abs(by_size.validation_mean.iloc[-2] - by_size.validation_mean.iloc[-1])
        <= float(config["gru"]["plateau_epsilon"])
    )
    stable = bool(best.validation_sd <= epsilon)
    competitive = bool(best.test_mean <= baselines["mlp"]["macro_log_loss"] + epsilon)
    ceiling = pd.DataFrame(
        [
            {"model": "finite_history", **baselines["finite_history"]},
            {"model": "mlp", **baselines["mlp"]},
            {
                "model": "gru",
                "validation_macro_log_loss": best.validation_mean,
                "macro_log_loss": best.test_mean,
                "seed_sd": best.test_sd,
                "parameter_count": int(best.parameter_count),
                "input_variant": best.input_variant,
                "hidden_size": int(best.hidden_size),
                "layers": int(best.layers),
                "learning_rate": best.learning_rate,
                "weight_decay": best.weight_decay,
                "stable_across_seeds": stable,
                "capacity_plateau": plateau,
                "competitive_with_mlp": competitive,
                "credible_recurrent_ceiling": bool(stable and plateau and competitive),
                "gru_improvement_over_finite_history": baselines["finite_history"]["macro_log_loss"] - best.test_mean,
                "gru_difference_from_mlp": best.test_mean - baselines["mlp"]["macro_log_loss"],
            },
        ]
    )
    best_mask = (
        (result_frame.input_variant == best.input_variant)
        & (result_frame.hidden_size == best.hidden_size)
        & (result_frame.layers == best.layers)
        & (result_frame.learning_rate == best.learning_rate)
        & (result_frame.weight_decay == best.weight_decay)
    )
    taskwise = []
    for row in result_frame[best_mask].itertuples():
        fit = fitted[
            (
                row.input_variant,
                int(row.hidden_size),
                int(row.layers),
                float(row.learning_rate),
                float(row.weight_decay),
                int(row.seed),
            )
        ]
        metrics = task_metrics(fit["test_predictions"])
        metrics["seed"] = int(row.seed)
        taskwise.append(metrics)
    gru_taskwise = (
        pd.concat(taskwise, ignore_index=True)
        .groupby("task", as_index=False)
        .agg(
            states=("states", "first"),
            log_loss=("log_loss", "mean"),
            log_loss_seed_sd=("log_loss", "std"),
            brier=("brier", "mean"),
            auc=("auc", "mean"),
            calibration_error=("calibration_error", "mean"),
            calibration_intercept=("calibration_intercept", "mean"),
            calibration_slope=("calibration_slope", "mean"),
            deviance_explained=("deviance_explained", "mean"),
        )
        .fillna({"log_loss_seed_sd": 0.0})
    )
    gru_taskwise.insert(0, "sharing", "fully_shared")
    gru_taskwise.insert(0, "model", "large_gru")
    return result_frame, stability, ceiling, pd.DataFrame(curves), gru_taskwise
