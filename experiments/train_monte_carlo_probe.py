"""Train stable supervised probes for realized future cumulative return."""

import argparse
import json
import math
import os
import time

from analysis.probe_mechanism import run_probe_mechanism_analysis
from experiments.train_value_probe import load_shards, parse_layers
from interventions.artifacts import load_frozen_probe, save_frozen_probe
from interventions.neuron_selection import dimension_mask, select_top_fraction
from interventions.supervised_probe import fit_supervised_probe, supervised_metrics


def layer_dataset(shards: list, layer: int, episode_ids: set):
    import torch

    states, targets, records = [], [], []
    for shard in shards:
        if shard["episode_id"] not in episode_ids:
            continue
        states.append(shard["activations"][:, layer, :].float())
        targets.append(
            torch.tensor(
                [float(row["future_cumulative_return"]) for row in shard["records"]]
            )
        )
        records.extend(shard["records"])
    if not states:
        raise ValueError("split contains no states")
    return {
        "states": torch.cat(states),
        "targets": torch.cat(targets),
        "records": records,
    }


def _loss_streak(record: dict) -> int:
    rewards = record["reward_history"]
    if isinstance(rewards, str):
        rewards = json.loads(rewards)
    streak = 0
    for reward in reversed(rewards):
        if float(reward) != -2:
            break
        streak += 1
    return streak


def behavior_features(records: list[dict]):
    import torch

    rows = []
    for record in records:
        previous = record.get("previous_outcome")
        initial = previous in (None, "")
        previous_value = 0.0 if initial else float(previous)
        log_round = math.log1p(float(record["round"]))
        rows.append(
            [
                1.0,
                previous_value,
                float(initial),
                float(_loss_streak(record)),
                log_round,
                log_round**2,
            ]
        )
    return torch.tensor(rows, dtype=torch.float64)


def fit_behavior_baseline(train: dict, validation: dict, test: dict) -> dict:
    import torch

    train_x = behavior_features(train["records"])
    train_y = train["targets"].double()
    coefficients = torch.linalg.pinv(train_x) @ train_y

    def evaluate(data):
        target = data["targets"].double()
        prediction = behavior_features(data["records"]) @ coefficients
        residual_sum = float((prediction - target).square().sum())
        centered_sum = float((target - target.mean()).square().sum())
        return {
            "mse": residual_sum / len(target),
            "r_squared": 1.0 - residual_sum / centered_sum if centered_sum else 0.0,
            "correlation": float(
                torch.corrcoef(torch.stack([prediction, target]))[0, 1]
            ),
        }

    return {
        "feature_names": [
            "intercept",
            "previous_outcome",
            "initial_state",
            "loss_streak",
            "log_round",
            "log_round_squared",
        ],
        "coefficients": coefficients.tolist(),
        "validation": evaluate(validation),
        "test": evaluate(test),
    }


def constant_baseline(train: dict, validation: dict, test: dict) -> dict:
    import torch

    train_mean = float(train["targets"].mean())

    def evaluate(data):
        target = data["targets"].float()
        prediction = torch.full_like(target, train_mean)
        residual_sum = float((prediction - target).square().sum())
        centered_sum = float((target - target.mean()).square().sum())
        return {
            "mse": residual_sum / len(target),
            "r_squared": 1.0 - residual_sum / centered_sum if centered_sum else 0.0,
        }

    return {
        "train_mean_future_return": train_mean,
        "validation": evaluate(validation),
        "test": evaluate(test),
    }


def _zero_unselected_weights(probe, mask) -> None:
    import torch

    with torch.no_grad():
        probe.hidden.weight[:, mask == 0] = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", default="artifacts/activation_bank")
    parser.add_argument("--split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--output-dir", default="artifacts/mc_value_probes")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=62026)
    args = parser.parse_args()

    import torch

    shards = load_shards(args.activation_dir)
    with open(args.split, encoding="utf-8") as handle:
        split = json.load(handle)
    split_sets = {key: set(value) for key, value in split.items()}
    all_ids = set().union(*split_sets.values())
    shard_ids = {shard["episode_id"] for shard in shards}
    if all_ids != shard_ids:
        raise ValueError("activation shards do not exactly match the frozen episode split")
    layer_count = int(shards[0]["activations"].shape[1])
    layers = parse_layers(args.layers, layer_count)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "episode_split.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(split, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Targets and behavioral baselines do not depend on layer.
    reference = {
        name: layer_dataset(shards, layers[0], episode_ids)
        for name, episode_ids in split_sets.items()
    }
    baselines = {
        "constant": constant_baseline(
            reference["train"], reference["validation"], reference["test"]
        ),
        "recent_history": fit_behavior_baseline(
            reference["train"], reference["validation"], reference["test"]
        ),
    }

    summaries = []
    total_started = time.perf_counter()
    for layer in layers:
        started = time.perf_counter()
        data = {
            name: layer_dataset(shards, layer, episode_ids)
            for name, episode_ids in split_sets.items()
        }
        full_result = fit_supervised_probe(
            data["train"]["states"],
            data["train"]["targets"],
            data["validation"]["states"],
            data["validation"]["targets"],
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed + layer,
        )
        indices = select_top_fraction(full_result.probe, args.top_fraction)
        mask = dimension_mask(full_result.probe, indices)
        sparse_result = fit_supervised_probe(
            data["train"]["states"],
            data["train"]["targets"],
            data["validation"]["states"],
            data["validation"]["targets"],
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed + 10_000 + layer,
            input_mask=mask,
        )
        _zero_unselected_weights(sparse_result.probe, mask)

        def metrics(result, name, input_mask=None):
            return supervised_metrics(
                result.probe,
                data[name]["states"],
                data[name]["targets"],
                target_mean=result.target_mean,
                target_std=result.target_std,
                input_mask=input_mask,
            )

        summary = {
            "layer": layer,
            "neuron_count": int(indices.numel()),
            "full_epochs_trained": full_result.epochs_trained,
            "sparse_epochs_trained": sparse_result.epochs_trained,
            "full_best_validation_mse_z": full_result.best_validation_mse,
            "sparse_best_validation_mse_z": sparse_result.best_validation_mse,
            "full": {
                "validation": metrics(full_result, "validation"),
                "test": metrics(full_result, "test"),
                "history": full_result.history,
            },
            "sparse": {
                "validation": metrics(sparse_result, "validation", mask),
                "test": metrics(sparse_result, "test", mask),
                "history": sparse_result.history,
            },
        }
        metadata = {
            "target": "realized_future_cumulative_return",
            "target_units": "z-scored using training episodes",
            "target_mean": sparse_result.target_mean,
            "target_std": sparse_result.target_std,
            "input_mask_refit": True,
            "summary": summary,
            "config": vars(args),
        }
        save_frozen_probe(
            os.path.join(args.output_dir, f"layer_{layer:02d}_sparse.pt"),
            sparse_result.probe,
            layer,
            indices,
            metadata,
        )
        save_frozen_probe(
            os.path.join(args.output_dir, f"layer_{layer:02d}_full.pt"),
            full_result.probe,
            layer,
            indices,
            {
                **metadata,
                "target_mean": full_result.target_mean,
                "target_std": full_result.target_std,
                "input_mask_refit": False,
            },
        )
        summaries.append(summary)
        print(
            f"MC layer {layer}: sparse validation R2="
            f"{summary['sparse']['validation']['r_squared']:.4f}; "
            f"test R2={summary['sparse']['test']['r_squared']:.4f}; "
            f"runtime={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    best = max(
        summaries, key=lambda item: item["sparse"]["validation"]["r_squared"]
    )
    sparse_path = os.path.join(
        args.output_dir, f"layer_{best['layer']:02d}_sparse.pt"
    )
    full_path = os.path.join(args.output_dir, f"layer_{best['layer']:02d}_full.pt")
    sparse_probe, layer, indices, sparse_artifact = load_frozen_probe(sparse_path)
    full_probe, full_layer, _full_indices, _full_artifact = load_frozen_probe(full_path)
    if full_layer != layer:
        raise ValueError("full and sparse Monte Carlo probes disagree on layer")
    save_frozen_probe(
        os.path.join(args.output_dir, "frozen_best.pt"),
        sparse_probe,
        layer,
        indices,
        {
            **sparse_artifact["metadata"],
            "selection_metric": "sparse_validation_future_return_r_squared",
            "layers": summaries,
            "baselines": baselines,
            "adaptive_evaluation_caveat": (
                "The original held-out split is reused after TD-probe diagnostics "
                "motivated this Monte Carlo analysis; treat results as exploratory."
            ),
        },
    )
    mechanism = run_probe_mechanism_analysis(
        shards,
        sparse_probe,
        layer,
        indices,
        split["test"],
        args.output_dir,
        full_probe=full_probe,
        probe_output_mean=float(sparse_artifact["metadata"]["target_mean"]),
        probe_output_std=float(sparse_artifact["metadata"]["target_std"]),
    )
    payload = {
        "target": "realized_future_cumulative_return",
        "target_interpretation": (
            "Monte Carlo return under the observed policy and outcome schedule, "
            "not a counterfactual continuation advantage"
        ),
        "episode_split": {
            "source": args.split,
            "train_episodes": len(split["train"]),
            "validation_episodes": len(split["validation"]),
            "test_episodes": len(split["test"]),
        },
        "best_layer": layer,
        "baselines": baselines,
        "layers": summaries,
        "probe_mechanism": mechanism,
        "adaptive_evaluation_caveat": sparse_artifact["metadata"].get(
            "adaptive_evaluation_caveat",
            "The analysis is exploratory because probe redesign followed inspection of the TD test result.",
        ),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        f"Monte Carlo probe runtime: {time.perf_counter() - total_started:.1f}s "
        f"for {len(layers)} layers; best layer={layer}",
        flush=True,
    )


if __name__ == "__main__":
    main()
