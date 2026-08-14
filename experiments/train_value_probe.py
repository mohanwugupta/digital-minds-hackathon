"""Train, prune, compare, and freeze one TD value probe per candidate layer."""

import argparse
import glob
import json
import os
from typing import Dict, List

from bandit.schemas import split_episode_ids
from interventions.artifacts import save_frozen_probe
from interventions.neuron_selection import evaluate_full_and_pruned, select_top_fraction
from interventions.value_probe import fit_value_probe


def load_shards(directory: str):
    import torch

    paths = sorted(glob.glob(os.path.join(directory, "episode_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no activation shards in {directory}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def layer_dataset(shards: list, layer: int, episode_ids: set) -> Dict[str, object]:
    import torch

    states, next_states, rewards, terminal = [], [], [], []
    for shard in shards:
        if shard["episode_id"] not in episode_ids:
            continue
        activations = shard["activations"][:, layer, :].float()
        successor = torch.zeros_like(activations)
        if len(activations) > 1:
            successor[:-1] = activations[1:]
        rows = shard["records"]
        states.append(activations)
        next_states.append(successor)
        rewards.append(torch.tensor([float(row["subsequent_reward"]) for row in rows]))
        terminal.append(torch.tensor([str(row["terminated"]).lower() == "true" for row in rows]))
    if not states:
        raise ValueError("split contains no states")
    return {
        "states": torch.cat(states),
        "next_states": torch.cat(next_states),
        "rewards": torch.cat(rewards),
        "terminal": torch.cat(terminal),
    }


def parse_layers(value: str, layer_count: int) -> List[int]:
    if value == "all":
        return list(range(layer_count))
    result = sorted({int(item) for item in value.split(",")})
    if not result or result[0] < 0 or result[-1] >= layer_count:
        raise ValueError(f"layers must fall within [0, {layer_count})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", default="artifacts/activation_bank")
    parser.add_argument("--output-dir", default="artifacts/value_probes")
    parser.add_argument("--layers", default="all", help="all or comma-separated zero-based layers")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=22026)
    args = parser.parse_args()

    shards = load_shards(args.activation_dir)
    episode_ids = [shard["episode_id"] for shard in shards]
    split = split_episode_ids(episode_ids, args.seed)
    layer_count = int(shards[0]["activations"].shape[1])
    layers = parse_layers(args.layers, layer_count)
    os.makedirs(args.output_dir, exist_ok=True)
    split_path = os.path.join(args.output_dir, "episode_split.json")
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump({"train": split.train, "validation": split.validation, "test": split.test}, handle, indent=2)

    summaries = []
    trained = {}
    for layer in layers:
        train = layer_dataset(shards, layer, set(split.train))
        validation = layer_dataset(shards, layer, set(split.validation))
        result = fit_value_probe(
            train,
            validation,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed + layer,
        )
        indices = select_top_fraction(result.probe, 0.01)
        validation_metrics = evaluate_full_and_pruned(result.probe, validation, indices)
        test = layer_dataset(shards, layer, set(split.test))
        test_metrics = evaluate_full_and_pruned(result.probe, test, indices)
        summary = {
            "layer": layer,
            "neuron_count": int(indices.numel()),
            "epochs_trained": result.epochs_trained,
            "best_validation_loss": result.best_validation_loss,
            "validation": validation_metrics,
            "test": test_metrics,
            "history": result.history,
        }
        summaries.append(summary)
        trained[layer] = (result.probe, indices, summary)
        save_frozen_probe(
            os.path.join(args.output_dir, f"layer_{layer:02d}.pt"),
            result.probe, layer, indices, summary,
        )
        print(f"layer {layer}: pruned validation TD MSE={validation_metrics['pruned_td_mse']:.6f}")

    best = min(summaries, key=lambda item: item["validation"]["pruned_td_mse"])
    probe, indices, _ = trained[best["layer"]]
    save_frozen_probe(
        os.path.join(args.output_dir, "frozen_best.pt"),
        probe, best["layer"], indices,
        {"selection_metric": "pruned_validation_td_mse", "layers": summaries, "config": vars(args)},
    )
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"best_layer": best["layer"], "layers": summaries}, handle, indent=2)


if __name__ == "__main__":
    main()
