"""Freeze validation-only magnitudes for all ridge causal directions."""

import argparse
import glob
import json
import os


DEFAULT_PROBES = {
    "persistence": "artifacts/linear_probes/frozen_best_persistence.pt",
    "generic_return": "artifacts/linear_probes/frozen_best_future_return.pt",
    "advantage": "artifacts/advantage_probes/frozen_best_advantage.pt",
}
EXPECTED_LAYERS = {"persistence": 31, "generic_return": 1, "advantage": 2}


def _layer(payload: dict, path: str) -> int:
    metadata = payload.get("metadata", {})
    layer = metadata.get("selected_layer", metadata.get("layer"))
    if layer is None:
        raise ValueError(f"ridge artifact does not identify its native layer: {path}")
    return int(layer)


def main() -> None:
    import torch

    from interventions.ridge_probe import load_ridge_probe
    from interventions.ridge_steering import calibrate_ridge_magnitude

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", default="artifacts/activation_bank")
    parser.add_argument("--split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--persistence-probe", default=DEFAULT_PROBES["persistence"])
    parser.add_argument("--generic-probe", default=DEFAULT_PROBES["generic_return"])
    parser.add_argument("--advantage-probe", default=DEFAULT_PROBES["advantage"])
    parser.add_argument("--decoded-sd-candidates", default="1,0.5,0.25,0.1")
    parser.add_argument("--max-relative-rms", type=float, default=0.25)
    parser.add_argument("--output", default="artifacts/causal_steering/calibration.json")
    args = parser.parse_args()

    with open(args.split, encoding="utf-8") as handle:
        validation_ids = set(json.load(handle)["validation"])
    probe_paths = {
        "persistence": args.persistence_probe,
        "generic_return": args.generic_probe,
        "advantage": args.advantage_probe,
    }
    loaded = {}
    needed_layers = set()
    for name, path in probe_paths.items():
        probe, payload = load_ridge_probe(path)
        layer = _layer(payload, path)
        if layer != EXPECTED_LAYERS[name]:
            raise ValueError(
                f"{name} frozen probe is layer {layer}; the preregistered "
                f"causal direction is layer {EXPECTED_LAYERS[name]}"
            )
        loaded[name] = (probe, payload, layer)
        needed_layers.add(layer)

    states_by_layer = {layer: [] for layer in needed_layers}
    validation_episode_count = 0
    for path in sorted(glob.glob(os.path.join(args.activation_dir, "episode_*.pt"))):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard["episode_id"] not in validation_ids:
            continue
        validation_episode_count += 1
        for layer in needed_layers:
            states_by_layer[layer].append(shard["activations"][:, layer, :].float())
    if validation_episode_count != len(validation_ids):
        raise ValueError(
            f"found {validation_episode_count}/{len(validation_ids)} validation episodes"
        )

    candidates = tuple(float(value) for value in args.decoded_sd_candidates.split(","))
    directions = {}
    for name, (probe, _payload, layer) in loaded.items():
        if not states_by_layer[layer]:
            raise ValueError(f"no validation activations found for layer {layer}")
        states = torch.cat(states_by_layer[layer])
        result = calibrate_ridge_magnitude(
            probe,
            states,
            decoded_sd_candidates=candidates,
            max_relative_rms=args.max_relative_rms,
        )
        directions[name] = {
            "probe_path": os.path.abspath(probe_paths[name]),
            "layer": layer,
            **result.to_dict(),
        }
    payload = {
        "selection_data": "probe validation episodes only",
        "split_path": os.path.abspath(args.split),
        "validation_episode_count": validation_episode_count,
        "decoded_sd_candidates": list(candidates),
        "max_relative_rms": args.max_relative_rms,
        "directions": directions,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
