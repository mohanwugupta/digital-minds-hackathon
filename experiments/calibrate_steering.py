"""Calibrate and freeze steering magnitude using probe-validation states only."""

import argparse
import glob
import json
import os

from interventions.artifacts import load_frozen_probe
from interventions.steering import calibrate_magnitude


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", default="artifacts/activation_bank")
    parser.add_argument("--probe", default="artifacts/value_probes/frozen_best.pt")
    parser.add_argument("--split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--output", default="artifacts/value_probes/steering_calibration.json")
    parser.add_argument("--candidates", default="0.01,0.025,0.05,0.1,0.2")
    parser.add_argument("--required-fraction", type=float, default=0.9)
    parser.add_argument("--max-relative-rms", type=float, default=0.25)
    args = parser.parse_args()

    probe, layer, indices, artifact = load_frozen_probe(args.probe)
    with open(args.split, encoding="utf-8") as handle:
        validation_ids = set(json.load(handle)["validation"])
    states = []
    for path in sorted(glob.glob(os.path.join(args.activation_dir, "episode_*.pt"))):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard["episode_id"] in validation_ids:
            states.append(shard["activations"][:, layer, :].float())
    if not states:
        raise ValueError("no validation states found for calibration")
    result = calibrate_magnitude(
        probe,
        torch.cat(states),
        indices,
        candidates=[float(value) for value in args.candidates.split(",")],
        required_fraction=args.required_fraction,
        max_relative_rms=args.max_relative_rms,
    )
    payload = {
        "magnitude": result.magnitude,
        "ordered_fraction": result.ordered_fraction,
        "relative_rms": result.relative_rms,
        "layer": layer,
        "neuron_indices": indices.tolist(),
        "probe_path": os.path.abspath(args.probe),
        "validation_episode_count": len(validation_ids),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
