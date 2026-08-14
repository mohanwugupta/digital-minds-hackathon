"""Run matched-seed complete episodes under a fixed steering condition."""

import argparse
import json
import os

from experiments.run_bandit_baseline import (
    append_records_csv, completed_episode_ids, episode_conditions, run_episode,
)


class SteeredDecisionModel:
    def __init__(self, model, probe, layer, neuron_indices, magnitude, alpha):
        self.model = model
        self.model_id = model.model_id
        self.probe = probe
        self.layer = layer
        self.neuron_indices = neuron_indices
        self.magnitude = magnitude
        self.alpha = float(alpha)

    def decision(self, messages):
        import torch
        from interventions.steering import build_value_direction, steer_hidden

        baseline = self.model.decision(messages)
        hidden = baseline["hidden_states"][self.layer].unsqueeze(0)
        direction = build_value_direction(
            self.probe, hidden, self.neuron_indices, magnitude=self.magnitude
        )
        with torch.no_grad():
            pre = float(self.probe(hidden).item())
            post = float(self.probe(steer_hidden(hidden, direction, self.alpha)).item())
        if self.alpha == 0.0:
            metrics = baseline
        else:
            def transform(current):
                return steer_hidden(
                    current, direction.to(current.device, current.dtype), self.alpha
                )
            metrics = self.model.decision(messages, layer=self.layer, transform=transform)
        return {
            **{key: value for key, value in metrics.items() if key != "hidden_states"},
            "layer": self.layer,
            "neuron_set": "value",
            "intervention_type": "value",
            "alpha": self.alpha,
            "probe_value_pre": pre,
            "probe_value_post": post,
        }


def matched_result_passed(path: str) -> bool:
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    return bool(result.get("primary_ordering_passed") and result.get("random_control_passed"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--probe", default="artifacts/value_probes/frozen_best.pt")
    parser.add_argument("--calibration", default="artifacts/value_probes/steering_calibration.json")
    parser.add_argument("--matched-analysis", required=True)
    parser.add_argument("--episodes-per-condition", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42026)
    parser.add_argument("--output-dir", default="artifacts/sequential")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--force", action="store_true", help="override the matched-state success gate")
    args = parser.parse_args()

    if not args.force and not matched_result_passed(args.matched_analysis):
        raise RuntimeError("matched-state value ordering and random-control gates must pass first")

    from experiments.runtime import save_run_metadata
    from interventions.artifacts import load_frozen_probe
    from models.hooked_qwen import HookedQwen

    probe, layer, indices, _ = load_frozen_probe(args.probe)
    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_metadata(os.path.join(args.output_dir, "metadata.json"), vars(args), model)
    for alpha in (-1.0, 0.0, 1.0):
        steered = SteeredDecisionModel(
            model, probe, layer, indices, float(calibration["magnitude"]), alpha
        )
        output = os.path.join(args.output_dir, f"alpha_{alpha:+.0f}.csv")
        done = completed_episode_ids(output)
        conditions = episode_conditions(args.episodes_per_condition, args.seed)
        for index, (p_a, p_b, seed, action_seed) in enumerate(conditions):
            # The exact bandit and action-sampling seeds are shared across alpha.
            base_id = f"seed-{seed}-pa-{p_a:.2f}-pb-{p_b:.2f}"
            if base_id in done:
                continue
            records = run_episode(
                steered, p_a, p_b, seed=seed, action_seed=action_seed
            )
            append_records_csv(output, records)
            print(f"alpha={alpha:+.0f} episode {index + 1}/{args.episodes_per_condition}", flush=True)


if __name__ == "__main__":
    main()
