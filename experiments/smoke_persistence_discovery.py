"""Small synthetic end-to-end smoke for Track C contrast orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.persistence_cross_generalization import (
    run_contrast_search,
    save_search_outputs,
)


def synthetic_contrast_bank(seed: int = 0):
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    families = {
        "bandit": ("continue_incentive", "stop_outside_option"),
        "foraging": ("search_cost", "outside_option"),
        "solvability": ("progress_evidence",),
    }
    contrasts = []
    common = torch.tensor([1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for task, manipulations in families.items():
        for manipulation in manipulations:
            for split, count in (("train", 8), ("validation", 5), ("test", 5)):
                for index in range(count):
                    layers = []
                    for layer in range(4):
                        layers.append(
                            (0.4 + 0.2 * layer) * common
                            + 0.03 * torch.randn(8, generator=generator)
                        )
                    contrasts.append(
                        {
                            "task": task,
                            "manipulation": manipulation,
                            "contrast_kind": "persistence",
                            "split": split,
                            "contrast_id": f"{task}-{manipulation}-{split}-{index}",
                            "cluster_id": f"{task}-{split}-{index}",
                            "activation_delta": torch.stack(layers),
                            "behavior_effect": 1.0,
                        }
                    )
    for nuisance_index, nuisance in enumerate(
        ("label", "arbitrary_choice", "terminality", "generic_value")
    ):
        for index in range(8):
            nuisance_direction = torch.zeros(8)
            nuisance_direction[4 + nuisance_index] = 1.0
            contrasts.append(
                {
                    "task": f"{nuisance}_control",
                    "manipulation": nuisance,
                    "contrast_kind": "nuisance",
                    "nuisance_type": nuisance,
                    "split": "test",
                    "contrast_id": f"{nuisance}-{index}",
                    "cluster_id": f"{nuisance}-{index}",
                    "activation_delta": torch.stack(
                        [nuisance_direction * (layer + 1) for layer in range(4)]
                    ),
                    "behavior_effect": 0.0,
                }
            )
    return contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="artifacts/persistence_discovery_smoke"
    )
    args = parser.parse_args()
    config = {
        "protocol_version": "task_general_persistence_discovery_smoke_v1",
        "model": "synthetic",
        "analysis_seed": 123,
        "search": {
            "ranks": [1, 2, 4],
            "feature_types": ["static", "displacement"],
            "matched_random_subspaces": 5,
            "bootstrap_samples": 40,
            "minimum_transfer": 0.25,
            "maximum_nuisance_fraction": 0.5,
        },
    }
    summary, artifacts = run_contrast_search(synthetic_contrast_bank(), config)
    save_search_outputs(summary, artifacts, args.output_dir)
    expected = (
        "persistence_discovery_summary.json",
        "persistence_candidate_subspaces.pt",
        "layerwise_transfer_map.csv",
        "layerwise_cross_task_transfer.svg",
        "persistence_discovery_report.md",
    )
    missing = [name for name in expected if not (Path(args.output_dir) / name).exists()]
    if missing:
        raise RuntimeError(f"persistence discovery smoke missed outputs: {missing}")
    print(json.dumps({"passed": True, "classification": summary["classification"]}))


if __name__ == "__main__":
    main()
