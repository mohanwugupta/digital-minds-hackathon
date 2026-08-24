"""Run synthetic recovery, real behavioral comparison, and conditional decoding."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.persistence_future_behavior import (
    add_future_behavior_targets,
    heldout_future_behavior_validation,
)
from analysis.persistence_latent_representation import (
    residualize_commitment_targets,
    search_latent_representation,
)
from analysis.persistence_latent_state import (
    compare_behavioral_architectures,
    fit_latent_state_model,
    real_task_latent_inputs,
    simulate_latent_trajectories,
)
from analysis.persistence_transition_analysis import transition_aligned_summary
from experiments.cross_task_utils import load_activation_shards
from experiments.train_value_probe import load_shards


def _correlation(left, right) -> float:
    import torch

    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    if float(x.std(unbiased=False)) == 0 or float(y.std(unbiased=False)) == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def run_synthetic_gates(config: dict) -> dict:
    synthetic_config = config["synthetic_recovery"]
    state_config = config["state_model"]
    expected = {
        "immediate": "immediate_decision",
        "choice_inertia": "choice_history_inertia",
        "latent_commitment": "latent_commitment",
        "generic_value": "generic_latent_value",
    }
    architectures = {}
    for offset, architecture in enumerate(synthetic_config["architectures"]):
        synthetic = simulate_latent_trajectories(
            architecture=architecture,
            tasks=tuple(synthetic_config["tasks"]),
            episodes_per_task=int(synthetic_config["episodes_per_task"]),
            decisions=int(synthetic_config["decisions_per_episode"]),
            rho=0.7,
            seed=int(config["analysis_seed"]) + offset,
        )
        comparison = compare_behavioral_architectures(
            synthetic["records"],
            feature_names=synthetic["feature_names"],
            generic_value_feature="relative_value",
            rho_grid=tuple(float(value) for value in state_config["rho_grid"]),
        )
        row = {
            "expected": expected[architecture],
            "selected": comparison["selected_model"],
            "architecture_identified": comparison["selected_model"]
            == expected[architecture],
            "comparison": comparison,
        }
        if architecture == "latent_commitment":
            fit = fit_latent_state_model(
                synthetic["records"],
                feature_names=synthetic["feature_names"],
                rho_grid=tuple(float(value) for value in state_config["rho_grid"]),
            )
            correlation = _correlation(
                [record["true_w"] for record in synthetic["records"]],
                fit["latent_state"],
            )
            row.update(
                {
                    "state_correlation": correlation,
                    "rho_error": abs(float(fit["rho"]) - 0.7),
                    "state_recovery_passed": correlation
                    >= float(synthetic_config["minimum_state_correlation"])
                    and abs(float(fit["rho"]) - 0.7)
                    <= float(synthetic_config["maximum_rho_error"]),
                }
            )
        architectures[architecture] = row
    return {
        "passed": all(row["architecture_identified"] for row in architectures.values())
        and architectures["latent_commitment"].get("state_recovery_passed", False),
        "architectures": architectures,
    }


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _records(shards):
    return [dict(record) for shard in shards for record in shard["records"]]


def run_real_latent_analysis(
    *, latent_config: dict, discovery_config: dict, output_dir: str
) -> dict:
    paths = discovery_config["paths"]
    bandit_shards = load_shards("artifacts/activation_bank")
    foraging_shards = load_activation_shards(paths["foraging_activations"])
    solvability_shards = load_activation_shards(paths["solvability_activations"])
    shards_by_task = {
        "bandit": bandit_shards,
        "foraging": foraging_shards,
        "solvability": solvability_shards,
    }
    splits_by_task = {
        "bandit": _load_json("artifacts/value_probes/episode_split.json"),
        "foraging": _load_json(paths["foraging_split"]),
        "solvability": _load_json(paths["solvability_split"]),
    }
    source_records = []
    for task in ("bandit", "foraging", "solvability"):
        for record in _records(shards_by_task[task]):
            record["task"] = task
            source_records.append(record)
    records, feature_names = real_task_latent_inputs(source_records)
    fit_episode_ids = {
        str(episode)
        for split in splits_by_task.values()
        for episode in split["train"]
    }
    test_episode_ids = {
        str(episode)
        for split in splits_by_task.values()
        for episode in split["test"]
    }
    state_config = latent_config["state_model"]
    fit = fit_latent_state_model(
        records,
        feature_names=feature_names,
        rho_grid=tuple(float(value) for value in state_config["rho_grid"]),
        fit_episode_ids=fit_episode_ids,
    )
    comparison = compare_behavioral_architectures(
        records,
        feature_names=feature_names,
        generic_value_feature="relative_value",
        rho_grid=tuple(float(value) for value in state_config["rho_grid"]),
    )
    future_records = add_future_behavior_targets(
        records, k_values=tuple(latent_config["future_behavior"]["horizons"])
    )
    future = heldout_future_behavior_validation(
        records=future_records,
        current_choice=[record["persistence_logit"] for record in future_records],
        latent_state=fit["latent_state"],
        future_outcome=[record["remaining_episode_length"] for record in future_records],
        train_episode_ids=fit_episode_ids,
        test_episode_ids=test_episode_ids,
        minimum_incremental_r_squared=float(
            latent_config["future_behavior"]["minimum_incremental_r_squared"]
        ),
    )
    behavioral_gate = (
        comparison["selected_model"] == "latent_commitment" and future["passed"]
    )
    transition = (
        transition_aligned_summary(future_records, fit["latent_state"])
        if behavioral_gate
        else {"status": "skipped_behavioral_gate_failed"}
    )
    internal = {"status": "skipped_behavioral_gate_failed"}
    probe_artifacts = None
    if behavioral_gate:
        residual = residualize_commitment_targets(
            future_records, fit["latent_state"], splits_by_task
        )
        internal, probe_artifacts = search_latent_representation(
            shards_by_task=shards_by_task,
            splits_by_task=splits_by_task,
            residual_target_by_state=residual,
            bootstrap_samples=2000,
            seed=int(latent_config["analysis_seed"]),
        )
    from experiments.runtime import run_metadata

    result = {
        "analysis_role": "exploratory_discovery",
        "model_comparison": comparison,
        "latent_fit": fit,
        "future_behavior_beyond_current_choice": future,
        "behavioral_gate_passed": behavioral_gate,
        "transition_analysis": transition,
        "internal_representation": internal,
        "candidate_taxonomy": (
            "latent_policy_state_representation"
            if behavioral_gate
            and internal.get("all_loto_clustered_intervals_positive")
            else "immediate_decision_or_input_value_representation"
        ),
        "provenance": run_metadata(
            {
                "analysis": "latent_policy_commitment",
                "protocol_version": latent_config["protocol_version"],
                "model": discovery_config["model"],
            }
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "latent_state_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "inferred_latent_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task",
                "episode_id",
                "state_id",
                "round",
                "persistence_logit",
                "latent_state",
                "remaining_episode_length",
            ),
        )
        writer.writeheader()
        for record, latent in zip(future_records, fit["latent_state"]):
            writer.writerow(
                {
                    "task": record["task"],
                    "episode_id": record["episode_id"],
                    "state_id": record["state_id"],
                    "round": record["round"],
                    "persistence_logit": record["persistence_logit"],
                    "latent_state": latent,
                    "remaining_episode_length": record["remaining_episode_length"],
                }
            )
    if probe_artifacts is not None:
        from experiments.runtime import atomic_torch_save

        atomic_torch_save(
            probe_artifacts, str(output / "latent_representation_probes.pt")
        )
    report = [
        "# Latent policy-commitment search",
        "",
        f"Selected behavioral architecture: **{comparison['selected_model']}**.",
        f"Future-behavior gate passed: **{future['passed']}**.",
        f"Track C1 behavioral gate passed: **{behavioral_gate}**.",
        f"Candidate taxonomy: **{result['candidate_taxonomy']}**.",
        "",
        "Internal all-layer decoding runs only when both behavioral gates pass.",
    ]
    (output / "latent_state_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_latent_state.yaml")
    parser.add_argument(
        "--discovery-config", default="config/persistence_discovery.yaml"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/persistence_discovery/latent_state"
    )
    parser.add_argument(
        "--synthetic-only", action="store_true", help="run the mandatory recovery gate only"
    )
    args = parser.parse_args()
    import yaml

    with open(args.config, encoding="utf-8") as handle:
        latent_config = yaml.safe_load(handle)
    with open(args.discovery_config, encoding="utf-8") as handle:
        discovery_config = yaml.safe_load(handle)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    synthetic = run_synthetic_gates(latent_config)
    (output / "synthetic_recovery.json").write_text(
        json.dumps(synthetic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not synthetic["passed"]:
        raise RuntimeError("synthetic latent-state recovery/confusion gate failed")
    if not args.synthetic_only:
        run_real_latent_analysis(
            latent_config=latent_config,
            discovery_config=discovery_config,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
