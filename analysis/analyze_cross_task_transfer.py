"""Evaluate strict bandit-to-foraging transfer, controls, and within-task ceiling."""

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes
from analysis.cross_task_integrity import (
    audit_cross_task_shards,
    require_behavioral_clearance,
)
from experiments.cross_task_utils import (
    layer_dataset,
    load_activation_shards,
    make_or_validate_split,
    probe_layer,
)
from experiments.runtime import run_metadata
from interventions.ridge_probe import load_ridge_probe, regression_metrics
from interventions.ridge_steering import matched_sign_random_directions


def affine_fit(prediction, target) -> dict[str, float]:
    prediction = [float(value) for value in prediction]
    target = [float(value) for value in target]
    mean_x, mean_y = statistics.mean(prediction), statistics.mean(target)
    denominator = sum((value - mean_x) ** 2 for value in prediction)
    slope = (
        sum(
            (value - mean_x) * (outcome - mean_y)
            for value, outcome in zip(prediction, target)
        )
        / denominator
        if denominator
        else 0.0
    )
    return {"intercept": mean_y - slope * mean_x, "slope": slope}


def apply_affine(prediction, calibration: dict[str, float]):
    import torch

    values = torch.as_tensor(prediction, dtype=torch.float32)
    return calibration["intercept"] + calibration["slope"] * values


def direction_score(states, probe, direction=None):
    """Score a raw activation direction with the frozen bandit origin/scale."""
    raw = (
        probe.raw_activation_direction().float()
        if direction is None
        else direction.detach().float().cpu()
    )
    return probe.target_mean + (states.float() - probe.state_mean) @ raw


def _bootstrap_metrics(
    prediction,
    target,
    episode_ids,
    *,
    samples: int,
    seed: int,
) -> dict:
    grouped = {}
    for index, episode_id in enumerate(episode_ids):
        grouped.setdefault(episode_id, []).append(index)
    episodes = sorted(grouped)
    rng = random.Random(seed)
    draws = {"r_squared": [], "correlation": []}
    for _ in range(samples):
        indices = [
            index
            for _episode in episodes
            for index in grouped[rng.choice(episodes)]
        ]
        metrics = regression_metrics(prediction[indices], target[indices])
        for key in draws:
            draws[key].append(metrics[key])
    result = {}
    for key, values in draws.items():
        ordered = sorted(values)
        result[key] = {
            "lower_95": ordered[int(0.025 * samples)],
            "upper_95": ordered[max(0, int(0.975 * samples) - 1)],
            "bootstrap_mean": statistics.mean(values),
        }
    return result


def _mapping_metrics(
    prediction,
    target,
    records: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    import torch

    result = {}
    mapping_ids = sorted({record["mapping_id"] for record in records})
    for mapping_id in mapping_ids:
        indices = [
            index
            for index, record in enumerate(records)
            if record["mapping_id"] == mapping_id
        ]
        result[mapping_id] = {
            "states": len(indices),
            **regression_metrics(
                prediction[torch.tensor(indices)], target[torch.tensor(indices)]
            ),
            "episode_bootstrap": _bootstrap_metrics(
                prediction[torch.tensor(indices)],
                target[torch.tensor(indices)],
                [records[index]["episode_id"] for index in indices],
                samples=bootstrap_samples,
                seed=seed + len(result),
            ),
        }
    return result


def behavioral_summary(shards: list[dict]) -> dict:
    records = [record for shard in shards for record in shard["records"]]
    probabilities = [float(record["p_stay"]) for record in records]
    logits = [float(record["persistence_logit"]) for record in records]
    episode_summaries = []
    for shard in shards:
        selected = shard["records"]
        episode_summaries.append(
            {
                "episode_id": shard["episode_id"],
                "decisions": len(selected),
                "left": selected[-1]["termination_reason"] == "leave",
                "mapping_id": shard["mapping_id"],
            }
        )
    initial_by_mapping = {}
    for record in records:
        if int(record["round"]) == 0:
            initial_by_mapping.setdefault(record["mapping_id"], []).append(
                float(record["p_stay"])
            )
    cells = {}
    for record in records:
        key = (
            float(record["initial_quality"]),
            float(record["depletion"]),
            int(record["outside_option"]),
            int(record["stay_cost"]),
        )
        cells.setdefault(key, []).append(float(record["persistence_logit"]))
    ordered_probabilities = sorted(probabilities)
    return {
        "level": "behavioral_generalization",
        "analysis_role": "exploratory_post_test_description",
        "states": len(records),
        "episodes": len(shards),
        "semantic_stay_choice_rate": statistics.mean(
            float(record["semantic_choice"] == "STAY") for record in records
        ),
        "episodes_ending_by_leave_rate": statistics.mean(
            float(row["left"]) for row in episode_summaries
        ),
        "mean_episode_decisions": statistics.mean(
            row["decisions"] for row in episode_summaries
        ),
        "p_stay": {
            "mean": statistics.mean(probabilities),
            "standard_deviation": statistics.pstdev(probabilities),
            "minimum": ordered_probabilities[0],
            "maximum": ordered_probabilities[-1],
            "p10": ordered_probabilities[int(0.10 * (len(probabilities) - 1))],
            "p90": ordered_probabilities[int(0.90 * (len(probabilities) - 1))],
        },
        "persistence_logit_standard_deviation": statistics.pstdev(logits),
        "initial_state_p_stay_by_mapping": {
            mapping: statistics.mean(values)
            for mapping, values in sorted(initial_by_mapping.items())
        },
        "condition_cell_means": [
            {
                "initial_quality": key[0],
                "depletion": key[1],
                "outside_option": key[2],
                "stay_cost": key[3],
                "states": len(values),
                "mean_persistence_logit": statistics.mean(values),
            }
            for key, values in sorted(cells.items())
        ],
    }


def evaluate_transfer(
    *,
    foraging_shards: list[dict],
    foraging_split: dict,
    control_shards: list[dict],
    control_split: dict,
    bandit_probe,
    bandit_layer: int,
    foraging_probe,
    foraging_layer: int,
    random_directions: int,
    random_seed: int,
    bootstrap_samples: int,
    thresholds: dict,
) -> dict:
    import torch

    foraging_validation = layer_dataset(
        foraging_shards,
        bandit_layer,
        set(foraging_split["validation"]),
        target_key="persistence_logit",
    )
    foraging_test = layer_dataset(
        foraging_shards,
        bandit_layer,
        set(foraging_split["test"]),
        target_key="persistence_logit",
    )
    validation_projection = bandit_probe.predict(foraging_validation["states"])
    strict_prediction = bandit_probe.predict(foraging_test["states"])
    strict = regression_metrics(strict_prediction, foraging_test["target"])
    calibration = affine_fit(validation_projection, foraging_validation["target"])
    calibrated_prediction = apply_affine(strict_prediction, calibration)
    calibrated = regression_metrics(calibrated_prediction, foraging_test["target"])

    ceiling_test = layer_dataset(
        foraging_shards,
        foraging_layer,
        set(foraging_split["test"]),
        target_key="persistence_logit",
    )
    if [record["state_id"] for record in ceiling_test["records"]] != [
        record["state_id"] for record in foraging_test["records"]
    ]:
        raise ValueError("foraging test-state order differs across probe layers")
    ceiling_prediction = foraging_probe.predict(ceiling_test["states"])
    ceiling = regression_metrics(ceiling_prediction, ceiling_test["target"])

    raw_direction = bandit_probe.raw_activation_direction().float()
    controls = matched_sign_random_directions(
        raw_direction,
        n_directions=random_directions,
        seed=random_seed,
    )
    random_metrics = [
        regression_metrics(
            direction_score(foraging_test["states"], bandit_probe, direction),
            foraging_test["target"],
        )
        for direction in controls
    ]
    correlation_95 = sorted(row["correlation"] for row in random_metrics)[
        max(0, math.ceil(0.95 * len(random_metrics)) - 1)
    ]

    control_test = layer_dataset(
        control_shards,
        bandit_layer,
        set(control_split["test"]),
        target_key="choice_logit",
    )
    control_prediction = bandit_probe.predict(control_test["states"])
    negative_control = regression_metrics(control_prediction, control_test["target"])

    mapping_metrics = _mapping_metrics(
        strict_prediction,
        foraging_test["target"],
        foraging_test["records"],
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 100,
    )
    control_mapping_metrics = _mapping_metrics(
        control_prediction,
        control_test["target"],
        control_test["records"],
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 200,
    )
    transfer_ratio = (
        strict["r_squared"] / ceiling["r_squared"]
        if strict["r_squared"] > 0 and ceiling["r_squared"] > 0
        else None
    )
    negative_absolute_limit = float(
        thresholds["negative_control_max_absolute_correlation"]
    )
    negative_relative_limit = float(
        thresholds["negative_control_relative_correlation_fraction"]
    )
    criteria = {
        "expected_direction": strict["correlation"] > 0,
        "label_reversal_consistency": all(
            row["correlation"] > 0
            and row["episode_bootstrap"]["correlation"]["lower_95"] > 0
            for row in mapping_metrics.values()
        ),
        "exceeds_random_95th_percentile": strict["correlation"] > correlation_95,
        "negative_control_absent_or_weaker": (
            abs(negative_control["correlation"]) <= negative_absolute_limit
            or abs(negative_control["correlation"])
            <= negative_relative_limit * max(strict["correlation"], 0.0)
        ),
        "at_least_half_ceiling": (
            transfer_ratio is not None
            and transfer_ratio >= float(thresholds["strong_transfer_ceiling_fraction"])
        ),
    }
    required = thresholds.get(
        "required_checks",
        (
            "expected_direction",
            "label_reversal_consistency",
            "exceeds_random_95th_percentile",
            "negative_control_absent_or_weaker",
        ),
    )
    unknown_checks = set(required) - set(criteria)
    if unknown_checks:
        raise ValueError(
            f"representational preregistration names unknown checks: {sorted(unknown_checks)}"
        )
    first_four = all(criteria[name] for name in required)
    classification = (
        "strong_transfer"
        if first_four and criteria["at_least_half_ceiling"]
        else "partial_transfer"
        if first_four
        else "no_convincing_transfer"
    )
    decision_matrix_outcome = (
        "representational_transfer_pending_causal_checkpoint"
        if classification in {"strong_transfer", "partial_transfer"}
        else "outcome_d_generic_binary_or_output_geometry"
        if not criteria["negative_control_absent_or_weaker"]
        else "outcome_c_task_specific_persistence_representations"
        if ceiling["r_squared"] > 0
        else "no_interpretable_cross_task_result"
    )
    return {
        "classification": classification,
        "decision_matrix_outcome": decision_matrix_outcome,
        "analysis_roles": {
            "strict_zero_shot": "confirmatory_primary",
            "mapping_consistency": "confirmatory",
            "matched_random_directions": "confirmatory",
            "negative_control": "confirmatory",
            "foraging_specific_ceiling": "confirmatory_comparison",
            "calibration_only": "exploratory_secondary",
        },
        "criteria": criteria,
        "strict_zero_shot": {
            **strict,
            "episode_bootstrap": _bootstrap_metrics(
                strict_prediction,
                foraging_test["target"],
                [record["episode_id"] for record in foraging_test["records"]],
                samples=bootstrap_samples,
                seed=random_seed + 1,
            ),
            "mapping_metrics": mapping_metrics,
        },
        "calibration_only": {"fit_on": "foraging_validation", **calibration, **calibrated},
        "foraging_specific_ceiling": {
            **ceiling,
            "selected_layer": foraging_layer,
            "episode_bootstrap": _bootstrap_metrics(
                ceiling_prediction,
                ceiling_test["target"],
                [record["episode_id"] for record in ceiling_test["records"]],
                samples=bootstrap_samples,
                seed=random_seed + 2,
            ),
        },
        "transfer_ratio": transfer_ratio,
        "matched_random_directions": {
            "count": len(random_metrics),
            "correlation_95th_percentile": correlation_95,
            "metrics": random_metrics,
        },
        "negative_control": {
            **negative_control,
            "mapping_metrics": control_mapping_metrics,
        },
        "test_counts": {
            "foraging_states": len(foraging_test["records"]),
            "foraging_episodes": len(set(foraging_split["test"])),
            "control_states": len(control_test["records"]),
            "control_episodes": len(set(control_split["test"])),
        },
    }


def load_bandit_within_task(path: str, selected_layer: int) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = next(
        (
            row["targets"]["persistence"]["test"]
            for row in payload["layers"]
            if int(row["layer"]) == int(selected_layer)
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            f"bandit metrics do not contain layer-{selected_layer} persistence test"
        )
    return {
        "analysis_role": "previously_frozen_reference",
        "selected_layer": int(selected_layer),
        **selected,
    }


def make_summary_figure(result: dict, path: Path) -> None:
    values = [
        ("Bandit", result["bandit_within_task"]["r_squared"], COLORS["reference"]),
        ("Zero-shot", result["strict_zero_shot"]["r_squared"], COLORS["observed"]),
        ("Ceiling", result["foraging_specific_ceiling"]["r_squared"], COLORS["both_positive"]),
        ("Control", result["negative_control"]["r_squared"], COLORS["both_negative"]),
    ]
    lower = min(-0.05, min(value for _name, value, _color in values) - 0.05)
    upper = max(1.0, max(value for _name, value, _color in values) + 0.05)
    svg = Svg(920, 620)
    svg.text(50, 42, "Track B representational checkpoint", "title")
    svg.text(50, 68, "Held-out R²; frozen bandit direction is strict zero-shot", "subtitle")
    sx, sy, box = axes(
        svg,
        70,
        105,
        780,
        410,
        "",
        "Evaluation",
        "R²",
        [(index, name) for index, (name, _value, _color) in enumerate(values)],
        [(lower, f"{lower:.2f}"), (0.0, "0"), (0.5, "0.5"), (1.0, "1.0")],
        (-0.6, 3.6),
        (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    bar_half_width = abs(sx(0.30) - sx(0.0))
    for index, (_name, value, color) in enumerate(values):
        x0, x1 = sx(index) - bar_half_width, sx(index) + bar_half_width
        y0, y1 = sy(0), sy(value)
        svg.rect(x0, min(y0, y1), x1 - x0, abs(y1 - y0), color, 0.9)
        svg.text(sx(index), min(y0, y1) - 9, f"{value:.3f}", "axis", "middle")
    svg.text(
        870,
        585,
        f"Classification: {result['classification'].replace('_', ' ')}",
        "note",
        "end",
    )
    svg.save(path)


def write_report(result: dict, path: Path) -> None:
    strict = result["strict_zero_shot"]
    ceiling = result["foraging_specific_ceiling"]
    control = result["negative_control"]
    ratio = result["transfer_ratio"]
    lines = [
        "# Bandit-to-foraging representational transfer",
        "",
        "## Level 1 — behavioral generalization",
        "",
        f"Foraging produced **{result['behavioral_generalization']['states']} states** across **{result['behavioral_generalization']['episodes']} episodes**; the semantic STAY choice rate was **{result['behavioral_generalization']['semantic_stay_choice_rate']:.3f}** and persistence-logit SD was **{result['behavioral_generalization']['persistence_logit_standard_deviation']:.3f}**.",
        "",
        "## Level 2 — representational generalization",
        "",
        f"Classification: **{result['classification'].replace('_', ' ')}**.",
        f"Decision-matrix status: **{result['decision_matrix_outcome'].replace('_', ' ')}**.",
        "",
        f"Strict zero-shot: R² **{strict['r_squared']:.3f}**, correlation **{strict['correlation']:.3f}**.",
        f"Bandit within-task reference: R² **{result['bandit_within_task']['r_squared']:.3f}** at layer {result['bandit_within_task']['selected_layer']}.",
        f"Foraging-specific ceiling: R² **{ceiling['r_squared']:.3f}** at layer {ceiling['selected_layer']}.",
        f"Transfer ratio: **{ratio:.3f}**." if ratio is not None else "Transfer ratio: undefined because one R² is non-positive.",
        f"Non-persistence control correlation: **{control['correlation']:.3f}**.",
        f"Strict zero-shot episode-bootstrap R² interval: **{strict['episode_bootstrap']['r_squared']['lower_95']:.3f} to {strict['episode_bootstrap']['r_squared']['upper_95']:.3f}**.",
        "",
        "## Preregistered checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in result["criteria"].items()
    )
    lines.extend(
        [
            "",
            "The strict result applies the original bandit standardization and frozen weights with no fitted foraging parameter. The affine fit is diagnostic only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foraging-bank", default="artifacts/cross_task/foraging_activation_bank"
    )
    parser.add_argument(
        "--control-bank", default="artifacts/cross_task/control_activation_bank"
    )
    parser.add_argument(
        "--foraging-split", default="artifacts/cross_task/foraging_episode_split.json"
    )
    parser.add_argument(
        "--control-split", default="artifacts/cross_task/control_episode_split.json"
    )
    parser.add_argument(
        "--bandit-probe", default="artifacts/linear_probes/frozen_best_persistence.pt"
    )
    parser.add_argument(
        "--foraging-probe",
        default="artifacts/cross_task/foraging_probes/frozen_best_persistence.pt",
    )
    parser.add_argument(
        "--behavioral-gate",
        default="artifacts/cross_task/behavioral/behavioral_validation_summary.json",
    )
    parser.add_argument(
        "--bandit-metrics", default="artifacts/linear_probes/metrics.json"
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output-dir", default="artifacts/cross_task/transfer")
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    behavioral_gate = require_behavioral_clearance(args.behavioral_gate)
    prereg = config["representational_transfer"]
    foraging_shards = load_activation_shards(args.foraging_bank)
    control_shards = load_activation_shards(args.control_bank)
    foraging_split = make_or_validate_split(
        foraging_shards, args.foraging_split, seed=int(config["split_seed"])
    )
    control_split = make_or_validate_split(
        control_shards, args.control_split, seed=int(config["split_seed"])
    )
    bandit_probe, bandit_payload = load_ridge_probe(args.bandit_probe)
    foraging_probe, foraging_payload = load_ridge_probe(args.foraging_probe)
    bandit_layer = probe_layer(bandit_payload, args.bandit_probe)
    foraging_layer = probe_layer(foraging_payload, args.foraging_probe)
    result = evaluate_transfer(
        foraging_shards=foraging_shards,
        foraging_split=foraging_split,
        control_shards=control_shards,
        control_split=control_split,
        bandit_probe=bandit_probe,
        bandit_layer=bandit_layer,
        foraging_probe=foraging_probe,
        foraging_layer=foraging_layer,
        random_directions=int(prereg["matched_random_directions"]),
        random_seed=int(config["analysis_seed"]),
        bootstrap_samples=int(prereg["episode_bootstrap_samples"]),
        thresholds=prereg,
    )
    result.update(
        {
            "bandit_probe": os.path.abspath(args.bandit_probe),
            "bandit_layer": bandit_layer,
            "strict_zero_shot_parameters_fit_on_foraging": 0,
            "preregistered_config": os.path.abspath(args.config),
            "provenance": run_metadata(
                {
                    "model": foraging_shards[0].get("model_id", config["model"]),
                    "analysis": "bandit_to_foraging_representational_transfer",
                    "config": os.path.abspath(args.config),
                }
            ),
            "development_behavioral_gate": behavioral_gate,
            "bandit_within_task": load_bandit_within_task(
                args.bandit_metrics, bandit_layer
            ),
            "behavioral_generalization": behavioral_summary(foraging_shards),
            "foraging_audit": audit_cross_task_shards(
                foraging_shards,
                "foraging",
                response_labels=tuple(
                    config["collection"]["foraging_response_labels"]
                ),
                expected_episodes=int(config["collection"]["foraging_episodes"]),
            ),
            "control_audit": audit_cross_task_shards(
                control_shards,
                "control",
                response_labels=tuple(
                    config["collection"]["control_response_labels"]
                ),
                expected_episodes=int(config["collection"]["control_episodes"]),
            ),
        }
    )
    if not result["foraging_audit"]["passed"] or not result["control_audit"]["passed"]:
        raise ValueError("cross-task counterbalancing audit failed")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "representational_transfer_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(result, output / "representational_transfer_report.md")
    make_summary_figure(result, output / "representational_transfer_summary.svg")
    print(json.dumps({"classification": result["classification"], "criteria": result["criteria"]}, indent=2))


if __name__ == "__main__":
    main()
