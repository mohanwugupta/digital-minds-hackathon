"""Evaluate frozen shared persistence directions on never-used held-out tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes
from analysis.cross_task_integrity import require_behavioral_clearance
from analysis.matched_label_integrity import audit_matched_label_shards
from analysis.shared_persistence_integrity import (
    validate_discovery_plan,
    validate_loto_folds,
)
from experiments.cross_task_utils import (
    layer_dataset as cross_task_layer_dataset,
    load_activation_shards,
    make_or_validate_split,
    probe_layer,
)
from experiments.runtime import run_metadata
from experiments.shared_persistence_utils import (
    load_task_shards,
    load_task_split,
    semantic_layer_dataset,
    validate_compatible_tasks,
)
from interventions.ridge_probe import load_ridge_probe, regression_metrics
from interventions.ridge_steering import matched_sign_random_directions


def _association_metrics(prediction, target) -> dict:
    ordinary = regression_metrics(prediction, target)
    return {
        "correlation": ordinary["correlation"],
        "association_r_squared": ordinary["correlation"] ** 2,
        "no_fit_mse": ordinary["mse"],
        "no_fit_r_squared": ordinary["r_squared"],
        "prediction_mean": ordinary["prediction_mean"],
        "prediction_std": ordinary["prediction_std"],
        "target_mean": ordinary["target_mean"],
        "target_std": ordinary["target_std"],
    }


def _cluster_bootstrap(
    prediction, target, episode_ids: list[str], *, samples: int, seed: int
) -> dict:
    import torch

    if samples < 20:
        raise ValueError("episode bootstrap requires at least 20 samples")
    groups: dict[str, list[int]] = {}
    for index, episode_id in enumerate(episode_ids):
        groups.setdefault(str(episode_id), []).append(index)
    episodes = sorted(groups)
    rng = random.Random(seed)
    draws = {"correlation": [], "association_r_squared": []}
    for _ in range(samples):
        indices = [
            index
            for _draw in episodes
            for index in groups[rng.choice(episodes)]
        ]
        selected = torch.tensor(indices, dtype=torch.long)
        metrics = _association_metrics(prediction[selected], target[selected])
        for key in draws:
            draws[key].append(metrics[key])
    result = {}
    for key, values in draws.items():
        ordered = sorted(values)
        result[key] = {
            "lower_95": ordered[int(0.025 * samples)],
            "upper_95": ordered[min(samples - 1, int(0.975 * samples))],
            "bootstrap_mean": statistics.mean(values),
        }
    result["cluster_unit"] = "counterbalanced_pair_when_available_else_episode"
    return result


def _mapping_associations(
    prediction, target, records: list[dict], *, samples: int, seed: int
) -> dict:
    import torch

    if not records or any("mapping_id" not in record for record in records):
        return {}
    result = {}
    for offset, mapping_id in enumerate(
        sorted({str(record["mapping_id"]) for record in records})
    ):
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["mapping_id"]) == mapping_id
        ]
        selected = torch.tensor(indices, dtype=torch.long)
        result[mapping_id] = {
            "states": len(indices),
            **_association_metrics(prediction[selected], target[selected]),
            "cluster_bootstrap": _cluster_bootstrap(
                prediction[selected],
                target[selected],
                [
                    records[index].get("pair_id", records[index]["episode_id"])
                    for index in indices
                ],
                samples=samples,
                seed=seed + offset,
            ),
        }
    return result


def _affine_fit(prediction, target) -> dict[str, float]:
    x = [float(value) for value in prediction]
    y = [float(value) for value in target]
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    slope = (
        sum((value - mean_x) * (outcome - mean_y) for value, outcome in zip(x, y))
        / denominator
        if denominator
        else 0.0
    )
    return {"intercept": mean_y - slope * mean_x, "slope": slope}


def _apply_affine(prediction, fit: dict[str, float]):
    import torch

    return fit["intercept"] + fit["slope"] * torch.as_tensor(
        prediction, dtype=torch.float32
    )


def _direction_score(states, probe, direction=None):
    raw = probe.raw_activation_direction().float() if direction is None else direction.float()
    return probe.target_mean + (states.float() - probe.state_mean) @ raw.cpu()


def _cosine(left, right) -> float:
    import torch

    left, right = left.float(), right.float()
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


def _matched_label_evaluation(
    shards: list[dict],
    task: str,
    layer: int,
    probe,
    *,
    samples: int,
    seed: int,
    max_projection_difference_sd: float,
    max_target_difference_sd: float,
    expected_source_state_ids: set[str],
) -> dict:
    audit = audit_matched_label_shards(shards, task)
    observed_source_state_ids = {
        str(record["source_state_id"])
        for shard in shards
        for record in shard["records"]
    }
    audit["expected_source_states"] = len(expected_source_state_ids)
    audit["observed_source_states"] = len(observed_source_state_ids)
    audit["source_state_coverage_passed"] = (
        observed_source_state_ids == expected_source_state_ids
    )
    audit["passed"] = audit["passed"] and audit["source_state_coverage_passed"]
    if not audit["passed"]:
        raise ValueError(f"matched {task} label replay audit failed: {audit}")
    data = cross_task_layer_dataset(
        shards,
        layer,
        {str(shard["episode_id"]) for shard in shards},
        target_key="persistence_logit",
    )
    prediction = probe.predict(data["states"])
    mapping = _mapping_associations(
        prediction,
        data["target"],
        data["records"],
        samples=samples,
        seed=seed,
    )
    grouped = {}
    for index, record in enumerate(data["records"]):
        grouped.setdefault(str(record["pair_id"]), []).append(index)
    projection_differences, target_differences = [], []
    for pair_id, indices in grouped.items():
        if len(indices) != 2:
            raise ValueError(f"matched label history {pair_id} lacks two variants")
        left, right = indices
        projection_differences.append(abs(float(prediction[left] - prediction[right])))
        target_differences.append(abs(float(data["target"][left] - data["target"][right])))
    prediction_sd = max(float(prediction.std(unbiased=False)), 1e-6)
    target_sd = max(float(data["target"].std(unbiased=False)), 1e-6)
    projection_gap = statistics.mean(projection_differences)
    target_gap = statistics.mean(target_differences)
    criteria = {
        "both_mappings_predict_positive": all(
            row["correlation"] > 0
            and row["cluster_bootstrap"]["correlation"]["lower_95"] > 0
            for row in mapping.values()
        ),
        "projection_invariant_to_mapping": projection_gap / prediction_sd
        <= float(max_projection_difference_sd),
        "semantic_target_invariant_to_mapping": target_gap / target_sd
        <= float(max_target_difference_sd),
    }
    return {
        "passed": all(criteria.values()),
        "audit": audit,
        "criteria": criteria,
        "pooled": _association_metrics(prediction, data["target"]),
        "mapping_associations": mapping,
        "mean_absolute_projection_mapping_difference": projection_gap,
        "projection_difference_in_pooled_sd": projection_gap / prediction_sd,
        "mean_absolute_target_mapping_difference": target_gap,
        "target_difference_in_pooled_sd": target_gap / target_sd,
    }


def _canonical_source_state_ids(
    shards: list[dict], test_episode_ids: set[str]
) -> set[str]:
    by_pair: dict[str, list[dict]] = {}
    for shard in shards:
        if str(shard["episode_id"]) in test_episode_ids:
            by_pair.setdefault(str(shard["pair_id"]), []).append(shard)
    selected = []
    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            raise ValueError(f"held-out pair {pair_id} lacks both label mappings")
        selected.append(min(pair, key=lambda shard: str(shard["mapping_id"])))
    return {
        str(record["state_id"])
        for shard in selected
        for record in shard["records"]
    }


def _load_ceiling(task: str, directories: dict[str, str]):
    path = os.path.join(directories[task], "frozen_best_persistence.pt")
    probe, payload = load_ridge_probe(path)
    if task != "bandit" and payload.get("metadata", {}).get(
        "test_evaluation_deferred"
    ) is not True:
        raise ValueError(
            f"{task} ceiling evaluated its held-out test split before the shared checkpoint"
        )
    return path, probe, probe_layer(payload, path)


def _evaluate_fold(
    *,
    discovery_tasks: tuple[str, ...],
    heldout_task: str,
    shared_artifact: str,
    shards_by_task: dict[str, list[dict]],
    splits_by_task: dict[str, dict],
    ceiling_directories: dict[str, str],
    matched_label_shards_by_task: dict[str, list[dict]],
    control_shards: list[dict],
    control_split: dict,
    terminality_shards: list[dict],
    terminality_split: dict,
    thresholds: dict,
    seed: int,
) -> dict:
    shared_probe, payload = load_ridge_probe(shared_artifact)
    layer = probe_layer(payload, shared_artifact)
    selected_discovery = next(
        row
        for row in payload["metadata"]["layers"]
        if int(row["layer"]) == layer
    )
    if selected_discovery.get("source_task_gate", {}).get("passed") is not True:
        raise ValueError("shared artifact did not clear every discovery source task")
    plan = validate_discovery_plan(
        discovery_tasks=discovery_tasks,
        heldout_task=heldout_task,
        layer_selection_tasks=tuple(payload["metadata"]["layer_selection_tasks"]),
    )
    heldout_test = semantic_layer_dataset(
        heldout_task,
        shards_by_task[heldout_task],
        layer,
        set(splits_by_task[heldout_task]["test"]),
    )
    prediction = shared_probe.predict(heldout_test["states"])
    strict = _association_metrics(prediction, heldout_test["target"])
    bootstrap_samples = int(thresholds["episode_bootstrap_samples"])
    strict_bootstrap = _cluster_bootstrap(
        prediction,
        heldout_test["target"],
        [
            record.get("pair_id", record["episode_id"])
            for record in heldout_test["records"]
        ],
        samples=bootstrap_samples,
        seed=seed,
    )
    mapping = _mapping_associations(
        prediction,
        heldout_test["target"],
        heldout_test["records"],
        samples=bootstrap_samples,
        seed=seed + 100,
    )
    matched_label = (
        _matched_label_evaluation(
            matched_label_shards_by_task[heldout_task],
            heldout_task,
            layer,
            shared_probe,
            samples=bootstrap_samples,
            seed=seed + 150,
            max_projection_difference_sd=float(
                thresholds["matched_label_max_projection_difference_sd"]
            ),
            max_target_difference_sd=float(
                thresholds["matched_label_max_target_difference_sd"]
            ),
            expected_source_state_ids=_canonical_source_state_ids(
                shards_by_task[heldout_task],
                set(splits_by_task[heldout_task]["test"]),
            ),
        )
        if heldout_task in matched_label_shards_by_task
        else None
    )

    heldout_validation = semantic_layer_dataset(
        heldout_task,
        shards_by_task[heldout_task],
        layer,
        set(splits_by_task[heldout_task]["validation"]),
    )
    affine = _affine_fit(
        shared_probe.predict(heldout_validation["states"]),
        heldout_validation["target"],
    )
    affine_metrics = regression_metrics(
        _apply_affine(prediction, affine), heldout_test["target"]
    )

    ceiling_path, ceiling_probe, ceiling_layer = _load_ceiling(
        heldout_task, ceiling_directories
    )
    ceiling_data = semantic_layer_dataset(
        heldout_task,
        shards_by_task[heldout_task],
        ceiling_layer,
        set(splits_by_task[heldout_task]["test"]),
    )
    if [record["state_id"] for record in ceiling_data["records"]] != [
        record["state_id"] for record in heldout_test["records"]
    ]:
        raise ValueError("held-out test state order changed between shared and ceiling layers")
    ceiling_prediction = ceiling_probe.predict(ceiling_data["states"])
    ceiling = _association_metrics(ceiling_prediction, ceiling_data["target"])
    ceiling_bootstrap = _cluster_bootstrap(
        ceiling_prediction,
        ceiling_data["target"],
        [
            record.get("pair_id", record["episode_id"])
            for record in ceiling_data["records"]
        ],
        samples=bootstrap_samples,
        seed=seed + 200,
    )
    ratio = (
        strict["association_r_squared"] / ceiling["association_r_squared"]
        if ceiling["association_r_squared"] > 0
        else None
    )

    random_directions = matched_sign_random_directions(
        shared_probe.raw_activation_direction().float(),
        n_directions=int(thresholds["matched_random_directions"]),
        seed=seed + 300,
    )
    random_metrics = [
        _association_metrics(
            _direction_score(heldout_test["states"], shared_probe, direction),
            heldout_test["target"],
        )
        for direction in random_directions
    ]
    random_95 = sorted(row["correlation"] for row in random_metrics)[
        max(0, math.ceil(0.95 * len(random_metrics)) - 1)
    ]

    control_data = cross_task_layer_dataset(
        control_shards,
        layer,
        set(control_split["test"]),
        target_key="choice_logit",
    )
    control_prediction = shared_probe.predict(control_data["states"])
    negative_control = _association_metrics(
        control_prediction, control_data["target"]
    )
    negative_control_bootstrap = _cluster_bootstrap(
        control_prediction,
        control_data["target"],
        [
            record.get("pair_id", record["episode_id"])
            for record in control_data["records"]
        ],
        samples=bootstrap_samples,
        seed=seed + 350,
    )
    terminality_data = cross_task_layer_dataset(
        terminality_shards,
        layer,
        set(terminality_split["test"]),
        target_key="terminality_logit",
    )
    terminality_prediction = shared_probe.predict(terminality_data["states"])
    terminality_control = _association_metrics(
        terminality_prediction, terminality_data["target"]
    )
    terminality_control_bootstrap = _cluster_bootstrap(
        terminality_prediction,
        terminality_data["target"],
        [
            record.get("pair_id", record["episode_id"])
            for record in terminality_data["records"]
        ],
        samples=bootstrap_samples,
        seed=seed + 375,
    )
    negative_abs = float(thresholds["negative_control_max_absolute_correlation"])
    negative_relative = float(
        thresholds["negative_control_relative_correlation_fraction"]
    )
    specificity_bound = max(
        negative_abs, negative_relative * max(strict["correlation"], 0.0)
    )
    binary_interval = negative_control_bootstrap["correlation"]
    terminality_interval = terminality_control_bootstrap["correlation"]
    label_consistency = matched_label["passed"] if matched_label is not None else None
    criteria = {
        "expected_direction": strict["correlation"] > 0
        and strict_bootstrap["correlation"]["lower_95"] > 0,
        "label_reversal_consistency": label_consistency,
        "exceeds_random_95th_percentile": strict["correlation"] > random_95,
        "negative_control_absent_or_weaker": (
            max(abs(binary_interval["lower_95"]), abs(binary_interval["upper_95"]))
            <= specificity_bound
        ),
        "terminality_control_absent_or_weaker": (
            max(
                abs(terminality_interval["lower_95"]),
                abs(terminality_interval["upper_95"]),
            )
            <= specificity_bound
        ),
        "at_least_half_ceiling": ratio is not None
        and ratio >= float(thresholds["strong_transfer_ceiling_fraction"]),
    }
    required = list(thresholds["primary_required_checks"])
    if label_consistency is None:
        required.remove("label_reversal_consistency")
    cleared = all(criteria[name] is True for name in required)
    classification = (
        "strong_shared_transfer"
        if cleared and criteria["at_least_half_ceiling"]
        else "partial_shared_transfer"
        if cleared
        else "no_convincing_shared_transfer"
    )
    return {
        **plan,
        "analysis_role": (
            "confirmatory_primary"
            if heldout_task == str(thresholds["primary_heldout_task"])
            else "secondary_leave_one_task_out_robustness"
        ),
        "classification": classification,
        "selected_layer": layer,
        "selected_alpha": shared_probe.alpha,
        "discovery_validation_by_task": selected_discovery["validation_by_task"],
        "discovery_source_task_gate": selected_discovery["source_task_gate"],
        "shared_probe": os.path.abspath(shared_artifact),
        "criteria": criteria,
        "required_checks": required,
        "strict_zero_shot": {
            **strict,
            "cluster_bootstrap": strict_bootstrap,
            "mapping_associations": mapping,
            "parameters_fit_on_heldout_task": 0,
        },
        "exact_matched_semantic_history_label_test": matched_label,
        "heldout_validation_affine_secondary": {
            "analysis_role": "exploratory_scale_calibration_not_used_for_clearance",
            "parameters_fit_on_heldout_validation": 2,
            **affine,
            **affine_metrics,
        },
        "heldout_task_specific_ceiling": {
            **ceiling,
            "selected_layer": ceiling_layer,
            "probe": os.path.abspath(ceiling_path),
            "cluster_bootstrap": ceiling_bootstrap,
        },
        "fraction_of_ceiling_association_r_squared": ratio,
        "matched_random_directions": {
            "count": len(random_metrics),
            "correlation_95th_percentile": random_95,
            "metrics": random_metrics,
        },
        "non_persistence_binary_control": {
            **negative_control,
            "states": len(control_data["records"]),
            "cluster_bootstrap": negative_control_bootstrap,
            "specificity_absolute_correlation_bound": specificity_bound,
            "response_labels_match_primary_heldout": heldout_task == "solvability",
        },
        "rule_determined_terminality_control": {
            **terminality_control,
            "states": len(terminality_data["records"]),
            "cluster_bootstrap": terminality_control_bootstrap,
            "specificity_absolute_correlation_bound": specificity_bound,
            "mapping_associations": _mapping_associations(
                terminality_prediction,
                terminality_data["target"],
                terminality_data["records"],
                samples=bootstrap_samples,
                seed=seed + 400,
            ),
            "response_labels_match_primary_heldout": heldout_task == "solvability",
        },
        "test_counts": {
            "states": len(heldout_test["records"]),
            "episodes": len(set(splits_by_task[heldout_task]["test"])),
        },
    }


def _direction_alignment(
    layer_count: int, directories: dict[str, str]
) -> list[dict]:
    rows = []
    for layer in range(layer_count):
        probes = {}
        for task, directory in directories.items():
            path = os.path.join(directory, f"layer_{layer:02d}_persistence.pt")
            if not os.path.exists(path):
                probes = {}
                break
            probes[task] = load_ridge_probe(path)[0].raw_activation_direction()
        if len(probes) != 3:
            continue
        rows.append(
            {
                "layer": layer,
                "bandit_foraging_cosine": _cosine(probes["bandit"], probes["foraging"]),
                "bandit_solvability_cosine": _cosine(
                    probes["bandit"], probes["solvability"]
                ),
                "foraging_solvability_cosine": _cosine(
                    probes["foraging"], probes["solvability"]
                ),
            }
        )
    return rows


def _write_alignment(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(result: dict, path: Path) -> None:
    folds = result["leave_one_task_out"]
    svg = Svg(1050, 650)
    svg.text(50, 42, "Shared persistence: leave-one-task-out transfer", "title")
    svg.text(
        50,
        68,
        "Frozen shared directions; pair/episode-clustered 95% intervals; task-specific ceiling shown",
        "subtitle",
    )
    sx, sy, box = axes(
        svg,
        55,
        95,
        930,
        470,
        "Held-out semantic persistence association",
        "Held-out task",
        "Correlation",
        [(index, row["heldout_task"].title()) for index, row in enumerate(folds)],
        [(-0.5, "-.5"), (0, "0"), (0.5, ".5"), (1, "1")],
        (-0.5, len(folds) - 0.5),
        (-0.55, 1.05),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, fold in enumerate(folds):
        strict = fold["strict_zero_shot"]
        lower = strict["cluster_bootstrap"]["correlation"]["lower_95"]
        upper = strict["cluster_bootstrap"]["correlation"]["upper_95"]
        value = strict["correlation"]
        ceiling = fold["heldout_task_specific_ceiling"]["correlation"]
        svg.line(sx(index), sy(lower), sx(index), sy(upper), COLORS["observed"], 2)
        svg.line(sx(index) - 7, sy(lower), sx(index) + 7, sy(lower), COLORS["observed"], 2)
        svg.line(sx(index) - 7, sy(upper), sx(index) + 7, sy(upper), COLORS["observed"], 2)
        svg.circle(sx(index), sy(value), 6, COLORS["observed"])
        svg.circle(sx(index) + 20, sy(ceiling), 5, COLORS["both_positive"])
        svg.text(sx(index), sy(value) - 12, f"{value:.3f}", anchor="middle")
    svg.line(710, 585, 732, 585, COLORS["observed"], 3)
    svg.text(738, 589, "Shared zero-shot", "legend")
    svg.circle(870, 585, 5, COLORS["both_positive"])
    svg.text(882, 589, "Within-task ceiling", "legend")
    svg.save(path)


def _make_specificity_figure(result: dict, path: Path) -> None:
    primary = result["primary_heldout_result"]
    strict = primary["strict_zero_shot"]
    entries = [
        ("Solvability", strict["correlation"], COLORS["observed"]),
        (
            "Random 95th",
            primary["matched_random_directions"]["correlation_95th_percentile"],
            COLORS["reference"],
        ),
        (
            "Binary control",
            primary["non_persistence_binary_control"]["correlation"],
            COLORS["both_negative"],
        ),
        (
            "Rule terminality",
            primary["rule_determined_terminality_control"]["correlation"],
            COLORS["both_negative"],
        ),
    ]
    interval = strict["cluster_bootstrap"]["correlation"]
    values = [value for _name, value, _color in entries]
    values.extend([interval["lower_95"], interval["upper_95"], 0.0])
    lower = min(-0.15, min(values) - 0.08)
    upper = max(0.30, max(values) + 0.08)
    svg = Svg(980, 620)
    svg.text(50, 42, "Shared persistence specificity", "title")
    svg.text(
        50,
        68,
        "Primary held-out association versus matched-random and negative controls",
        "subtitle",
    )
    sx, sy, box = axes(
        svg,
        70,
        105,
        830,
        410,
        "",
        "Condition",
        "Correlation",
        [(index, name) for index, (name, _value, _color) in enumerate(entries)],
        [(lower, f"{lower:.2f}"), (0.0, "0"), (upper, f"{upper:.2f}")],
        (-0.6, 3.6),
        (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (_name, value, color) in enumerate(entries):
        svg.circle(sx(index), sy(value), 7, color)
        svg.text(sx(index), sy(value) - 12, f"{value:.3f}", anchor="middle")
    svg.line(
        sx(0),
        sy(interval["lower_95"]),
        sx(0),
        sy(interval["upper_95"]),
        COLORS["observed"],
        2.5,
    )
    svg.save(path)


def _write_report(result: dict, path: Path) -> None:
    primary = result["primary_heldout_result"]
    strict = primary["strict_zero_shot"]
    interval = strict["cluster_bootstrap"]["correlation"]
    lines = [
        "# Shared persistence representational test",
        "",
        "The primary direction was discovered using task-balanced Bandit + Foraging training targets. Solvability supplied no direction, layer, regularization, centering, or scaling parameter for the strict test.",
        "",
        f"Primary classification: **{primary['classification'].replace('_', ' ')}**.",
        f"Held-out Solvability correlation: **{strict['correlation']:.3f}** (counterbalanced-pair-clustered 95% CI **{interval['lower_95']:.3f} to {interval['upper_95']:.3f}**).",
        f"Scale-free variance association (correlation²): **{strict['association_r_squared']:.3f}**.",
        f"Fraction of Solvability-specific ceiling correlation²: **{primary['fraction_of_ceiling_association_r_squared']:.3f}**."
        if primary["fraction_of_ceiling_association_r_squared"] is not None
        else "Fraction of ceiling: undefined.",
        f"Selected discovery-only layer / ridge penalty: **{primary['selected_layer']} / {primary['selected_alpha']:g}**.",
        "",
        "## Discovery source-task gate",
        "",
    ]
    for task, metrics in primary["discovery_validation_by_task"].items():
        random_95 = primary["discovery_source_task_gate"][
            "random_correlation_95th_by_task"
        ][task]
        lines.append(
            f"- {task.title()}: validation r **{metrics['correlation']:.3f}**, "
            f"R² **{metrics['r_squared']:.3f}**, matched-random 95th percentile "
            f"**{random_95:.3f}**."
        )
    lines.extend([
        "",
        "## Primary checks",
        "",
    ])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in primary["criteria"].items()
    )
    matched = primary["exact_matched_semantic_history_label_test"]
    if matched is not None:
        lines.extend(["", "## Exact matched-history label replay", ""])
        for mapping_id, metrics in matched["mapping_associations"].items():
            mapping_interval = metrics["cluster_bootstrap"]["correlation"]
            lines.append(
                f"- {mapping_id}: r **{metrics['correlation']:.3f}** "
                f"(95% CI **{mapping_interval['lower_95']:.3f}, "
                f"{mapping_interval['upper_95']:.3f}**)."
            )
        lines.extend(
            [
                f"- Mean paired projection gap: **{matched['projection_difference_in_pooled_sd']:.3f} pooled SD**.",
                f"- Mean paired semantic-target gap: **{matched['target_difference_in_pooled_sd']:.3f} pooled SD**.",
            ]
        )
    lines.extend(
        [
            "",
            "## Specificity controls",
            "",
            f"- Arbitrary binary-choice r: **{primary['non_persistence_binary_control']['correlation']:.3f}** "
            f"(95% CI **{primary['non_persistence_binary_control']['cluster_bootstrap']['correlation']['lower_95']:.3f}, "
            f"{primary['non_persistence_binary_control']['cluster_bootstrap']['correlation']['upper_95']:.3f}**).",
            f"- Rule-determined terminality r: **{primary['rule_determined_terminality_control']['correlation']:.3f}** "
            f"(95% CI **{primary['rule_determined_terminality_control']['cluster_bootstrap']['correlation']['lower_95']:.3f}, "
            f"{primary['rule_determined_terminality_control']['cluster_bootstrap']['correlation']['upper_95']:.3f}**).",
        ]
    )
    lines.extend(["", "## Leave-one-task-out robustness", ""])
    for fold in result["leave_one_task_out"]:
        fold_interval = fold["strict_zero_shot"]["cluster_bootstrap"][
            "correlation"
        ]
        lines.append(
            f"- Held out {fold['heldout_task']}: r **{fold['strict_zero_shot']['correlation']:.3f}** "
            f"(95% CI **{fold_interval['lower_95']:.3f}, {fold_interval['upper_95']:.3f}**), "
            f"layer **{fold['selected_layer']}**, {fold['classification'].replace('_', ' ')} "
            f"({fold['analysis_role'].replace('_', ' ')})."
        )
    lines.extend(
        [
            "",
            "The validation-affine results are scale diagnostics only. Clearance uses the strict correlation, exact matched-history label replays, matched random directions, the arbitrary binary-choice control, and the rule-determined terminality control. The old Bandit-only transfer is not part of this gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-probe-dir", required=True)
    parser.add_argument("--bandit-bank", default="artifacts/activation_bank")
    parser.add_argument("--bandit-split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument("--foraging-bank", required=True)
    parser.add_argument("--foraging-split", required=True)
    parser.add_argument("--solvability-bank", required=True)
    parser.add_argument("--solvability-split", required=True)
    parser.add_argument("--control-bank", required=True)
    parser.add_argument("--control-split", required=True)
    parser.add_argument("--terminality-bank", required=True)
    parser.add_argument("--terminality-split", required=True)
    parser.add_argument("--foraging-label-bank", required=True)
    parser.add_argument("--solvability-label-bank", required=True)
    parser.add_argument("--bandit-probe-dir", default="artifacts/linear_probes")
    parser.add_argument("--foraging-probe-dir", required=True)
    parser.add_argument("--solvability-probe-dir", required=True)
    parser.add_argument("--behavioral-gate", required=True)
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import yaml

    gate = require_behavioral_clearance(args.behavioral_gate)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    thresholds = config["shared_persistence_transfer"]
    validate_loto_folds(
        ("bandit", "foraging", "solvability"),
        thresholds["leave_one_task_out_folds"],
    )
    task_paths = {
        "bandit": (args.bandit_bank, args.bandit_split),
        "foraging": (args.foraging_bank, args.foraging_split),
        "solvability": (args.solvability_bank, args.solvability_split),
    }
    shards = {
        task: load_task_shards(task, bank) for task, (bank, _split) in task_paths.items()
    }
    layer_count, _hidden_width = validate_compatible_tasks(shards)
    splits = {
        task: load_task_split(
            task, shards[task], split, seed=int(config["split_seed"])
        )
        for task, (_bank, split) in task_paths.items()
    }
    control_shards = load_activation_shards(args.control_bank)
    control_split = make_or_validate_split(
        control_shards, args.control_split, seed=int(config["split_seed"])
    )
    terminality_shards = load_activation_shards(args.terminality_bank)
    terminality_split = make_or_validate_split(
        terminality_shards,
        args.terminality_split,
        seed=int(config["split_seed"]),
    )
    matched_label_shards = {
        "foraging": load_activation_shards(args.foraging_label_bank),
        "solvability": load_activation_shards(args.solvability_label_bank),
    }
    ceiling_directories = {
        "bandit": args.bandit_probe_dir,
        "foraging": args.foraging_probe_dir,
        "solvability": args.solvability_probe_dir,
    }
    folds = []
    for index, fold_specification in enumerate(thresholds["leave_one_task_out_folds"]):
        heldout = str(fold_specification["heldout"])
        folds.append(
            _evaluate_fold(
                discovery_tasks=tuple(fold_specification["discovery"]),
                heldout_task=heldout,
                shared_artifact=os.path.join(
                    args.shared_probe_dir, f"heldout_{heldout}.pt"
                ),
                shards_by_task=shards,
                splits_by_task=splits,
                ceiling_directories=ceiling_directories,
                matched_label_shards_by_task=matched_label_shards,
                control_shards=control_shards,
                control_split=control_split,
                terminality_shards=terminality_shards,
                terminality_split=terminality_split,
                thresholds=thresholds,
                seed=int(config["analysis_seed"]) + 1000 * index,
            )
        )
    primary_heldout = str(thresholds["primary_heldout_task"])
    primary = next(fold for fold in folds if fold["heldout_task"] == primary_heldout)
    alignment = _direction_alignment(layer_count, ceiling_directories)
    result = {
        "classification": primary["classification"],
        "analysis_role": "primary_shared_heldout_transfer_with_loto_robustness",
        "primary_discovery_tasks": thresholds["primary_discovery_tasks"],
        "primary_heldout_task": primary_heldout,
        "heldout_task_parameters_fit": 0,
        "primary_heldout_result": primary,
        "leave_one_task_out": folds,
        "loto_all_correlations_positive": all(
            fold["strict_zero_shot"]["correlation"] > 0 for fold in folds
        ),
        "loto_all_clustered_intervals_positive": all(
            fold["strict_zero_shot"]["cluster_bootstrap"]["correlation"][
                "lower_95"
            ]
            > 0
            for fold in folds
        ),
        "task_specific_direction_alignment": alignment,
        "development_behavioral_gate": gate,
        "preregistered_config": os.path.abspath(args.config),
        "provenance": run_metadata(
            {
                "analysis": "shared_persistence_heldout_transfer",
                "model": shards["foraging"][0].get("model_id", config["model"]),
                "config": os.path.abspath(args.config),
            }
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "shared_persistence_transfer_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_alignment(alignment, output / "task_direction_alignment.csv")
    _write_report(result, output / "shared_persistence_transfer_report.md")
    _make_figure(result, output / "shared_persistence_transfer.svg")
    _make_specificity_figure(result, output / "shared_persistence_specificity.svg")
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "primary_correlation": primary["strict_zero_shot"]["correlation"],
                "loto_all_clustered_intervals_positive": result[
                    "loto_all_clustered_intervals_positive"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
