"""All-layer contrast-subspace, transfer, and nuisance-specificity search."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from statistics import mean

from analysis.persistence_isolation import (
    validate_manipulation_holdout,
    validate_task_holdout,
)
from analysis.persistence_specificity import classify_candidate
from analysis.persistence_subspace import (
    evaluate_subspace,
    fit_balanced_subspace,
    matched_random_subspace_scores,
    validate_initial_rank,
)


def _stack(rows, *, layer: int, feature_type: str):
    import torch

    if not rows:
        raise ValueError("cannot stack an empty contrast subset")
    layer_count = int(rows[0]["activation_delta"].shape[0])
    if feature_type == "static":
        if not 0 <= layer < layer_count:
            raise IndexError("static contrast layer is out of range")
        values = [row["activation_delta"][layer].float() for row in rows]
    elif feature_type == "displacement":
        if not 0 <= layer < layer_count - 1:
            raise IndexError("displacement transition is out of range")
        values = [
            (
                row["activation_delta"][layer + 1]
                - row["activation_delta"][layer]
            ).float()
            for row in rows
        ]
    else:
        raise ValueError("feature_type must be static or displacement")
    stacked = torch.stack(values)
    return stacked.to("cuda") if torch.cuda.is_available() else stacked


def _fit_and_evaluate(train_rows, evaluation_rows, *, layer, feature_type, rank):
    train = _stack(train_rows, layer=layer, feature_type=feature_type)
    evaluation = _stack(
        evaluation_rows, layer=layer, feature_type=feature_type
    )
    candidate = fit_balanced_subspace(train, train_rows, rank=rank)
    metrics = evaluate_subspace(candidate, evaluation)
    return candidate, metrics


def _component_mean(rows: list[dict], key: str) -> float:
    if not rows:
        raise ValueError(f"no rows available for component {key!r}")
    return mean(float(row[key]) for row in rows)


def _cluster_bootstrap_subspace(
    candidate,
    rows,
    *,
    layer,
    feature_type,
    samples,
    seed,
):
    import torch

    if samples < 20:
        raise ValueError("contrast bootstrap requires at least 20 samples")
    features = _stack(rows, layer=layer, feature_type=feature_type)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["cluster_id"]), []).append(index)
    cluster_ids = sorted(groups)
    rng = random.Random(seed)
    draws = {"captured_energy_fraction": [], "positive_projection_fraction": []}
    for _ in range(samples):
        indices = [
            index
            for _draw in cluster_ids
            for index in groups[rng.choice(cluster_ids)]
        ]
        selected = torch.tensor(indices, dtype=torch.long, device=features.device)
        metrics = evaluate_subspace(candidate, features[selected])
        for key in draws:
            draws[key].append(float(metrics[key]))
    result = {"samples": samples, "clusters": len(cluster_ids)}
    for key, values in draws.items():
        ordered = sorted(values)
        result[key] = {
            "bootstrap_mean": mean(values),
            "lower_95": ordered[int(0.025 * samples)],
            "upper_95": ordered[min(samples - 1, int(0.975 * samples))],
        }
    return result


def _grid(layer_count: int, ranks, feature_types):
    for feature_type in feature_types:
        maximum = layer_count if feature_type == "static" else layer_count - 1
        for layer in range(maximum):
            for rank in ranks:
                yield feature_type, layer, validate_initial_rank(rank)


def _strict_fold(
    *,
    contrasts,
    source_filter,
    heldout_filter,
    layer_count,
    ranks,
    feature_types,
    plan,
    bootstrap_samples,
    seed,
):
    source_train = [
        row
        for row in contrasts
        if row.get("contrast_kind") == "persistence"
        and row["split"] == "train"
        and source_filter(row)
    ]
    source_validation = [
        row
        for row in contrasts
        if row.get("contrast_kind") == "persistence"
        and row["split"] == "validation"
        and source_filter(row)
    ]
    heldout_test = [
        row
        for row in contrasts
        if row.get("contrast_kind") == "persistence"
        and row["split"] == "test"
        and heldout_filter(row)
    ]
    if not source_train or not source_validation or not heldout_test:
        raise ValueError(f"strict fold has an empty split: {plan}")
    candidates = []
    artifacts = {}
    for feature_type, layer, rank in _grid(layer_count, ranks, feature_types):
        try:
            candidate, validation = _fit_and_evaluate(
                source_train,
                source_validation,
                layer=layer,
                feature_type=feature_type,
                rank=rank,
            )
        except ValueError as error:
            if "exceeds available" in str(error):
                continue
            raise
        key = f"{feature_type}-L{layer:02d}-k{rank}"
        artifacts[key] = candidate
        candidates.append(
            {
                "key": key,
                "feature_type": feature_type,
                "layer": layer,
                "rank": rank,
                "source_validation": validation,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["source_validation"]["positive_projection_fraction"],
            row["source_validation"]["captured_energy_fraction"],
            -row["rank"],
            -row["layer"],
        ),
    )
    heldout = evaluate_subspace(
        artifacts[selected["key"]],
        _stack(
            heldout_test,
            layer=selected["layer"],
            feature_type=selected["feature_type"],
        ),
    )
    heldout_bootstrap = _cluster_bootstrap_subspace(
        artifacts[selected["key"]],
        heldout_test,
        layer=selected["layer"],
        feature_type=selected["feature_type"],
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "plan": plan,
        "selection_rule": (
            "lexicographic source-validation positive fraction, captured energy, "
            "lower rank, earlier layer"
        ),
        "selected": selected,
        "heldout_test": heldout,
        "heldout_cluster_bootstrap": heldout_bootstrap,
        "counts": {
            "source_train": len(source_train),
            "source_validation": len(source_validation),
            "heldout_test": len(heldout_test),
        },
        "heldout_parameters_fit": 0,
        "candidate": artifacts[selected["key"]],
    }


def run_contrast_search(contrasts: list[dict], config: dict) -> tuple[dict, dict]:
    """Run wide maps plus strict LOMO/LOTO selections and specificity tests."""

    import torch

    required_config = {"analysis_seed", "model", "protocol_version", "search"}
    missing_config = sorted(required_config.difference(config))
    if missing_config:
        raise ValueError(
            f"contrast search config is missing required keys: {missing_config}"
        )
    if not contrasts:
        raise ValueError("contrast search requires a nonempty bank")
    shapes = {
        tuple(int(value) for value in row["activation_delta"].shape)
        for row in contrasts
    }
    if len(shapes) != 1:
        raise ValueError(f"contrast bank mixes activation shapes: {sorted(shapes)}")
    layer_count, width = next(iter(shapes))
    search = config["search"]
    ranks = tuple(int(rank) for rank in search["ranks"])
    feature_types = tuple(str(value) for value in search["feature_types"])
    persistence = [row for row in contrasts if row.get("contrast_kind") == "persistence"]
    nuisance = [row for row in contrasts if row.get("contrast_kind") == "nuisance"]
    tasks = sorted({str(row["task"]) for row in persistence})
    manipulations = sorted({str(row["manipulation"]) for row in persistence})
    if set(tasks) != {"bandit", "foraging", "solvability"}:
        raise ValueError(f"persistence search requires B/F/S; observed {tasks}")
    nuisance_types = sorted({str(row["nuisance_type"]) for row in nuisance})
    required_nuisance = {"label", "arbitrary_choice", "terminality", "generic_value"}
    if set(nuisance_types) != required_nuisance:
        raise ValueError(
            f"specificity bank must contain {sorted(required_nuisance)}; observed {nuisance_types}"
        )

    training = [row for row in persistence if row["split"] == "train"]
    validation = [row for row in persistence if row["split"] == "validation"]
    nuisance_test = [row for row in nuisance if row["split"] == "test"]
    if not training or not validation or not nuisance_test:
        raise ValueError("wide search has an empty train/validation/nuisance-test split")

    layerwise, candidate_artifacts = [], {}
    for feature_type, layer, rank in _grid(layer_count, ranks, feature_types):
        try:
            candidate, validation_metrics = _fit_and_evaluate(
                training,
                validation,
                layer=layer,
                feature_type=feature_type,
                rank=rank,
            )
        except ValueError as error:
            if "exceeds available" in str(error):
                continue
            raise
        key = f"{feature_type}-L{layer:02d}-k{rank}"
        candidate_artifacts[key] = candidate
        control_metrics = {}
        for nuisance_type in sorted(required_nuisance):
            rows = [row for row in nuisance_test if row["nuisance_type"] == nuisance_type]
            control_metrics[nuisance_type] = evaluate_subspace(
                candidate,
                _stack(rows, layer=layer, feature_type=feature_type),
            )
        random_control = matched_random_subspace_scores(
            width=width,
            rank=rank,
            evaluation_deltas=_stack(
                validation, layer=layer, feature_type=feature_type
            ),
            count=int(search["matched_random_subspaces"]),
            seed=int(config["analysis_seed"]) + layer * 17 + rank,
        )
        layerwise.append(
            {
                "key": key,
                "feature_type": feature_type,
                "layer": layer,
                "rank": rank,
                "persistence_validation": validation_metrics,
                "nuisance_test": control_metrics,
                "matched_random": random_control,
            }
        )

    manipulation_folds = []
    for heldout in manipulations:
        discovery = tuple(value for value in manipulations if value != heldout)
        plan = validate_manipulation_holdout(
            discovery_manipulations=discovery,
            heldout_manipulation=heldout,
            selection_manipulations=discovery,
        )
        manipulation_folds.append(
            _strict_fold(
                contrasts=contrasts,
                source_filter=lambda row, values=set(discovery): row["manipulation"] in values,
                heldout_filter=lambda row, value=heldout: row["manipulation"] == value,
                layer_count=layer_count,
                ranks=ranks,
                feature_types=feature_types,
                plan=plan,
                bootstrap_samples=int(search["bootstrap_samples"]),
                seed=int(config["analysis_seed"]) + len(manipulation_folds),
            )
        )

    task_folds = []
    for heldout in tasks:
        discovery = tuple(value for value in tasks if value != heldout)
        plan = validate_task_holdout(
            discovery_tasks=discovery,
            heldout_task=heldout,
            selection_tasks=discovery,
        )
        task_folds.append(
            _strict_fold(
                contrasts=contrasts,
                source_filter=lambda row, values=set(discovery): row["task"] in values,
                heldout_filter=lambda row, value=heldout: row["task"] == value,
                layer_count=layer_count,
                ranks=ranks,
                feature_types=feature_types,
                plan=plan,
                bootstrap_samples=int(search["bootstrap_samples"]),
                seed=int(config["analysis_seed"]) + 100 + len(task_folds),
            )
        )

    # Attach same-hyperparameter transfer maps. Each fold representation is fit
    # only on that fold's source training data; held-out scores never affect the
    # representation itself.
    for row in layerwise:
        lomo_scores, loto_scores = [], []
        for heldout in manipulations:
            source = [
                item for item in training if item["manipulation"] != heldout
            ]
            target = [
                item
                for item in persistence
                if item["split"] == "test" and item["manipulation"] == heldout
            ]
            _candidate, metrics = _fit_and_evaluate(
                source,
                target,
                layer=row["layer"],
                feature_type=row["feature_type"],
                rank=row["rank"],
            )
            lomo_scores.append(metrics)
        for heldout in tasks:
            source = [item for item in training if item["task"] != heldout]
            target = [
                item
                for item in persistence
                if item["split"] == "test" and item["task"] == heldout
            ]
            _candidate, metrics = _fit_and_evaluate(
                source,
                target,
                layer=row["layer"],
                feature_type=row["feature_type"],
                rank=row["rank"],
            )
            loto_scores.append(metrics)
        row["cross_manipulation_transfer"] = {
            "captured_energy_fraction": _component_mean(
                lomo_scores, "captured_energy_fraction"
            ),
            "positive_projection_fraction": _component_mean(
                lomo_scores, "positive_projection_fraction"
            ),
        }
        row["cross_task_transfer"] = {
            "captured_energy_fraction": _component_mean(
                loto_scores, "captured_energy_fraction"
            ),
            "positive_projection_fraction": _component_mean(
                loto_scores, "positive_projection_fraction"
            ),
        }
        nuisance_sensitivity = {
            name: row["nuisance_test"][name]["captured_energy_fraction"]
            for name in required_nuisance
        }
        row["decision"] = classify_candidate(
            persistence_sensitivity=row["persistence_validation"][
                "captured_energy_fraction"
            ],
            cross_manipulation_transfer=row["cross_manipulation_transfer"][
                "captured_energy_fraction"
            ],
            cross_task_transfer=row["cross_task_transfer"][
                "captured_energy_fraction"
            ],
            nuisance_sensitivity=nuisance_sensitivity,
            minimum_transfer=float(search["minimum_transfer"]),
            maximum_nuisance_fraction=float(
                search["maximum_nuisance_fraction"]
            ),
            positive_projection_fraction=min(
                row["persistence_validation"]["positive_projection_fraction"],
                row["cross_manipulation_transfer"]["positive_projection_fraction"],
                row["cross_task_transfer"]["positive_projection_fraction"],
            ),
        )

    passing = [row for row in layerwise if row["decision"]["causal_gate_passed"]]
    strongest = max(
        layerwise,
        key=lambda row: (
            row["cross_task_transfer"]["captured_energy_fraction"],
            row["cross_manipulation_transfer"]["captured_energy_fraction"],
        ),
    )
    strongest_candidate = candidate_artifacts[strongest["key"]]
    strongest_uncertainty = {
        "persistence_validation": _cluster_bootstrap_subspace(
            strongest_candidate,
            validation,
            layer=strongest["layer"],
            feature_type=strongest["feature_type"],
            samples=int(search["bootstrap_samples"]),
            seed=int(config["analysis_seed"]) + 200,
        )
    }
    for nuisance_type in sorted(required_nuisance):
        rows = [row for row in nuisance_test if row["nuisance_type"] == nuisance_type]
        strongest_uncertainty[nuisance_type] = _cluster_bootstrap_subspace(
            strongest_candidate,
            rows,
            layer=strongest["layer"],
            feature_type=strongest["feature_type"],
            samples=int(search["bootstrap_samples"]),
            seed=int(config["analysis_seed"]) + 300 + len(strongest_uncertainty),
        )
    from experiments.runtime import run_metadata

    summary = {
        "analysis_role": "exploratory_discovery",
        "classification": (
            "persistence_specific_candidate_found"
            if passing
            else "no_persistence_specific_candidate"
        ),
        "causal_gate_passed": bool(passing),
        "tasks": tasks,
        "manipulations": manipulations,
        "nuisance_types": nuisance_types,
        "activation_shape": [layer_count, width],
        "candidate_count": len(layerwise),
        "layerwise_candidates": layerwise,
        "leave_one_manipulation_out": [
            {key: value for key, value in fold.items() if key != "candidate"}
            for fold in manipulation_folds
        ],
        "leave_one_task_out": [
            {key: value for key, value in fold.items() if key != "candidate"}
            for fold in task_folds
        ],
        "passing_candidate_keys": [row["key"] for row in passing],
        "strongest_cross_task_candidate_key": strongest["key"],
        "strongest_candidate_clustered_uncertainty": strongest_uncertainty,
        "conditional_next_step": (
            "eligible_for_targeted_existing_task_causal_design"
            if passing
            else "stop_causal_pipeline"
        ),
        "task4_status": (
            "candidate_may_be_frozen_for_task4_design"
            if passing
            else "do_not_design_task4_yet"
        ),
        "provenance": run_metadata(
            {
                "analysis": "persistence_contrast_search",
                "protocol_version": config["protocol_version"],
                "model": config["model"],
            }
        ),
    }
    artifacts = {
        "wide_candidates": candidate_artifacts,
        "lomo_selected": {
            str(fold["plan"]["heldout_manipulation"]): fold["candidate"]
            for fold in manipulation_folds
        },
        "loto_selected": {
            str(fold["plan"]["heldout_task"]): fold["candidate"]
            for fold in task_folds
        },
    }
    return summary, artifacts


def _flatten_candidate(row: dict) -> dict:
    output = {
        "key": row["key"],
        "feature_type": row["feature_type"],
        "layer": row["layer"],
        "rank": row["rank"],
        "persistence_sensitivity": row["persistence_validation"][
            "captured_energy_fraction"
        ],
        "persistence_positive_fraction": row["persistence_validation"][
            "positive_projection_fraction"
        ],
        "cross_manipulation_transfer": row["cross_manipulation_transfer"][
            "captured_energy_fraction"
        ],
        "cross_task_transfer": row["cross_task_transfer"][
            "captured_energy_fraction"
        ],
        "random_95th": row["matched_random"][
            "captured_energy_fraction_95th"
        ],
        "classification": row["decision"]["classification"],
    }
    for name, metrics in row["nuisance_test"].items():
        output[f"nuisance_{name}"] = metrics["captured_energy_fraction"]
    return output


def _write_svg(rows: list[dict], path: str) -> None:
    """Write a dependency-free layer/rank/feature transfer trajectory."""

    width, height = 1100, 620
    margin = 70
    colors = {
        ("static", 1): "#264653",
        ("static", 2): "#2a9d8f",
        ("static", 4): "#8ab17d",
        ("displacement", 1): "#e76f51",
        ("displacement", 2): "#f4a261",
        ("displacement", 4): "#e9c46a",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="34" font-family="sans-serif" font-size="22">Cross-task persistence-contrast transfer by layer</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = height - margin - value * (height - 2 * margin)
        lines.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#dddddd"/>')
        lines.append(f'<text x="{margin-12}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    for series_index, ((feature, rank), color) in enumerate(colors.items()):
        selected = sorted(
            (row for row in rows if row["feature_type"] == feature and row["rank"] == rank),
            key=lambda row: row["layer"],
        )
        points = []
        for row in selected:
            x = margin + row["layer"] / 31 * (width - 2 * margin)
            value = row["cross_task_transfer"]["captured_energy_fraction"]
            y = height - margin - value * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = 55 + 20 * series_index
        lines.append(f'<line x1="760" y1="{legend_y}" x2="790" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="798" y="{legend_y+4}" font-family="sans-serif" font-size="12">{feature}, k={rank}</text>')
    lines.extend(
        [
            f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Layer / depth transition</text>',
            f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Mean LOTO captured energy</text>',
            '</svg>',
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_search_outputs(summary: dict, artifacts: dict, output_dir: str) -> None:
    from experiments.runtime import atomic_torch_save

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "persistence_discovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    atomic_torch_save(artifacts, str(output / "persistence_candidate_subspaces.pt"))
    flattened = [_flatten_candidate(row) for row in summary["layerwise_candidates"]]
    with (output / "layerwise_transfer_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    _write_svg(
        summary["layerwise_candidates"],
        str(output / "layerwise_cross_task_transfer.svg"),
    )
    best = max(
        summary["layerwise_candidates"],
        key=lambda row: (
            row["cross_task_transfer"]["captured_energy_fraction"],
            row["cross_manipulation_transfer"]["captured_energy_fraction"],
        ),
    )
    report = [
        "# Task-general persistence discovery",
        "",
        f"Classification: **{summary['classification']}**.",
        f"Causal gate passed: **{summary['causal_gate_passed']}**.",
        "",
        "## Strongest cross-task candidate (exploratory)",
        "",
        f"- Feature/layer/rank: **{best['feature_type']} / {best['layer']} / {best['rank']}**.",
        f"- Persistence sensitivity: **{best['persistence_validation']['captured_energy_fraction']:.3f}**.",
        f"- Cross-manipulation transfer: **{best['cross_manipulation_transfer']['captured_energy_fraction']:.3f}**.",
        f"- Cross-task transfer: **{best['cross_task_transfer']['captured_energy_fraction']:.3f}**.",
        "",
        "## Specificity",
        "",
    ]
    for name, metrics in best["nuisance_test"].items():
        report.append(f"- {name}: **{metrics['captured_energy_fraction']:.3f}**.")
    report.extend(
        [
            "",
            "All Bandit/Foraging/Solvability results are exploratory because all three tasks influenced method development.",
            f"Conditional next step: **{summary['conditional_next_step']}**.",
        ]
    )
    (output / "persistence_discovery_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
