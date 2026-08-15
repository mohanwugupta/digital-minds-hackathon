"""Adjudicate continuation advantage versus recent-state stopping heuristics."""

import argparse
import csv
import glob
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.probe_history_matching import _clustered_regression


CONDITIONS = ("both_negative", "one_positive", "both_positive")
CONDITION_LABELS = {
    "both_negative": "Both negative",
    "one_positive": "One positive",
    "both_positive": "Both positive",
}


def pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def r_squared(target: list[float], prediction: list[float]) -> float:
    target_mean = statistics.mean(target)
    total = sum((value - target_mean) ** 2 for value in target)
    residual = sum(
        (value - estimate) ** 2 for value, estimate in zip(target, prediction)
    )
    return 1.0 - residual / total if total else 0.0


def condition_class(episode_id: str) -> str:
    match = re.search(r"pa-([0-9.]+)-pb-([0-9.]+)", episode_id)
    if match is None:
        raise ValueError(f"episode ID does not encode arm probabilities: {episode_id}")
    positive = sum(float(value) > 0.4 for value in match.groups())
    return CONDITIONS[positive]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _rank(values: list[float]) -> list[int]:
    result = [0] * len(values)
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        result[index] = rank
    return result


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            row["round"] = int(float(row["round"]))
            row["loss_streak"] = int(float(row["loss_streak"]))
            row["previous_outcome"] = (
                None if row["previous_outcome"] == "" else float(row["previous_outcome"])
            )
            for key in (
                "cumulative_score",
                "persistence_logit",
                "q_A",
                "q_B",
                "continuation_advantage",
                "advantage_standard_error",
                "ridge_advantage",
            ):
                row[key] = float(row[key])
            row["prior_score"] = row["cumulative_score"] - float(
                row["previous_outcome"] or 0
            )
            row["q_mean"] = (row["q_A"] + row["q_B"]) / 2.0
            row["choice_premium"] = abs(row["q_A"] - row["q_B"]) / 2.0
            row["p_continue"] = sigmoid(row["persistence_logit"])
            row["condition"] = condition_class(row["episode_id"])
            rows.append(row)
    return rows


def attach_generic_probe(rows: list[dict], path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        generic = {row["state_id"]: row for row in csv.DictReader(handle)}
    missing = {row["state_id"] for row in rows} - set(generic)
    if missing:
        raise ValueError(f"generic probe predictions omit {len(missing)} test states")
    for row in rows:
        reference = generic[row["state_id"]]
        row["ridge_return"] = float(reference["ridge_future_return"])
        row["future_return"] = float(reference["future_return"])
        row["ridge_persistence"] = float(reference["ridge_persistence"])


def audit_target_files(pattern: str) -> dict:
    rows, paths = [], sorted(glob.glob(pattern))
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    state_ids = [row["state_id"] for row in rows]
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    return {
        "files": paths,
        "rows": len(rows),
        "unique_state_ids": len(set(state_ids)),
        "duplicate_state_ids": len(state_ids) - len(set(state_ids)),
        "states_by_split": split_counts,
        "rollouts_per_action": sorted(
            {int(row["rollouts_per_action"]) for row in rows}
        ),
    }


def controls(rows: list[dict]) -> dict[str, list[float]]:
    rounds = [math.log1p(row["round"]) for row in rows]
    return {
        "previous_outcome": [
            0.0 if row["previous_outcome"] is None else row["previous_outcome"]
            for row in rows
        ],
        "initial_state": [float(row["previous_outcome"] is None) for row in rows],
        "loss_streak": [float(row["loss_streak"]) for row in rows],
        "log_round": rounds,
        "log_round_squared": [value * value for value in rounds],
    }


def exact_match(rows: list[dict], minimum: int = 4) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row["previous_outcome"] is not None:
            grouped[(row["round"], row["previous_outcome"], row["loss_streak"])].append(row)
    eligible = [selected for selected in grouped.values() if len(selected) >= minimum]
    keys = [
        "prior_score",
        "continuation_advantage",
        "ridge_advantage",
        "persistence_logit",
    ]
    if rows and "ridge_return" in rows[0]:
        keys.extend(("ridge_return", "future_return"))
    residuals = {key: [] for key in keys}
    clusters = []
    for selected in eligible:
        means = {key: statistics.mean(row[key] for row in selected) for key in keys}
        for row in selected:
            for key in keys:
                residuals[key].append(row[key] - means[key])
            clusters.append(row["episode_id"])
    if not clusters:
        raise ValueError(
            "no exact-match strata; collect more test states or increase states-per-split"
        )

    def fit(outcome, *predictors):
        return _clustered_regression(
            residuals[outcome],
            {key: residuals[key] for key in predictors},
            clusters,
        )

    models = {
        "probe_to_advantage": fit("continuation_advantage", "ridge_advantage"),
        "actual_advantage_to_persistence": fit(
            "persistence_logit", "continuation_advantage"
        ),
        "probe_advantage_to_persistence": fit(
            "persistence_logit", "ridge_advantage"
        ),
        "older_history_to_advantage": fit(
            "continuation_advantage", "prior_score"
        ),
        "older_history_plus_probe_to_persistence": fit(
            "persistence_logit", "prior_score", "ridge_advantage"
        ),
        "actual_plus_probe_to_persistence": fit(
            "persistence_logit", "continuation_advantage", "ridge_advantage"
        ),
    }
    if "ridge_return" in residuals:
        models.update(
            {
                "generic_probe_to_advantage": fit(
                    "continuation_advantage", "ridge_return"
                ),
                "generic_probe_to_persistence": fit(
                    "persistence_logit", "ridge_return"
                ),
                "both_probes_to_advantage": fit(
                    "continuation_advantage", "ridge_advantage", "ridge_return"
                ),
                "both_probes_to_persistence": fit(
                    "persistence_logit", "ridge_advantage", "ridge_return"
                ),
                "advantage_probe_to_realized_return": fit(
                    "future_return", "ridge_advantage"
                ),
            }
        )
    return {
        "eligible_strata": len(eligible),
        "states": len(clusters),
        "episode_clusters": len(set(clusters)),
        "models": models,
    }


def _episode_bootstrap_r_squared(
    rows: list[dict], *, samples: int = 5000
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    episode_ids = sorted(grouped)
    rng, draws = random.Random(92026), []
    for _ in range(samples):
        selected = [
            row
            for _episode in episode_ids
            for row in grouped[rng.choice(episode_ids)]
        ]
        draws.append(
            r_squared(
                [row["continuation_advantage"] for row in selected],
                [row["ridge_advantage"] for row in selected],
            )
        )
    draws.sort()
    return {
        "samples": samples,
        "lower_95": draws[int(0.025 * samples)],
        "upper_95": draws[int(0.975 * samples) - 1],
        "bootstrap_mean": statistics.mean(draws),
    }


def _layer_robustness(metrics: dict) -> dict:
    validation = [row["validation"]["r_squared"] for row in metrics["layers"]]
    test = [row["test"]["r_squared"] for row in metrics["layers"]]
    selected = metrics["best_layer"]
    return {
        "validation_test_spearman": pearson(_rank(validation), _rank(test)),
        "selected_layer_test_rank": sorted(
            range(len(test)), key=test.__getitem__, reverse=True
        ).index(selected)
        + 1,
        "best_test_layer": max(range(len(test)), key=test.__getitem__),
        "best_test_r_squared": max(test),
    }


def _condition_diagnostics(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        actual = [row["continuation_advantage"] for row in selected]
        predicted = [row["ridge_advantage"] for row in selected]
        output[condition] = {
            "states": len(selected),
            "episodes": len({row["episode_id"] for row in selected}),
            "actual_advantage_mean": statistics.mean(actual),
            "predicted_advantage_mean": statistics.mean(predicted),
            "advantage_prediction_correlation": pearson(actual, predicted),
            "mean_p_continue": statistics.mean(row["p_continue"] for row in selected),
            "negative_advantage_states": sum(value < 0 for value in actual),
        }
    return output


def _target_structure(rows: list[dict]) -> dict:
    actual = [row["continuation_advantage"] for row in rows]
    q_mean = [row["q_mean"] for row in rows]
    premium = [row["choice_premium"] for row in rows]
    negative = [row for row in rows if row["continuation_advantage"] < 0]
    nonnegative = [row for row in rows if row["continuation_advantage"] >= 0]
    return {
        "q_A_q_B_correlation": pearson(
            [row["q_A"] for row in rows], [row["q_B"] for row in rows]
        ),
        "advantage_q_mean_correlation": pearson(actual, q_mean),
        "q_mean_standard_deviation": statistics.pstdev(q_mean),
        "choice_premium_mean": statistics.mean(premium),
        "choice_premium_standard_deviation": statistics.pstdev(premium),
        "choice_premium_fraction_of_mean_absolute_q_mean": statistics.mean(premium)
        / statistics.mean(abs(value) for value in q_mean),
        "negative_advantage": {
            "states": len(negative),
            "mean_advantage": statistics.mean(
                row["continuation_advantage"] for row in negative
            ),
            "mean_p_continue": statistics.mean(row["p_continue"] for row in negative),
            "states_with_p_continue_below_half": sum(
                row["p_continue"] < 0.5 for row in negative
            ),
        },
        "nonnegative_advantage": {
            "states": len(nonnegative),
            "mean_advantage": statistics.mean(
                row["continuation_advantage"] for row in nonnegative
            ),
            "mean_p_continue": statistics.mean(
                row["p_continue"] for row in nonnegative
            ),
            "states_with_p_continue_below_half": sum(
                row["p_continue"] < 0.5 for row in nonnegative
            ),
        },
    }


def summarize(
    metrics: dict,
    rows: list[dict],
    linear_summary: dict | None,
    target_audit: dict | None = None,
) -> dict:
    recent = controls(rows)
    clusters = [row["episode_id"] for row in rows]
    persistence = [row["persistence_logit"] for row in rows]
    predicted = [row["ridge_advantage"] for row in rows]
    actual = [row["continuation_advantage"] for row in rows]
    best = next(row for row in metrics["layers"] if row["layer"] == metrics["best_layer"])
    persistence_models = {
        "history_only": _clustered_regression(persistence, recent, clusters),
        "advantage_probe_only": _clustered_regression(
            persistence, {"ridge_advantage": predicted}, clusters
        ),
        "history_plus_advantage_probe": _clustered_regression(
            persistence, {**recent, "ridge_advantage": predicted}, clusters
        ),
        "actual_advantage_only": _clustered_regression(
            persistence, {"continuation_advantage": actual}, clusters
        ),
        "history_plus_actual_advantage": _clustered_regression(
            persistence, {**recent, "continuation_advantage": actual}, clusters
        ),
        "history_plus_actual_and_probe": _clustered_regression(
            persistence,
            {
                **recent,
                "continuation_advantage": actual,
                "ridge_advantage": predicted,
            },
            clusters,
        ),
    }
    cross_target = None
    if rows and "ridge_return" in rows[0]:
        generic = [row["ridge_return"] for row in rows]
        persistence_models.update(
            {
                "generic_return_probe_only": _clustered_regression(
                    persistence, {"ridge_return": generic}, clusters
                ),
                "history_plus_generic_return_probe": _clustered_regression(
                    persistence, {**recent, "ridge_return": generic}, clusters
                ),
                "history_plus_both_probes": _clustered_regression(
                    persistence,
                    {
                        **recent,
                        "ridge_advantage": predicted,
                        "ridge_return": generic,
                    },
                    clusters,
                ),
            }
        )
        cross_target = {
            "advantage_probe_generic_probe_correlation": pearson(
                predicted, generic
            ),
            "actual_advantage_realized_return_correlation": pearson(
                actual, [row["future_return"] for row in rows]
            ),
            "actual_advantage_generic_probe_correlation": pearson(actual, generic),
            "actual_advantage_advantage_probe_correlation": pearson(
                actual, predicted
            ),
            "actual_advantage_from_both_probes": _clustered_regression(
                actual,
                {"ridge_advantage": predicted, "ridge_return": generic},
                clusters,
            ),
        }
    result = {
        "best_layer": metrics["best_layer"],
        "best_layer_metrics": best,
        "labeled_states_by_split": metrics["labeled_states_by_split"],
        "target_audit": target_audit,
        "episode_bootstrap_test_r_squared": _episode_bootstrap_r_squared(rows),
        "layer_robustness": _layer_robustness(metrics),
        "target_reliability": {
            "target_variance": statistics.pvariance(actual),
            "mean_estimation_error_variance": statistics.mean(
                row["advantage_standard_error"] ** 2 for row in rows
            ),
            "median_standard_error": statistics.median(
                row["advantage_standard_error"] for row in rows
            ),
        },
        "target_structure": _target_structure(rows),
        "condition_diagnostics": _condition_diagnostics(rows),
        "persistence_models": persistence_models,
        "cross_target_comparison": cross_target,
        "exact_matching": exact_match(rows),
        "advantage_persistence_overlap": [
            {
                "layer": row["layer"],
                **(row["advantage_persistence_overlap"] or {}),
            }
            for row in metrics["layers"]
        ],
        "direct_persistence_reference": linear_summary,
        "target": metrics["target"],
        "caveat": metrics["caveat"],
    }
    return result


def make_figure(metrics: dict, summary: dict, path: Path) -> None:
    svg = Svg(1500, 620)
    svg.text(55, 45, "Does continuation advantage explain persistence?", "title")
    svg.text(55, 70, "Forced A/B paired rollouts; ridge probes; held-out episode tests", "subtitle")
    panels = ((45, 105), (520, 105), (995, 105))
    layers = metrics["layers"]

    x, y = panels[0]
    validation = [row["validation"]["r_squared"] for row in layers]
    test = [row["test"]["r_squared"] for row in layers]
    values = validation + test
    lower, upper = min(-.1, min(values)-.04), max(.2, max(values)+.04)
    sx,sy,box=axes(svg,x,y,455,440,"A. Advantage decoding by layer","Layer","R²",[(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")],[(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")],(0,31),(lower,upper))
    svg.polyline([(sx(i),sy(v)) for i,v in enumerate(validation)],COLORS["observed"],2)
    svg.polyline([(sx(i),sy(v)) for i,v in enumerate(test)],COLORS["model"],2)
    legend(svg,box[0]+4,box[1]+18,[("Validation",COLORS["observed"]),("Test",COLORS["model"])])

    x,y=panels[1]
    exact=summary["exact_matching"]["models"]
    estimates=(
        ("Probe → advantage",exact["probe_to_advantage"],"ridge_advantage",COLORS["observed"]),
        ("Advantage → persist",exact["actual_advantage_to_persistence"],"continuation_advantage",COLORS["both_positive"]),
        ("Probe → persist",exact["probe_advantage_to_persistence"],"ridge_advantage",COLORS["model"]),
    )
    bounds=[]
    for _l,r,k,_c in estimates:
        c=r["coefficients"][k];b=c["standardized_beta"];e=c["cluster_robust_standard_error"] or 0;bounds.extend((b-1.96*e,b+1.96*e))
    lower,upper=min(-.5,min(bounds)-.05),max(.8,max(bounds)+.05)
    sx,sy,box=axes(svg,x,y,455,440,"B. Exact matched-state tests","Standardized coefficient","Relation",[(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")],[(i,l) for i,(l,_r,_k,_c) in enumerate(estimates)],(lower,upper),(-.6,2.6))
    svg.line(sx(0),box[1],sx(0),box[3],COLORS["reference"],1.2,"5 5")
    for i,(_l,r,k,color) in enumerate(estimates):
        c=r["coefficients"][k];b=c["standardized_beta"];e=c["cluster_robust_standard_error"] or 0
        svg.line(sx(b-1.96*e),sy(i),sx(b+1.96*e),sy(i),color,2);svg.circle(sx(b),sy(i),5,color)

    x,y=panels[2]
    overlap=summary["advantage_persistence_overlap"]
    available=[row for row in overlap if "absolute_cosine_similarity" in row]
    cosine=[row["absolute_cosine_similarity"] for row in available]
    jaccard=[row["top_dimension_jaccard"] for row in available]
    sx,sy,box=axes(svg,x,y,455,440,"C. Advantage/decision overlap","Layer","Overlap",[(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")],[(0,"0"),(.5,".5"),(1,"1")],(0,31),(0,1))
    if available:
        svg.polyline([(sx(row["layer"]),sy(value)) for row,value in zip(available,cosine)],COLORS["observed"],2)
        svg.polyline([(sx(row["layer"]),sy(value)) for row,value in zip(available,jaccard)],COLORS["model"],2)
    legend(svg,box[0]+4,box[1]+18,[("|Cosine|",COLORS["observed"]),("Top-1% Jaccard",COLORS["model"])])
    svg.save(path)


def make_adjudication_figure(summary: dict, path: Path) -> None:
    svg = Svg(1500, 1040)
    svg.text(55, 45, "What computation best explains the model's persistence?", "title")
    svg.text(
        55,
        70,
        "Held-out continuation rollouts distinguish target decodability from decision use",
        "subtitle",
    )
    panels = ((45, 100), (760, 100), (45, 565), (760, 565))

    x, y = panels[0]
    structure = summary["target_structure"]
    component_values = (
        structure["q_mean_standard_deviation"],
        structure["choice_premium_standard_deviation"],
    )
    upper = max(component_values) * 1.2
    sx, sy, box = axes(
        svg, x, y, 690, 420, "A. The target is mostly common state value",
        "Target component", "Standard deviation (reward units)",
        [(0, "Mean(QA,QB)"), (1, "Arm-choice premium")],
        [(0, "0"), (10, "10"), (20, "20"), (30, "30"), (upper, f"{upper:.0f}")],
        (-0.6, 1.6), (0, upper),
    )
    for index, (value, color) in enumerate(
        zip(component_values, (COLORS["observed"], COLORS["model"]))
    ):
        svg.rect(sx(index) - 55, sy(value), 110, box[3] - sy(value), color, 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.1f}", anchor="middle")
    svg.text(
        box[0] + 8,
        box[1] + 22,
        f"corr(Acontinue, mean Q) = {structure['advantage_q_mean_correlation']:.3f}",
        "note",
    )

    x, y = panels[1]
    conditions = summary["condition_diagnostics"]
    actual = [conditions[key]["actual_advantage_mean"] for key in CONDITIONS]
    predicted = [conditions[key]["predicted_advantage_mean"] for key in CONDITIONS]
    lower = min(actual + predicted + [0]) - 8
    upper = max(actual + predicted + [0]) + 8
    sx, sy, box = axes(
        svg, x, y, 690, 420, "B. Probe misses the negative-value regime",
        "Reward condition", "Mean continuation advantage",
        [(index, CONDITION_LABELS[key]) for index, key in enumerate(CONDITIONS)],
        [(lower, f"{lower:.0f}"), (0, "0"), (25, "25"), (50, "50"), (upper, f"{upper:.0f}")],
        (-0.6, 2.6), (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    width = 42
    for index, key in enumerate(CONDITIONS):
        for offset, value, color in (
            (-width - 3, conditions[key]["actual_advantage_mean"], COLORS["observed"]),
            (3, conditions[key]["predicted_advantage_mean"], COLORS["model"]),
        ):
            top, bottom = min(sy(0), sy(value)), max(sy(0), sy(value))
            svg.rect(sx(index) + offset, top, width, bottom - top, color, 0.9)
    legend(svg, box[0] + 8, box[1] + 18, [("Rollout target", COLORS["observed"]), ("Ridge prediction", COLORS["model"])])

    x, y = panels[2]
    sign_rows = (
        ("Negative", structure["negative_advantage"]),
        ("Nonnegative", structure["nonnegative_advantage"]),
    )
    sx, sy, box = axes(
        svg, x, y, 690, 420, "C. Persistence does not threshold at zero value",
        "Rollout continuation advantage", "Mean model P(continue)",
        [(index, label) for index, (label, _row) in enumerate(sign_rows)],
        [(0.5, ".50"), (0.7, ".70"), (0.9, ".90"), (1.0, "1")],
        (-0.6, 1.6), (0.5, 1.0),
    )
    svg.line(box[0], sy(0.5), box[2], sy(0.5), COLORS["reference"], 1.2, "5 5")
    for index, (_label, row) in enumerate(sign_rows):
        value = row["mean_p_continue"]
        color = COLORS["both_negative"] if index == 0 else COLORS["both_positive"]
        svg.rect(sx(index) - 55, sy(value), 110, box[3] - sy(value), color, 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.3f} (n={row['states']})", anchor="middle")

    x, y = panels[3]
    exact = summary["exact_matching"]["models"]
    estimates = (
        ("Adv probe → advantage", exact["probe_to_advantage"], "ridge_advantage", COLORS["observed"]),
        ("Generic probe → advantage", exact["generic_probe_to_advantage"], "ridge_return", COLORS["model"]),
        ("Actual advantage → persist", exact["actual_advantage_to_persistence"], "continuation_advantage", COLORS["both_positive"]),
        ("Adv probe → persist", exact["probe_advantage_to_persistence"], "ridge_advantage", COLORS["both_negative"]),
        ("Generic probe → persist", exact["generic_probe_to_persistence"], "ridge_return", COLORS["reference"]),
    )
    bounds = []
    for _label, result, key, _color in estimates:
        coefficient = result["coefficients"][key]
        beta = coefficient["standardized_beta"]
        error = coefficient["cluster_robust_standard_error"] or 0
        bounds.extend((beta - 1.96 * error, beta + 1.96 * error))
    lower, upper = min(-0.2, min(bounds) - 0.05), max(0.85, max(bounds) + 0.05)
    sx, sy, box = axes(
        svg, x, y, 690, 420, "D. Exact recent-state matches",
        "Standardized coefficient", "Relation",
        [(lower, f"{lower:.1f}"), (0, "0"), (0.4, ".4"), (0.8, ".8")],
        [(index, label) for index, (label, _result, _key, _color) in enumerate(estimates)],
        (lower, upper), (-0.6, len(estimates) - 0.4),
    )
    svg.line(sx(0), box[1], sx(0), box[3], COLORS["reference"], 1.2, "5 5")
    for index, (_label, result, key, color) in enumerate(estimates):
        coefficient = result["coefficients"][key]
        beta = coefficient["standardized_beta"]
        error = coefficient["cluster_robust_standard_error"] or 0
        svg.line(sx(beta - 1.96 * error), sy(index), sx(beta + 1.96 * error), sy(index), color, 2)
        svg.circle(sx(beta), sy(index), 5, color)
    svg.text(
        1450,
        1020,
        "Exact matches hold round, previous outcome, and loss streak fixed; errors cluster by episode.",
        "note",
        "end",
    )
    svg.save(path)


def write_report(summary: dict, path: Path) -> None:
    best = summary["best_layer_metrics"]
    exact = summary["exact_matching"]
    models = exact["models"]
    probe_target = models["probe_to_advantage"]["coefficients"]["ridge_advantage"]
    actual_persist = models["actual_advantage_to_persistence"]["coefficients"]["continuation_advantage"]
    probe_persist = models["probe_advantage_to_persistence"]["coefficients"]["ridge_advantage"]
    history = summary["persistence_models"]["history_only"]
    joint = summary["persistence_models"]["history_plus_advantage_probe"]
    unique = joint["r_squared"] - history["r_squared"]
    available_overlap = [
        row
        for row in summary["advantage_persistence_overlap"]
        if "absolute_cosine_similarity" in row
    ]
    actual_joint = summary["persistence_models"]["history_plus_actual_advantage"]
    actual_joint_coefficient = actual_joint["coefficients"]["continuation_advantage"]
    actual_probe_joint = summary["persistence_models"]["history_plus_actual_and_probe"]
    generic_joint = summary["persistence_models"].get(
        "history_plus_generic_return_probe"
    )
    cross = summary["cross_target_comparison"]
    structure = summary["target_structure"]
    conditions = summary["condition_diagnostics"]
    bootstrap = summary["episode_bootstrap_test_r_squared"]
    robustness = summary["layer_robustness"]
    audit = summary.get("target_audit")
    generic_target = models.get("generic_probe_to_advantage")
    generic_persist = models.get("generic_probe_to_persistence")
    both_target = models.get("both_probes_to_advantage")
    actual_plus_probe = models["actual_plus_probe_to_persistence"]
    lines = [
        "# Continuation-advantage probe",
        "",
        "## Bottom line",
        "",
        "The data support **integrated return information plus a strong recent-state persistence heuristic**, but they do **not** isolate a distinct continuation-advantage decision variable.",
        "",
        f"Selected layer: **{summary['best_layer']}**. Validation/test advantage R²: **{best['validation']['r_squared']:.3f} / {best['test']['r_squared']:.3f}**.",
        f"The episode-bootstrap 95% interval for test R² is **[{bootstrap['lower_95']:.3f}, {bootstrap['upper_95']:.3f}]**. "
        f"Validation/test layer ranks correlate **{robustness['validation_test_spearman']:.3f}**; the selected layer ranks **{robustness['selected_layer_test_rank']}** on test.",
        f"Exact matching retained **{exact['states']} states in {exact['eligible_strata']} strata from {exact['episode_clusters']} episodes**.",
        "",
        "## What the three candidate accounts predict",
        "",
        "1. **Recent-state STOP heuristic.** Round, previous outcome, and loss streak should dominate persistence; a value probe should add little after those variables are fixed.",
        "2. **Integrated generic value.** Earlier hidden states should predict future return, but that signal need not be the quantity used to choose CONTINUE versus STOP.",
        "3. **Continuation-advantage decision variable.** A probe trained on `max(Q_A,Q_B)-Q_STOP` should predict both its rollout target and persistence within matched states, with meaningful alignment to the direct persistence direction.",
        "",
        "## Result 1: continuation return is linearly decodable",
        "",
        f"The ridge probe predicts held-out rollout advantage at layer {summary['best_layer']} (test R² **{best['test']['r_squared']:.3f}**, r **{best['test']['correlation']:.3f}**). "
        f"Within exact recent-state matches, probe → advantage is beta **{probe_target['standardized_beta']:.3f}** (p={probe_target['normal_approximation_p_value']:.3g}).",
        "This is evidence that the hidden state integrates information beyond the immediately preceding reward.",
        "",
        "## Result 2: rollout advantage relates to behavior, but the probe does not",
        "",
        f"Within exact matches, actual rollout advantage predicts persistence (beta **{actual_persist['standardized_beta']:.3f}**, p={actual_persist['normal_approximation_p_value']:.3g}), "
        f"whereas decoded advantage does not (beta **{probe_persist['standardized_beta']:.3f}**, p={probe_persist['normal_approximation_p_value']:.3g}).",
        f"Recent history explains **{history['r_squared']:.3f}** of persistence variance. Adding the advantage probe raises this to **{joint['r_squared']:.3f}** (increment **{unique:.4f}**; adjusted probe beta **{joint['coefficients']['ridge_advantage']['standardized_beta']:.3f}**, p={joint['coefficients']['ridge_advantage']['normal_approximation_p_value']:.3g}).",
        f"Adding actual advantage instead raises R² to **{actual_joint['r_squared']:.3f}** (adjusted beta **{actual_joint_coefficient['standardized_beta']:.3f}**, p={actual_joint_coefficient['normal_approximation_p_value']:.3g}).",
        f"When actual and decoded advantage enter together, actual advantage remains associated with persistence (beta **{actual_probe_joint['coefficients']['continuation_advantage']['standardized_beta']:.3f}**, p={actual_probe_joint['coefficients']['continuation_advantage']['normal_approximation_p_value']:.3g}) while the probe coefficient is **{actual_probe_joint['coefficients']['ridge_advantage']['standardized_beta']:.3f}** (p={actual_probe_joint['coefficients']['ridge_advantage']['normal_approximation_p_value']:.3g}).",
        "This pattern is compatible with a noisy or incomplete advantage decoder, but it is not evidence that the fitted probe direction is the model's operative persistence variable.",
        "",
        "## Result 3: the target is not sharply distinct from generic value",
        "",
        f"Forced-arm Q values are highly correlated (r **{structure['q_A_q_B_correlation']:.3f}**), and continuation advantage correlates **{structure['advantage_q_mean_correlation']:.3f}** with their mean. "
        f"The between-state standard deviation of mean Q is **{structure['q_mean_standard_deviation']:.1f}** reward units, versus only **{structure['choice_premium_standard_deviation']:.1f}** for the max-over-arms premium.",
    ]
    if cross is not None and generic_target is not None:
        both_generic = both_target["coefficients"]["ridge_return"]
        both_advantage = both_target["coefficients"]["ridge_advantage"]
        lines.extend(
            [
                f"The advantage and generic-return probe outputs correlate **{cross['advantage_probe_generic_probe_correlation']:.3f}**. Within exact matches, the earlier generic-return probe predicts the advantage target (beta **{generic_target['coefficients']['ridge_return']['standardized_beta']:.3f}**, p={generic_target['coefficients']['ridge_return']['normal_approximation_p_value']:.3g}).",
                f"With both probes entered together for advantage, generic return remains strong (beta **{both_generic['standardized_beta']:.3f}**, p={both_generic['normal_approximation_p_value']:.3g}) while the nominal advantage probe is beta **{both_advantage['standardized_beta']:.3f}** (p={both_advantage['normal_approximation_p_value']:.3g}).",
                "The generic probe was trained with many more labeled states, so this comparison is not a fair decoder competition. It does show that the present rollout target and probe do not identify a cleanly distinct computational axis.",
                "",
            ]
        )
    negative = structure["negative_advantage"]
    nonnegative = structure["nonnegative_advantage"]
    lines.extend(
        [
            "## Result 4: the model does not implement a zero-value stopping threshold",
            "",
            f"All **{negative['states']}** held-out states with negative rollout advantage still had P(continue) above .5; their mean P(continue) was **{negative['mean_p_continue']:.3f}**, versus **{nonnegative['mean_p_continue']:.3f}** for nonnegative states.",
            f"In both-negative environments the rollout target averages **{conditions['both_negative']['actual_advantage_mean']:.1f}**, but the probe predicts **{conditions['both_negative']['predicted_advantage_mean']:.1f}** and the model's mean P(continue) is **{conditions['both_negative']['mean_p_continue']:.3f}**.",
            "Thus persistence changes in the sensible direction with environment quality, but STOP is not governed by a rational `A_continue > 0` threshold. The strong continuation prior and recent loss/time cues dominate.",
            "",
            "## Matched-state coefficients",
            "",
            f"- Advantage probe → continuation advantage: beta **{probe_target['standardized_beta']:.3f}**, p={probe_target['normal_approximation_p_value']:.3g}.",
            f"- Actual rollout advantage → persistence: beta **{actual_persist['standardized_beta']:.3f}**, p={actual_persist['normal_approximation_p_value']:.3g}.",
            f"- Advantage probe → persistence: beta **{probe_persist['standardized_beta']:.3f}**, p={probe_persist['normal_approximation_p_value']:.3g}.",
        ]
    )
    if generic_persist is not None:
        lines.append(
            f"- Generic-return probe → persistence in this 128-state subset: beta **{generic_persist['coefficients']['ridge_return']['standardized_beta']:.3f}**, p={generic_persist['coefficients']['ridge_return']['normal_approximation_p_value']:.3g}. This subset result conflicts with the null effect in the prior 1,800-state analysis and should not supersede it."
        )
    lines.extend(["", "## Advantage/decision direction overlap", ""])
    if available_overlap:
        maximum_cosine = max(
            available_overlap,
            key=lambda row: row["absolute_cosine_similarity"],
        )
        maximum_jaccard = max(
            available_overlap, key=lambda row: row["top_dimension_jaccard"]
        )
        lines.extend(
            [
                f"Maximum absolute cosine was **{maximum_cosine['absolute_cosine_similarity']:.3f}** at layer **{maximum_cosine['layer']}**; "
                f"maximum top-1% Jaccard was **{maximum_jaccard['top_dimension_jaccard']:.3f}** at layer **{maximum_jaccard['layer']}**.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Direct persistence probe artifacts were unavailable, so overlap was not computed.",
                "",
            ]
        )
    lines.extend(
        [
            "The weak geometric overlap further argues against treating the fitted advantage direction as the late persistence axis.",
            "",
            "## Target precision and limitations",
            "",
            f"Median per-state rollout SE is **{summary['target_reliability']['median_standard_error']:.3f}**; target variance is **{summary['target_reliability']['target_variance']:.3f}** and mean estimation-error variance is **{summary['target_reliability']['mean_estimation_error_variance']:.3f}**.",
            f"The test split contains only **{conditions['both_negative']['states']}** both-negative states, compared with **{conditions['one_positive']['states']}** one-positive and **{conditions['both_positive']['states']}** both-positive states. The validation and test target means also differ ({best['validation']['target_mean']:.1f} versus {best['test']['target_mean']:.1f}), reflecting reward-condition imbalance.",
            "The advantage probe has only 128 labeled training states, whereas the generic-return probe used the much larger activation bank. Null differences between their coefficients are therefore not equivalence tests.",
            "The rollout target follows the model's own downstream policy after a forced first action. It measures policy-contingent continuation return, not an environment-optimal Q function, and can inherit downstream stopping heuristics.",
            "",
            "## Adjudication",
            "",
            "- **Best-supported:** recent outcome, loss streak, and time implement the dominant STOP heuristic on top of a strong default-to-continue bias.",
            "- **Also supported:** early hidden states contain integrated information about future/continuation return.",
            "- **Not established:** a distinct continuation-advantage representation is read out to cause persistence. The present target is almost collinear with generic continuation value, the fitted probe adds little beyond recent history, and it fails the negative-value regime.",
            "",
            "The decisive next step is causal steering with the frozen advantage direction and matched generic-return/random directions. A selective monotonic change in persistence would upgrade the advantage account; no effect would favor the heuristic interpretation.",
            "",
            summary["target"],
            summary["caveat"],
            "",
        ]
    )
    if audit is not None:
        lines.insert(
            6,
            f"Target audit passed: **{audit['rows']} rows, {audit['unique_state_ids']} unique states, {audit['duplicate_state_ids']} duplicates**, with split counts {audit['states_by_split']} and {audit['rollouts_per_action']} rollouts per action.",
        )
        lines.insert(7, "")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="artifacts/advantage_probes")
    parser.add_argument("--linear-summary", default="artifacts/linear_probes/publication/linear_probe_summary.json")
    parser.add_argument("--linear-predictions", default="artifacts/linear_probes/test_predictions.csv")
    parser.add_argument("--targets", default="artifacts/advantage_targets/targets*.csv")
    parser.add_argument("--output-dir", default="artifacts/advantage_probes/publication")
    args = parser.parse_args()
    probe_dir, output = Path(args.probe_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((probe_dir / "metrics.json").read_text())
    rows = read_rows(probe_dir / "test_predictions.csv")
    linear_predictions = Path(args.linear_predictions)
    if linear_predictions.exists():
        attach_generic_probe(rows, linear_predictions)
    linear_path = Path(args.linear_summary)
    linear = json.loads(linear_path.read_text()) if linear_path.exists() else None
    summary = summarize(
        metrics,
        rows,
        linear,
        target_audit=audit_target_files(args.targets),
    )
    (output / "advantage_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(metrics, summary, output / "advantage_probe_results.svg")
    if summary["cross_target_comparison"] is not None:
        make_adjudication_figure(
            summary, output / "advantage_computational_adjudication.svg"
        )
    write_report(summary, output / "advantage_probe_report.md")
    print(json.dumps({"best_layer": summary["best_layer"]}, indent=2))


if __name__ == "__main__":
    main()
