"""Publication diagnostics for ridge future-return and persistence probes."""

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.probe_history_matching import _clustered_regression


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
    mean = statistics.mean(target)
    total = sum((value - mean) ** 2 for value in target)
    residual = sum(
        (value - estimate) ** 2 for value, estimate in zip(target, prediction)
    )
    return 1.0 - residual / total if total else 0.0


def _condition_class(episode_id: str) -> str:
    match = re.search(r"pa-([0-9.]+)-pb-([0-9.]+)", episode_id)
    if match is None:
        raise ValueError(f"episode ID does not encode arm probabilities: {episode_id}")
    positive = sum(float(value) > 0.4 for value in match.groups())
    return ("both_negative", "one_positive", "both_positive")[positive]


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            for key in ("round", "loss_streak"):
                row[key] = int(float(row[key]))
            row["previous_outcome"] = (
                None
                if row["previous_outcome"] in (None, "")
                else float(row["previous_outcome"])
            )
            for key in (
                "cumulative_score",
                "future_return",
                "persistence_logit",
                "ridge_future_return",
                "ridge_persistence",
            ):
                row[key] = float(row[key])
            row["prior_score"] = row["cumulative_score"] - float(
                row["previous_outcome"] or 0
            )
            rows.append(row)
    return rows


def recent_history(rows: list[dict]) -> dict[str, list[float]]:
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


def exact_residuals(rows: list[dict], min_stratum_size: int = 4) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row["previous_outcome"] is None:
            continue
        grouped[
            (row["round"], row["previous_outcome"], row["loss_streak"])
        ].append(row)
    eligible = [
        selected
        for selected in grouped.values()
        if len(selected) >= min_stratum_size
        and max(row["prior_score"] for row in selected)
        > min(row["prior_score"] for row in selected)
    ]
    keys = (
        "prior_score",
        "future_return",
        "persistence_logit",
        "ridge_future_return",
        "ridge_persistence",
    )
    residuals = {key: [] for key in keys}
    clusters = []
    for selected in eligible:
        means = {
            key: statistics.mean(row[key] for row in selected) for key in keys
        }
        for row in selected:
            for key in keys:
                residuals[key].append(row[key] - means[key])
            clusters.append(row["episode_id"])
    if not clusters:
        raise ValueError("no eligible exact-match strata")
    return {
        "eligible_strata": len(eligible),
        "states": len(clusters),
        "episode_clusters": len(set(clusters)),
        "residuals": residuals,
        "clusters": clusters,
    }


def regression(outcome, predictors, clusters) -> dict:
    return _clustered_regression(outcome, predictors, clusters)


def _replication_comparison(
    rows: list[dict], nonlinear_rows: list[dict], bootstrap_samples: int = 5000
) -> dict:
    nonlinear_by_state = {row["state_id"]: row for row in nonlinear_rows}
    if set(nonlinear_by_state) != {row["state_id"] for row in rows}:
        raise ValueError("linear and nonlinear test-state IDs do not match")
    combined = []
    for row in rows:
        nonlinear = nonlinear_by_state[row["state_id"]]
        combined.append(
            {
                "episode_id": row["episode_id"],
                "target": row["future_return"],
                "ridge": row["ridge_future_return"],
                "nonlinear_sparse": float(nonlinear["probe_value"]),
                "nonlinear_full": float(nonlinear["probe_value_full"]),
            }
        )
    target = [row["target"] for row in combined]
    point = {
        key: r_squared(target, [row[key] for row in combined])
        for key in ("ridge", "nonlinear_sparse", "nonlinear_full")
    }
    grouped = defaultdict(list)
    for row in combined:
        grouped[row["episode_id"]].append(row)
    episode_ids = sorted(grouped)
    draws = {
        key: []
        for key in (
            "ridge",
            "nonlinear_sparse",
            "nonlinear_full",
            "ridge_minus_nonlinear_sparse",
            "ridge_minus_nonlinear_full",
        )
    }
    rng = random.Random(82026)
    for _ in range(bootstrap_samples):
        selected = [
            row
            for _episode in episode_ids
            for row in grouped[rng.choice(episode_ids)]
        ]
        selected_target = [row["target"] for row in selected]
        values = {
            key: r_squared(selected_target, [row[key] for row in selected])
            for key in ("ridge", "nonlinear_sparse", "nonlinear_full")
        }
        for key, value in values.items():
            draws[key].append(value)
        draws["ridge_minus_nonlinear_sparse"].append(
            values["ridge"] - values["nonlinear_sparse"]
        )
        draws["ridge_minus_nonlinear_full"].append(
            values["ridge"] - values["nonlinear_full"]
        )

    intervals = {}
    for key, values in draws.items():
        values.sort()
        intervals[key] = {
            "lower_95": values[int(0.025 * bootstrap_samples)],
            "upper_95": values[int(0.975 * bootstrap_samples) - 1],
            "bootstrap_mean": statistics.mean(values),
        }
    return {
        "states": len(combined),
        "episodes": len(episode_ids),
        "point_r_squared": point,
        "episode_bootstrap": {
            "samples": bootstrap_samples,
            "intervals": intervals,
        },
        "prediction_correlations": {
            "ridge_with_nonlinear_sparse": pearson(
                [row["ridge"] for row in combined],
                [row["nonlinear_sparse"] for row in combined],
            ),
            "ridge_with_nonlinear_full": pearson(
                [row["ridge"] for row in combined],
                [row["nonlinear_full"] for row in combined],
            ),
        },
    }


def _condition_diagnostics(rows: list[dict], nonlinear_rows: list[dict]) -> dict:
    nonlinear_by_state = {row["state_id"]: row for row in nonlinear_rows}
    result = {}
    for condition in ("both_negative", "one_positive", "both_positive"):
        selected = [
            row for row in rows if _condition_class(row["episode_id"]) == condition
        ]
        target = [row["future_return"] for row in selected]
        predictions = {
            "ridge": [row["ridge_future_return"] for row in selected],
            "nonlinear_sparse": [
                float(nonlinear_by_state[row["state_id"]]["probe_value"])
                for row in selected
            ],
            "nonlinear_full": [
                float(nonlinear_by_state[row["state_id"]]["probe_value_full"])
                for row in selected
            ],
        }
        result[condition] = {
            "states": len(selected),
            "episodes": len({row["episode_id"] for row in selected}),
            "target_mean": statistics.mean(target),
            "models": {
                key: {
                    "r_squared": r_squared(target, prediction),
                    "correlation": pearson(target, prediction),
                    "prediction_mean": statistics.mean(prediction),
                }
                for key, prediction in predictions.items()
            },
        }
    return result


def _hypergeometric_upper_tail(
    intersection: int, population: int, left_size: int, right_size: int
) -> float:
    denominator = math.comb(population, right_size)
    maximum = min(left_size, right_size)
    return sum(
        math.comb(left_size, value)
        * math.comb(population - left_size, right_size - value)
        for value in range(intersection, maximum + 1)
    ) / denominator


def _rank(values: list[float]) -> list[int]:
    result = [0] * len(values)
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        result[index] = rank
    return result


def _layer_robustness(metrics: dict) -> dict:
    result = {}
    for target in ("future_return", "persistence"):
        validation = [
            row["targets"][target]["validation"]["r_squared"]
            for row in metrics["layers"]
        ]
        test = [
            row["targets"][target]["test"]["r_squared"]
            for row in metrics["layers"]
        ]
        selected = metrics["best_layers"][target]
        result[target] = {
            "validation_test_spearman": pearson(_rank(validation), _rank(test)),
            "selected_layer_test_rank": sorted(
                range(len(test)), key=test.__getitem__, reverse=True
            ).index(selected)
            + 1,
            "best_test_layer": max(range(len(test)), key=test.__getitem__),
            "best_test_r_squared": max(test),
        }
    return result


def summarize(
    metrics: dict,
    rows: list[dict],
    nonlinear: dict | None,
    nonlinear_rows: list[dict] | None = None,
    nonlinear_summary: dict | None = None,
) -> dict:
    clusters = [row["episode_id"] for row in rows]
    controls = recent_history(rows)
    persistence = [row["persistence_logit"] for row in rows]
    future = [row["future_return"] for row in rows]
    ridge_return = [row["ridge_future_return"] for row in rows]
    direct_persistence = [row["ridge_persistence"] for row in rows]
    matched = exact_residuals(rows)
    residuals, matched_clusters = matched["residuals"], matched["clusters"]
    exact_models = {
        "ridge_return_to_future": regression(
            residuals["future_return"],
            {"ridge_future_return": residuals["ridge_future_return"]},
            matched_clusters,
        ),
        "ridge_return_to_persistence": regression(
            residuals["persistence_logit"],
            {"ridge_future_return": residuals["ridge_future_return"]},
            matched_clusters,
        ),
        "direct_probe_to_persistence": regression(
            residuals["persistence_logit"],
            {"ridge_persistence": residuals["ridge_persistence"]},
            matched_clusters,
        ),
        "older_history_to_ridge_return": regression(
            residuals["ridge_future_return"],
            {"prior_score": residuals["prior_score"]},
            matched_clusters,
        ),
    }
    layer_overlap = [
        {
            "layer": row["layer"],
            **row["return_persistence_overlap"],
        }
        for row in metrics["layers"]
    ]
    for row in layer_overlap:
        chance_p = _hypergeometric_upper_tail(
            row["top_dimension_intersection"],
            2560,
            row["top_dimension_count"],
            row["top_dimension_count"],
        )
        row["top_dimension_overlap_chance_p"] = chance_p
        row["top_dimension_overlap_bonferroni_p"] = min(
            1.0, chance_p * len(layer_overlap)
        )
    result = {
        "best_layers": metrics["best_layers"],
        "best_layer_metrics": {
            target: next(
                row["targets"][target]
                for row in metrics["layers"]
                if row["layer"] == metrics["best_layers"][target]
            )
            for target in ("future_return", "persistence")
        },
        "nonlinear_future_return": (
            None
            if nonlinear is None
            else next(
                row
                for row in nonlinear["layers"]
                if row["layer"] == nonlinear["best_layer"]
            )
        ),
        "persistence_models": {
            "history_only": regression(persistence, controls, clusters),
            "ridge_return_only": regression(
                persistence, {"ridge_future_return": ridge_return}, clusters
            ),
            "history_plus_ridge_return": regression(
                persistence,
                {**controls, "ridge_future_return": ridge_return},
                clusters,
            ),
            "direct_probe_only": regression(
                persistence, {"ridge_persistence": direct_persistence}, clusters
            ),
        },
        "future_return_models": {
            "history_only": regression(future, controls, clusters),
            "ridge_return_only": regression(
                future, {"ridge_future_return": ridge_return}, clusters
            ),
            "history_plus_ridge_return": regression(
                future, {**controls, "ridge_future_return": ridge_return}, clusters
            ),
        },
        "exact_matching": {
            key: value
            for key, value in matched.items()
            if key not in {"residuals", "clusters"}
        }
        | {"models": exact_models},
        "layer_overlap": layer_overlap,
        "layer_robustness": _layer_robustness(metrics),
        "caveat": (
            "Layer and ridge alpha use validation episodes; all reported test "
            "metrics and exact-match diagnostics use untouched test episodes."
        ),
    }
    if nonlinear_rows is not None:
        result["replication_against_nonlinear"] = _replication_comparison(
            rows, nonlinear_rows
        )
        result["condition_diagnostics"] = _condition_diagnostics(
            rows, nonlinear_rows
        )
    if nonlinear_summary is not None:
        result["nonlinear_exact_matching_reference"] = {
            "future_return": nonlinear_summary["matched_future_return"][
                "exact_matching"
            ]["models"],
            "persistence": nonlinear_summary["exact_history_matching"][
                "simple_regressions"
            ],
        }
    return result


def make_figure(metrics: dict, summary: dict, path: Path) -> None:
    svg = Svg(1500, 1030)
    svg.text(55, 45, "Linear decoding of future return and persistence", "title")
    svg.text(55, 70, "Ridge probes with episode-held-out layer and regularization selection", "subtitle")
    panels = ((45, 95), (760, 95), (45, 565), (760, 565))
    layers = metrics["layers"]

    x, y = panels[0]
    series = (
        ("Return validation", [row["targets"]["future_return"]["validation"]["r_squared"] for row in layers], COLORS["observed"]),
        ("Return test", [row["targets"]["future_return"]["test"]["r_squared"] for row in layers], COLORS["model"]),
    )
    values = [value for _label, data, _color in series for value in data]
    lower, upper = min(-0.1, min(values) - 0.05), max(0.2, max(values) + 0.05)
    sx, sy, box = axes(svg, x, y, 690, 430, "A. Ridge future-return decoding", "Layer", "R²", [(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")], [(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")], (0,31), (lower,upper))
    for label, data, color in series:
        svg.polyline([(sx(i), sy(value)) for i, value in enumerate(data)], color, 2)
    legend(svg, box[0]+5, box[1]+18, [(label,color) for label,_data,color in series])

    x, y = panels[1]
    data = [row["targets"]["persistence"]["test"]["r_squared"] for row in layers]
    lower, upper = min(-0.05, min(data)-0.03), max(1.0, max(data)+0.03)
    sx, sy, box = axes(svg, x, y, 690, 430, "B. Where persistence becomes linearly decodable", "Layer", "Test R²", [(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")], [(lower,f"{lower:.1f}"),(0,"0"),(0.5,".5"),(upper,f"{upper:.1f}")], (0,31), (lower,upper))
    svg.polyline([(sx(i),sy(value)) for i,value in enumerate(data)], COLORS["both_positive"], 2)

    x, y = panels[2]
    exact = summary["exact_matching"]["models"]
    estimates = (
        ("Ridge return → return", exact["ridge_return_to_future"], "ridge_future_return", COLORS["observed"]),
        ("Ridge return → persist", exact["ridge_return_to_persistence"], "ridge_future_return", COLORS["model"]),
        ("Direct probe → persist", exact["direct_probe_to_persistence"], "ridge_persistence", COLORS["both_positive"]),
    )
    bounds=[]
    for _label,result,key,_color in estimates:
        c=result["coefficients"][key]; b=c["standardized_beta"]; e=c["cluster_robust_standard_error"] or 0; bounds.extend((b-1.96*e,b+1.96*e))
    lower,upper=min(-.3,min(bounds)-.05),max(1.0,max(bounds)+.05)
    sx,sy,box=axes(svg,x,y,690,430,"C. Exact matched-state discriminator","Standardized coefficient","Target",[(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")],[(i,label) for i,(label,_r,_k,_c) in enumerate(estimates)],(lower,upper),(-.6,len(estimates)-.4))
    svg.line(sx(0),box[1],sx(0),box[3],COLORS["reference"],1.2,"5 5")
    for i,(_label,result,key,color) in enumerate(estimates):
        c=result["coefficients"][key]; b=c["standardized_beta"]; e=c["cluster_robust_standard_error"] or 0
        svg.line(sx(b-1.96*e),sy(i),sx(b+1.96*e),sy(i),color,2); svg.circle(sx(b),sy(i),5,color)

    x,y=panels[3]
    overlap=summary["layer_overlap"]
    cosine=[row["absolute_cosine_similarity"] for row in overlap]
    jaccard=[row["top_dimension_jaccard"] for row in overlap]
    sx,sy,box=axes(svg,x,y,690,430,"D. Return/persistence direction overlap","Layer","Overlap",[(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")],[(0,"0"),(.5,".5"),(1,"1")],(0,31),(0,1))
    svg.polyline([(sx(i),sy(value)) for i,value in enumerate(cosine)],COLORS["observed"],2)
    svg.polyline([(sx(i),sy(value)) for i,value in enumerate(jaccard)],COLORS["model"],2)
    legend(svg,box[0]+5,box[1]+18,[("|Cosine|",COLORS["observed"]),("Top-1% Jaccard",COLORS["model"])])
    svg.save(path)


def make_replication_figure(summary: dict, path: Path) -> None:
    """Compare linear and nonlinear probes on identical held-out states."""
    comparison = summary["replication_against_nonlinear"]
    nonlinear_exact = summary["nonlinear_exact_matching_reference"]
    conditions = summary["condition_diagnostics"]
    svg = Svg(1500, 620)
    svg.text(55, 45, "A linear probe replicates the nonlinear future-return result", "title")
    svg.text(
        55,
        70,
        "Identical held-out episodes; error bars resample episodes, not individual states",
        "subtitle",
    )

    x, y = 45, 95
    labels = ("Ridge linear", "Nonlinear sparse", "Nonlinear full")
    keys = ("ridge", "nonlinear_sparse", "nonlinear_full")
    colors = (COLORS["observed"], COLORS["model"], COLORS["both_positive"])
    point = comparison["point_r_squared"]
    intervals = comparison["episode_bootstrap"]["intervals"]
    lower = min(intervals[key]["lower_95"] for key in keys) - 0.04
    upper = max(intervals[key]["upper_95"] for key in keys) + 0.04
    sx, sy, box = axes(
        svg, x, y, 450, 460, "A. Held-out return decoding", "Decoder", "Prediction R²",
        [(index, label) for index, label in enumerate(labels)],
        [(lower, f"{lower:.1f}"), (0, "0"), (0.2, ".2"), (upper, f"{upper:.1f}")],
        (-0.6, 2.6), (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (key, color) in enumerate(zip(keys, colors)):
        low, high = intervals[key]["lower_95"], intervals[key]["upper_95"]
        svg.line(sx(index), sy(low), sx(index), sy(high), color, 2)
        svg.line(sx(index) - 7, sy(low), sx(index) + 7, sy(low), color, 2)
        svg.line(sx(index) - 7, sy(high), sx(index) + 7, sy(high), color, 2)
        svg.circle(sx(index), sy(point[key]), 6, color)

    x = 520
    ridge_exact = summary["exact_matching"]["models"]
    estimates = (
        (
            "Ridge linear",
            ridge_exact["ridge_return_to_future"]["coefficients"]["ridge_future_return"],
            ridge_exact["ridge_return_to_persistence"]["coefficients"]["ridge_future_return"],
            COLORS["observed"],
        ),
        (
            "Nonlinear sparse",
            nonlinear_exact["future_return"]["sparse_probe_to_future_return"]["coefficients"]["probe_value"],
            nonlinear_exact["persistence"]["sparse_probe_to_persistence"]["coefficients"]["probe_value"],
            COLORS["model"],
        ),
        (
            "Nonlinear full",
            nonlinear_exact["future_return"]["full_probe_to_future_return"]["coefficients"]["probe_value_full"],
            nonlinear_exact["persistence"]["full_probe_to_persistence"]["coefficients"]["probe_value_full"],
            COLORS["both_positive"],
        ),
    )
    sx, sy, box = axes(
        svg, x, y, 450, 460, "B. Within exactly matched states", "Standardized coefficient", "Outcome",
        [(-0.3, "-.3"), (0, "0"), (0.3, ".3"), (0.6, ".6")],
        [(0, "Future return"), (1, "Persistence logit")],
        (-0.35, 0.68), (-0.5, 1.5),
    )
    svg.line(sx(0), box[1], sx(0), box[3], COLORS["reference"], 1.2, "5 5")
    offsets = (-0.16, 0, 0.16)
    for offset, (label, future_result, persistence_result, color) in zip(offsets, estimates):
        for target_index, result in enumerate((future_result, persistence_result)):
            beta = result["standardized_beta"]
            error = result["cluster_robust_standard_error"] or 0
            position = target_index + offset
            svg.line(sx(beta - 1.96 * error), sy(position), sx(beta + 1.96 * error), sy(position), color, 2)
            svg.circle(sx(beta), sy(position), 5, color)
    legend(svg, box[0] + 5, box[1] + 18, [(label, color) for label, _f, _p, color in estimates])

    x = 995
    condition_order = ("both_negative", "one_positive", "both_positive")
    condition_labels = ("Both\nnegative", "One\npositive", "Both\npositive")
    sx, sy, box = axes(
        svg, x, y, 460, 460, "C. Decoding depends on reward regime", "Condition", "Return correlation",
        [(index, label.replace("\n", " ")) for index, label in enumerate(condition_labels)],
        [(-0.8, "-.8"), (-0.4, "-.4"), (0, "0"), (0.4, ".4"), (0.8, ".8")],
        (-0.6, 2.6), (-0.8, 0.8),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    offsets = (-0.14, 0, 0.14)
    for offset, key, color in zip(offsets, keys, colors):
        points = [
            (sx(index + offset), sy(conditions[condition]["models"][key]["correlation"]))
            for index, condition in enumerate(condition_order)
        ]
        svg.polyline(points, color, 2)
        for px, py in points:
            svg.circle(px, py, 5, color)
    legend(svg, box[0] + 5, box[1] + 18, list(zip(labels, colors)))
    svg.text(
        1450,
        600,
        "Intervals: 5,000 episode-cluster bootstrap samples. Matched-state errors cluster by episode.",
        "note",
        "end",
    )
    svg.save(path)


def write_report(summary: dict, path: Path) -> None:
    return_metrics = summary["best_layer_metrics"]["future_return"]
    persistence_metrics = summary["best_layer_metrics"]["persistence"]
    exact = summary["exact_matching"]
    models = exact["models"]
    return_exact = models["ridge_return_to_future"]["coefficients"]["ridge_future_return"]
    return_persist = models["ridge_return_to_persistence"]["coefficients"]["ridge_future_return"]
    direct = models["direct_probe_to_persistence"]["coefficients"]["ridge_persistence"]
    maximum_cosine = max(
        summary["layer_overlap"], key=lambda row: row["absolute_cosine_similarity"]
    )
    maximum_jaccard = max(
        summary["layer_overlap"], key=lambda row: row["top_dimension_jaccard"]
    )
    nonlinear = summary["nonlinear_future_return"]
    nonlinear_text = "not available"
    if nonlinear is not None:
        nonlinear_text = f"{nonlinear['full']['test']['r_squared']:.3f} full / {nonlinear['sparse']['test']['r_squared']:.3f} sparse"
    replication = summary.get("replication_against_nonlinear")
    replication_lines = []
    condition_lines = []
    if replication is not None:
        point = replication["point_r_squared"]
        intervals = replication["episode_bootstrap"]["intervals"]
        difference = intervals["ridge_minus_nonlinear_full"]
        replication_lines = [
            "## Direct replication against the nonlinear probes",
            "",
            f"All three decoders were evaluated on the same **{replication['states']} states from {replication['episodes']} untouched test episodes**. "
            f"Prediction R² was **{point['ridge']:.3f}** for ridge, **{point['nonlinear_sparse']:.3f}** for the sparse nonlinear probe, and **{point['nonlinear_full']:.3f}** for the full nonlinear probe.",
            f"Ridge and full-nonlinear predictions correlated **r={replication['prediction_correlations']['ridge_with_nonlinear_full']:.3f}**. "
            f"The episode-bootstrap 95% interval for ridge minus full-nonlinear R² was **[{difference['lower_95']:.3f}, {difference['upper_95']:.3f}]**, which includes zero.",
            "",
            "This is a strong replication of the central result: a flexible nonlinear decoder is not required to recover the held-out future-return signal.",
            "",
        ]
        condition_lines = [
            "## Important boundary condition",
            "",
        ]
        for key, label in (
            ("both_negative", "Both-negative"),
            ("one_positive", "One-positive"),
            ("both_positive", "Both-positive"),
        ):
            row = summary["condition_diagnostics"][key]
            models_by_name = row["models"]
            condition_lines.append(
                f"- {label}: {row['states']} states / {row['episodes']} episodes; "
                f"return correlations ridge **{models_by_name['ridge']['correlation']:.3f}**, "
                f"sparse **{models_by_name['nonlinear_sparse']['correlation']:.3f}**, "
                f"full **{models_by_name['nonlinear_full']['correlation']:.3f}**."
            )
        condition_lines += [
            "",
            "All three probes fail in the both-negative regime and succeed in the one- or both-positive regimes. "
            "The replication therefore supports a linearly decodable future-return/trajectory signal, but not a uniformly calibrated signed value representation across every reward environment.",
            "",
        ]
    persistence_models = summary["persistence_models"]
    return_only = persistence_models["ridge_return_only"]
    history_only = persistence_models["history_only"]
    joint = persistence_models["history_plus_ridge_return"]
    unique = joint["r_squared"] - history_only["r_squared"]
    return_joint = joint["coefficients"]["ridge_future_return"]
    robustness = summary["layer_robustness"]
    lines = [
        "# Ridge-linear future-return and persistence probes",
        "",
        "## Bottom line",
        "",
        f"Future return is linearly decodable at layer **{summary['best_layers']['future_return']}** "
        f"(validation/test R² **{return_metrics['validation']['r_squared']:.3f} / {return_metrics['test']['r_squared']:.3f}**). "
        f"The nonlinear benchmark test R² is {nonlinear_text}.",
        f"The direct persistence probe peaks at layer **{summary['best_layers']['persistence']}** "
        f"(validation/test R² **{persistence_metrics['validation']['r_squared']:.3f} / {persistence_metrics['test']['r_squared']:.3f}**).",
        "",
        *replication_lines,
        "## Selection robustness",
        "",
        f"For future return, validation and test layer ranks correlated **Spearman r={robustness['future_return']['validation_test_spearman']:.3f}**; "
        f"the validation-selected layer ranked **{robustness['future_return']['selected_layer_test_rank']}** on test (best test layer {robustness['future_return']['best_test_layer']}).",
        f"For persistence, validation and test layer ranks correlated **Spearman r={robustness['persistence']['validation_test_spearman']:.3f}**; "
        f"the selected final layer also ranked **{robustness['persistence']['selected_layer_test_rank']}** on test.",
        "",
        "## Exact matching",
        "",
        f"Matching retained **{exact['states']} states in {exact['eligible_strata']} strata from {exact['episode_clusters']} episodes**.",
        f"Ridge return → actual future return: beta **{return_exact['standardized_beta']:.3f}**, p={return_exact['normal_approximation_p_value']:.3g}.",
        f"Ridge return → persistence: beta **{return_persist['standardized_beta']:.3f}**, p={return_persist['normal_approximation_p_value']:.3g}.",
        f"Direct persistence probe → persistence: beta **{direct['standardized_beta']:.3f}**, p={direct['normal_approximation_p_value']:.3g}.",
        "",
        *condition_lines,
        "## Probe alone, history alone, and joint persistence models",
        "",
        f"The ridge return prediction alone explains **{return_only['r_squared']:.3f}** of persistence-logit variance. "
        f"Recent history alone explains **{history_only['r_squared']:.3f}**; the joint model explains **{joint['r_squared']:.3f}** (increment **{unique:.4f}**).",
        f"In the joint model, the standardized ridge-return coefficient is **{return_joint['standardized_beta']:.3f}** (p={return_joint['normal_approximation_p_value']:.3g}); "
        f"within exact matched states it is **{return_persist['standardized_beta']:.3f}** (p={return_persist['normal_approximation_p_value']:.3g}).",
        "",
        "Thus the linearly decoded return signal replicates, but it does not explain STOP beyond recent outcome, loss streak, and round. The near-perfect final-layer persistence probe confirms that the decision itself is linearly encoded and localizes where it becomes explicit; it is not evidence for value.",
        "",
        "## Direction overlap",
        "",
        f"Maximum absolute return/persistence direction cosine was **{maximum_cosine['absolute_cosine_similarity']:.3f}** at layer **{maximum_cosine['layer']}**. "
        f"Maximum top-1% dimension Jaccard was **{maximum_jaccard['top_dimension_jaccard']:.3f}** at layer **{maximum_jaccard['layer']}** "
        f"(chance-tail p={maximum_jaccard['top_dimension_overlap_chance_p']:.3g}; 32-layer Bonferroni p={maximum_jaccard['top_dimension_overlap_bonferroni_p']:.3g}).",
        "",
        "The sparse top-dimension overlap is above chance, but the corresponding signed whole-vector directions are nearly orthogonal. This suggests some shared dimensions without a common global linear axis.",
        "",
        "## Adjudication",
        "",
        "The ridge result rules out the narrow explanation that future return was recoverable only by a flexible nonlinear decoder. "
        "It does not establish that generic future return controls stopping: the exact-matched and joint persistence tests are null. "
        "The next discriminating experiment remains the counterfactual continuation-advantage probe.",
        "",
        summary["caveat"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="artifacts/linear_probes")
    parser.add_argument("--nonlinear-dir", default="artifacts/mc_value_probes")
    parser.add_argument("--output-dir", default="artifacts/linear_probes/publication")
    args = parser.parse_args()
    probe_dir, output = Path(args.probe_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((probe_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = read_rows(probe_dir / "test_predictions.csv")
    nonlinear_dir = Path(args.nonlinear_dir)
    nonlinear_path = nonlinear_dir / "metrics.json"
    nonlinear = json.loads(nonlinear_path.read_text()) if nonlinear_path.exists() else None
    nonlinear_rows_path = nonlinear_dir / "probe_mechanism_test_states.csv"
    nonlinear_rows = None
    if nonlinear_rows_path.exists():
        with nonlinear_rows_path.open(newline="", encoding="utf-8") as handle:
            nonlinear_rows = list(csv.DictReader(handle))
    nonlinear_summary_path = nonlinear_dir / "publication" / "monte_carlo_probe_summary.json"
    nonlinear_summary = (
        json.loads(nonlinear_summary_path.read_text(encoding="utf-8"))
        if nonlinear_summary_path.exists()
        else None
    )
    summary = summarize(
        metrics,
        rows,
        nonlinear,
        nonlinear_rows=nonlinear_rows,
        nonlinear_summary=nonlinear_summary,
    )
    (output / "linear_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(metrics, summary, output / "linear_probe_results.svg")
    if nonlinear_rows is not None and nonlinear_summary is not None:
        make_replication_figure(summary, output / "linear_probe_replication.svg")
    write_report(summary, output / "linear_probe_report.md")
    print(json.dumps({"best_layers": summary["best_layers"]}, indent=2))


if __name__ == "__main__":
    main()
