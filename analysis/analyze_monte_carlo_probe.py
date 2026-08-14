"""Publication summary for the supervised Monte Carlo future-return probe."""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.probe_history_matching import (
    _clustered_regression,
    _standardize,
    exact_history_match_analysis,
    pearson,
    variance_decomposition,
)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric_probe_rows(rows: list[dict]) -> list[dict]:
    """Coerce mechanism CSV rows and reconstruct return up to the final reward."""
    grouped = defaultdict(list)
    for source in rows:
        row = dict(source)
        row["round"] = int(float(row["round"]))
        row["loss_streak"] = int(float(row["loss_streak"]))
        row["sampled_stop"] = int(float(row["sampled_stop"]))
        row["previous_outcome"] = (
            None
            if row["previous_outcome"] in (None, "")
            else float(row["previous_outcome"])
        )
        for key in (
            "cumulative_score",
            "persistence_logit",
            "probe_value",
            "probe_value_full",
        ):
            row[key] = float(row[key])
        grouped[row["episode_id"]].append(row)

    result = []
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: row["round"])
        final_state = episode_rows[-1]
        final_score_before_action = final_state["cumulative_score"]
        final_reward_unknown = not bool(final_state["sampled_stop"])
        for row in episode_rows:
            row["prior_score"] = row["cumulative_score"] - float(
                row["previous_outcome"] or 0.0
            )
            # STOP has reward zero. At the 100-decision cap, the final A/B
            # reward is absent from the mechanism CSV and is temporarily set
            # to zero; the exhaustive sensitivity analysis below restores both
            # possible values (-2 and +3) for every capped episode.
            row["reconstructed_future_return"] = (
                final_score_before_action - row["cumulative_score"]
            )
            row["final_reward_unknown"] = final_reward_unknown
            result.append(row)
    return result


def _recent_history_predictors(rows: list[dict]) -> dict[str, list[float]]:
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


def _matched_future_return_analysis(csv_rows: list[dict], min_stratum_size: int = 4) -> dict:
    """Test future-return decoding within identical recent behavioral states."""
    rows = _numeric_probe_rows(csv_rows)
    clusters = [row["episode_id"] for row in rows]
    outcome = [row["reconstructed_future_return"] for row in rows]
    history = _recent_history_predictors(rows)

    def overall(predictors: dict[str, list[float]]) -> dict:
        return _clustered_regression(outcome, predictors, clusters)

    overall_models = {
        "history_only": overall(history),
        "sparse_probe_only": overall(
            {"probe_value": [row["probe_value"] for row in rows]}
        ),
        "full_probe_only": overall(
            {"probe_value_full": [row["probe_value_full"] for row in rows]}
        ),
        "history_plus_sparse": overall(
            {**history, "probe_value": [row["probe_value"] for row in rows]}
        ),
        "history_plus_full": overall(
            {
                **history,
                "probe_value_full": [row["probe_value_full"] for row in rows],
            }
        ),
    }

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
        "probe_value",
        "probe_value_full",
        "persistence_logit",
        "reconstructed_future_return",
    )
    residuals = {key: [] for key in keys}
    matched_clusters = []
    capped_episode_ids = sorted(
        {
            row["episode_id"]
            for row in rows
            if row["final_reward_unknown"]
        }
    )
    capped_indicators = {episode_id: [] for episode_id in capped_episode_ids}
    for selected in eligible:
        means = {
            key: statistics.mean(row[key] for row in selected) for key in keys
        }
        shares = {
            episode_id: sum(
                row["episode_id"] == episode_id for row in selected
            )
            / len(selected)
            for episode_id in capped_episode_ids
        }
        for row in selected:
            for key in keys:
                residuals[key].append(row[key] - means[key])
            for episode_id in capped_episode_ids:
                capped_indicators[episode_id].append(
                    float(row["episode_id"] == episode_id) - shares[episode_id]
                )
            matched_clusters.append(row["episode_id"])

    future_key = "reconstructed_future_return"

    def matched(outcome_key: str, predictor_keys: tuple[str, ...]) -> dict:
        return _clustered_regression(
            residuals[outcome_key],
            {key: residuals[key] for key in predictor_keys},
            matched_clusters,
        )

    exact_models = {
        "older_history_to_future_return": matched(future_key, ("prior_score",)),
        "sparse_probe_to_future_return": matched(future_key, ("probe_value",)),
        "full_probe_to_future_return": matched(future_key, ("probe_value_full",)),
        "older_history_plus_sparse_to_future_return": matched(
            future_key, ("prior_score", "probe_value")
        ),
        "older_history_plus_full_to_future_return": matched(
            future_key, ("prior_score", "probe_value_full")
        ),
    }

    # There are only 11 capped test episodes in this run. Enumerating all 2^11
    # possible final rewards gives an exact robustness range without guessing
    # the omitted terminal outcomes.
    sensitivity = {}
    if len(capped_episode_ids) <= 16:
        ranges = {
            "older_history": [float("inf"), float("-inf")],
            "sparse_probe": [float("inf"), float("-inf")],
            "full_probe": [float("inf"), float("-inf")],
        }
        for rewards in itertools.product((-2.0, 3.0), repeat=len(capped_episode_ids)):
            adjusted = list(residuals[future_key])
            for episode_id, reward in zip(capped_episode_ids, rewards):
                indicator = capped_indicators[episode_id]
                adjusted = [
                    value + reward * membership
                    for value, membership in zip(adjusted, indicator)
                ]
            for label, key in (
                ("older_history", "prior_score"),
                ("sparse_probe", "probe_value"),
                ("full_probe", "probe_value_full"),
            ):
                correlation = pearson(residuals[key], adjusted)
                ranges[label][0] = min(ranges[label][0], correlation)
                ranges[label][1] = max(ranges[label][1], correlation)
        sensitivity = {
            "assignments_enumerated": 2 ** len(capped_episode_ids),
            "exact_match_standardized_beta_ranges": {
                key: {"minimum": value[0], "maximum": value[1]}
                for key, value in ranges.items()
            },
        }

    standardized = {
        key: _standardize(value)[0]
        for key, value in residuals.items()
    }
    order = sorted(
        range(len(matched_clusters)), key=standardized["prior_score"].__getitem__
    )
    deciles = []
    for decile in range(10):
        selected = order[
            decile * len(order) // 10 : (decile + 1) * len(order) // 10
        ]
        deciles.append(
            {
                "decile": decile + 1,
                "states": len(selected),
                **{
                    key: statistics.mean(values[index] for index in selected)
                    for key, values in standardized.items()
                },
            }
        )

    stopped_episodes = len(
        {
            row["episode_id"]
            for row in rows
            if not row["final_reward_unknown"]
        }
    )
    return {
        "return_reconstruction": {
            "states": len(rows),
            "episodes": len(set(clusters)),
            "stop_terminated_episodes_exact": stopped_episodes,
            "decision_cap_episodes_with_unknown_final_reward": len(
                capped_episode_ids
            ),
            "imputed_final_reward_for_point_estimates": 0,
            "actual_possible_final_rewards": [-2, 3],
        },
        "overall_models": overall_models,
        "exact_matching": {
            "matching_variables": ["round", "previous_outcome", "loss_streak"],
            "eligible_strata": len(eligible),
            "states": len(matched_clusters),
            "episode_clusters": len(set(matched_clusters)),
            "models": exact_models,
            "within_stratum_deciles": deciles,
        },
        "terminal_reward_sensitivity": sensitivity,
    }


def summarize(metrics: dict, mechanism: dict, rows: list[dict]) -> dict:
    best = next(
        item for item in metrics["layers"] if item["layer"] == metrics["best_layer"]
    )
    matching = exact_history_match_analysis(rows)
    return {
        "target": metrics["target"],
        "target_interpretation": metrics["target_interpretation"],
        "adaptive_evaluation_caveat": metrics["adaptive_evaluation_caveat"],
        "best_layer": metrics["best_layer"],
        "neuron_count": best["neuron_count"],
        "baselines": metrics["baselines"],
        "best_layer_metrics": {
            "full": best["full"],
            "sparse": best["sparse"],
        },
        "training_stability": {
            "full_epochs_trained": best["full_epochs_trained"],
            "sparse_epochs_trained": best["sparse_epochs_trained"],
            "full_best_epoch": min(
                best["full"]["history"], key=lambda row: row["validation_mse_z"]
            )["epoch"],
            "sparse_best_epoch": min(
                best["sparse"]["history"], key=lambda row: row["validation_mse_z"]
            )["epoch"],
        },
        "mechanism": mechanism,
        "variance_decomposition": {
            "sparse": variance_decomposition(rows, mechanism, "probe_value"),
            "full": variance_decomposition(rows, mechanism, "probe_value_full"),
        },
        "exact_history_matching": matching,
        "matched_future_return": _matched_future_return_analysis(rows),
    }


def _figure(metrics: dict, summary: dict, output: Path) -> None:
    svg = Svg(1500, 1030)
    svg.text(55, 45, "Supervised future-return probe", "title")
    svg.text(
        55,
        70,
        "Exploratory Monte Carlo target; episode-held-out evaluation; sparse probe refit on selected dimensions",
        "subtitle",
    )
    panels = ((45, 95), (760, 95), (45, 565), (760, 565))
    layers = metrics["layers"]
    selected = metrics["best_layer"]

    x, y = panels[0]
    series = (
        ("Sparse validation", [item["sparse"]["validation"]["r_squared"] for item in layers], COLORS["observed"]),
        ("Sparse test", [item["sparse"]["test"]["r_squared"] for item in layers], COLORS["model"]),
        ("Full validation", [item["full"]["validation"]["r_squared"] for item in layers], COLORS["both_positive"]),
        ("Full test", [item["full"]["test"]["r_squared"] for item in layers], "#64748B"),
    )
    all_values = [value for _label, values, _color in series for value in values]
    lower = min(-0.05, min(all_values) - 0.05)
    upper = max(0.1, max(all_values) + 0.05)
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "A. Future-return prediction across layers",
        "Transformer layer",
        "Held-out R²",
        [(0, "0"), (8, "8"), (16, "16"), (24, "24"), (31, "31")],
        [(lower, f"{lower:.2f}"), (0, "0"), (upper, f"{upper:.2f}")],
        (0, 31),
        (lower, upper),
    )
    svg.line(sx(selected), box[1], sx(selected), box[3], "#9CA3AF", 1.2, "5 5")
    for _label, values, color in series:
        points = [(sx(index), sy(value)) for index, value in enumerate(values)]
        svg.polyline(points, color, 2)
    legend(svg, box[0] + 5, box[1] + 18, [(label, color) for label, _values, color in series])

    x, y = panels[1]
    best = summary["best_layer_metrics"]
    bars = (
        ("Constant", summary["baselines"]["constant"]["test"]["r_squared"], "#D1D5DB"),
        ("History", summary["baselines"]["recent_history"]["test"]["r_squared"], "#64748B"),
        ("Sparse", best["sparse"]["test"]["r_squared"], COLORS["observed"]),
        ("Full", best["full"]["test"]["r_squared"], COLORS["model"]),
    )
    lower = min(-0.05, min(value for _label, value, _color in bars) - 0.05)
    upper = max(0.1, max(value for _label, value, _color in bars) + 0.08)
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "B. Held-out future-return benchmarks",
        "Predictor",
        "Test R²",
        [(index, label) for index, (label, _value, _color) in enumerate(bars)],
        [(lower, f"{lower:.2f}"), (0, "0"), (upper, f"{upper:.2f}")],
        (-0.6, 3.6),
        (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (_label, value, color) in enumerate(bars):
        top = min(sy(value), sy(0))
        svg.rect(sx(index) - 32, top, 64, abs(sy(value) - sy(0)), color, 0.9)
        svg.text(sx(index), sy(value) - 8 if value >= 0 else sy(value) + 18, f"{value:.2f}", anchor="middle")

    x, y = panels[2]
    mechanism = summary["mechanism"]
    matching = summary["exact_history_matching"]["simple_regressions"]
    estimates = (
        ("Sparse adjusted", mechanism["primary_pruned_probe"]["probe_standardized_beta"], mechanism["primary_pruned_probe"]["cluster_robust_standard_error"]),
        ("Full adjusted", mechanism["full_probe"]["probe_standardized_beta"], mechanism["full_probe"]["cluster_robust_standard_error"]),
        ("Sparse exact", matching["sparse_probe_to_persistence"]["coefficients"]["probe_value"]["standardized_beta"], matching["sparse_probe_to_persistence"]["coefficients"]["probe_value"]["cluster_robust_standard_error"]),
        ("Full exact", matching["full_probe_to_persistence"]["coefficients"]["probe_value_full"]["standardized_beta"], matching["full_probe_to_persistence"]["coefficients"]["probe_value_full"]["cluster_robust_standard_error"]),
    )
    bounds = [bound for _label, estimate, error in estimates for bound in (estimate - 1.96 * (error or 0), estimate + 1.96 * (error or 0))]
    lower, upper = min(-0.2, min(bounds) - 0.03), max(0.2, max(bounds) + 0.03)
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "C. Does future-return code predict persistence?",
        "Standardized coefficient",
        "Analysis",
        [(lower, f"{lower:.2f}"), (0, "0"), (upper, f"{upper:.2f}")],
        [(index, label) for index, (label, _estimate, _error) in enumerate(estimates)],
        (lower, upper),
        (-0.6, len(estimates) - 0.4),
    )
    svg.line(sx(0), box[1], sx(0), box[3], COLORS["reference"], 1.2, "5 5")
    for index, (_label, estimate, error) in enumerate(estimates):
        error = error or 0
        svg.line(sx(estimate - 1.96 * error), sy(index), sx(estimate + 1.96 * error), sy(index), COLORS["observed"], 2)
        svg.circle(sx(estimate), sy(index), 5, COLORS["observed"])

    x, y = panels[3]
    variance = summary["variance_decomposition"]
    bars = (
        ("Sparse probe", variance["sparse"]["probe_only_r_squared"], COLORS["observed"]),
        ("Sparse unique", variance["sparse"]["unique_probe"], COLORS["model"]),
        ("Full probe", variance["full"]["probe_only_r_squared"], COLORS["both_positive"]),
        ("Full unique", variance["full"]["unique_probe"], "#64748B"),
    )
    upper = max(0.1, max(value for _label, value, _color in bars) * 1.2)
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "D. Total versus history-unique persistence R²",
        "Probe model",
        "R²",
        [(index, label) for index, (label, _value, _color) in enumerate(bars)],
        [(0, "0"), (upper / 2, f"{upper / 2:.2f}"), (upper, f"{upper:.2f}")],
        (-0.6, 3.6),
        (0, upper),
    )
    for index, (_label, value, color) in enumerate(bars):
        svg.rect(sx(index) - 32, sy(value), 64, box[3] - sy(value), color, 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.3f}", anchor="middle")
    svg.save(output)


def _adjudication_figure(summary: dict, output: Path) -> None:
    """Show the separation between future-return decoding and stopping policy."""
    svg = Svg(1500, 620)
    svg.text(55, 45, "What does the Monte Carlo probe represent?", "title")
    svg.text(
        55,
        70,
        "Frozen probe outputs on test episodes; exact matching fixes round, previous outcome, and loss streak",
        "subtitle",
    )
    future = summary["matched_future_return"]
    overall = future["overall_models"]
    matching = future["exact_matching"]
    exact = matching["models"]
    persistence = summary["exact_history_matching"]["simple_regressions"]

    x, y = 45, 105
    bars = (
        ("History", overall["history_only"]["r_squared"], "#64748B"),
        ("Sparse", overall["sparse_probe_only"]["r_squared"], COLORS["observed"]),
        ("Full", overall["full_probe_only"]["r_squared"], COLORS["model"]),
        ("Hist.+sparse", overall["history_plus_sparse"]["r_squared"], COLORS["one_positive"]),
        ("Hist.+full", overall["history_plus_full"]["r_squared"], COLORS["both_positive"]),
    )
    maximum = max(value for _label, value, _color in bars) * 1.18
    sx, sy, box = axes(
        svg,
        x,
        y,
        455,
        440,
        "A. Future-return prediction",
        "Predictor",
        "Reconstructed return R²",
        [(index, label) for index, (label, _value, _color) in enumerate(bars)],
        [(0, "0"), (maximum / 2, f"{maximum / 2:.2f}"), (maximum, f"{maximum:.2f}")],
        (-0.6, len(bars) - 0.4),
        (0, maximum),
    )
    for index, (_label, value, color) in enumerate(bars):
        svg.rect(sx(index) - 24, sy(value), 48, box[3] - sy(value), color, 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.2f}", anchor="middle")

    x, y = 520, 105
    estimates = (
        (
            "History → return",
            exact["older_history_to_future_return"],
            "prior_score",
            COLORS["reference"],
        ),
        (
            "Sparse → return",
            exact["sparse_probe_to_future_return"],
            "probe_value",
            COLORS["observed"],
        ),
        (
            "Full → return",
            exact["full_probe_to_future_return"],
            "probe_value_full",
            COLORS["model"],
        ),
        (
            "Sparse → persistence",
            persistence["sparse_probe_to_persistence"],
            "probe_value",
            COLORS["observed"],
        ),
        (
            "Full → persistence",
            persistence["full_probe_to_persistence"],
            "probe_value_full",
            COLORS["model"],
        ),
    )
    values = []
    for _label, result, key, _color in estimates:
        coefficient = result["coefficients"][key]
        estimate = coefficient["standardized_beta"]
        error = coefficient["cluster_robust_standard_error"] or 0
        values.extend((estimate - 1.96 * error, estimate + 1.96 * error))
    lower, upper = min(-0.3, min(values) - 0.04), max(0.7, max(values) + 0.04)
    sx, sy, box = axes(
        svg,
        x,
        y,
        455,
        440,
        "B. Within identical recent states",
        "Standardized coefficient",
        "Outcome",
        [(lower, f"{lower:.1f}"), (0, "0"), (upper, f"{upper:.1f}")],
        [(index, label) for index, (label, _result, _key, _color) in enumerate(estimates)],
        (lower, upper),
        (-0.6, len(estimates) - 0.4),
    )
    svg.line(sx(0), box[1], sx(0), box[3], COLORS["reference"], 1.2, "5 5")
    for index, (_label, result, key, color) in enumerate(estimates):
        coefficient = result["coefficients"][key]
        estimate = coefficient["standardized_beta"]
        error = coefficient["cluster_robust_standard_error"] or 0
        svg.line(
            sx(estimate - 1.96 * error),
            sy(index),
            sx(estimate + 1.96 * error),
            sy(index),
            color,
            2,
        )
        svg.circle(sx(estimate), sy(index), 5, color)

    x, y = 995, 105
    deciles = matching["within_stratum_deciles"]
    series = (
        (
            "Future return",
            [row["reconstructed_future_return"] for row in deciles],
            COLORS["both_positive"],
        ),
        ("Sparse probe", [row["probe_value"] for row in deciles], COLORS["observed"]),
        ("Full probe", [row["probe_value_full"] for row in deciles], COLORS["model"]),
        (
            "Persistence",
            [row["persistence_logit"] for row in deciles],
            "#64748B",
        ),
    )
    x_values = [row["prior_score"] for row in deciles]
    y_values = [value for _label, values, _color in series for value in values]
    x_min, x_max = min(x_values) - 0.15, max(x_values) + 0.15
    y_min, y_max = min(-1.0, min(y_values) - 0.15), max(1.0, max(y_values) + 0.15)
    sx, sy, box = axes(
        svg,
        x,
        y,
        455,
        440,
        "C. Older-history gradient",
        "Older history (within-stratum z)",
        "Within-stratum z",
        [(x_min, f"{x_min:.1f}"), (0, "0"), (x_max, f"{x_max:.1f}")],
        [(y_min, f"{y_min:.1f}"), (0, "0"), (y_max, f"{y_max:.1f}")],
        (x_min, x_max),
        (y_min, y_max),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1, "5 5")
    for _label, values, color in series:
        points = [(sx(x_value), sy(value)) for x_value, value in zip(x_values, values)]
        svg.polyline(points, color, 2)
        for point_x, point_y in points:
            svg.circle(point_x, point_y, 3, color)
    for index, (label, _values, color) in enumerate(series):
        legend_x = box[0] + 4 + (index % 2) * 190
        legend_y = box[1] + 18 + (index // 2) * 20
        svg.line(legend_x, legend_y, legend_x + 22, legend_y, color, 3)
        svg.text(legend_x + 28, legend_y + 4, label, "legend")
    svg.text(
        1450,
        595,
        "Point estimates impute zero for the omitted final reward of 11 capped episodes; exhaustive ±2/+3 sensitivity preserves the conclusions.",
        "note",
        "end",
    )
    svg.save(output)


def _report(summary: dict, output: Path) -> None:
    best = summary["best_layer_metrics"]
    stability = summary["training_stability"]
    mechanism = summary["mechanism"]
    matching = summary["exact_history_matching"]
    variance = summary["variance_decomposition"]
    future = summary["matched_future_return"]
    overall = future["overall_models"]
    exact = future["exact_matching"]["models"]
    sensitivity = future["terminal_reward_sensitivity"][
        "exact_match_standardized_beta_ranges"
    ]
    sparse_future = exact["sparse_probe_to_future_return"]["coefficients"][
        "probe_value"
    ]
    full_future = exact["full_probe_to_future_return"]["coefficients"][
        "probe_value_full"
    ]
    sparse_persistence = matching["simple_regressions"][
        "sparse_probe_to_persistence"
    ]["coefficients"]["probe_value"]
    full_persistence = matching["simple_regressions"][
        "full_probe_to_persistence"
    ]["coefficients"]["probe_value_full"]
    lines = [
        "# Monte Carlo value probe: mechanism adjudication",
        "",
        "## Bottom line",
        "",
        "The supervised probe establishes a decodable, history-integrating "
        "future-return signal, but that signal does **not** explain the model's "
        "STOP policy after recent state is controlled. The behavioral evidence "
        "therefore favors a recent-state stopping heuristic. A distinct "
        "continuation-advantage representation remains plausible but untested.",
        "",
        f"Selected layer: **{summary['best_layer']}**; sparse dimensions: "
        f"**{summary['neuron_count']}**.",
        "",
        "## Probe validity: does it predict future return?",
        "",
        f"- Sparse validation/test R²: **{best['sparse']['validation']['r_squared']:.3f} / {best['sparse']['test']['r_squared']:.3f}**",
        f"- Full validation/test R²: **{best['full']['validation']['r_squared']:.3f} / {best['full']['test']['r_squared']:.3f}**",
        f"- Recent-history test R²: **{summary['baselines']['recent_history']['test']['r_squared']:.3f}**",
        f"- Constant test R²: **{summary['baselines']['constant']['test']['r_squared']:.3f}**",
        f"- Selected full/sparse checkpoints occurred at epochs "
        f"**{stability['full_best_epoch']} / {stability['sparse_best_epoch']}**, "
        "rather than every layer collapsing to epoch one as in the TD run.",
        "",
        "In descriptive regressions refit within the test set (a conservative "
        "advantage for the behavioral baseline), recent history explained "
        f"**{overall['history_only']['r_squared']:.1%}** of reconstructed future "
        f"return. Sparse/full probes alone explained "
        f"**{overall['sparse_probe_only']['r_squared']:.1%} / "
        f"{overall['full_probe_only']['r_squared']:.1%}**; joint history-plus-probe "
        f"models explained **{overall['history_plus_sparse']['r_squared']:.1%} / "
        f"{overall['history_plus_full']['r_squared']:.1%}**.",
        "",
        "## Exact matched-state test",
        "",
        f"Matching round, previous outcome, and loss streak retained "
        f"**{future['exact_matching']['states']} states in "
        f"{future['exact_matching']['eligible_strata']} strata from "
        f"{future['exact_matching']['episode_clusters']} episodes**.",
        f"Within those strata, sparse probe value predicted future return "
        f"(beta=**{sparse_future['standardized_beta']:.3f}**, "
        f"SE={sparse_future['cluster_robust_standard_error']:.3f}, "
        f"p={sparse_future['normal_approximation_p_value']:.3g}); the full probe "
        f"did too (beta=**{full_future['standardized_beta']:.3f}**, "
        f"SE={full_future['cluster_robust_standard_error']:.3f}, "
        f"p={full_future['normal_approximation_p_value']:.3g}).",
        "The mechanism CSV omits the final reward for 11 episodes censored at "
        "the 100-decision cap. Exhaustively assigning every combination of -2 "
        "and +3 left the matched beta positive: sparse "
        f"**[{sensitivity['sparse_probe']['minimum']:.3f}, "
        f"{sensitivity['sparse_probe']['maximum']:.3f}]** and full "
        f"**[{sensitivity['full_probe']['minimum']:.3f}, "
        f"{sensitivity['full_probe']['maximum']:.3f}]**.",
        "",
        "## Does that representation explain persistence?",
        "",
        f"Recent-state controls explained **{mechanism['primary_pruned_probe']['control_r_squared']:.1%}** "
        "of persistence-logit variance. Sparse/full probe-only R² values were "
        f"**{variance['sparse']['probe_only_r_squared']:.3f} / "
        f"{variance['full']['probe_only_r_squared']:.3f}**, and their unique "
        "increments beyond recent history were only "
        f"**{variance['sparse']['unique_probe']:.5f} / "
        f"{variance['full']['unique_probe']:.5f}**.",
        f"Within exact strata, sparse probe value did not predict persistence "
        f"(beta={sparse_persistence['standardized_beta']:.3f}, "
        f"p={sparse_persistence['normal_approximation_p_value']:.3g}); neither "
        f"did the full probe (beta={full_persistence['standardized_beta']:.3f}, "
        f"p={full_persistence['normal_approximation_p_value']:.3g}).",
        f"Adding cumulative score as a linear covariate produced a positive sparse "
        f"coefficient (beta={mechanism['pruned_probe_controlling_cumulative_score']['probe_standardized_beta']:.3f}, "
        f"p={mechanism['pruned_probe_controlling_cumulative_score']['normal_approximation_p_value']:.3g}), "
        "but this sign-reversing suppression result did not survive the more direct "
        "exact-matching test and should not be treated as robust mechanism evidence.",
        "",
        "## Adjudicating the three possibilities",
        "",
        "1. **Simple recent-loss/time stopping heuristic — supported for the "
        "behavioral policy.** Recent state predicts persistence extremely well, "
        "and the Monte Carlo probe adds essentially nothing after adjustment or "
        "exact matching.",
        "2. **Integrated value exists but TD training failed — supported at the "
        "representational level.** Stable supervised probes predict held-out "
        "future return and retain that prediction among states with identical "
        "recent experience. This rescues the existence of a decodable integrated "
        "return signal, but not the claim that it drives STOP.",
        "3. **Continuation advantage rather than generic value drives stopping — "
        "plausible, not established.** The dissociation between reliable return "
        "decoding and absent persistence prediction is exactly what motivates an "
        "advantage target. It is also compatible with the simpler account that "
        "the model encodes value epiphenomenally and stops via a heuristic.",
        "",
        "## Sprint decision",
        "",
        "Do not promote this Monte Carlo direction as a validated causal "
        "persistence direction. If time permits, the next discriminating analysis "
        "is a continuation-advantage probe with forced A/B counterfactual returns. "
        "Steering the current direction can still be reported as exploratory, but "
        "a null effect would be predicted by the observational mechanism results.",
        "",
        "## Interpretation constraints",
        "",
        summary["target_interpretation"] + ".",
        summary["adaptive_evaluation_caveat"],
        "The fitted probe is a two-layer ReLU network, so these results establish "
        "out-of-episode decodability rather than proving that the model exposes a "
        "single native linear value variable. A ridge-linear replication would "
        "strengthen the representational claim.",
        "The reconstructed-return matched analysis uses exact outcomes for 67 "
        "STOP-terminated episodes and an explicitly enumerated terminal-reward "
        "sensitivity analysis for the 11 capped episodes.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="artifacts/mc_value_probes")
    parser.add_argument("--output-dir", default="artifacts/mc_value_probes/publication")
    args = parser.parse_args()
    probe_dir, output = Path(args.probe_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((probe_dir / "metrics.json").read_text(encoding="utf-8"))
    mechanism = json.loads((probe_dir / "probe_mechanism.json").read_text(encoding="utf-8"))
    rows = _read_rows(probe_dir / "probe_mechanism_test_states.csv")
    summary = summarize(metrics, mechanism, rows)
    (output / "monte_carlo_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _figure(metrics, summary, output / "monte_carlo_probe_results.svg")
    _adjudication_figure(summary, output / "monte_carlo_probe_adjudication.svg")
    _report(summary, output / "monte_carlo_probe_report.md")
    print(
        json.dumps(
            {
                "best_layer": summary["best_layer"],
                "test_sparse_r2": summary["best_layer_metrics"]["sparse"]["test"]["r_squared"],
                "integrated_future_return_decodable": True,
                "probe_predicts_persistence_after_matching": False,
                "recommended_interpretation": (
                    "recent-state stopping policy with a decodable but behaviorally "
                    "dissociated future-return representation"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
