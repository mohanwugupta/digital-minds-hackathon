"""Create publication-style diagnostics for a completed value-probe run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.probe_history_matching import (
    exact_history_match_analysis,
    variance_decomposition,
)


ROUND_BINS = (
    (0, 0, "0"),
    (1, 5, "1–5"),
    (6, 10, "6–10"),
    (11, 25, "11–25"),
    (26, 50, "26–50"),
    (51, 99, "51–99"),
)
SCORE_BINS = (
    (-10_000, -1, "<0"),
    (0, 9, "0–9"),
    (10, 29, "10–29"),
    (30, 10_000, "30+"),
)


def _read_rows(path: Path) -> list[dict]:
    numeric = (
        "round",
        "loss_streak",
        "cumulative_score",
        "persistence_logit",
        "p_stop",
        "probe_value",
        "probe_value_full",
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def _rank(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _z_values(rows: list[dict], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    mean = statistics.mean(values)
    scale = statistics.pstdev(values)
    return [(value - mean) / scale for value in values]


def _binned_z_means(rows: list[dict], bins, grouping_key: str) -> list[dict]:
    standardized = {
        key: _z_values(rows, key)
        for key in ("probe_value", "probe_value_full", "persistence_logit")
    }
    result = []
    for lower, upper, label in bins:
        indices = [
            index
            for index, row in enumerate(rows)
            if lower <= float(row[grouping_key]) <= upper
        ]
        result.append(
            {
                "label": label,
                "states": len(indices),
                **{
                    key: statistics.mean(values[index] for index in indices)
                    for key, values in standardized.items()
                },
            }
        )
    return result


def summarize(metrics: dict, mechanism: dict, calibration: dict, rows: list[dict]):
    layers = metrics["layers"]
    selected_layer = int(metrics["best_layer"])
    selected = layers[selected_layer]
    validation_pruned = [item["validation"]["pruned_td_mse"] for item in layers]
    test_pruned = [item["test"]["pruned_td_mse"] for item in layers]
    validation_order = sorted(range(len(layers)), key=validation_pruned.__getitem__)
    best_epoch_one = sum(
        min(item["history"], key=lambda value: value["validation_loss"])["epoch"] == 1
        for item in layers
    )
    raw_correlations = {
        "sparse_probe_with_persistence": _pearson(
            [row["probe_value"] for row in rows],
            [row["persistence_logit"] for row in rows],
        ),
        "full_probe_with_persistence": _pearson(
            [row["probe_value_full"] for row in rows],
            [row["persistence_logit"] for row in rows],
        ),
        "sparse_probe_with_round": _pearson(
            [row["probe_value"] for row in rows], [row["round"] for row in rows]
        ),
        "full_probe_with_round": _pearson(
            [row["probe_value_full"] for row in rows],
            [row["round"] for row in rows],
        ),
    }
    return {
        "selection": {
            "layer": selected_layer,
            "neuron_count": int(mechanism["neuron_count"]),
            "validation_pruned_td_mse": selected["validation"]["pruned_td_mse"],
            "test_pruned_td_mse": selected["test"]["pruned_td_mse"],
            "validation_full_td_mse": selected["validation"]["full_td_mse"],
            "test_full_td_mse": selected["test"]["full_td_mse"],
            "validation_gap_to_second_layer": (
                validation_pruned[validation_order[1]]
                - validation_pruned[selected_layer]
            ),
            "pruned_validation_layer_range": max(validation_pruned)
            - min(validation_pruned),
            "pruned_test_layer_range": max(test_pruned) - min(test_pruned),
            "validation_test_layer_rank_correlation": _pearson(
                _rank(validation_pruned), _rank(test_pruned)
            ),
            "layers_best_at_epoch_one": best_epoch_one,
            "layer_count": len(layers),
        },
        "mechanism": {
            "test_episodes": mechanism["episodes"],
            "test_states": mechanism["states"],
            "control_r_squared": mechanism["primary_pruned_probe"][
                "control_r_squared"
            ],
            "primary_sparse": mechanism["primary_pruned_probe"],
            "with_score_control": mechanism[
                "pruned_probe_controlling_cumulative_score"
            ],
            "within_loss": mechanism["within_previous_outcome"]["last_loss"],
            "within_gain": mechanism["within_previous_outcome"]["last_gain"],
            "full_probe": mechanism["full_probe"],
            "history_within_loss": mechanism["probe_encodes_cumulative_history"][
                "last_loss"
            ],
            "evidence_flags": mechanism["evidence_flags"],
        },
        "calibration": {
            **calibration,
            "selected_fraction": mechanism["neuron_count"] / 2560,
        },
        "probe_output_scale": {
            key: {
                "mean": statistics.mean(row[key] for row in rows),
                "standard_deviation": statistics.pstdev(row[key] for row in rows),
                "minimum": min(row[key] for row in rows),
                "maximum": max(row[key] for row in rows),
            }
            for key in ("probe_value", "probe_value_full")
        },
        "raw_correlations": raw_correlations,
        "round_bins": _binned_z_means(rows, ROUND_BINS, "round"),
        "score_bins": _binned_z_means(rows, SCORE_BINS, "cumulative_score"),
        "variance_decomposition": {
            "sparse": variance_decomposition(rows, mechanism, "probe_value"),
            "full": variance_decomposition(rows, mechanism, "probe_value_full"),
        },
        "exact_history_matching": exact_history_match_analysis(rows),
        "interpretation": {
            "integrated_value_supported": False,
            "causal_value_claim_ready": False,
            "recommended_status": "pause confirmatory value steering",
            "reason": (
                "The probe is dominated by round/history covariation, the full "
                "probe adds no held-out persistence signal after controls, the "
                "sparse residual effect is small and negative, and all layer "
                "training runs selected epoch one before TD loss deteriorated."
            ),
        },
    }


def _layer_figure(metrics: dict, mechanism: dict, output: Path) -> None:
    layers = metrics["layers"]
    selected = int(metrics["best_layer"])
    x_values = [item["layer"] for item in layers]
    validation_pruned = [item["validation"]["pruned_td_mse"] for item in layers]
    test_pruned = [item["test"]["pruned_td_mse"] for item in layers]
    svg = Svg(1500, 1030)
    svg.text(55, 45, "Value-probe training and held-out mechanism diagnostics", "title")
    svg.text(
        55,
        70,
        "Qwen3.5-4B; validation-selected layer and neurons; mechanism tests use untouched test episodes",
        "subtitle",
    )
    panels = ((45, 95), (760, 95), (45, 565), (760, 565))

    x, y = panels[0]
    lower = min([*validation_pruned, *test_pruned]) - 0.025
    upper = max([*validation_pruned, *test_pruned]) + 0.025
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "A. Sparse-probe TD error is nearly flat across layers",
        "Transformer layer",
        "TD MSE",
        [(0, "0"), (8, "8"), (16, "16"), (24, "24"), (31, "31")],
        [
            (lower, f"{lower:.2f}"),
            ((lower + upper) / 2, f"{(lower + upper) / 2:.2f}"),
            (upper, f"{upper:.2f}"),
        ],
        (0, 31),
        (lower, upper),
    )
    svg.line(sx(selected), box[1], sx(selected), box[3], "#9CA3AF", 1.2, "5 5")
    for values, color in (
        (validation_pruned, COLORS["observed"]),
        (test_pruned, COLORS["model"]),
    ):
        points = [(sx(layer), sy(value)) for layer, value in zip(x_values, values)]
        svg.polyline(points, color, 2)
        for point_x, point_y in points:
            svg.circle(point_x, point_y, 2.7, color)
    svg.circle(sx(selected), sy(validation_pruned[selected]), 6, COLORS["observed"])
    svg.text(sx(selected) + 8, sy(validation_pruned[selected]) - 8, "Selected: L17", "note")
    legend(
        svg,
        box[0] + 12,
        box[1] + 18,
        [("Validation", COLORS["observed"]), ("Test", COLORS["model"])],
    )

    x, y = panels[1]
    selected_metrics = layers[selected]
    bars = [
        selected_metrics["validation"]["full_td_mse"],
        selected_metrics["validation"]["pruned_td_mse"],
        selected_metrics["test"]["full_td_mse"],
        selected_metrics["test"]["pruned_td_mse"],
    ]
    upper = max(bars) * 1.15
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "B. Masking 99% of inputs lowers TD error",
        "Layer 17 evaluation",
        "TD MSE",
        [(0, "Val full"), (1, "Val sparse"), (2, "Test full"), (3, "Test sparse")],
        [(0, "0"), (upper / 2, f"{upper / 2:.1f}"), (upper, f"{upper:.1f}")],
        (-0.6, 3.6),
        (0, upper),
    )
    for index, value in enumerate(bars):
        color = COLORS["observed"] if index % 2 else "#94A3B8"
        svg.rect(sx(index) - 32, sy(value), 64, box[3] - sy(value), color, 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.2f}", anchor="middle")

    x, y = panels[2]
    regressions = [
        ("Sparse", mechanism["primary_pruned_probe"]),
        ("+ score", mechanism["pruned_probe_controlling_cumulative_score"]),
        ("Last loss", mechanism["within_previous_outcome"]["last_loss"]),
        ("Last gain", mechanism["within_previous_outcome"]["last_gain"]),
        ("Full", mechanism["full_probe"]),
    ]
    intervals = []
    for _label, item in regressions:
        estimate = item["probe_standardized_beta"]
        error = item["cluster_robust_standard_error"] or 0
        intervals.extend((estimate - 1.96 * error, estimate + 1.96 * error))
    lower = min(-0.05, min(intervals) - 0.03)
    upper = max(0.05, max(intervals) + 0.03)
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "C. Adjusted probe–persistence effects",
        "Standardized probe coefficient",
        "Held-out analysis",
        [
            (lower, f"{lower:.2f}"),
            (0, "0"),
            (upper, f"{upper:.2f}"),
        ],
        [(index, label) for index, (label, _) in enumerate(regressions)],
        (lower, upper),
        (-0.6, len(regressions) - 0.4),
    )
    svg.line(sx(0), box[1], sx(0), box[3], COLORS["reference"], 1.2, "5 5")
    for index, (_label, item) in enumerate(regressions):
        estimate = item["probe_standardized_beta"]
        error = item["cluster_robust_standard_error"] or 0
        low, high = estimate - 1.96 * error, estimate + 1.96 * error
        svg.line(sx(low), sy(index), sx(high), sy(index), COLORS["observed"], 2)
        svg.line(sx(low), sy(index) - 6, sx(low), sy(index) + 6, COLORS["observed"], 2)
        svg.line(sx(high), sy(index) - 6, sx(high), sy(index) + 6, COLORS["observed"], 2)
        svg.circle(sx(estimate), sy(index), 5, COLORS["observed"])

    x, y = panels[3]
    delta_values = [item["delta_r_squared"] for _label, item in regressions]
    upper = max(0.012, max(delta_values) * 1.25)
    sx, sy, box = axes(
        svg,
        x,
        y,
        690,
        430,
        "D. Incremental variance is below the 1% criterion",
        "Held-out analysis",
        "Delta R²",
        [(index, label) for index, (label, _) in enumerate(regressions)],
        [(0, "0"), (0.005, ".005"), (0.01, ".010"), (upper, f"{upper:.3f}")],
        (-0.6, len(regressions) - 0.4),
        (0, upper),
    )
    svg.line(box[0], sy(0.01), box[2], sy(0.01), "#D55E00", 1.5, "6 4")
    for index, value in enumerate(delta_values):
        svg.rect(sx(index) - 28, sy(value), 56, box[3] - sy(value), COLORS["model"], 0.9)
        svg.text(sx(index), sy(value) - 7, f"{value:.3f}", anchor="middle")
    svg.text(box[0] + 8, sy(0.01) - 7, "Prespecified minimum", "note")
    svg.save(output)


def _variance_matching_figure(summary: dict, output: Path) -> None:
    svg = Svg(1500, 570)
    svg.text(55, 45, "Where does probe prediction come from?", "title")
    matching = summary["exact_history_matching"]
    svg.text(
        55,
        70,
        f"Held-out test data; exact matching retains {matching['matched_states']} states in "
        f"{matching['eligible_strata']} strata from {matching['episode_clusters']} episodes",
        "subtitle",
    )
    panels = ((45, 100), (515, 100), (985, 100))
    variance = summary["variance_decomposition"]

    x, y = panels[0]
    groups = ("sparse", "full")
    fields = (
        ("probe_only_r_squared", "Probe", COLORS["observed"]),
        ("history_only_r_squared", "History", "#64748B"),
        ("joint_r_squared", "Joint", COLORS["model"]),
    )
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "A. Total held-out persistence variance",
        "Probe representation",
        "R²",
        [(0, "Sparse"), (1, "Full")],
        [(0, "0"), (0.25, ".25"), (0.5, ".50"), (0.75, ".75"), (0.85, ".85")],
        (-0.5, 1.5),
        (0, 0.85),
    )
    offsets = (-0.19, 0, 0.19)
    for field_index, (field, _label, color) in enumerate(fields):
        for group_index, group in enumerate(groups):
            value = variance[group][field]
            center = sx(group_index) + offsets[field_index] * (box[2] - box[0]) / 2
            svg.rect(center - 22, sy(value), 44, box[3] - sy(value), color, 0.9)
    legend(svg, box[0] + 8, box[1] + 18, [(label, color) for _field, label, color in fields])

    x, y = panels[1]
    partitions = (
        ("shared_probe_history", "Shared", COLORS["observed"]),
        ("unique_probe", "Unique probe", COLORS["model"]),
        ("unique_history", "Unique history", "#64748B"),
        ("unexplained", "Unexplained", "#D1D5DB"),
    )
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "B. Commonality decomposition",
        "Probe representation",
        "Share of persistence variance",
        [(0, "Sparse"), (1, "Full")],
        [(0, "0"), (0.25, ".25"), (0.5, ".50"), (0.75, ".75"), (1, "1")],
        (-0.5, 1.5),
        (0, 1),
    )
    for group_index, group in enumerate(groups):
        cumulative = 0.0
        for field, _label, color in partitions:
            value = max(0.0, variance[group][field])
            svg.rect(
                sx(group_index) - 42,
                sy(cumulative + value),
                84,
                sy(cumulative) - sy(cumulative + value),
                color,
                0.9,
            )
            cumulative += value
    legend(
        svg,
        box[0] + 2,
        box[1] + 18,
        [(label, color) for _field, label, color in partitions],
    )

    x, y = panels[2]
    simple = matching["simple_regressions"]
    estimates = (
        (
            "History→sparse",
            simple["older_history_to_sparse_probe"]["coefficients"]["prior_score"],
        ),
        (
            "History→full",
            simple["older_history_to_full_probe"]["coefficients"]["prior_score"],
        ),
        (
            "History→persist",
            simple["older_history_to_persistence"]["coefficients"]["prior_score"],
        ),
        (
            "Sparse→persist",
            simple["sparse_probe_to_persistence"]["coefficients"]["probe_value"],
        ),
        (
            "Full→persist",
            simple["full_probe_to_persistence"]["coefficients"]["probe_value_full"],
        ),
    )
    intervals = []
    for _label, item in estimates:
        error = item["cluster_robust_standard_error"] or 0
        intervals.extend(
            (item["standardized_beta"] - 1.96 * error, item["standardized_beta"] + 1.96 * error)
        )
    lower = min(-0.3, min(intervals) - 0.03)
    upper = max(0.5, max(intervals) + 0.03)
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "C. Exact recent-state matching",
        "Standardized coefficient",
        "Relationship",
        [(lower, f"{lower:.1f}"), (0, "0"), (upper, f"{upper:.1f}")],
        [(index, label) for index, (label, _item) in enumerate(estimates)],
        (lower, upper),
        (-0.6, len(estimates) - 0.4),
    )
    svg.line(sx(0), box[1], sx(0), box[3], COLORS["reference"], 1.2, "5 5")
    for index, (_label, item) in enumerate(estimates):
        estimate = item["standardized_beta"]
        error = item["cluster_robust_standard_error"] or 0
        low, high = estimate - 1.96 * error, estimate + 1.96 * error
        svg.line(sx(low), sy(index), sx(high), sy(index), COLORS["observed"], 2)
        svg.line(sx(low), sy(index) - 6, sx(low), sy(index) + 6, COLORS["observed"], 2)
        svg.line(sx(high), sy(index) - 6, sx(high), sy(index) + 6, COLORS["observed"], 2)
        svg.circle(sx(estimate), sy(index), 5, COLORS["observed"])
    svg.save(output)


def _history_figure(summary: dict, output: Path) -> None:
    svg = Svg(1500, 570)
    svg.text(55, 45, "Why the raw probe–persistence association is misleading", "title")
    svg.text(
        55,
        70,
        "Held-out test states; lines are bin means after standardizing each variable over all states",
        "subtitle",
    )
    panels = ((45, 100), (515, 100), (985, 100))
    line_specs = (
        ("probe_value", "Sparse probe", COLORS["observed"]),
        ("probe_value_full", "Full probe", COLORS["model"]),
        ("persistence_logit", "Persistence", COLORS["both_positive"]),
    )
    for panel_index, (title, bins) in enumerate(
        (("A. All three variables rise with round", summary["round_bins"]),
         ("B. All three variables rise with score", summary["score_bins"]))
    ):
        x, y = panels[panel_index]
        all_values = [item[key] for item in bins for key, _label, _color in line_specs]
        lower = min(-0.5, min(all_values) - 0.15)
        upper = max(0.5, max(all_values) + 0.15)
        sx, sy, box = axes(
            svg,
            x,
            y,
            430,
            410,
            title,
            "Round bin" if panel_index == 0 else "Cumulative-score bin",
            "Mean standardized value",
            [(index, item["label"]) for index, item in enumerate(bins)],
            [(lower, f"{lower:.1f}"), (0, "0"), (upper, f"{upper:.1f}")],
            (-0.4, len(bins) - 0.6),
            (lower, upper),
        )
        svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
        for key, _label, color in line_specs:
            points = [(sx(index), sy(item[key])) for index, item in enumerate(bins)]
            svg.polyline(points, color, 2.2)
            for px, py in points:
                svg.circle(px, py, 4, color)
        legend(svg, box[0] + 8, box[1] + 18, [(label, color) for _key, label, color in line_specs])

    x, y = panels[2]
    raw = summary["raw_correlations"]
    mechanism = summary["mechanism"]
    values = [
        raw["sparse_probe_with_persistence"],
        mechanism["primary_sparse"]["probe_standardized_beta"],
        raw["full_probe_with_persistence"],
        mechanism["full_probe"]["probe_standardized_beta"],
    ]
    lower, upper = -0.25, 0.8
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "C. Adjustment reverses the apparent relation",
        "Estimate",
        "Association with persistence",
        [(0, "Sparse raw"), (1, "Sparse adj."), (2, "Full raw"), (3, "Full adj.")],
        [(lower, f"{lower:.2f}"), (0, "0"), (0.4, ".40"), (upper, f"{upper:.2f}")],
        (-0.6, 3.6),
        (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, value in enumerate(values):
        color = COLORS["observed"] if index < 2 else COLORS["model"]
        top = min(sy(value), sy(0))
        height = abs(sy(value) - sy(0))
        svg.rect(sx(index) - 28, top, 56, height, color, 0.9)
        svg.text(
            sx(index),
            sy(value) - 8 if value >= 0 else sy(value) + 18,
            f"{value:.2f}",
            anchor="middle",
        )
    svg.text(
        box[0] + 8,
        box[1] + 20,
        "Raw: Pearson r. Adjusted: standardized beta controlling recent outcome, streak, and round.",
        "note",
    )
    svg.save(output)


def _report(summary: dict, output: Path) -> None:
    selection = summary["selection"]
    mechanism = summary["mechanism"]
    primary = mechanism["primary_sparse"]
    within_loss = mechanism["within_loss"]
    full = mechanism["full_probe"]
    history_loss = mechanism["history_within_loss"]
    calibration = summary["calibration"]
    raw = summary["raw_correlations"]
    output_scale = summary["probe_output_scale"]
    variance = summary["variance_decomposition"]
    matching = summary["exact_history_matching"]
    matched_simple = matching["simple_regressions"]
    lines = [
        "# Value-probe result: held-out diagnostic",
        "",
        "## Bottom line",
        "",
        "The run successfully found a differentiable sparse direction and calibrated "
        "a tiny intervention, but it did **not** validate that direction as an "
        "integrated task-value representation. Confirmatory steering should be "
        "paused or explicitly relabeled exploratory until the probe target/training "
        "is repaired.",
        "",
        "## Probe fitting",
        "",
        f"- Selected layer: **{selection['layer']} / 31**; selected dimensions: "
        f"**{selection['neuron_count']} / 2,560**.",
        f"- Sparse validation/test TD MSE: **{selection['validation_pruned_td_mse']:.3f} / "
        f"{selection['test_pruned_td_mse']:.3f}**.",
        f"- Full validation/test TD MSE at layer 17: **{selection['validation_full_td_mse']:.3f} / "
        f"{selection['test_full_td_mse']:.3f}**.",
        f"- Layer 17 beat the runner-up validation layer by only "
        f"**{selection['validation_gap_to_second_layer']:.3f} MSE**; the entire "
        f"32-layer sparse range was **{selection['pruned_validation_layer_range']:.3f}**.",
        f"- Validation/test layer rankings were nevertheless similar "
        f"(Spearman r={selection['validation_test_layer_rank_correlation']:.2f}), "
        "and layer 17 also had the lowest sparse test MSE.",
        f"- **{selection['layers_best_at_epoch_one']} / {selection['layer_count']}** "
        "layers had their best validation checkpoint at epoch 1, after which TD "
        "loss deteriorated before patience stopped training at epoch 11.",
        "- No constant or recent-history TD baseline was stored, so the absolute "
        "MSE cannot establish predictive value by itself.",
        f"- Masking reduced the probe-output SD from "
        f"**{output_scale['probe_value_full']['standard_deviation']:.3f}** to "
        f"**{output_scale['probe_value']['standard_deviation']:.3f}**. The lower "
        "sparse TD error therefore largely accompanies shrinkage toward zero, "
        "rather than preservation of the full probe's value scale.",
        "",
        "## Held-out mechanism test",
        "",
        f"The prespecified recent-history controls already explained "
        f"**{100 * mechanism['control_r_squared']:.1f}%** of persistence-logit "
        "variance. Adding the sparse probe explained only "
        f"**{100 * primary['delta_r_squared']:.2f} percentage points** "
        f"(beta={primary['probe_standardized_beta']:.3f}, "
        f"SE={primary['cluster_robust_standard_error']:.3f}, "
        f"p={primary['normal_approximation_p_value']:.3g}), and the coefficient "
        "was opposite the predicted direction.",
        f"Within states that had just received -2, the sparse coefficient was "
        f"**{within_loss['probe_standardized_beta']:.3f}** "
        f"(delta R²={within_loss['delta_r_squared']:.3f}, "
        f"p={within_loss['normal_approximation_p_value']:.3g}), again negative.",
        f"The full probe added essentially no adjusted persistence information "
        f"(beta={full['probe_standardized_beta']:.3f}, "
        f"delta R²={full['delta_r_squared']:.4f}, "
        f"p={full['normal_approximation_p_value']:.3g}).",
        f"The closest positive integrated-history result was cumulative score "
        f"predicting sparse probe output within last-loss states "
        f"(beta={history_loss['predictor_standardized_beta']:.3f}, "
        f"delta R²={history_loss['delta_r_squared']:.3f}), but its "
        f"p={history_loss['normal_approximation_p_value']:.3g} did not cross the "
        "prespecified .05 threshold.",
        "",
        "## Why the unadjusted pattern looks convincing",
        "",
        f"Raw probe–persistence correlations were positive (sparse r="
        f"{raw['sparse_probe_with_persistence']:.2f}; full r="
        f"{raw['full_probe_with_persistence']:.2f}), but both probes strongly "
        f"tracked round (sparse r={raw['sparse_probe_with_round']:.2f}; full r="
        f"{raw['full_probe_with_round']:.2f}). Once round and recent reward history "
        "were controlled, the apparent positive relationship vanished or reversed.",
        "",
        "## Probe-only, history-only, and joint models",
        "",
        f"The sparse probe alone explained **{100 * variance['sparse']['probe_only_r_squared']:.1f}%** "
        f"of persistence variance, while the full probe alone explained "
        f"**{100 * variance['full']['probe_only_r_squared']:.1f}%**. Behavioral "
        f"history alone explained **{100 * variance['sparse']['history_only_r_squared']:.1f}%**. "
        f"The joint models explained **{100 * variance['sparse']['joint_r_squared']:.1f}%** "
        f"(sparse) and **{100 * variance['full']['joint_r_squared']:.1f}%** (full).",
        f"For the sparse probe, **{100 * variance['sparse']['shared_probe_history']:.1f} "
        "percentage points** were shared with history and only "
        f"**{100 * variance['sparse']['unique_probe']:.2f} points** were unique. "
        f"For the full probe, **{100 * variance['full']['shared_probe_history']:.1f} "
        "points** were shared and **"
        f"{100 * variance['full']['unique_probe']:.03f} points** were unique.",
        "The probe-only models are therefore meaningful descriptions of total "
        "prediction, while the joint models show that nearly all of that prediction "
        "duplicates information already available in recent history and round.",
        "",
        "## Exact matched-history test",
        "",
        f"Matching exactly on round, previous outcome, and loss streak retained "
        f"**{matching['matched_states']} states in {matching['eligible_strata']} strata "
        f"from {matching['episode_clusters']} episodes**. Older history was measured "
        "as cumulative score before the immediately preceding outcome.",
        f"Older history strongly predicted the full probe "
        f"(beta={matched_simple['older_history_to_full_probe']['coefficients']['prior_score']['standardized_beta']:.3f}, "
        f"p={matched_simple['older_history_to_full_probe']['coefficients']['prior_score']['normal_approximation_p_value']:.3g}) "
        "but not the sparse probe or persistence. Within exact strata, neither the "
        f"sparse probe (beta={matched_simple['sparse_probe_to_persistence']['coefficients']['probe_value']['standardized_beta']:.3f}, "
        f"p={matched_simple['sparse_probe_to_persistence']['coefficients']['probe_value']['normal_approximation_p_value']:.3g}) "
        f"nor the full probe (beta={matched_simple['full_probe_to_persistence']['coefficients']['probe_value_full']['standardized_beta']:.3f}, "
        f"p={matched_simple['full_probe_to_persistence']['coefficients']['probe_value_full']['normal_approximation_p_value']:.3g}) "
        "predicted persistence.",
        "This supports a recent-state stopping heuristic over the claim that the "
        "current probe-defined value representation drives persistence. It also "
        "shows that the unpruned hidden-state-derived probe retains accumulated "
        "history; what remains unresolved is whether that history code represents "
        "expected future return.",
        "",
        "## Calibration is not construct validation",
        "",
        f"Magnitude **{calibration['magnitude']}** ordered probe outputs on "
        f"**{100 * calibration['ordered_fraction']:.0f}%** of validation states. "
        f"Its reported relative RMS was **{calibration['relative_rms']:.6f}**. "
        "This verifies that stepping along the probe gradient changes the probe in "
        "the intended mathematical direction; it does not show that the probe "
        "represents value or that the perturbation will materially change action logits.",
        "",
        "## Recommended sprint decision",
        "",
        "1. Do not present the current layer-17 direction as a validated integrated-"
        "value representation.",
        "2. Before causal confirmation, compare against a zero/constant TD baseline "
        "and a recent-outcome-plus-round baseline.",
        "3. Replace unstable online bootstrapping with a frozen-target TD procedure "
        "or a supervised Monte Carlo future-return probe using the already stored "
        "future returns, then rerun the same untouched-test mechanism diagnostic.",
        "4. Run causal steering only if the repaired probe adds positive held-out "
        "signal beyond recent history. Otherwise, any steering run should be labeled "
        "an exploratory perturbation of a probe-defined direction, not value steering.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="artifacts/value_probes")
    parser.add_argument(
        "--output-dir", default="artifacts/value_probes/publication"
    )
    args = parser.parse_args()
    probe_dir = Path(args.probe_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((probe_dir / "metrics.json").read_text(encoding="utf-8"))
    mechanism = json.loads(
        (probe_dir / "probe_mechanism.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (probe_dir / "steering_calibration.json").read_text(encoding="utf-8")
    )
    rows = _read_rows(probe_dir / "probe_mechanism_test_states.csv")
    summary = summarize(metrics, mechanism, calibration, rows)
    (output / "probe_results_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "probe_variance_and_matching.json").write_text(
        json.dumps(
            {
                "variance_decomposition": summary["variance_decomposition"],
                "exact_history_matching": summary["exact_history_matching"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _layer_figure(metrics, mechanism, output / "probe_training_and_mechanism.svg")
    _history_figure(summary, output / "probe_history_confounding.svg")
    _variance_matching_figure(summary, output / "probe_variance_and_matching.svg")
    _report(summary, output / "probe_results_report.md")
    print(json.dumps(summary["interpretation"], indent=2))


if __name__ == "__main__":
    main()
