"""Trace factorial incentive effects through frozen layerwise persistence probes."""

import argparse
import csv
import glob
import json
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes
from analysis.analyze_value_dissociation import (
    _state_fixed_regression,
    audit_rows,
)
from analysis.layerwise_detection import summarize_layerwise_detection


def read_rows(pattern: str) -> tuple[list[dict], list[int]]:
    rows, projection_fields = [], None
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = [
                field
                for field in (reader.fieldnames or [])
                if field.startswith("layer_") and field.endswith("_projection")
            ]
            if projection_fields is None:
                projection_fields = fields
            elif fields != projection_fields:
                raise ValueError("layerwise projection shards have different columns")
            for source in reader:
                row = dict(source)
                for key in (
                    "stop_payoff",
                    "continue_bonus",
                    "relative_incentive",
                    "common_incentive",
                ):
                    row[key] = int(float(row[key]))
                row["persistence_logit"] = float(row["persistence_logit"])
                for field in fields:
                    row[field] = float(row[field])
                rows.append(row)
    if not rows or not projection_fields:
        raise FileNotFoundError(f"no layerwise projections match {pattern}")
    layers = [int(field.split("_")[1]) for field in projection_fields]
    if layers != list(range(max(layers) + 1)):
        raise ValueError("projection columns do not cover contiguous layers from zero")
    return rows, layers


def audit_source_coverage(
    rows: list[dict], source_pattern: str, *, allow_partial: bool = False
) -> dict:
    expected = set()
    for path in sorted(glob.glob(source_pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            expected.update(
                (row["state_id"], row["stop_payoff"], row["continue_bonus"])
                for row in csv.DictReader(handle)
            )
    if not expected:
        raise FileNotFoundError(
            f"no source factorial rows match {source_pattern}"
        )
    observed = {
        (
            row["state_id"],
            str(row["stop_payoff"]),
            str(row["continue_bonus"]),
        )
        for row in rows
    }
    # Normalize source strings such as "-10.0" to the representation used by
    # the parsed projection rows.
    expected = {
        (state_id, str(int(float(stop))), str(int(float(continued))))
        for state_id, stop, continued in expected
    }
    missing, extra = expected - observed, observed - expected
    if extra or (missing and not allow_partial):
        raise ValueError(
            "layerwise replay does not match the retained factorial: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return {
        "expected_cells": len(expected),
        "observed_cells": len(observed),
        "expected_states": len({key[0] for key in expected}),
        "observed_states": len({key[0] for key in observed}),
        "missing_cells": len(missing),
        "extra_cells": len(extra),
        "allow_partial": allow_partial,
    }


def _coefficient(model: dict, predictor: str) -> dict:
    item = model["coefficients"][predictor]
    predictor_scale = item["predictor_scale"]
    conversion = model["outcome_scale"] / predictor_scale if predictor_scale else 0.0
    raw_slope = item["standardized_beta"] * conversion
    raw_standard_error = (
        item["cluster_robust_standard_error"] * conversion
        if item["cluster_robust_standard_error"] is not None
        else None
    )
    return {
        **item,
        "standardized_beta_lower_95": (
            item["standardized_beta"]
            - 1.96 * item["cluster_robust_standard_error"]
            if item["cluster_robust_standard_error"] is not None
            else None
        ),
        "standardized_beta_upper_95": (
            item["standardized_beta"]
            + 1.96 * item["cluster_robust_standard_error"]
            if item["cluster_robust_standard_error"] is not None
            else None
        ),
        "raw_slope": raw_slope,
        "raw_cluster_robust_standard_error": raw_standard_error,
        "raw_slope_lower_95": (
            raw_slope - 1.96 * raw_standard_error
            if raw_standard_error is not None
            else None
        ),
        "raw_slope_upper_95": (
            raw_slope + 1.96 * raw_standard_error
            if raw_standard_error is not None
            else None
        ),
    }


def analyze_rows(rows: list[dict], layers: list[int]) -> dict:
    audit = audit_rows(rows)
    if (
        audit["incomplete_states"]
        or audit["duplicate_cells"]
        or audit["history_hash_failures"]
        or audit["context_uniqueness_failures"]
    ):
        raise ValueError(f"factorial audit failed: {audit}")
    behavior_factors = _state_fixed_regression(
        rows, "persistence_logit", ("stop_payoff", "continue_bonus")
    )
    behavior_relative = _state_fixed_regression(
        rows, "persistence_logit", ("relative_incentive",)
    )
    behavior = {
        "stop": _coefficient(behavior_factors, "stop_payoff"),
        "continue": _coefficient(behavior_factors, "continue_bonus"),
        "relative_incentive_r_squared": behavior_relative["r_squared"],
    }
    results = []
    for layer in layers:
        outcome = f"layer_{layer:02d}_projection"
        factors = _state_fixed_regression(
            rows, outcome, ("stop_payoff", "continue_bonus")
        )
        relative = _state_fixed_regression(
            rows, outcome, ("relative_incentive",)
        )
        stop = _coefficient(factors, "stop_payoff")
        continued = _coefficient(factors, "continue_bonus")
        results.append(
            {
                "layer": layer,
                "stop": stop,
                "continue": continued,
                "relative_incentive_r_squared": relative["r_squared"],
                "normalized_to_behavior": {
                    "stop_raw_slope": (
                        stop["raw_slope"] / behavior["stop"]["raw_slope"]
                        if behavior["stop"]["raw_slope"]
                        else None
                    ),
                    "continue_raw_slope": (
                        continued["raw_slope"]
                        / behavior["continue"]["raw_slope"]
                        if behavior["continue"]["raw_slope"]
                        else None
                    ),
                    "relative_incentive_r_squared": (
                        relative["r_squared"]
                        / behavior["relative_incentive_r_squared"]
                        if behavior["relative_incentive_r_squared"]
                        else None
                    ),
                },
            }
        )
    result = {
        "analysis": "representational trajectory; not a computation-location test",
        "probes": "frozen bandit persistence probes; no factorial retraining",
        "audit": audit,
        "behavioral_reference": behavior,
        "layers": results,
    }
    result["detection_summary"] = summarize_layerwise_detection(results)
    return result


def make_figure(result: dict, path: Path) -> None:
    svg = Svg(1500, 560)
    svg.text(50, 42, "Incentive effects emerge along frozen persistence directions", "title")
    svg.text(
        50,
        68,
        "State fixed effects; normalized to the final behavioral persistence-logit effect",
        "subtitle",
    )
    panels = (
        (35, "A. STOP payoff", "stop_raw_slope", COLORS["both_negative"]),
        (525, "B. CONTINUE bonus", "continue_raw_slope", COLORS["both_positive"]),
        (1015, "C. Relative-incentive R²", "relative_incentive_r_squared", COLORS["observed"]),
    )
    layers = result["layers"]
    for x, title, key, color in panels:
        values = [row["normalized_to_behavior"][key] for row in layers]
        finite = [value for value in values if value is not None]
        lower, upper = min(0.0, min(finite)), max(1.0, max(finite))
        padding = max(0.05, (upper - lower) * 0.08)
        sx, sy, box = axes(
            svg,
            x,
            95,
            450,
            405,
            title,
            "Layer",
            "Behavior-normalized effect",
            [(layers[0]["layer"], str(layers[0]["layer"])), (layers[-1]["layer"], str(layers[-1]["layer"]))],
            [(lower - padding, f"{lower - padding:.1f}"), (1.0, "1.0"), (upper + padding, f"{upper + padding:.1f}")],
            (layers[0]["layer"], layers[-1]["layer"]),
            (lower - padding, upper + padding),
        )
        svg.line(box[0], sy(1.0), box[2], sy(1.0), COLORS["reference"], 1.2, "5 5")
        points = [
            (sx(row["layer"]), sy(row["normalized_to_behavior"][key]))
            for row in layers
            if row["normalized_to_behavior"][key] is not None
        ]
        svg.polyline(points, color, 2)
        for px, py in points:
            svg.circle(px, py, 3, color)
    svg.text(
        1450,
        540,
        "Trajectory indicates expression along a direction, not where persistence is computed.",
        "note",
        "end",
    )
    svg.save(path)


def make_raw_figure(result: dict, path: Path) -> None:
    svg = Svg(1500, 560)
    svg.text(50, 42, "Factorial effects along frozen persistence directions", "title")
    svg.text(
        50,
        68,
        "State fixed effects; coefficient intervals use episode-clustered standard errors",
        "subtitle",
    )
    layers = result["layers"]
    panels = (
        (35, "A. STOP payoff", "stop", COLORS["both_negative"]),
        (525, "B. CONTINUE bonus", "continue", COLORS["both_positive"]),
    )
    for x, title, key, color in panels:
        intervals = []
        for row in layers:
            coefficient = row[key]["standardized_beta"]
            error = row[key]["cluster_robust_standard_error"] or 0.0
            intervals.extend((coefficient - 1.96 * error, coefficient + 1.96 * error))
        lower, upper = min(0.0, min(intervals)), max(0.0, max(intervals))
        padding = max(0.03, (upper - lower) * 0.08)
        sx, sy, box = axes(
            svg,
            x,
            95,
            450,
            405,
            title,
            "Layer",
            "Standardized beta",
            [(layers[0]["layer"], str(layers[0]["layer"])), (layers[-1]["layer"], str(layers[-1]["layer"]))],
            [(lower - padding, f"{lower - padding:.2f}"), (0.0, "0"), (upper + padding, f"{upper + padding:.2f}")],
            (layers[0]["layer"], layers[-1]["layer"]),
            (lower - padding, upper + padding),
        )
        svg.line(box[0], sy(0.0), box[2], sy(0.0), COLORS["reference"], 1.2, "5 5")
        points = []
        for row in layers:
            coefficient = row[key]["standardized_beta"]
            error = row[key]["cluster_robust_standard_error"] or 0.0
            px, py = sx(row["layer"]), sy(coefficient)
            svg.line(px, sy(coefficient - 1.96 * error), px, sy(coefficient + 1.96 * error), color, 1)
            points.append((px, py))
        svg.polyline(points, color, 2)
        for px, py in points:
            svg.circle(px, py, 3, color)

    values = [row["relative_incentive_r_squared"] for row in layers]
    upper = max(0.05, max(values) * 1.08)
    sx, sy, box = axes(
        svg,
        1015,
        95,
        450,
        405,
        "C. Relative-incentive fit",
        "Layer",
        "Within-state R²",
        [(layers[0]["layer"], str(layers[0]["layer"])), (layers[-1]["layer"], str(layers[-1]["layer"]))],
        [(0.0, "0"), (upper, f"{upper:.2f}")],
        (layers[0]["layer"], layers[-1]["layer"]),
        (0.0, upper),
    )
    points = [
        (sx(row["layer"]), sy(row["relative_incentive_r_squared"]))
        for row in layers
    ]
    svg.polyline(points, COLORS["observed"], 2)
    for px, py in points:
        svg.circle(px, py, 3, COLORS["observed"])
    svg.text(
        1450,
        540,
        "Frozen probes were trained only on the original bandit activation bank.",
        "note",
        "end",
    )
    svg.save(path)


def write_table(result: dict, path: Path) -> None:
    detection = result["detection_summary"]
    stop_nominal = set(detection["stop"]["nominal_detectable_layers"])
    stop_holm = set(detection["stop"]["holm_detectable_layers"])
    continue_nominal = set(detection["continue"]["nominal_detectable_layers"])
    continue_holm = set(detection["continue"]["holm_detectable_layers"])
    fields = (
        "layer",
        "beta_stop",
        "se_stop",
        "lower_95_stop",
        "upper_95_stop",
        "p_stop",
        "nominal_detectable_stop",
        "holm_detectable_stop",
        "beta_continue",
        "se_continue",
        "lower_95_continue",
        "upper_95_continue",
        "p_continue",
        "nominal_detectable_continue",
        "holm_detectable_continue",
        "relative_r_squared",
        "normalized_stop",
        "normalized_continue",
        "normalized_relative_r_squared",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["layers"]:
            writer.writerow(
                {
                    "layer": row["layer"],
                    "beta_stop": row["stop"]["standardized_beta"],
                    "se_stop": row["stop"]["cluster_robust_standard_error"],
                    "lower_95_stop": row["stop"]["standardized_beta_lower_95"],
                    "upper_95_stop": row["stop"]["standardized_beta_upper_95"],
                    "p_stop": row["stop"]["normal_approximation_p_value"],
                    "nominal_detectable_stop": row["layer"] in stop_nominal,
                    "holm_detectable_stop": row["layer"] in stop_holm,
                    "beta_continue": row["continue"]["standardized_beta"],
                    "se_continue": row["continue"]["cluster_robust_standard_error"],
                    "lower_95_continue": row["continue"]["standardized_beta_lower_95"],
                    "upper_95_continue": row["continue"]["standardized_beta_upper_95"],
                    "p_continue": row["continue"]["normal_approximation_p_value"],
                    "nominal_detectable_continue": row["layer"] in continue_nominal,
                    "holm_detectable_continue": row["layer"] in continue_holm,
                    "relative_r_squared": row["relative_incentive_r_squared"],
                    "normalized_stop": row["normalized_to_behavior"]["stop_raw_slope"],
                    "normalized_continue": row["normalized_to_behavior"]["continue_raw_slope"],
                    "normalized_relative_r_squared": row["normalized_to_behavior"]["relative_incentive_r_squared"],
                }
            )


def _layer_text(value) -> str:
    return "none" if value is None else str(value)


def write_report(result: dict, path: Path) -> None:
    detection = result["detection_summary"]
    stop = detection["stop"]
    continued = detection["continue"]
    relative = detection["relative_incentive"]
    audit = result["audit"]
    coverage = result.get("source_coverage")
    lines = [
        "# Track A — factorial incentive trajectory across layers",
        "",
        f"The analysis contains **{audit['complete_states']} complete states**, "
        f"**{audit['episodes']} episode clusters**, and **{audit['rows']} factorial rows**.",
        "All projections use the independently trained, frozen persistence probe from the corresponding layer; no probe was retrained on factorial data.",
        "",
        "## Detectability",
        "",
        f"- STOP first has the expected negative sign with a nominal two-sided 95% clustered interval excluding zero at layer **{_layer_text(stop['first_nominal_detectable_layer'])}**; the first Holm-corrected layer is **{_layer_text(stop['first_holm_detectable_layer'])}** and the first sustained nominal layer is **{_layer_text(stop['first_sustained_nominal_layer'])}**.",
        f"- CONTINUE first has the expected positive sign with a nominal two-sided 95% clustered interval excluding zero at layer **{_layer_text(continued['first_nominal_detectable_layer'])}**; the first Holm-corrected layer is **{_layer_text(continued['first_holm_detectable_layer'])}** and the first sustained nominal layer is **{_layer_text(continued['first_sustained_nominal_layer'])}**.",
        "- Holm–Bonferroni correction is applied separately to the 32 layer tests for each incentive effect.",
        "",
        "## Evolution toward the final layer",
        "",
        f"- At layer {stop['final_layer']}, the STOP raw slope is **{stop['final_raw_slope']:.6g}** and its magnitude is **{stop['final_behavior_normalized_effect']:.3f}×** the behavioral persistence-logit slope.",
        f"- At layer {continued['final_layer']}, the CONTINUE raw slope is **{continued['final_raw_slope']:.6g}** and its magnitude is **{continued['final_behavior_normalized_effect']:.3f}×** the behavioral persistence-logit slope.",
        f"- Relative incentive reaches 25%, 50%, and 75% of its layer-{relative['final_layer']} R² at layers **{_layer_text(relative['first_quarter_final_layer'])}**, **{_layer_text(relative['first_half_final_layer'])}**, and **{_layer_text(relative['first_three_quarters_final_layer'])}**. Final within-state R² is **{relative['final_r_squared']:.3f}**.",
        "",
        "These onset layers identify where incentive-induced differences become expressed along frozen persistence directions. They do not identify where persistence is computed.",
        "",
    ]
    if coverage is not None:
        lines[4:4] = [
            f"Source coverage: **{coverage['observed_cells']}/{coverage['expected_cells']} cells** and **{coverage['observed_states']}/{coverage['expected_states']} states**.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/value_dissociation/layerwise_projections*.csv"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/value_dissociation/layerwise_publication"
    )
    parser.add_argument(
        "--source-factorial",
        default="artifacts/value_dissociation/factorial_shard_*.csv",
    )
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, layers = read_rows(args.input)
    if args.expected_layers > 0 and len(layers) != args.expected_layers:
        raise ValueError(
            f"expected {args.expected_layers} layers, found {len(layers)}"
        )
    result = analyze_rows(rows, layers)
    result["source_coverage"] = audit_source_coverage(
        rows, args.source_factorial, allow_partial=args.allow_partial
    )
    (output / "factorial_layerwise_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_table(result, output / "factorial_layerwise_effects.csv")
    write_report(result, output / "factorial_layerwise_report.md")
    make_raw_figure(result, output / "factorial_layerwise_effects.svg")
    make_figure(result, output / "factorial_layerwise_trajectory.svg")
    print(
        json.dumps(
            {
                "audit": result["audit"],
                "layers": layers,
                "detection_summary": result["detection_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
