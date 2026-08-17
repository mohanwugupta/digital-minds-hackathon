"""Analyze causal transfer of the shared direction to held-out Solvability."""

import argparse
import json
from pathlib import Path

from analysis.analyze_cross_task_causal import (
    _expected_test_state_ids,
    analyze,
    read_rows,
    summarize_task,
)
from analysis.analyze_pilot_detailed import COLORS, Svg, axes
from analysis.shared_persistence_integrity import require_shared_clearance
from experiments.cross_task_utils import load_activation_shards, make_or_validate_split
from experiments.runtime import run_metadata


def control_specificity(
    target_effect: float,
    control_effect: float,
    maximum_absolute_effect: float,
    relative_effect_fraction: float,
) -> bool:
    """Require a control effect to be absolutely small or target-relative weak."""
    return bool(
        abs(float(control_effect)) <= float(maximum_absolute_effect)
        or abs(float(control_effect))
        <= float(relative_effect_fraction) * abs(float(target_effect))
    )


def _write_report(result: dict, path: Path) -> None:
    target = result["solvability"]["target"]
    interval = target["episode_bootstrap"]
    lines = [
        "# Shared persistence causal transfer",
        "",
        f"Classification: **{result['classification'].replace('_', ' ')}**.",
        "The direction was learned from Bandit + Foraging, frozen, and its magnitude was calibrated only on Solvability validation states.",
        "",
        f"+λ minus −λ changed semantic TRY-AGAIN probability by **{target['mean_probability_difference']:.4f}**.",
        f"Counterbalanced-pair-clustered 95% interval: **{interval['lower_95']:.4f} to {interval['upper_95']:.4f}**.",
        f"State-level monotonic fraction: **{target['state_monotonic_fraction']:.3f}**.",
        f"Arbitrary binary-control effect: **{result['negative_control']['target']['mean_probability_difference']:.4f}**.",
        f"Rule-determined PROCEED/END effect: **{result['terminality_control']['target']['mean_probability_difference']:.4f}**.",
        "Control-specificity clearance uses the larger absolute endpoint of each pair-clustered 95% interval, not only its point estimate.",
        "",
        "## Required checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in result["criteria"].items()
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figure(result: dict, path: Path) -> None:
    target = result["solvability"]["target"]
    control = result["negative_control"]["target"]
    terminality = result["terminality_control"]["target"]
    random_95 = result["solvability"]["random_controls"][
        "probability_difference_95th_percentile"
    ]
    entries = [
        ("Solvability", target["mean_probability_difference"], COLORS["observed"]),
        ("Random 95th", random_95, COLORS["reference"]),
        ("Binary control", control["mean_probability_difference"], COLORS["both_negative"]),
        ("Rule terminality", terminality["mean_probability_difference"], COLORS["both_negative"]),
    ]
    interval = target["episode_bootstrap"]
    values = [value for _name, value, _color in entries]
    values.extend([interval["lower_95"], interval["upper_95"], 0.0])
    lower, upper = min(values) - 0.03, max(values) + 0.03
    if lower == upper:
        lower, upper = -0.1, 0.1
    svg = Svg(920, 620)
    svg.text(50, 42, "Shared persistence causal transfer", "title")
    svg.text(50, 68, "+λ minus −λ semantic-choice probability", "subtitle")
    sx, sy, box = axes(
        svg,
        70,
        105,
        780,
        410,
        "",
        "Condition",
        "Probability effect",
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
    svg.text(
        870,
        585,
        f"Classification: {result['classification'].replace('_', ' ')}",
        "note",
        "end",
    )
    svg.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvability-input", required=True)
    parser.add_argument("--control-input", required=True)
    parser.add_argument("--terminality-input", required=True)
    parser.add_argument("--solvability-bank", required=True)
    parser.add_argument("--control-bank", required=True)
    parser.add_argument("--terminality-bank", required=True)
    parser.add_argument("--solvability-split", required=True)
    parser.add_argument("--control-split", required=True)
    parser.add_argument("--terminality-split", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--representational-summary", required=True)
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import yaml

    representational = require_shared_clearance(args.representational_summary)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    if (
        calibration.get("status") != "valid"
        or calibration.get("selection_data") != "solvability validation states only"
        or calibration.get("test_states_inspected") is not False
        or calibration.get("causal_target_task") != "solvability"
    ):
        raise ValueError("causal analysis requires valid Solvability-validation calibration")
    causal = config["causal_transfer"]
    solvability_shards = load_activation_shards(args.solvability_bank)
    control_shards = load_activation_shards(args.control_bank)
    terminality_shards = load_activation_shards(args.terminality_bank)
    solvability_split = make_or_validate_split(
        solvability_shards, args.solvability_split, seed=int(config["split_seed"])
    )
    control_split = make_or_validate_split(
        control_shards, args.control_split, seed=int(config["split_seed"])
    )
    terminality_split = make_or_validate_split(
        terminality_shards,
        args.terminality_split,
        seed=int(config["split_seed"]),
    )
    solvability_rows = read_rows(args.solvability_input)
    control_rows = read_rows(args.control_input)
    terminality_rows = read_rows(args.terminality_input)
    expected_solvability = _expected_test_state_ids(
        solvability_shards, solvability_split
    )
    expected_control = _expected_test_state_ids(control_shards, control_split)
    expected_terminality = _expected_test_state_ids(
        terminality_shards, terminality_split
    )
    observed_solvability = {row["state_id"] for row in solvability_rows}
    observed_control = {row["state_id"] for row in control_rows}
    observed_terminality = {row["state_id"] for row in terminality_rows}
    if (
        observed_solvability != expected_solvability
        or observed_control != expected_control
        or observed_terminality != expected_terminality
    ):
        raise ValueError(
            "shared causal test-state coverage failed: "
            f"solvability missing/extra={len(expected_solvability - observed_solvability)}/"
            f"{len(observed_solvability - expected_solvability)}, control missing/extra="
            f"{len(expected_control - observed_control)}/"
            f"{len(observed_control - expected_control)}, terminality missing/extra="
            f"{len(expected_terminality - observed_terminality)}/"
            f"{len(observed_terminality - expected_terminality)}"
        )
    result = analyze(
        solvability_rows,
        control_rows,
        bootstrap_samples=int(causal["episode_bootstrap_samples"]),
        seed=int(config["analysis_seed"]) + 30_000,
        expected_random_directions=int(causal["matched_random_directions"]),
        baseline_tolerance=float(causal["baseline_tolerance"]),
        negative_control_max_absolute_effect=float(
            causal["negative_control_max_absolute_probability_effect"]
        ),
        negative_control_relative_effect_fraction=float(
            causal["negative_control_relative_effect_fraction"]
        ),
        require_negative_control=True,
    )
    result["solvability"] = result.pop("foraging")
    terminality = summarize_task(
        terminality_rows,
        bootstrap_samples=int(causal["episode_bootstrap_samples"]),
        seed=int(config["analysis_seed"]) + 30_002,
        expected_random_directions=int(causal["matched_random_directions"]),
        baseline_tolerance=float(causal["baseline_tolerance"]),
    )
    target_effect = result["solvability"]["target"]["mean_probability_difference"]
    result["terminality_control"] = terminality
    result["analysis_roles"]["terminality_control"] = "confirmatory"
    binary_interval = result["negative_control"]["target"]["episode_bootstrap"]
    terminality_interval = terminality["target"]["episode_bootstrap"]
    binary_upper_absolute = max(
        abs(binary_interval["lower_95"]), abs(binary_interval["upper_95"])
    )
    terminality_upper_absolute = max(
        abs(terminality_interval["lower_95"]),
        abs(terminality_interval["upper_95"]),
    )
    result["negative_control"]["upper_95_absolute_probability_effect"] = (
        binary_upper_absolute
    )
    result["terminality_control"]["upper_95_absolute_probability_effect"] = (
        terminality_upper_absolute
    )
    result["criteria"]["negative_control_specificity"] = control_specificity(
        target_effect,
        binary_upper_absolute,
        float(causal["negative_control_max_absolute_probability_effect"]),
        float(causal["negative_control_relative_effect_fraction"]),
    )
    result["criteria"]["terminality_control_specificity"] = control_specificity(
        target_effect,
        terminality_upper_absolute,
        float(causal["negative_control_max_absolute_probability_effect"]),
        float(causal["negative_control_relative_effect_fraction"]),
    )
    calibration_valid = (
        result["criteria"]["decoded_quantity_ordered"]
        and result["criteria"]["zero_steering_reproduces_baseline"]
    )
    result["classification"] = (
        "causal_transfer"
        if all(result["criteria"].values())
        else "invalid_or_inconclusive"
        if not calibration_valid
        else "no_convincing_causal_transfer"
    )
    if set(causal["shared_required_checks"]) != set(result["criteria"]):
        raise ValueError("causal preregistration and implemented criteria differ")
    result.update(
        {
            "analysis_role": "heldout_shared_direction_causal_transfer",
            "representational_gate": {
                "classification": representational["classification"],
                "heldout_task_parameters_fit": representational[
                    "heldout_task_parameters_fit"
                ],
            },
            "calibration": calibration,
            "decision_matrix_outcome": (
                "task_general_persistence_related_causal_direction"
                if result["classification"] == "causal_transfer"
                else "shared_prediction_without_heldout_causal_control"
                if result["classification"] == "no_convincing_causal_transfer"
                else "causal_checkpoint_invalid_or_inconclusive"
            ),
            "test_state_coverage": {
                "solvability_expected": len(expected_solvability),
                "solvability_observed": len(observed_solvability),
                "control_expected": len(expected_control),
                "control_observed": len(observed_control),
                "terminality_expected": len(expected_terminality),
                "terminality_observed": len(observed_terminality),
                "passed": True,
            },
            "preregistered_config": str(Path(args.config).resolve()),
            "provenance": run_metadata(
                {
                    "model": solvability_shards[0].get("model_id", config["model"]),
                    "analysis": "shared_persistence_solvability_causal_transfer",
                    "config": str(Path(args.config).resolve()),
                }
            ),
        }
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "shared_causal_transfer_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(result, output / "shared_causal_transfer_report.md")
    _make_figure(result, output / "shared_causal_transfer.svg")
    print(json.dumps({"classification": result["classification"], "criteria": result["criteria"]}, indent=2))


if __name__ == "__main__":
    main()
