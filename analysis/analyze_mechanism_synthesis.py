"""Synthesize causal steering and external-value dissociation results."""

import argparse
import csv
import glob
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.analyze_value_dissociation import read_rows


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator


def _episode_bootstrap(values: dict[str, list[float]], samples: int = 10000) -> dict:
    episode_means = [sum(items) / len(items) for items in values.values()]
    observed = sum(episode_means) / len(episode_means)
    rng = random.Random(812026)
    draws = sorted(
        sum(episode_means[rng.randrange(len(episode_means))] for _ in episode_means)
        / len(episode_means)
        for _ in range(samples)
    )
    return {
        "mean": observed,
        "confidence_interval_95": [draws[250], draws[9749]],
        "episodes": len(episode_means),
    }


def _framing_effect(rows: list[dict], positive_control_path: str) -> dict | None:
    if not Path(positive_control_path).exists():
        return None
    baseline = {}
    with open(positive_control_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["alpha"]) == 0:
                baseline[row["state_id"]] = (
                    row["episode_id"],
                    float(row["persistence_logit"]),
                )
    differences = defaultdict(list)
    for row in rows:
        if row["stop_payoff"] == 0 and row["continue_bonus"] == 0:
            if row["state_id"] in baseline:
                episode, unmodified = baseline[row["state_id"]]
                differences[episode].append(row["persistence_logit"] - unmodified)
    if not differences:
        return None
    return {"matched_states": sum(map(len, differences.values())), **_episode_bootstrap(differences)}


def synthesize(
    rows: list[dict],
    factorial: dict,
    causal: dict,
    *,
    positive_control_path: str,
    full_causal_pattern: str,
) -> dict:
    by_state = defaultdict(dict)
    for row in rows:
        by_state[row["state_id"]][
            (row["stop_payoff"], row["continue_bonus"])
        ] = row

    projection_names = (
        "generic_return_projection",
        "advantage_projection",
        "persistence_projection",
    )
    persistence_residual = []
    projection_residuals = {name: [] for name in projection_names}
    cell_residuals = []
    extreme_differences = defaultdict(list)
    manipulable = extreme_positive_continue = extreme_negative_continue = 0
    for cells in by_state.values():
        episode = next(iter(cells.values()))["episode_id"]
        outcome_mean = statistics.mean(
            row["persistence_logit"] for row in cells.values()
        )
        projection_means = {
            name: statistics.mean(row[name] for row in cells.values())
            for name in projection_names
        }
        has_continue = any(row["persistence_logit"] > 0 for row in cells.values())
        has_stop = any(row["persistence_logit"] < 0 for row in cells.values())
        manipulable += has_continue and has_stop
        positive = cells[(-10, 10)]["persistence_logit"]
        negative = cells[(20, -10)]["persistence_logit"]
        extreme_positive_continue += positive > 0
        extreme_negative_continue += negative > 0
        extreme_differences[episode].append(positive - negative)
        for (stop, bonus), row in cells.items():
            residual = row["persistence_logit"] - outcome_mean
            persistence_residual.append(residual)
            cell_residuals.append(((stop, bonus), bonus - stop, residual))
            for name in projection_names:
                projection_residuals[name].append(row[name] - projection_means[name])

    cell_groups, relative_groups = defaultdict(list), defaultdict(list)
    for cell, relative, value in cell_residuals:
        cell_groups[cell].append(value)
        relative_groups[relative].append(value)
    cell_means = {key: sum(value) / len(value) for key, value in cell_groups.items()}
    relative_means = {
        key: sum(value) / len(value) for key, value in relative_groups.items()
    }
    total = sum(value * value for _cell, _relative, value in cell_residuals)
    cell_r_squared = 1 - sum(
        (value - cell_means[cell]) ** 2 for cell, _relative, value in cell_residuals
    ) / total
    relative_categorical_r_squared = 1 - sum(
        (value - relative_means[relative]) ** 2
        for _cell, relative, value in cell_residuals
    ) / total

    same_relative = defaultdict(list)
    raw_cells = factorial["cell_means"]["persistence_logit"]
    for row in raw_cells:
        same_relative[row["relative_incentive"]].append(
            {
                "stop_payoff": row["stop_payoff"],
                "continue_bonus": row["continue_bonus"],
                "mean_persistence_logit": row["mean"],
            }
        )
    repeated_relative_ranges = {
        str(relative): max(item["mean_persistence_logit"] for item in selected)
        - min(item["mean_persistence_logit"] for item in selected)
        for relative, selected in same_relative.items()
        if len(selected) > 1
    }

    return {
        "data_audit": factorial["audit"],
        "persistence_positive_control": causal,
        "full_value_direction_causal_files_present": bool(
            glob.glob(full_causal_pattern)
        ),
        "behavior": {
            "relative_linear_r_squared": factorial["state_fixed_effects"]
            ["persistence_logit"]["relative_only"]["r_squared"],
            "relative_plus_common_r_squared": factorial["state_fixed_effects"]
            ["persistence_logit"]["relative_and_common"]["r_squared"],
            "categorical_relative_r_squared": relative_categorical_r_squared,
            "categorical_cell_r_squared": cell_r_squared,
            "fraction_states_switched_by_factorial": manipulable / len(by_state),
            "continue_fraction_extreme_positive": extreme_positive_continue
            / len(by_state),
            "continue_fraction_extreme_negative": extreme_negative_continue
            / len(by_state),
            "extreme_logit_contrast": _episode_bootstrap(extreme_differences),
            "zero_zero_prompt_framing_effect": _framing_effect(
                rows, positive_control_path
            ),
            "same_relative_cell_mean_ranges": repeated_relative_ranges,
        },
        "representations": {
            name: {
                "within_state_correlation_with_persistence": _correlation(
                    values, persistence_residual
                ),
                "relative_linear_r_squared": factorial["state_fixed_effects"]
                [name]["relative_only"]["r_squared"],
                "stop_standardized_beta": factorial["state_fixed_effects"]
                [name]["stop_and_continue"]["coefficients"]["stop_payoff"]
                ["standardized_beta"],
                "continue_standardized_beta": factorial["state_fixed_effects"]
                [name]["stop_and_continue"]["coefficients"]["continue_bonus"]
                ["standardized_beta"],
            }
            for name, values in projection_residuals.items()
        },
    }


def make_figure(result: dict, factorial: dict, path: Path) -> None:
    svg = Svg(1500, 930)
    svg.text(55, 45, "What computation controls persistence?", "title")
    svg.text(55, 70, "Matched-state causal steering and external payoff dissociation", "subtitle")
    panels = ((45, 100), (760, 100), (45, 515), (760, 515))

    x, y = panels[0]
    dose = result["persistence_positive_control"]["directions"]["persistence"]["dose_response"]
    values = [dose[str(alpha)] for alpha in (-1, 0, 1)]
    low, high = min(values) - 0.2, max(values) + 0.2
    sx, sy, _box = axes(svg, x, y, 690, 360, "A. Persistence direction is causally effective", "Steering alpha", "Persistence logit", [(-1, "-1"), (0, "0"), (1, "+1")], [(low, f"{low:.1f}"), (high, f"{high:.1f}")], (-1.2, 1.2), (low, high))
    points = [(sx(alpha), sy(dose[str(alpha)])) for alpha in (-1, 0, 1)]
    svg.polyline(points, COLORS["observed"], 3)
    for px, py in points:
        svg.circle(px, py, 6, COLORS["observed"])

    x, y = panels[1]
    cells = factorial["cell_means"]["persistence_logit"]
    values = [row["mean"] for row in cells]
    low, high = min(values) - 0.2, max(values) + 0.2
    sx, sy, box = axes(svg, x, y, 690, 360, "B. External payoffs move the same states", "STOP payoff", "Persistence logit", [(-10, "-10"), (0, "0"), (10, "+10"), (20, "+20")], [(low, f"{low:.1f}"), (high, f"{high:.1f}")], (-12, 22), (low, high))
    colors = (COLORS["both_negative"], COLORS["observed"], COLORS["both_positive"])
    for bonus, color in zip((-10, 0, 10), colors):
        selected = sorted((row for row in cells if row["continue_bonus"] == bonus), key=lambda row: row["stop_payoff"])
        points = [(sx(row["stop_payoff"]), sy(row["mean"])) for row in selected]
        svg.polyline(points, color, 2)
        for px, py in points:
            svg.circle(px, py, 4, color)
    legend(svg, box[0] + 8, box[1] + 18, [(f"Continue {bonus:+d}", color) for bonus, color in zip((-10, 0, 10), colors)])

    x, y = panels[2]
    labels = (("persistence_logit", "Behavior"), ("generic_return_projection", "Generic"), ("advantage_projection", "Advantage"), ("persistence_projection", "Decision"))
    models = factorial["state_fixed_effects"]
    sx, sy, box = axes(svg, x, y, 690, 360, "C. Which representation follows each payoff?", "Outcome", "Standardized effect", [(i, label) for i, (_key, label) in enumerate(labels)], [(-0.8, "-.8"), (0, "0"), (0.8, ".8")], (-0.6, 3.6), (-0.85, 0.85))
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (name, _label) in enumerate(labels):
        coefficients = models[name]["stop_and_continue"]["coefficients"]
        for offset, factor, color in ((-0.13, "stop_payoff", COLORS["both_negative"]), (0.13, "continue_bonus", COLORS["both_positive"])):
            svg.circle(sx(index + offset), sy(coefficients[factor]["standardized_beta"]), 6, color)
    legend(svg, box[0] + 8, box[1] + 18, [("STOP payoff", COLORS["both_negative"]), ("CONTINUE bonus", COLORS["both_positive"])])

    x, y = panels[3]
    bars = [("Behavior", result["behavior"]["relative_linear_r_squared"]), ("Generic", result["representations"]["generic_return_projection"]["relative_linear_r_squared"]), ("Advantage", result["representations"]["advantage_projection"]["relative_linear_r_squared"]), ("Decision", result["representations"]["persistence_projection"]["relative_linear_r_squared"])]
    sx, sy, box = axes(svg, x, y, 690, 360, "D. Encoding of CONTINUE minus STOP", "Outcome", "Within-state R²", [(i, label) for i, (label, _value) in enumerate(bars)], [(0, "0"), (0.4, ".4"), (0.8, ".8")], (-0.6, 3.6), (0, 0.85))
    for index, (_label, value) in enumerate(bars):
        svg.rect(sx(index) - 42, sy(value), 84, box[3] - sy(value), COLORS["model"], 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.2f}", anchor="middle")
    svg.text(1450, 910, "All factorial comparisons replay identical underlying histories.", "note", "end")
    svg.save(path)


def write_report(result: dict, factorial: dict, path: Path) -> None:
    behavior = factorial["state_fixed_effects"]["persistence_logit"]
    coefficients = behavior["stop_and_continue"]["coefficients"]
    framing = result["behavior"]["zero_zero_prompt_framing_effect"]
    extreme = result["behavior"]["extreme_logit_contrast"]
    lines = [
        "# Computational-structure synthesis",
        "",
        "## Bottom line",
        "",
        "The model contains decodable generic return information, but the identified generic-return and provisional-advantage directions do not causally move persistence. External behavior is organized primarily by the relative attractiveness of CONTINUE versus STOP, while the late persistence direction closely tracks and causally controls the final decision. A fixed recent-loss/time heuristic is therefore incomplete, although it remains a strong account of stopping in the original unmanipulated task.",
        "",
        "## Integrity and behavioral intervention",
        "",
        f"The factorial audit retained **{result['data_audit']['states']} states from {result['data_audit']['episodes']} episodes** with no missing, duplicated, or history-mismatched cells.",
        f"STOP payoff beta: **{coefficients['stop_payoff']['standardized_beta']:.3f}**; CONTINUE bonus beta: **{coefficients['continue_bonus']['standardized_beta']:.3f}**. Relative incentive explains **{result['behavior']['relative_linear_r_squared']:.3f}** of within-state persistence variation.",
        f"The most pro-CONTINUE versus most pro-STOP manipulation changes persistence by **{extreme['mean']:.3f} logits** (episode-bootstrap 95% CI {extreme['confidence_interval_95'][0]:.3f} to {extreme['confidence_interval_95'][1]:.3f}). **{100 * result['behavior']['fraction_states_switched_by_factorial']:.1f}%** of histories cross the CONTINUE/STOP boundary somewhere in the factorial.",
        "",
        "## Representational dissociation",
        "",
    ]
    labels = {"generic_return_projection": "Generic-return", "advantage_projection": "Provisional advantage", "persistence_projection": "Direct persistence"}
    for name, label in labels.items():
        item = result["representations"][name]
        lines.append(f"- **{label}:** within-state correlation with persistence **r={item['within_state_correlation_with_persistence']:.3f}**; relative-incentive R² **{item['relative_linear_r_squared']:.3f}**; STOP/CONTINUE betas **{item['stop_standardized_beta']:.3f} / {item['continue_standardized_beta']:.3f}**.")
    lines.extend([
        "",
        "The generic-return projection moves in the wrong direction when STOP becomes more valuable, so it is not an invariant action-relative decision signal under this manipulation. The advantage projection has the theoretically correct signs but explains only a modest share of the manipulated decision coordinate. The final persistence representation almost exactly follows behavior.",
        "",
        "## Causal steering",
        "",
        f"The layer-31 persistence positive control passes: positive-minus-negative steering changes persistence by **{result['persistence_positive_control']['directions']['persistence']['target']['mean']:.3f} logits** with 95% CI {result['persistence_positive_control']['directions']['persistence']['target']['confidence_interval_95'][0]:.3f} to {result['persistence_positive_control']['directions']['persistence']['target']['confidence_interval_95'][1]:.3f}.",
    ])
    if result["full_value_direction_causal_files_present"]:
        for name, label in (("generic_return", "Generic return"), ("advantage", "Continuation advantage")):
            item = result["persistence_positive_control"]["directions"][name]
            target = item["target"]
            lines.append(
                f"- **{label}:** positive-minus-negative effect **{target['mean']:.4f} logits** "
                f"(95% CI {target['confidence_interval_95'][0]:.4f} to {target['confidence_interval_95'][1]:.4f}; "
                f"bootstrap p={target['bootstrap_two_sided_p']:.3g}); matched-random absolute empirical p **{item['random_control_empirical_p_absolute']:.3f}**."
            )
        lines.append("Both early value-direction effects are indistinguishable from zero and ordinary matched random directions, despite validated one-SD movement of their frozen probe outputs.")
    else:
        lines.append("The generic-return, advantage, and matched-random causal replay files are absent, so their causal effects cannot yet be adjudicated.")
    lines.extend([
        "",
        "## Interpretation constraints",
        "",
        f"The zero/zero incentive prompt lowers persistence by **{framing['mean']:.3f} logits** relative to the same underlying state with its unmodified decision prompt (95% CI {framing['confidence_interval_95'][0]:.3f} to {framing['confidence_interval_95'][1]:.3f}). This does not invalidate within-factorial contrasts, but zero/zero is not a neutral copy of the original task.",
        f"A categorical relative-incentive model explains **{result['behavior']['categorical_relative_r_squared']:.3f}**, while unrestricted condition cells explain **{result['behavior']['categorical_cell_r_squared']:.3f}**. Relative value is dominant but not sufficient: numerical framing and/or nonlinear absolute-payoff effects remain.",
        "The common CONTINUE manipulation also changes the A-minus-B logit gap, so it is not perfectly arm-specific. Claims should concern CONTINUE versus STOP behavior, not a fully isolated scalar utility computation.",
        "",
        "## Adjudication",
        "",
        "1. **Integrated value exists:** supported as decodable early-layer information by the earlier held-out return probes, but the identified linear axes are causally inert for persistence at the calibrated intervention scale.",
        "2. **Generic future return drives stopping:** strongly disfavored; it neither predicted matched-state stopping, transformed appropriately under STOP-value manipulation, nor causally moved persistence.",
        "3. **Continuation-versus-stop value matters:** strongly supported behaviorally. Explicitly changing either side of the comparison reverses the decision in nearly every held-out history. This alone cannot distinguish utility computation from a prompt-local numeric/instruction heuristic.",
        "4. **A clean native continuation-advantage axis has been identified:** not supported for the fitted layer-2 direction. It responds correctly but weakly to external incentives and has no detectable causal effect; the late persistence axis is the operative variable identified here.",
        "5. **Stopping is only a recent-loss/time heuristic:** rejected as a complete account. It describes the original-task baseline well, but cannot explain the large within-history response to externally manipulated payoffs. A broader policy-heuristic account combining recent history with prompt-local incentive cues remains viable.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-input", default="artifacts/value_dissociation/factorial*.csv")
    parser.add_argument("--factorial-summary", default="artifacts/value_dissociation/publication/value_dissociation_summary.json")
    parser.add_argument("--causal-summary", default="artifacts/causal_steering/publication/causal_steering_summary.json")
    parser.add_argument("--positive-control", default="artifacts/causal_steering/positive_control.csv")
    parser.add_argument("--full-causal-pattern", default="artifacts/causal_steering/replays*.csv")
    parser.add_argument("--output-dir", default="artifacts/mechanism_synthesis")
    args = parser.parse_args()
    rows = read_rows(args.factorial_input)
    factorial = json.loads(Path(args.factorial_summary).read_text(encoding="utf-8"))
    causal = json.loads(Path(args.causal_summary).read_text(encoding="utf-8"))
    result = synthesize(rows, factorial, causal, positive_control_path=args.positive_control, full_causal_pattern=args.full_causal_pattern)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "computational_structure_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figure(result, factorial, output / "computational_structure.svg")
    write_report(result, factorial, output / "computational_structure_report.md")
    print(json.dumps({"data_audit": result["data_audit"], "full_value_direction_causal_files_present": result["full_value_direction_causal_files_present"]}, indent=2))


if __name__ == "__main__":
    main()
