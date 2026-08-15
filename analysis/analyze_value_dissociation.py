"""Analyze repeated-state STOP-payoff × CONTINUE-bonus dissociation."""

import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.probe_history_matching import _clustered_regression


OUTCOMES = (
    "persistence_logit",
    "generic_return_projection",
    "advantage_projection",
    "persistence_projection",
    "arm_logit_gap",
)
EXPECTED_CELLS = {
    (stop, bonus)
    for stop in (-10, 0, 10, 20)
    for bonus in (-10, 0, 10)
}


def _loss_streak(row: dict) -> int:
    if "loss_streak" in row and row["loss_streak"] not in (None, ""):
        return int(float(row["loss_streak"]))
    rewards = row.get("reward_history", [])
    if isinstance(rewards, str):
        rewards = json.loads(rewards)
    streak = 0
    for reward in reversed(rewards):
        if float(reward) != -2:
            break
        streak += 1
    return streak


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                row = dict(source)
                for key in ("stop_payoff", "continue_bonus", "relative_incentive", "common_incentive", "round"):
                    row[key] = int(float(row[key]))
                for key in (
                    "persistence_logit",
                    "p_continue",
                    "generic_return_projection",
                    "advantage_projection",
                    "persistence_projection",
                    "logit_A",
                    "logit_B",
                ):
                    row[key] = float(row[key])
                row["previous_outcome"] = (
                    None
                    if row.get("previous_outcome") in (None, "")
                    else float(row["previous_outcome"])
                )
                row["loss_streak"] = _loss_streak(row)
                row["arm_logit_gap"] = row["logit_A"] - row["logit_B"]
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no value-dissociation rows match {pattern}")
    return rows


def audit_rows(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
    complete, history_failures, context_failures = 0, 0, 0
    for selected in grouped.values():
        cells = {(row["stop_payoff"], row["continue_bonus"]) for row in selected}
        complete += cells == EXPECTED_CELLS and len(selected) == 12
        history_failures += len({row["history_hash"] for row in selected}) != 1
        context_failures += len({row["context_hash"] for row in selected}) != len(selected)
    keys = [
        (row["state_id"], row["stop_payoff"], row["continue_bonus"])
        for row in rows
    ]
    return {
        "rows": len(rows),
        "states": len(grouped),
        "episodes": len({row["episode_id"] for row in rows}),
        "complete_states": complete,
        "incomplete_states": len(grouped) - complete,
        "history_hash_failures": history_failures,
        "context_uniqueness_failures": context_failures,
        "duplicate_cells": len(keys) - len(set(keys)),
    }


def _state_fixed_regression(
    rows: list[dict], outcome: str, predictors: tuple[str, ...]
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
    residual_outcome, residual_predictors, clusters = [], {key: [] for key in predictors}, []
    for selected in grouped.values():
        outcome_mean = statistics.mean(row[outcome] for row in selected)
        predictor_means = {
            key: statistics.mean(row[key] for row in selected) for key in predictors
        }
        for row in selected:
            residual_outcome.append(row[outcome] - outcome_mean)
            for key in predictors:
                residual_predictors[key].append(row[key] - predictor_means[key])
            clusters.append(row["episode_id"])
    result = _clustered_regression(
        residual_outcome, residual_predictors, clusters
    )
    for key, coefficient in result["coefficients"].items():
        scale = coefficient["predictor_scale"]
        coefficient["raw_slope"] = (
            coefficient["standardized_beta"] * result["outcome_scale"] / scale
            if scale
            else 0.0
        )
    result["fixed_effect"] = "state"
    return result


def _recent_controls(rows: list[dict]) -> dict[str, list[float]]:
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


def _cell_means(rows: list[dict], outcome: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stop_payoff"], row["continue_bonus"])].append(row[outcome])
    return [
        {
            "stop_payoff": stop,
            "continue_bonus": bonus,
            "relative_incentive": bonus - stop,
            "mean": statistics.mean(grouped[(stop, bonus)]),
            "states": len(grouped[(stop, bonus)]),
        }
        for stop, bonus in sorted(EXPECTED_CELLS)
    ]


def analyze_rows(rows: list[dict]) -> dict:
    for row in rows:
        if "arm_logit_gap" not in row:
            row["arm_logit_gap"] = float(row["logit_A"]) - float(row["logit_B"])
    audit = audit_rows(rows)
    if (
        audit["incomplete_states"]
        or audit["duplicate_cells"]
        or audit["history_hash_failures"]
        or audit["context_uniqueness_failures"]
    ):
        raise ValueError(f"factorial audit failed: {audit}")
    state_fixed = {}
    for outcome in OUTCOMES:
        state_fixed[outcome] = {
            "stop_and_continue": _state_fixed_regression(
                rows, outcome, ("stop_payoff", "continue_bonus")
            ),
            "relative_only": _state_fixed_regression(
                rows, outcome, ("relative_incentive",)
            ),
            "relative_and_common": _state_fixed_regression(
                rows, outcome, ("relative_incentive", "common_incentive")
            ),
            "stop_only": _state_fixed_regression(rows, outcome, ("stop_payoff",)),
            "continue_only": _state_fixed_regression(
                rows, outcome, ("continue_bonus",)
            ),
        }
    recent = _recent_controls(rows)
    persistence = [row["persistence_logit"] for row in rows]
    clusters = [row["episode_id"] for row in rows]
    pooled = {
        "stop_and_continue": _clustered_regression(
            persistence,
            {
                "stop_payoff": [row["stop_payoff"] for row in rows],
                "continue_bonus": [row["continue_bonus"] for row in rows],
                **recent,
            },
            clusters,
        ),
        "relative_and_common": _clustered_regression(
            persistence,
            {
                "relative_incentive": [row["relative_incentive"] for row in rows],
                "common_incentive": [row["common_incentive"] for row in rows],
                **recent,
            },
            clusters,
        ),
        "relative_only": _clustered_regression(
            persistence,
            {
                "relative_incentive": [row["relative_incentive"] for row in rows],
                **recent,
            },
            clusters,
        ),
    }
    return {
        "audit": audit,
        "state_fixed_effects": state_fixed,
        "pooled_behavior_with_recent_history": pooled,
        "cell_means": {
            outcome: _cell_means(rows, outcome) for outcome in OUTCOMES
        },
        "collinearity_note": (
            "continue_bonus - stop_payoff is exactly collinear with the two "
            "factor terms. Models use either STOP+CONTINUE or relative+common "
            "parameterizations, never all three in one design matrix."
        ),
    }


def make_figure(result: dict, path: Path) -> None:
    svg = Svg(1500, 1040)
    svg.text(55, 45, "External value dissociation reveals the persistence computation", "title")
    svg.text(55, 70, "Identical histories under 12 STOP-payoff × CONTINUE-bonus conditions", "subtitle")
    panels = ((45, 100), (760, 100), (45, 565), (760, 565))
    behavior_cells = result["cell_means"]["persistence_logit"]

    x, y = panels[0]
    values = [row["mean"] for row in behavior_cells]
    lower, upper = min(values) - 0.2, max(values) + 0.2
    sx, sy, box = axes(
        svg, x, y, 690, 420, "A. Independent payoff dose responses",
        "STOP payoff", "Mean persistence logit",
        [(-10, "-10"), (0, "0"), (10, "+10"), (20, "+20")],
        [(lower, f"{lower:.1f}"), (upper, f"{upper:.1f}")],
        (-12, 22), (lower, upper),
    )
    colors = (COLORS["both_negative"], COLORS["observed"], COLORS["both_positive"])
    for bonus, color in zip((-10, 0, 10), colors):
        selected = sorted(
            (row for row in behavior_cells if row["continue_bonus"] == bonus),
            key=lambda row: row["stop_payoff"],
        )
        points = [(sx(row["stop_payoff"]), sy(row["mean"])) for row in selected]
        svg.polyline(points, color, 2)
        for px, py in points:
            svg.circle(px, py, 4, color)
    legend(svg, box[0] + 5, box[1] + 18, [(f"Continue {bonus:+d}", color) for bonus, color in zip((-10, 0, 10), colors)])

    x, y = panels[1]
    relative_grouped = defaultdict(list)
    for row in behavior_cells:
        relative_grouped[row["relative_incentive"]].append(row["mean"])
    relative = sorted(
        (value, statistics.mean(means)) for value, means in relative_grouped.items()
    )
    values = [mean for _value, mean in relative]
    lower, upper = min(values) - 0.2, max(values) + 0.2
    sx, sy, box = axes(
        svg, x, y, 690, 420, "B. Persistence by relative incentive",
        "CONTINUE bonus - STOP payoff", "Mean persistence logit",
        [(value, f"{value:+d}") for value, _mean in relative],
        [(lower, f"{lower:.1f}"), (upper, f"{upper:.1f}")],
        (min(value for value, _ in relative) - 2, max(value for value, _ in relative) + 2),
        (lower, upper),
    )
    points = [(sx(value), sy(mean)) for value, mean in relative]
    svg.polyline(points, COLORS["observed"], 2)
    for px, py in points:
        svg.circle(px, py, 5, COLORS["observed"])

    x, y = panels[2]
    outcomes = (
        ("persistence_logit", "Behavior"),
        ("generic_return_projection", "Generic return"),
        ("advantage_projection", "Advantage"),
        ("persistence_projection", "Persistence repr."),
    )
    sx, sy, box = axes(
        svg, x, y, 690, 420, "C. Experimental-factor coefficients",
        "Representation / behavior", "Standardized coefficient",
        [(index, label) for index, (_key, label) in enumerate(outcomes)],
        [(-1, "-1"), (-0.5, "-.5"), (0, "0"), (0.5, ".5"), (1, "1")],
        (-0.6, len(outcomes) - 0.4), (-1.1, 1.1),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (outcome, _label) in enumerate(outcomes):
        coefficients = result["state_fixed_effects"][outcome]["stop_and_continue"]["coefficients"]
        for offset, key, color in (
            (-0.13, "stop_payoff", COLORS["both_negative"]),
            (0.13, "continue_bonus", COLORS["both_positive"]),
        ):
            item = coefficients[key]
            beta, error = item["standardized_beta"], item["cluster_robust_standard_error"] or 0
            svg.line(sx(index + offset), sy(beta - 1.96 * error), sx(index + offset), sy(beta + 1.96 * error), color, 2)
            svg.circle(sx(index + offset), sy(beta), 5, color)
    legend(svg, box[0] + 5, box[1] + 18, [("STOP payoff", COLORS["both_negative"]), ("CONTINUE bonus", COLORS["both_positive"])])

    x, y = panels[3]
    models = result["state_fixed_effects"]["persistence_logit"]
    model_rows = (
        ("STOP only", models["stop_only"]["r_squared"]),
        ("CONTINUE only", models["continue_only"]["r_squared"]),
        ("Relative only", models["relative_only"]["r_squared"]),
        ("Relative + common", models["relative_and_common"]["r_squared"]),
    )
    upper = max(value for _label, value in model_rows) * 1.15
    sx, sy, box = axes(
        svg, x, y, 690, 420, "D. Which value coordinate organizes behavior?",
        "Model", "Within-state R²",
        [(index, label) for index, (label, _value) in enumerate(model_rows)],
        [(0, "0"), (upper, f"{upper:.2f}")],
        (-0.6, len(model_rows) - 0.4), (0, upper),
    )
    for index, (_label, value) in enumerate(model_rows):
        svg.rect(sx(index) - 45, sy(value), 90, box[3] - sy(value), COLORS["model"], 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.2f}", anchor="middle")
    svg.text(1450, 1020, "State fixed effects; confidence intervals cluster repeated cells by episode.", "note", "end")
    svg.save(path)


def write_report(result: dict, path: Path) -> None:
    models = result["state_fixed_effects"]
    behavior = models["persistence_logit"]["stop_and_continue"]
    stop = behavior["coefficients"]["stop_payoff"]
    continued = behavior["coefficients"]["continue_bonus"]
    relative = models["persistence_logit"]["relative_only"]
    arm = models["arm_logit_gap"]["stop_and_continue"]
    lines = [
        "# STOP-payoff × CONTINUE-bonus value dissociation",
        "",
        f"Audit: **{result['audit']['complete_states']} complete states / {result['audit']['episodes']} episodes**, "
        f"{result['audit']['duplicate_cells']} duplicate cells and {result['audit']['history_hash_failures']} history failures.",
        "",
        "## Behavioral causal effects",
        "",
        f"STOP payoff: standardized beta **{stop['standardized_beta']:.3f}**, raw slope **{stop['raw_slope']:.4f} logit/point**, p={stop['normal_approximation_p_value']:.3g}.",
        f"CONTINUE bonus: standardized beta **{continued['standardized_beta']:.3f}**, raw slope **{continued['raw_slope']:.4f} logit/point**, p={continued['normal_approximation_p_value']:.3g}.",
        f"Relative incentive alone explains **{relative['r_squared']:.3f}** of within-state persistence variation; STOP-only and CONTINUE-only R² are **{models['persistence_logit']['stop_only']['r_squared']:.3f}** and **{models['persistence_logit']['continue_only']['r_squared']:.3f}**.",
        "",
        "## Representational response",
        "",
    ]
    for outcome, label in (
        ("generic_return_projection", "Generic-return direction"),
        ("advantage_projection", "Provisional advantage direction"),
        ("persistence_projection", "Direct persistence direction"),
    ):
        coefficients = models[outcome]["stop_and_continue"]["coefficients"]
        lines.append(
            f"- {label}: STOP beta **{coefficients['stop_payoff']['standardized_beta']:.3f}**; "
            f"CONTINUE beta **{coefficients['continue_bonus']['standardized_beta']:.3f}**; "
            f"relative-only R² **{models[outcome]['relative_only']['r_squared']:.3f}**."
        )
    lines.extend(
        [
            "",
            "## Manipulation specificity",
            "",
            f"Because the continuation bonus applies equally to A and B, their relative logit should be stable. Arm-gap STOP/CONTINUE betas are **{arm['coefficients']['stop_payoff']['standardized_beta']:.3f} / {arm['coefficients']['continue_bonus']['standardized_beta']:.3f}**.",
            "",
            result["collinearity_note"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/value_dissociation/factorial*.csv")
    parser.add_argument("--output-dir", default="artifacts/value_dissociation/publication")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = analyze_rows(read_rows(args.input))
    (output / "value_dissociation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(result, output / "value_dissociation_results.svg")
    write_report(result, output / "value_dissociation_report.md")
    print(json.dumps({"audit": result["audit"]}, indent=2))


if __name__ == "__main__":
    main()
