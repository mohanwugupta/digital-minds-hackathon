"""Analyze matched ridge-direction steering with episode-bootstrap inference."""

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend


DIRECTIONS = ("persistence", "generic_return", "advantage")
LABELS = {
    "persistence": "Persistence",
    "generic_return": "Generic return",
    "advantage": "Continuation advantage",
}


def read_rows(pattern: str) -> list[dict]:
    rows = []
    retained = (
        "state_id",
        "episode_id",
        "direction_name",
        "control_type",
        "control_id",
        "context_hash",
        "logit_A",
        "logit_B",
        "logit_C",
        "alpha",
        "persistence_logit",
        "p_continue",
        "probe_value_pre",
        "probe_value_post",
        "direction_l2_norm",
        "intervention_relative_rms",
    )
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                row = {key: source[key] for key in retained}
                for key in (
                    "alpha",
                    "persistence_logit",
                    "p_continue",
                    "probe_value_pre",
                    "probe_value_post",
                    "direction_l2_norm",
                    "intervention_relative_rms",
                ):
                    row[key] = float(row[key])
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no causal steering rows match {pattern}")
    return rows


def audit_rows(rows: list[dict]) -> dict:
    keys = [
        (
            row["state_id"],
            row["direction_name"],
            row["control_type"],
            row["control_id"],
            row["alpha"],
        )
        for row in rows
    ]
    contexts = defaultdict(set)
    for row in rows:
        contexts[row["state_id"]].add(row["context_hash"])
    baseline_by_state = defaultdict(set)
    for row in rows:
        if row["alpha"] == 0:
            baseline_by_state[row["state_id"]].add(
                (row["logit_A"], row["logit_B"], row["logit_C"])
            )
    target_groups = defaultdict(dict)
    all_groups = defaultdict(set)
    for row in rows:
        all_groups[
            (
                row["state_id"],
                row["direction_name"],
                row["control_type"],
                row["control_id"],
            )
        ].add(row["alpha"])
        if row["control_type"] == "target":
            target_groups[(row["state_id"], row["direction_name"])][row["alpha"]] = row
    unordered = 0
    for group in target_groups.values():
        if set(group) != {-1.0, 0.0, 1.0}:
            unordered += 1
            continue
        if not (
            group[1.0]["probe_value_post"]
            > group[0.0]["probe_value_post"]
            > group[-1.0]["probe_value_post"]
        ):
            unordered += 1
    return {
        "rows": len(rows),
        "unique_keys": len(set(keys)),
        "duplicate_keys": len(keys) - len(set(keys)),
        "states": len({row["state_id"] for row in rows}),
        "episodes": len({row["episode_id"] for row in rows}),
        "contexts_per_state": sorted({len(value) for value in contexts.values()}),
        "alpha_zero_exact_across_conditions": all(
            len(value) == 1 for value in baseline_by_state.values()
        ),
        "incomplete_alpha_groups": sum(
            values != {-1.0, 0.0, 1.0} for values in all_groups.values()
        ),
        "target_probe_ordering_failures": unordered,
    }


def _episode_effects(rows: list[dict]) -> tuple[dict[str, float], int]:
    by_state = defaultdict(dict)
    episode = {}
    for row in rows:
        by_state[row["state_id"]][row["alpha"]] = row["persistence_logit"]
        episode[row["state_id"]] = row["episode_id"]
    state_effects = {
        state_id: values[1.0] - values[-1.0]
        for state_id, values in by_state.items()
        if {-1.0, 0.0, 1.0}.issubset(values)
    }
    grouped = defaultdict(list)
    for state_id, effect in state_effects.items():
        grouped[episode[state_id]].append(effect)
    return {
        episode_id: statistics.mean(values) for episode_id, values in grouped.items()
    }, len(state_effects)


def _bootstrap(values: dict[str, float], samples: int = 10000) -> dict:
    episode_ids = sorted(values)
    observed = statistics.mean(values.values())
    rng, draws = random.Random(102026), []
    for _ in range(samples):
        draws.append(
            statistics.mean(values[rng.choice(episode_ids)] for _ in episode_ids)
        )
    draws.sort()
    nonpositive = sum(value <= 0 for value in draws) / samples
    nonnegative = sum(value >= 0 for value in draws) / samples
    return {
        "mean": observed,
        "confidence_interval_95": [
            draws[int(0.025 * samples)],
            draws[int(0.975 * samples) - 1],
        ],
        "bootstrap_two_sided_p": min(1.0, 2 * min(nonpositive, nonnegative)),
        "episode_count": len(episode_ids),
        "bootstrap_samples": samples,
    }


def _dose_response(rows: list[dict]) -> dict:
    by_episode_alpha = defaultdict(list)
    for row in rows:
        by_episode_alpha[(row["episode_id"], row["alpha"])].append(
            row["persistence_logit"]
        )
    return {
        str(int(alpha)): statistics.mean(
            statistics.mean(values)
            for (episode, selected_alpha), values in by_episode_alpha.items()
            if selected_alpha == alpha
        )
        for alpha in (-1.0, 0.0, 1.0)
    }


def analyze(rows: list[dict]) -> dict:
    audit = audit_rows(rows)
    if (
        audit["duplicate_keys"]
        or audit["contexts_per_state"] != [1]
        or audit["incomplete_alpha_groups"]
    ):
        raise ValueError(f"causal replay audit failed: {audit}")
    available = [name for name in DIRECTIONS if any(row["direction_name"] == name for row in rows)]
    directions = {}
    for name in available:
        target = [
            row
            for row in rows
            if row["direction_name"] == name and row["control_type"] == "target"
        ]
        episode_effects, states = _episode_effects(target)
        target_inference = _bootstrap(episode_effects)
        random_effects = []
        control_ids = sorted(
            {
                row["control_id"]
                for row in rows
                if row["direction_name"] == name and row["control_type"] == "random"
            }
        )
        for control_id in control_ids:
            selected = [
                row
                for row in rows
                if row["direction_name"] == name
                and row["control_type"] == "random"
                and row["control_id"] == control_id
            ]
            effects, _states = _episode_effects(selected)
            random_effects.append(
                {"control_id": control_id, "mean_effect": statistics.mean(effects.values())}
            )
        target_effect = target_inference["mean"]
        directions[name] = {
            "matched_states": states,
            "target": target_inference,
            "dose_response": _dose_response(target),
            "random_controls": random_effects,
            "random_control_empirical_p_one_sided": (
                (1 + sum(row["mean_effect"] >= target_effect for row in random_effects))
                / (1 + len(random_effects))
                if random_effects
                else None
            ),
            "random_control_empirical_p_absolute": (
                (1 + sum(abs(row["mean_effect"]) >= abs(target_effect) for row in random_effects))
                / (1 + len(random_effects))
                if random_effects
                else None
            ),
        }
    persistence = directions.get("persistence")
    positive_control_passed = bool(
        persistence
        and audit["alpha_zero_exact_across_conditions"]
        and audit["target_probe_ordering_failures"] == 0
        and persistence["dose_response"]["1"]
        > persistence["dose_response"]["0"]
        > persistence["dose_response"]["-1"]
        and persistence["target"]["confidence_interval_95"][0] > 0
    )
    return {
        "audit": audit,
        "directions": directions,
        "positive_control_passed": positive_control_passed,
        "value_direction_interpretation_allowed": positive_control_passed,
        "interpretation_gate": (
            "Persistence-direction dose ordering and positive episode-bootstrap "
            "95% interval must pass before interpreting return/advantage nulls."
        ),
    }


def make_figure(result: dict, path: Path) -> None:
    svg = Svg(1500, 620)
    svg.text(55, 45, "Causal steering of the CONTINUE-versus-STOP decision", "title")
    svg.text(55, 70, "Frozen native-layer ridge directions; matched held-out states", "subtitle")
    panels = ((45, 100), (520, 100), (995, 100))
    names = [name for name in DIRECTIONS if name in result["directions"]]

    x, y = panels[0]
    intervals = [result["directions"][name]["target"] for name in names]
    bounds = [value for item in intervals for value in item["confidence_interval_95"]]
    lower, upper = min(bounds + [0]) - 0.1, max(bounds + [0]) + 0.1
    sx, sy, box = axes(
        svg, x, y, 455, 440, "A. Positive-minus-negative steering",
        "Direction", "Persistence-logit effect",
        [(index, LABELS[name]) for index, name in enumerate(names)],
        [(lower, f"{lower:.2f}"), (0, "0"), (upper, f"{upper:.2f}")],
        (-0.6, max(0.6, len(names) - 0.4)), (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (name, item) in enumerate(zip(names, intervals)):
        low, high = item["confidence_interval_95"]
        color = (COLORS["both_positive"], COLORS["observed"], COLORS["model"])[index]
        svg.line(sx(index), sy(low), sx(index), sy(high), color, 2)
        svg.circle(sx(index), sy(item["mean"]), 6, color)

    x, y = panels[1]
    random_values = [
        row["mean_effect"]
        for name in names
        for row in result["directions"][name]["random_controls"]
    ]
    all_effects = random_values + [result["directions"][name]["target"]["mean"] for name in names]
    lower, upper = min(all_effects + [0]) - 0.1, max(all_effects + [0]) + 0.1
    sx, sy, box = axes(
        svg, x, y, 455, 440, "B. Matched random-direction controls",
        "Direction family", "Persistence-logit effect",
        [(index, LABELS[name]) for index, name in enumerate(names)],
        [(lower, f"{lower:.2f}"), (0, "0"), (upper, f"{upper:.2f}")],
        (-0.6, max(0.6, len(names) - 0.4)), (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, name in enumerate(names):
        for control_index, row in enumerate(result["directions"][name]["random_controls"]):
            jitter = ((control_index % 5) - 2) * 0.035
            svg.circle(sx(index + jitter), sy(row["mean_effect"]), 3, "#94A3B8")
        svg.circle(sx(index), sy(result["directions"][name]["target"]["mean"]), 6, COLORS["observed"])

    x, y = panels[2]
    dose_values = [
        value
        for name in names
        for value in result["directions"][name]["dose_response"].values()
    ]
    lower, upper = min(dose_values) - 0.1, max(dose_values) + 0.1
    sx, sy, box = axes(
        svg, x, y, 455, 440, "C. Full steering dose response",
        "Steering alpha", "Mean persistence logit",
        [(-1, "-1"), (0, "0"), (1, "+1")],
        [(lower, f"{lower:.2f}"), (upper, f"{upper:.2f}")],
        (-1.25, 1.25), (lower, upper),
    )
    colors = (COLORS["both_positive"], COLORS["observed"], COLORS["model"])
    for name, color in zip(names, colors):
        dose = result["directions"][name]["dose_response"]
        points = [(sx(alpha), sy(dose[str(alpha)])) for alpha in (-1, 0, 1)]
        svg.polyline(points, color, 2)
        for px, py in points:
            svg.circle(px, py, 4, color)
    legend(svg, box[0] + 5, box[1] + 18, [(LABELS[name], color) for name, color in zip(names, colors)])
    svg.text(1450, 600, "Intervals resample episodes; states are paired across alpha.", "note", "end")
    svg.save(path)


def write_report(result: dict, path: Path) -> None:
    lines = [
        "# Causal ridge-steering analysis",
        "",
        f"Positive-control gate passed: **{result['positive_control_passed']}**.",
        f"Alpha-zero exact reproduction: **{result['audit']['alpha_zero_exact_across_conditions']}**.",
        f"Target probe-ordering failures: **{result['audit']['target_probe_ordering_failures']}**.",
        "",
    ]
    for name in DIRECTIONS:
        if name not in result["directions"]:
            continue
        item = result["directions"][name]
        target = item["target"]
        lines.extend(
            [
                f"## {LABELS[name]}",
                "",
                f"Positive-minus-negative persistence-logit effect: **{target['mean']:.4f}** "
                f"(episode-bootstrap 95% CI {target['confidence_interval_95'][0]:.4f} to {target['confidence_interval_95'][1]:.4f}; p={target['bootstrap_two_sided_p']:.3g}).",
                f"Matched states/episodes: **{item['matched_states']} / {target['episode_count']}**.",
                f"Random controls: **{len(item['random_controls'])}**; one-sided empirical p **{item['random_control_empirical_p_one_sided']}**; absolute empirical p **{item['random_control_empirical_p_absolute']}**.",
                "",
            ]
        )
    lines.extend([result["interpretation_gate"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/causal_steering/replays*.csv")
    parser.add_argument("--output-dir", default="artifacts/causal_steering/publication")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = analyze(read_rows(args.input))
    (output / "causal_steering_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(result, output / "causal_steering_results.svg")
    write_report(result, output / "causal_steering_report.md")
    print(json.dumps({"positive_control_passed": result["positive_control_passed"]}, indent=2))


if __name__ == "__main__":
    main()
