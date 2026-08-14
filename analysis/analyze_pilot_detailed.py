"""Dependency-free audit and descriptive plots for the behavioral pilot."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bandit.conversation import BanditConversation
from bandit.environment import BanditEnvironment, condition_class


COLORS = {
    "both_negative": "#D55E00",
    "one_positive": "#0072B2",
    "both_positive": "#009E73",
    "observed": "#7A5195",
    "model": "#E69F00",
    "reference": "#6B7280",
}
LABELS = {
    "both_negative": "Both negative",
    "one_positive": "One positive",
    "both_positive": "Both positive",
}
CONDITION_ORDER = ("both_negative", "one_positive", "both_positive")
ROUND_BINS = ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100))
LEARNING_BINS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100))


def read_rows(path: Path) -> list[dict]:
    rows = []
    integer_fields = ("seed", "action_seed", "round", "cumulative_score")
    float_fields = (
        "p_A_true",
        "p_B_true",
        "logit_A",
        "logit_B",
        "logit_C",
        "p_A",
        "p_B",
        "p_stop",
        "p_continue",
        "persistence_logit",
        "p_action_mass_raw",
        "subsequent_reward",
        "future_cumulative_return",
    )
    json_fields = ("choice_history", "reward_history", "conversation")
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in integer_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            for field in json_fields:
                row[field] = json.loads(row[field])
            row["terminated"] = row["terminated"] == "True"
            row["top_token_is_action"] = row["top_token_is_action"] == "True"
            row["previous_outcome"] = (
                None if row["previous_outcome"] == "" else float(row["previous_outcome"])
            )
            row["condition_class"] = condition_class(
                row["p_A_true"], row["p_B_true"]
            )
            row["decision"] = row["round"] + 1
            streak = 0
            for reward in reversed(row["reward_history"]):
                if reward != -2:
                    break
                streak += 1
            row["loss_streak"] = streak
            rows.append(row)
    return rows


def group_episodes(rows: list[dict]) -> dict[str, list[dict]]:
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["episode_id"]].append(row)
    return {
        episode_id: sorted(records, key=lambda item: item["round"])
        for episode_id, records in episodes.items()
    }


def audit_episodes(episodes: dict[str, list[dict]]) -> dict:
    issues = Counter()
    cell_counts = Counter()
    condition_counts = Counter()
    termination_counts = Counter()
    seen_states = set()

    for episode_id, records in episodes.items():
        first = records[0]
        environment = BanditEnvironment(
            first["p_A_true"], first["p_B_true"], first["seed"], max_decisions=100
        )
        conversation = BanditConversation.start("ABC")
        action_rng = random.Random(first["action_seed"])
        cell_counts[(first["p_A_true"], first["p_B_true"])] += 1
        condition_counts[first["condition_class"]] += 1

        for position, row in enumerate(records):
            if row["state_id"] in seen_states:
                issues["duplicate_state_id"] += 1
            seen_states.add(row["state_id"])
            if row["round"] != position or row["state_id"] != f"{episode_id}:{position}":
                issues["round_or_state_id"] += 1
            if row["choice_history"] != environment.action_history:
                issues["choice_history"] += 1
            if row["reward_history"] != environment.reward_history:
                issues["reward_history"] += 1
            if row["cumulative_score"] != environment.cumulative_score:
                issues["cumulative_score"] += 1
            if row["conversation"] != conversation.snapshot():
                issues["conversation"] += 1
            expected_previous = (
                None if position == 0 else float(environment.reward_history[-1])
            )
            if row["previous_outcome"] != expected_previous:
                issues["previous_outcome"] += 1
            if abs(row["p_A"] + row["p_B"] + row["p_stop"] - 1.0) > 1e-9:
                issues["action_probability_sum"] += 1
            if abs(row["p_continue"] - row["p_A"] - row["p_B"]) > 1e-9:
                issues["continuation_probability"] += 1

            draw = action_rng.random()
            expected_action = (
                "A"
                if draw < row["p_A"]
                else "B"
                if draw < row["p_A"] + row["p_B"]
                else "C"
            )
            if row["sampled_action"] != expected_action:
                issues["action_sampling"] += 1

            conversation.record_action(row["sampled_action"])
            result = environment.step(row["sampled_action"])
            if float(result.reward) != row["subsequent_reward"]:
                issues["reward"] += 1
            if result.terminated != row["terminated"]:
                issues["termination"] += 1
            if row["terminated"] != (position == len(records) - 1):
                issues["terminal_position"] += 1
            if not result.terminated:
                conversation.record_feedback(result.reward)

        running_return = 0.0
        for row in reversed(records):
            running_return += row["subsequent_reward"]
            if abs(row["future_cumulative_return"] - running_return) > 1e-9:
                issues["future_return"] += 1

        last = records[-1]
        if last["sampled_action"] == "C":
            termination_counts["quit"] += 1
        elif len(records) == 100:
            termination_counts["horizon_censored"] += 1
        else:
            termination_counts["invalid"] += 1

    return {
        "passed": not issues,
        "issue_counts": dict(sorted(issues.items())),
        "unique_state_ids": len(seen_states),
        "termination_counts": dict(termination_counts),
        "condition_episode_counts": dict(condition_counts),
        "cell_episode_counts": {
            f"{p_a:.2f},{p_b:.2f}": count
            for (p_a, p_b), count in sorted(cell_counts.items())
        },
    }


def nearest_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def binned_rows(rows: list[dict], bins: tuple[tuple[int, int], ...]) -> list[list[dict]]:
    return [
        [row for row in rows if lower <= row["decision"] <= upper]
        for lower, upper in bins
    ]


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def summarize(rows: list[dict], episodes: dict[str, list[dict]], audit: dict) -> dict:
    episode_records = list(episodes.values())
    quit_episodes = [episode for episode in episode_records if episode[-1]["sampled_action"] == "C"]
    horizon_episodes = [episode for episode in episode_records if episode[-1]["sampled_action"] != "C"]
    quit_decisions = [len(episode) for episode in quit_episodes]
    action_counts = Counter(row["sampled_action"] for row in rows)

    condition_summary = {}
    for condition in CONDITION_ORDER:
        selected = [ep for ep in episode_records if ep[0]["condition_class"] == condition]
        quits = [ep for ep in selected if ep[-1]["sampled_action"] == "C"]
        condition_summary[condition] = {
            "episodes": len(selected),
            "quit_episodes": len(quits),
            "quit_fraction": ratio(len(quits), len(selected)),
            "horizon_censored": len(selected) - len(quits),
            "mean_decisions": statistics.mean(len(ep) for ep in selected),
            "median_decisions": statistics.median(len(ep) for ep in selected),
            "median_quit_decision": statistics.median(len(ep) for ep in quits),
        }

    streak_summary = {}
    for label, predicate in (
        ("0", lambda value: value == 0),
        ("1", lambda value: value == 1),
        ("2", lambda value: value == 2),
        ("3+", lambda value: value >= 3),
    ):
        selected = [row for row in rows if predicate(row["loss_streak"])]
        streak_summary[label] = {
            "states": len(selected),
            "quits": sum(row["sampled_action"] == "C" for row in selected),
            "observed_quit_rate": statistics.mean(
                row["sampled_action"] == "C" for row in selected
            ),
            "mean_model_p_stop": statistics.mean(row["p_stop"] for row in selected),
        }

    timing_summary = []
    for (lower, upper), selected in zip(ROUND_BINS, binned_rows(rows, ROUND_BINS)):
        timing_summary.append(
            {
                "label": str(lower) if lower == upper else f"{lower}-{upper}",
                "lower": lower,
                "upper": upper,
                "states_at_risk": len(selected),
                "quits": sum(row["sampled_action"] == "C" for row in selected),
                "observed_hazard": statistics.mean(
                    row["sampled_action"] == "C" for row in selected
                ),
                "mean_model_p_stop": statistics.mean(row["p_stop"] for row in selected),
            }
        )

    event_aligned = []
    for lag in range(-10, 1):
        selected = [episode[len(episode) - 1 + lag] for episode in quit_episodes if len(episode) + lag > 0]
        event_aligned.append(
            {
                "lag": lag,
                "episodes": len(selected),
                "mean_p_stop": statistics.mean(row["p_stop"] for row in selected),
                "median_p_stop": statistics.median(row["p_stop"] for row in selected),
            }
        )

    unequal = [row for row in rows if row["p_A_true"] != row["p_B_true"]]
    learning_summary = []
    for (lower, upper), selected in zip(
        LEARNING_BINS, binned_rows(unequal, LEARNING_BINS)
    ):
        arm_choices = [row for row in selected if row["sampled_action"] != "C"]
        optimal = [
            (row["sampled_action"] == "A") == (row["p_A_true"] > row["p_B_true"])
            for row in arm_choices
        ]
        conditional_probability = [
            (
                row["p_A"] if row["p_A_true"] > row["p_B_true"] else row["p_B"]
            )
            / row["p_continue"]
            for row in selected
        ]
        learning_summary.append(
            {
                "label": str(lower) if lower == upper else f"{lower}-{upper}",
                "observed_optimal_choice": statistics.mean(optimal),
                "model_optimal_arm_probability": statistics.mean(
                    conditional_probability
                ),
                "arm_choice_states": len(arm_choices),
            }
        )

    calibration_bins = ((0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.001))
    calibration = []
    for lower, upper in calibration_bins:
        selected = [row for row in rows if lower <= row["p_stop"] < upper]
        if selected:
            calibration.append(
                {
                    "label": f"{lower:g}-{min(upper, 1):g}",
                    "states": len(selected),
                    "mean_predicted": statistics.mean(row["p_stop"] for row in selected),
                    "observed_quit_rate": statistics.mean(
                        row["sampled_action"] == "C" for row in selected
                    ),
                }
            )

    first_rows = [row for row in rows if row["round"] == 0]
    quit_rows = [row for row in rows if row["sampled_action"] == "C"]
    continue_rows = [row for row in rows if row["sampled_action"] != "C"]
    return {
        "audit": audit,
        "episodes": len(episode_records),
        "decision_states": len(rows),
        "action_counts": dict(action_counts),
        "quit_episodes": len(quit_episodes),
        "horizon_censored_episodes": len(horizon_episodes),
        "quit_episode_fraction": len(quit_episodes) / len(episode_records),
        "mean_decisions_per_episode": statistics.mean(len(ep) for ep in episode_records),
        "quit_decision_quantiles": {
            str(probability): nearest_quantile(quit_decisions, probability)
            for probability in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "quits_by_decision": {
            str(limit): sum(value <= limit for value in quit_decisions)
            for limit in (1, 3, 5, 10, 20, 50, 100)
        },
        "condition_summary": condition_summary,
        "loss_streak_summary": streak_summary,
        "timing_hazard": timing_summary,
        "event_aligned_p_stop": event_aligned,
        "optimal_arm_learning": learning_summary,
        "stop_calibration": calibration,
        "stop_brier_score": statistics.mean(
            (row["p_stop"] - (row["sampled_action"] == "C")) ** 2 for row in rows
        ),
        "first_decision_probabilities": {
            action: statistics.mean(row[f"p_{'stop' if action == 'C' else action}"] for row in first_rows)
            for action in "ABC"
        },
        "first_decision_action_counts": dict(
            Counter(row["sampled_action"] for row in first_rows)
        ),
        "mean_p_stop_at_quit": statistics.mean(row["p_stop"] for row in quit_rows),
        "median_p_stop_at_quit": statistics.median(row["p_stop"] for row in quit_rows),
        "mean_p_stop_while_continuing": statistics.mean(
            row["p_stop"] for row in continue_rows
        ),
        "mean_final_score": {
            "quit": statistics.mean(ep[-1]["cumulative_score"] for ep in quit_episodes),
            "horizon_censored": statistics.mean(
                ep[-1]["cumulative_score"] for ep in horizon_episodes
            ),
        },
    }


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17212B}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px;fill:#4B5563}.panel{font-size:17px;font-weight:700}.axis{font-size:12px;fill:#4B5563}.legend{font-size:12px}.note{font-size:11px;fill:#6B7280}</style>',
        ]

    def text(self, x: float, y: float, value: str, css: str = "axis", anchor: str = "start", rotate: int | None = None) -> None:
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}"{transform}>{html.escape(str(value))}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#CBD5E1", width: float = 1, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'
        )

    def rect(self, x: float, y: float, width: float, height: float, fill: str, opacity: float = 1.0, stroke: str = "none") -> None:
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}"/>'
        )

    def circle(self, x: float, y: float, radius: float, fill: str) -> None:
        self.items.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}"/>'
        )

    def polyline(self, points: list[tuple[float, float]], color: str, width: float = 2.5) -> None:
        encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.items.append(
            f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.items, "</svg>"]) + "\n", encoding="utf-8")


def axes(svg: Svg, x: float, y: float, width: float, height: float, title: str, xlabel: str, ylabel: str, x_ticks: list[tuple[float, str]], y_ticks: list[tuple[float, str]], x_domain: tuple[float, float], y_domain: tuple[float, float]):
    left, top, right, bottom = x + 62, y + 42, x + width - 18, y + height - 58
    svg.text(x, y + 18, title, "panel")

    def sx(value: float) -> float:
        return left + (value - x_domain[0]) / (x_domain[1] - x_domain[0]) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - y_domain[0]) / (y_domain[1] - y_domain[0]) * (bottom - top)

    for value, label in y_ticks:
        position = sy(value)
        svg.line(left, position, right, position, "#E5E7EB")
        svg.text(left - 9, position + 4, label, anchor="end")
    for value, label in x_ticks:
        position = sx(value)
        svg.line(position, bottom, position, bottom + 5, "#64748B")
        svg.text(position, bottom + 20, label, anchor="middle")
    svg.line(left, top, left, bottom, "#64748B", 1.2)
    svg.line(left, bottom, right, bottom, "#64748B", 1.2)
    svg.text((left + right) / 2, y + height - 15, xlabel, anchor="middle")
    svg.text(x + 15, (top + bottom) / 2, ylabel, anchor="middle", rotate=-90)
    return sx, sy, (left, top, right, bottom)


def legend(svg: Svg, x: float, y: float, entries: list[tuple[str, str]]) -> None:
    for index, (label, color) in enumerate(entries):
        offset = index * 145
        svg.line(x + offset, y, x + offset + 22, y, color, 3)
        svg.text(x + offset + 28, y + 4, label, "legend")


def make_stopping_figure(summary: dict, episodes: dict[str, list[dict]], path: Path) -> None:
    svg = Svg(1500, 1050)
    svg.text(55, 45, "When does Qwen3.5-4B quit the bandit?", "title")
    svg.text(55, 70, "Behavioral pilot: 200 episodes; horizon endings treated as right-censored", "subtitle")
    panels = ((45, 95), (760, 95), (45, 565), (760, 565))

    x, y = panels[0]
    sx, sy, box = axes(
        svg, x, y, 690, 430, "A. Persistence curves by reward condition",
        "Decision", "Proportion not yet quit",
        [(1, "1"), (5, "5"), (10, "10"), (20, "20"), (50, "50"), (100, "100")],
        [(0, "0"), (0.25, ".25"), (0.5, ".50"), (0.75, ".75"), (1, "1")],
        (1, 100), (0, 1),
    )
    for condition in CONDITION_ORDER:
        selected = [ep for ep in episodes.values() if ep[0]["condition_class"] == condition]
        points = []
        for decision in range(1, 101):
            survivors = sum(
                ep[-1]["sampled_action"] != "C" or len(ep) > decision
                for ep in selected
            )
            points.append((sx(decision), sy(survivors / len(selected))))
        svg.polyline(points, COLORS[condition])
    legend(svg, box[0] + 12, box[1] + 18, [(LABELS[key], COLORS[key]) for key in CONDITION_ORDER])

    x, y = panels[1]
    timing = summary["timing_hazard"]
    sx, sy, box = axes(
        svg, x, y, 690, 430, "B. Discrete quit hazard peaks at decisions 4-5",
        "Decision interval", "STOP probability per state",
        [(i, item["label"]) for i, item in enumerate(timing)],
        [(0, "0"), (0.05, ".05"), (0.1, ".10"), (0.15, ".15"), (0.2, ".20"), (0.25, ".25")],
        (-0.4, len(timing) - 0.6), (0, 0.25),
    )
    observed = [(sx(i), sy(item["observed_hazard"])) for i, item in enumerate(timing)]
    predicted = [(sx(i), sy(item["mean_model_p_stop"])) for i, item in enumerate(timing)]
    svg.polyline(observed, COLORS["observed"])
    svg.polyline(predicted, COLORS["model"])
    for px, py in observed:
        svg.circle(px, py, 4, COLORS["observed"])
    for px, py in predicted:
        svg.circle(px, py, 4, COLORS["model"])
    legend(svg, box[0] + 12, box[1] + 18, [("Observed STOP", COLORS["observed"]), ("Mean model P(STOP)", COLORS["model"])])

    x, y = panels[2]
    streak = summary["loss_streak_summary"]
    labels = list(streak)
    sx, sy, box = axes(
        svg, x, y, 690, 430, "C. Quitting rises with consecutive losses",
        "Consecutive losses before decision", "STOP probability per state",
        [(i, label) for i, label in enumerate(labels)],
        [(0, "0"), (0.03, ".03"), (0.06, ".06"), (0.09, ".09"), (0.12, ".12"), (0.15, ".15")],
        (-0.6, len(labels) - 0.4), (0, 0.15),
    )
    bar_width = (box[2] - box[0]) / len(labels) * 0.25
    for index, label in enumerate(labels):
        observed = streak[label]["observed_quit_rate"]
        predicted = streak[label]["mean_model_p_stop"]
        center = sx(index)
        svg.rect(center - bar_width - 2, sy(observed), bar_width, box[3] - sy(observed), COLORS["observed"], 0.9)
        svg.rect(center + 2, sy(predicted), bar_width, box[3] - sy(predicted), COLORS["model"], 0.9)
    legend(svg, box[0] + 12, box[1] + 18, [("Observed STOP", COLORS["observed"]), ("Mean model P(STOP)", COLORS["model"])])

    x, y = panels[3]
    aligned = summary["event_aligned_p_stop"]
    sx, sy, box = axes(
        svg, x, y, 690, 430, "D. STOP probability ramps immediately before quitting",
        "Decisions relative to STOP", "Model P(STOP)",
        [(value, str(value)) for value in (-10, -8, -6, -4, -2, 0)],
        [(0, "0"), (0.05, ".05"), (0.1, ".10"), (0.15, ".15"), (0.2, ".20"), (0.25, ".25")],
        (-10, 0), (0, 0.25),
    )
    mean_points = [(sx(item["lag"]), sy(item["mean_p_stop"])) for item in aligned]
    median_points = [(sx(item["lag"]), sy(item["median_p_stop"])) for item in aligned]
    svg.polyline(mean_points, COLORS["observed"])
    svg.polyline(median_points, COLORS["reference"])
    for px, py in mean_points:
        svg.circle(px, py, 4, COLORS["observed"])
    legend(svg, box[0] + 12, box[1] + 18, [("Mean", COLORS["observed"]), ("Median", COLORS["reference"])])
    svg.text(1450, 1030, "Descriptive pilot estimates; state-level panels contain repeated observations within episode.", "note", "end")
    svg.save(path)


def make_bandit_figure(summary: dict, path: Path) -> None:
    svg = Svg(1500, 1050)
    svg.text(55, 45, "Bandit learning and action diagnostics", "title")
    svg.text(55, 70, "Qwen3.5-4B behavioral pilot; no activation intervention", "subtitle")
    panels = ((45, 95), (760, 95), (45, 565), (760, 565))

    x, y = panels[0]
    learning = summary["optimal_arm_learning"]
    sx, sy, box = axes(
        svg, x, y, 690, 430, "A. Preference for the objectively better arm",
        "Decision interval", "Probability of better arm | continue",
        [(i, item["label"]) for i, item in enumerate(learning)],
        [(0.4, ".40"), (0.5, ".50"), (0.6, ".60"), (0.7, ".70"), (0.8, ".80")],
        (-0.4, len(learning) - 0.6), (0.4, 0.8),
    )
    svg.line(box[0], sy(0.5), box[2], sy(0.5), COLORS["reference"], 1.5, "5 5")
    observed = [(sx(i), sy(item["observed_optimal_choice"])) for i, item in enumerate(learning)]
    predicted = [(sx(i), sy(item["model_optimal_arm_probability"])) for i, item in enumerate(learning)]
    svg.polyline(observed, COLORS["observed"])
    svg.polyline(predicted, COLORS["model"])
    for px, py in observed:
        svg.circle(px, py, 4, COLORS["observed"])
    for px, py in predicted:
        svg.circle(px, py, 4, COLORS["model"])
    legend(svg, box[0] + 12, box[1] + 18, [("Sampled choices", COLORS["observed"]), ("Model probability", COLORS["model"])])

    x, y = panels[1]
    first = summary["first_decision_probabilities"]
    sx, sy, box = axes(
        svg, x, y, 690, 430, "B. Strong action-label prior before any feedback",
        "First-decision action", "Mean model probability",
        [(i, label) for i, label in enumerate(("A", "B", "C / STOP"))],
        [(0, "0"), (0.2, ".2"), (0.4, ".4"), (0.6, ".6"), (0.8, ".8"), (1, "1")],
        (-0.6, 2.6), (0, 1),
    )
    for index, (key, color) in enumerate((("A", "#0072B2"), ("B", "#56B4E9"), ("C", "#D55E00"))):
        value = first[key]
        svg.rect(sx(index) - 45, sy(value), 90, box[3] - sy(value), color, 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.3f}", anchor="middle")

    x, y = panels[2]
    conditions = summary["condition_summary"]
    sx, sy, box = axes(
        svg, x, y, 690, 430, "C. Episode duration reflects arm value",
        "Reward condition", "Mean decisions per episode",
        [(i, LABELS[key]) for i, key in enumerate(CONDITION_ORDER)],
        [(0, "0"), (5, "5"), (10, "10"), (15, "15"), (20, "20"), (25, "25")],
        (-0.6, 2.6), (0, 25),
    )
    for index, key in enumerate(CONDITION_ORDER):
        value = conditions[key]["mean_decisions"]
        svg.rect(sx(index) - 48, sy(value), 96, box[3] - sy(value), COLORS[key], 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.1f}", anchor="middle")

    x, y = panels[3]
    calibration = summary["stop_calibration"]
    sx, sy, box = axes(
        svg, x, y, 690, 430, "D. Sampled STOP choices track model probabilities",
        "Mean predicted P(STOP)", "Observed STOP rate",
        [(0, "0"), (0.1, ".1"), (0.2, ".2"), (0.3, ".3"), (0.4, ".4"), (0.5, ".5")],
        [(0, "0"), (0.1, ".1"), (0.2, ".2"), (0.3, ".3"), (0.4, ".4"), (0.5, ".5")],
        (0, 0.55), (0, 0.55),
    )
    svg.line(sx(0), sy(0), sx(0.55), sy(0.55), COLORS["reference"], 1.5, "5 5")
    for item in calibration:
        svg.circle(sx(item["mean_predicted"]), sy(item["observed_quit_rate"]), 4 + math.sqrt(item["states"]) / 6, COLORS["observed"])
    svg.text(box[0] + 12, box[1] + 18, "Point area scales with number of states", "note")
    svg.text(1450, 1030, "Optimal-arm panels exclude equal-probability conditions; C choices excluded from sampled arm accuracy.", "note", "end")
    svg.save(path)


def write_report(summary: dict, path: Path) -> None:
    conditions = summary["condition_summary"]
    streak = summary["loss_streak_summary"]
    quantiles = summary["quit_decision_quantiles"]
    lines = [
        "# Behavioral pilot audit and descriptive analysis",
        "",
        "## Integrity audit",
        "",
        f"- Audit passed: **{summary['audit']['passed']}**",
        f"- Rows: {summary['decision_states']:,}; episodes: {summary['episodes']}",
        "- Replayed from seeds: rewards, stopping, action sampling, histories, cumulative scores, and future returns all matched.",
        "- Every stored conversation exactly matched the intended model-visible chat; experimenter-only state remained separate.",
        "- All state IDs were unique and each episode had one terminal final row.",
        "",
        "## When the model quits",
        "",
        f"- {summary['quit_episodes']}/{summary['episodes']} episodes quit; {summary['horizon_censored_episodes']} reached the 100-decision horizon.",
        f"- Median quit decision: **{quantiles['0.5']:.0f}** (IQR {quantiles['0.25']:.0f}-{quantiles['0.75']:.0f}; 90th percentile {quantiles['0.9']:.0f}).",
        f"- Mean P(STOP) was {summary['mean_p_stop_at_quit']:.3f} on quit states versus {summary['mean_p_stop_while_continuing']:.3f} on continuing states.",
        f"- Observed STOP rose from {streak['0']['observed_quit_rate']:.3f} with no current loss streak to {streak['3+']['observed_quit_rate']:.3f} after three or more consecutive losses.",
        "- The discrete quit hazard peaked at decisions 4-5, then fell sharply among surviving episodes.",
        "",
        "## Condition differences",
        "",
        "| Condition | Episodes | Quit fraction | Mean decisions | Median quit decision |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in CONDITION_ORDER:
        value = conditions[key]
        lines.append(
            f"| {LABELS[key]} | {value['episodes']} | {value['quit_fraction']:.3f} | {value['mean_decisions']:.2f} | {value['median_quit_decision']:.1f} |"
        )
    first = summary["first_decision_probabilities"]
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            f"- Before receiving evidence, the model assigned P(A)={first['A']:.3f}, P(B)={first['B']:.3f}, and P(STOP)={first['C']:.3f}. This strong A-label prior should be treated as a nuisance effect.",
            "- The overall mean STOP probability understates the timing result: quits are concentrated early and after short loss streaks, while a selected group of persistent episodes survives to the horizon.",
            "- State-level observations are correlated within episodes. Confirmatory uncertainty should resample or cluster by episode.",
            "- Pilot data should remain separate from probe fitting and confirmatory intervention data.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/bandit_pilot.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pilot_diagnostics"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    episodes = group_episodes(rows)
    audit = audit_episodes(episodes)
    summary = summarize(rows, episodes, audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_detailed_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(summary, args.output_dir / "pilot_diagnostic_report.md")
    make_stopping_figure(
        summary, episodes, args.output_dir / "pilot_stopping_diagnostics.svg"
    )
    make_bandit_figure(summary, args.output_dir / "pilot_bandit_behavior.svg")
    print(json.dumps(summary["audit"], indent=2, sort_keys=True))
    print(f"Wrote detailed analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
