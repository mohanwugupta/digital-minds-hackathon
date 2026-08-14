"""Episode-clustered matched-state persistence and random-control analysis."""

import argparse
import json
import math
import os
from pathlib import Path


def _paired_episode_effect(group, upper: float, lower: float):
    required_columns = {"episode_id", "state_id", "alpha", "persistence_logit"}
    missing = required_columns - set(group.columns)
    if missing:
        raise ValueError(f"matched analysis is missing columns: {sorted(missing)}")
    pivot = group.pivot(
        index=["episode_id", "state_id"],
        columns="alpha",
        values="persistence_logit",
    ).dropna()
    if not {upper, lower}.issubset(set(pivot.columns)):
        raise ValueError(f"matched intervention is missing alpha {lower} or {upper}")
    state_effect = pivot[upper] - pivot[lower]
    # Episodes, rather than states, are the independent sampling units. Giving
    # each episode equal weight also prevents long-persistence episodes from
    # dominating the causal estimate merely because they contribute more states.
    return pivot, state_effect.groupby(level="episode_id").mean()


def _episode_inference(episode_effects) -> dict:
    """One-sample t inference over independent episode-level paired effects."""
    from scipy import stats

    values = episode_effects.astype(float)
    count = int(len(values))
    if count < 2:
        raise ValueError("at least two matched episodes are required for inference")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(count))
    if standard_error == 0:
        statistic = None
        p_value = 0.0 if mean != 0 else 1.0
        interval = [mean, mean]
    else:
        statistic = mean / standard_error
        p_value = float(2 * stats.t.sf(abs(statistic), df=count - 1))
        critical = float(stats.t.ppf(0.975, df=count - 1))
        interval = [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ]
    return {
        "episodes": count,
        "mean": mean,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "degrees_of_freedom": count - 1,
        "two_sided_p_value": p_value,
        "confidence_interval_95": interval,
    }


def _episode_balanced_alpha_mean(pivot, alpha: float) -> float:
    return float(pivot[alpha].groupby(level="episode_id").mean().mean())


def analyze_frame(frame) -> dict:
    value = frame[frame["intervention_type"] == "value"]
    value_pivot, total_episode_effect = _paired_episode_effect(value, 1.0, -1.0)
    required = {-1.0, 0.0, 1.0}
    if not required.issubset(set(value_pivot.columns)):
        raise ValueError("value intervention is missing one or more alpha conditions")

    positive_episode_effect = (
        (value_pivot[1.0] - value_pivot[0.0])
        .groupby(level="episode_id")
        .mean()
    )
    negative_episode_effect = (
        (value_pivot[0.0] - value_pivot[-1.0])
        .groupby(level="episode_id")
        .mean()
    )
    positive_inference = _episode_inference(positive_episode_effect)
    negative_inference = _episode_inference(negative_episode_effect)
    total_inference = _episode_inference(total_episode_effect)
    mean_logits = {
        "negative": _episode_balanced_alpha_mean(value_pivot, -1.0),
        "control": _episode_balanced_alpha_mean(value_pivot, 0.0),
        "positive": _episode_balanced_alpha_mean(value_pivot, 1.0),
    }
    ordered = bool(
        mean_logits["positive"] > mean_logits["control"] > mean_logits["negative"]
        and positive_inference["mean"] > 0
        and negative_inference["mean"] > 0
    )
    # Prespecified sprint gate: monotonic adjacent means plus a nonzero total
    # contrast using episodes as independent units. Sequential work remains
    # blocked unless this and the random-neuron specificity gate both pass.
    primary_passed = bool(
        ordered
        and total_inference["two_sided_p_value"] < 0.05
        and total_inference["confidence_interval_95"][0] > 0
    )

    random_effects = []
    random_frame = frame[frame["intervention_type"] == "random"]
    for _neuron_set, group in random_frame.groupby("neuron_set"):
        try:
            _pivot, episode_effect = _paired_episode_effect(group, 1.0, -1.0)
        except ValueError:
            continue
        random_effects.append(float(episode_effect.mean()))
    value_effect = total_inference["mean"]
    # With 20 controls, 1 / (20 + 1) = .0476 is the smallest attainable
    # corrected empirical p-value; the value effect must exceed every control.
    empirical_p = (
        (1 + sum(effect >= value_effect for effect in random_effects))
        / (1 + len(random_effects))
        if random_effects
        else None
    )
    return {
        "inference_unit": "episode",
        "episode_weighting": "equal",
        "n_matched_states": int(len(value_pivot)),
        "n_matched_episodes": int(len(total_episode_effect)),
        "mean_persistence_logit": mean_logits,
        "mean_positive_vs_control": positive_inference["mean"],
        "mean_control_vs_negative": negative_inference["mean"],
        "mean_positive_vs_negative": value_effect,
        "episode_clustered_inference": {
            "positive_vs_control": positive_inference,
            "control_vs_negative": negative_inference,
            "positive_vs_negative": total_inference,
        },
        "primary_monotonic_ordering": ordered,
        "primary_ordering_passed": primary_passed,
        "primary_gate": (
            "monotonic episode-balanced means and positive-vs-negative 95% "
            "episode-level CI strictly above zero"
        ),
        "random_set_count": len(random_effects),
        "random_effects": random_effects,
        "random_control_empirical_p": empirical_p,
        "random_control_passed": bool(
            len(random_effects) >= 20
            and empirical_p is not None
            and empirical_p <= 0.05
        ),
    }


def _padded_domain(values: list[float], *, include_zero: bool = False):
    selected = [*values, 0.0] if include_zero else list(values)
    lower, upper = min(selected), max(selected)
    padding = max(0.05, (upper - lower) * 0.2)
    return lower - padding, upper + padding


def _make_figure(result: dict, path: Path) -> None:
    from analysis.analyze_pilot_detailed import COLORS, Svg, axes

    svg = Svg(1500, 570)
    svg.text(55, 45, "Matched-state causal effect on persistence", "title")
    svg.text(
        55,
        70,
        f"{result['n_matched_episodes']} held-out episodes; equal episode weighting",
        "subtitle",
    )
    panels = ((45, 100), (515, 100), (985, 100))

    x, y = panels[0]
    logits = result["mean_persistence_logit"]
    alpha_values = [logits["negative"], logits["control"], logits["positive"]]
    lower, upper = _padded_domain(alpha_values)
    midpoint = (lower + upper) / 2
    sx, sy, _box = axes(
        svg,
        x,
        y,
        430,
        410,
        "A. Persistence logit by steering",
        "Steering alpha",
        "Mean persistence logit",
        [(-1, "Negative"), (0, "Control"), (1, "Positive")],
        [
            (lower, f"{lower:.2f}"),
            (midpoint, f"{midpoint:.2f}"),
            (upper, f"{upper:.2f}"),
        ],
        (-1.25, 1.25),
        (lower, upper),
    )
    points = [
        (sx(alpha), sy(value))
        for alpha, value in zip((-1, 0, 1), alpha_values)
    ]
    svg.polyline(points, COLORS["observed"])
    for px, py in points:
        svg.circle(px, py, 5, COLORS["observed"])

    x, y = panels[1]
    inference = result["episode_clustered_inference"]
    contrasts = [
        ("+ vs 0", inference["positive_vs_control"]),
        ("0 vs -", inference["control_vs_negative"]),
        ("+ vs -", inference["positive_vs_negative"]),
    ]
    bounds = [
        bound
        for _, item in contrasts
        for bound in item["confidence_interval_95"]
    ]
    lower, upper = _padded_domain(bounds, include_zero=True)
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "B. Episode-level paired effects",
        "Contrast",
        "Persistence-logit difference",
        [(index, label) for index, (label, _) in enumerate(contrasts)],
        [(lower, f"{lower:.2f}"), (0, "0"), (upper, f"{upper:.2f}")],
        (-0.6, 2.6),
        (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (_label, item) in enumerate(contrasts):
        low, high = item["confidence_interval_95"]
        center = sx(index)
        svg.line(center, sy(low), center, sy(high), COLORS["model"], 2)
        svg.line(center - 7, sy(low), center + 7, sy(low), COLORS["model"], 2)
        svg.line(center - 7, sy(high), center + 7, sy(high), COLORS["model"], 2)
        svg.circle(center, sy(item["mean"]), 5, COLORS["model"])

    x, y = panels[2]
    random_effects = result["random_effects"]
    value_effect = result["mean_positive_vs_negative"]
    lower, upper = _padded_domain(
        [*random_effects, value_effect], include_zero=True
    )
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "C. Value neurons vs random sets",
        "Random-neuron set",
        "Positive-minus-negative effect",
        [(1, "1"), (10, "10"), (20, "20")],
        [(lower, f"{lower:.2f}"), (0, "0"), (upper, f"{upper:.2f}")],
        (0, max(21, len(random_effects) + 1)),
        (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, effect in enumerate(random_effects, 1):
        svg.circle(sx(index), sy(effect), 4, "#94A3B8")
    svg.line(
        box[0], sy(value_effect), box[2], sy(value_effect), COLORS["observed"], 2.5
    )
    svg.text(box[0] + 8, sy(value_effect) - 8, "Value-neuron effect", "note")
    svg.text(
        1450,
        550,
        "Error bars: 95% t intervals over episode-averaged paired effects.",
        "note",
        "end",
    )
    svg.save(path)


def _write_report(result: dict, path: Path) -> None:
    total = result["episode_clustered_inference"]["positive_vs_negative"]
    lines = [
        "# Matched-state causal persistence result",
        "",
        f"Data: **{result['n_matched_episodes']} episodes / "
        f"{result['n_matched_states']} states**",
        "",
        f"Positive minus negative steering: **{total['mean']:.4f}** "
        f"(95% CI {total['confidence_interval_95'][0]:.4f} to "
        f"{total['confidence_interval_95'][1]:.4f}; "
        f"p={total['two_sided_p_value']:.3g}).",
        "",
        f"- Monotonic ordering: **{result['primary_monotonic_ordering']}**",
        f"- Primary episode-level gate: **{result['primary_ordering_passed']}**",
        f"- Random controls: **{result['random_set_count']}**",
        f"- Random-control empirical p: **{result['random_control_empirical_p']}**",
        f"- Specificity gate: **{result['random_control_passed']}**",
        "",
        "Inference averages paired state effects within each episode and then "
        "weights episodes equally. Sequential steering remains gated on both "
        "the primary and random-control results.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/matched_intervention.csv")
    parser.add_argument("--output", default="artifacts/matched_analysis.json")
    parser.add_argument("--figure", default="artifacts/matched_analysis.svg")
    parser.add_argument("--report", default="artifacts/matched_analysis.md")
    args = parser.parse_args()
    result = analyze_frame(pd.read_csv(args.input))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _make_figure(result, Path(args.figure))
    _write_report(result, Path(args.report))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
