"""Publication summary for the supervised Monte Carlo future-return probe."""

import argparse
import csv
import json
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.probe_history_matching import (
    exact_history_match_analysis,
    variance_decomposition,
)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        "mechanism": mechanism,
        "variance_decomposition": {
            "sparse": variance_decomposition(rows, mechanism, "probe_value"),
            "full": variance_decomposition(rows, mechanism, "probe_value_full"),
        },
        "exact_history_matching": matching,
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


def _report(summary: dict, output: Path) -> None:
    best = summary["best_layer_metrics"]
    mechanism = summary["mechanism"]
    matching = summary["exact_history_matching"]
    lines = [
        "# Exploratory supervised future-return probe",
        "",
        f"Selected layer: **{summary['best_layer']}**; sparse dimensions: **{summary['neuron_count']}**",
        "",
        "## Future-return prediction",
        "",
        f"- Sparse validation/test R²: **{best['sparse']['validation']['r_squared']:.3f} / {best['sparse']['test']['r_squared']:.3f}**",
        f"- Full validation/test R²: **{best['full']['validation']['r_squared']:.3f} / {best['full']['test']['r_squared']:.3f}**",
        f"- Recent-history test R²: **{summary['baselines']['recent_history']['test']['r_squared']:.3f}**",
        f"- Constant test R²: **{summary['baselines']['constant']['test']['r_squared']:.3f}**",
        "",
        "## Persistence mechanism",
        "",
        f"Sparse adjusted beta: **{mechanism['primary_pruned_probe']['probe_standardized_beta']:.3f}**; "
        f"delta R²: **{mechanism['primary_pruned_probe']['delta_r_squared']:.3f}**; "
        f"p: **{mechanism['primary_pruned_probe']['normal_approximation_p_value']:.3g}**.",
        f"Exact matching retained **{matching['matched_states']} states / {matching['episode_clusters']} episodes**.",
        "",
        "## Interpretation constraints",
        "",
        summary["target_interpretation"] + ".",
        summary["adaptive_evaluation_caveat"],
        "A continuation-advantage target remains a separate, later analysis.",
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
    _report(summary, output / "monte_carlo_probe_report.md")
    print(json.dumps({"best_layer": summary["best_layer"], "test_sparse_r2": summary["best_layer_metrics"]["sparse"]["test"]["r_squared"]}, indent=2))


if __name__ == "__main__":
    main()
