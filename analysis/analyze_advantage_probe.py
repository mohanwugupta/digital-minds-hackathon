"""Adjudicate continuation advantage versus recent-state stopping heuristics."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes, legend
from analysis.probe_history_matching import _clustered_regression


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            row["round"] = int(float(row["round"]))
            row["loss_streak"] = int(float(row["loss_streak"]))
            row["previous_outcome"] = (
                None if row["previous_outcome"] == "" else float(row["previous_outcome"])
            )
            for key in (
                "cumulative_score",
                "persistence_logit",
                "q_A",
                "q_B",
                "continuation_advantage",
                "advantage_standard_error",
                "ridge_advantage",
            ):
                row[key] = float(row[key])
            row["prior_score"] = row["cumulative_score"] - float(
                row["previous_outcome"] or 0
            )
            rows.append(row)
    return rows


def controls(rows: list[dict]) -> dict[str, list[float]]:
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


def exact_match(rows: list[dict], minimum: int = 4) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row["previous_outcome"] is not None:
            grouped[(row["round"], row["previous_outcome"], row["loss_streak"])].append(row)
    eligible = [selected for selected in grouped.values() if len(selected) >= minimum]
    keys = (
        "prior_score",
        "continuation_advantage",
        "ridge_advantage",
        "persistence_logit",
    )
    residuals = {key: [] for key in keys}
    clusters = []
    for selected in eligible:
        means = {key: statistics.mean(row[key] for row in selected) for key in keys}
        for row in selected:
            for key in keys:
                residuals[key].append(row[key] - means[key])
            clusters.append(row["episode_id"])
    if not clusters:
        raise ValueError(
            "no exact-match strata; collect more test states or increase states-per-split"
        )

    def fit(outcome, *predictors):
        return _clustered_regression(
            residuals[outcome],
            {key: residuals[key] for key in predictors},
            clusters,
        )

    return {
        "eligible_strata": len(eligible),
        "states": len(clusters),
        "episode_clusters": len(set(clusters)),
        "models": {
            "probe_to_advantage": fit("continuation_advantage", "ridge_advantage"),
            "actual_advantage_to_persistence": fit(
                "persistence_logit", "continuation_advantage"
            ),
            "probe_advantage_to_persistence": fit(
                "persistence_logit", "ridge_advantage"
            ),
            "older_history_to_advantage": fit(
                "continuation_advantage", "prior_score"
            ),
            "older_history_plus_probe_to_persistence": fit(
                "persistence_logit", "prior_score", "ridge_advantage"
            ),
        },
    }


def summarize(metrics: dict, rows: list[dict], linear_summary: dict | None) -> dict:
    recent = controls(rows)
    clusters = [row["episode_id"] for row in rows]
    persistence = [row["persistence_logit"] for row in rows]
    predicted = [row["ridge_advantage"] for row in rows]
    actual = [row["continuation_advantage"] for row in rows]
    best = next(row for row in metrics["layers"] if row["layer"] == metrics["best_layer"])
    return {
        "best_layer": metrics["best_layer"],
        "best_layer_metrics": best,
        "labeled_states_by_split": metrics["labeled_states_by_split"],
        "target_reliability": {
            "target_variance": statistics.pvariance(actual),
            "mean_estimation_error_variance": statistics.mean(
                row["advantage_standard_error"] ** 2 for row in rows
            ),
            "median_standard_error": statistics.median(
                row["advantage_standard_error"] for row in rows
            ),
        },
        "persistence_models": {
            "history_only": _clustered_regression(persistence, recent, clusters),
            "advantage_probe_only": _clustered_regression(
                persistence, {"ridge_advantage": predicted}, clusters
            ),
            "history_plus_advantage_probe": _clustered_regression(
                persistence, {**recent, "ridge_advantage": predicted}, clusters
            ),
            "actual_advantage_only": _clustered_regression(
                persistence, {"continuation_advantage": actual}, clusters
            ),
        },
        "exact_matching": exact_match(rows),
        "advantage_persistence_overlap": [
            {
                "layer": row["layer"],
                **(row["advantage_persistence_overlap"] or {}),
            }
            for row in metrics["layers"]
        ],
        "direct_persistence_reference": linear_summary,
        "target": metrics["target"],
        "caveat": metrics["caveat"],
    }


def make_figure(metrics: dict, summary: dict, path: Path) -> None:
    svg = Svg(1500, 620)
    svg.text(55, 45, "Does continuation advantage explain persistence?", "title")
    svg.text(55, 70, "Forced A/B paired rollouts; ridge probes; held-out episode tests", "subtitle")
    panels = ((45, 105), (520, 105), (995, 105))
    layers = metrics["layers"]

    x, y = panels[0]
    validation = [row["validation"]["r_squared"] for row in layers]
    test = [row["test"]["r_squared"] for row in layers]
    values = validation + test
    lower, upper = min(-.1, min(values)-.04), max(.2, max(values)+.04)
    sx,sy,box=axes(svg,x,y,455,440,"A. Advantage decoding by layer","Layer","R²",[(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")],[(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")],(0,31),(lower,upper))
    svg.polyline([(sx(i),sy(v)) for i,v in enumerate(validation)],COLORS["observed"],2)
    svg.polyline([(sx(i),sy(v)) for i,v in enumerate(test)],COLORS["model"],2)
    legend(svg,box[0]+4,box[1]+18,[("Validation",COLORS["observed"]),("Test",COLORS["model"])])

    x,y=panels[1]
    exact=summary["exact_matching"]["models"]
    estimates=(
        ("Probe → advantage",exact["probe_to_advantage"],"ridge_advantage",COLORS["observed"]),
        ("Advantage → persist",exact["actual_advantage_to_persistence"],"continuation_advantage",COLORS["both_positive"]),
        ("Probe → persist",exact["probe_advantage_to_persistence"],"ridge_advantage",COLORS["model"]),
    )
    bounds=[]
    for _l,r,k,_c in estimates:
        c=r["coefficients"][k];b=c["standardized_beta"];e=c["cluster_robust_standard_error"] or 0;bounds.extend((b-1.96*e,b+1.96*e))
    lower,upper=min(-.5,min(bounds)-.05),max(.8,max(bounds)+.05)
    sx,sy,box=axes(svg,x,y,455,440,"B. Exact matched-state tests","Standardized coefficient","Relation",[(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")],[(i,l) for i,(l,_r,_k,_c) in enumerate(estimates)],(lower,upper),(-.6,2.6))
    svg.line(sx(0),box[1],sx(0),box[3],COLORS["reference"],1.2,"5 5")
    for i,(_l,r,k,color) in enumerate(estimates):
        c=r["coefficients"][k];b=c["standardized_beta"];e=c["cluster_robust_standard_error"] or 0
        svg.line(sx(b-1.96*e),sy(i),sx(b+1.96*e),sy(i),color,2);svg.circle(sx(b),sy(i),5,color)

    x,y=panels[2]
    overlap=summary["advantage_persistence_overlap"]
    available=[row for row in overlap if "absolute_cosine_similarity" in row]
    cosine=[row["absolute_cosine_similarity"] for row in available]
    jaccard=[row["top_dimension_jaccard"] for row in available]
    sx,sy,box=axes(svg,x,y,455,440,"C. Advantage/decision overlap","Layer","Overlap",[(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")],[(0,"0"),(.5,".5"),(1,"1")],(0,31),(0,1))
    if available:
        svg.polyline([(sx(row["layer"]),sy(value)) for row,value in zip(available,cosine)],COLORS["observed"],2)
        svg.polyline([(sx(row["layer"]),sy(value)) for row,value in zip(available,jaccard)],COLORS["model"],2)
    legend(svg,box[0]+4,box[1]+18,[("|Cosine|",COLORS["observed"]),("Top-1% Jaccard",COLORS["model"])])
    svg.save(path)


def write_report(summary: dict, path: Path) -> None:
    best = summary["best_layer_metrics"]
    exact = summary["exact_matching"]
    models = exact["models"]
    probe_target = models["probe_to_advantage"]["coefficients"]["ridge_advantage"]
    actual_persist = models["actual_advantage_to_persistence"]["coefficients"]["continuation_advantage"]
    probe_persist = models["probe_advantage_to_persistence"]["coefficients"]["ridge_advantage"]
    history = summary["persistence_models"]["history_only"]
    joint = summary["persistence_models"]["history_plus_advantage_probe"]
    unique = joint["r_squared"] - history["r_squared"]
    available_overlap = [
        row
        for row in summary["advantage_persistence_overlap"]
        if "absolute_cosine_similarity" in row
    ]
    if (
        probe_target["normal_approximation_p_value"] < .05
        and probe_persist["normal_approximation_p_value"] < .05
        and probe_persist["standardized_beta"] > 0
    ):
        verdict = "Continuation advantage is decodable and positively tracks persistence within matched states."
    elif probe_target["normal_approximation_p_value"] < .05:
        verdict = "Continuation advantage is decodable but does not track persistence; the simple heuristic account is strengthened."
    else:
        verdict = "The probe does not reliably decode continuation advantage, so the mechanism test is inconclusive."
    lines = [
        "# Continuation-advantage probe",
        "",
        "## Bottom line",
        "",
        verdict,
        "",
        f"Selected layer: **{summary['best_layer']}**. Validation/test advantage R²: **{best['validation']['r_squared']:.3f} / {best['test']['r_squared']:.3f}**.",
        f"Exact matching retained **{exact['states']} states in {exact['eligible_strata']} strata from {exact['episode_clusters']} episodes**.",
        "",
        "## Critical matched-state coefficients",
        "",
        f"- Probe → continuation advantage: beta **{probe_target['standardized_beta']:.3f}**, p={probe_target['normal_approximation_p_value']:.3g}.",
        f"- Actual rollout advantage → persistence: beta **{actual_persist['standardized_beta']:.3f}**, p={actual_persist['normal_approximation_p_value']:.3g}.",
        f"- Advantage probe → persistence: beta **{probe_persist['standardized_beta']:.3f}**, p={probe_persist['normal_approximation_p_value']:.3g}.",
        f"- Unique persistence R² beyond recent history: **{unique:.4f}**.",
        "",
        "## Advantage/decision direction overlap",
        "",
    ]
    if available_overlap:
        maximum_cosine = max(
            available_overlap,
            key=lambda row: row["absolute_cosine_similarity"],
        )
        maximum_jaccard = max(
            available_overlap, key=lambda row: row["top_dimension_jaccard"]
        )
        lines.extend(
            [
                f"Maximum absolute cosine was **{maximum_cosine['absolute_cosine_similarity']:.3f}** at layer **{maximum_cosine['layer']}**; "
                f"maximum top-1% Jaccard was **{maximum_jaccard['top_dimension_jaccard']:.3f}** at layer **{maximum_jaccard['layer']}**.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Direct persistence probe artifacts were unavailable, so overlap was not computed.",
                "",
            ]
        )
    lines.extend([
        "## Target precision",
        "",
        f"Median per-state rollout SE: **{summary['target_reliability']['median_standard_error']:.3f}**; target variance: **{summary['target_reliability']['target_variance']:.3f}**; mean estimation-error variance: **{summary['target_reliability']['mean_estimation_error_variance']:.3f}**.",
        "",
        summary["target"],
        summary["caveat"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="artifacts/advantage_probes")
    parser.add_argument("--linear-summary", default="artifacts/linear_probes/publication/linear_probe_summary.json")
    parser.add_argument("--output-dir", default="artifacts/advantage_probes/publication")
    args = parser.parse_args()
    probe_dir, output = Path(args.probe_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((probe_dir / "metrics.json").read_text())
    rows = read_rows(probe_dir / "test_predictions.csv")
    linear_path = Path(args.linear_summary)
    linear = json.loads(linear_path.read_text()) if linear_path.exists() else None
    summary = summarize(metrics, rows, linear)
    (output / "advantage_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(metrics, summary, output / "advantage_probe_results.svg")
    write_report(summary, output / "advantage_probe_report.md")
    print(json.dumps({"best_layer": summary["best_layer"]}, indent=2))


if __name__ == "__main__":
    main()
