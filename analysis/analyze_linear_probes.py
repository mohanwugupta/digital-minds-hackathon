"""Publication diagnostics for ridge future-return and persistence probes."""

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
            for key in ("round", "loss_streak"):
                row[key] = int(float(row[key]))
            row["previous_outcome"] = (
                None
                if row["previous_outcome"] in (None, "")
                else float(row["previous_outcome"])
            )
            for key in (
                "cumulative_score",
                "future_return",
                "persistence_logit",
                "ridge_future_return",
                "ridge_persistence",
            ):
                row[key] = float(row[key])
            row["prior_score"] = row["cumulative_score"] - float(
                row["previous_outcome"] or 0
            )
            rows.append(row)
    return rows


def recent_history(rows: list[dict]) -> dict[str, list[float]]:
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


def exact_residuals(rows: list[dict], min_stratum_size: int = 4) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row["previous_outcome"] is None:
            continue
        grouped[
            (row["round"], row["previous_outcome"], row["loss_streak"])
        ].append(row)
    eligible = [
        selected
        for selected in grouped.values()
        if len(selected) >= min_stratum_size
        and max(row["prior_score"] for row in selected)
        > min(row["prior_score"] for row in selected)
    ]
    keys = (
        "prior_score",
        "future_return",
        "persistence_logit",
        "ridge_future_return",
        "ridge_persistence",
    )
    residuals = {key: [] for key in keys}
    clusters = []
    for selected in eligible:
        means = {
            key: statistics.mean(row[key] for row in selected) for key in keys
        }
        for row in selected:
            for key in keys:
                residuals[key].append(row[key] - means[key])
            clusters.append(row["episode_id"])
    if not clusters:
        raise ValueError("no eligible exact-match strata")
    return {
        "eligible_strata": len(eligible),
        "states": len(clusters),
        "episode_clusters": len(set(clusters)),
        "residuals": residuals,
        "clusters": clusters,
    }


def regression(outcome, predictors, clusters) -> dict:
    return _clustered_regression(outcome, predictors, clusters)


def summarize(metrics: dict, rows: list[dict], nonlinear: dict | None) -> dict:
    clusters = [row["episode_id"] for row in rows]
    controls = recent_history(rows)
    persistence = [row["persistence_logit"] for row in rows]
    future = [row["future_return"] for row in rows]
    ridge_return = [row["ridge_future_return"] for row in rows]
    direct_persistence = [row["ridge_persistence"] for row in rows]
    matched = exact_residuals(rows)
    residuals, matched_clusters = matched["residuals"], matched["clusters"]
    exact_models = {
        "ridge_return_to_future": regression(
            residuals["future_return"],
            {"ridge_future_return": residuals["ridge_future_return"]},
            matched_clusters,
        ),
        "ridge_return_to_persistence": regression(
            residuals["persistence_logit"],
            {"ridge_future_return": residuals["ridge_future_return"]},
            matched_clusters,
        ),
        "direct_probe_to_persistence": regression(
            residuals["persistence_logit"],
            {"ridge_persistence": residuals["ridge_persistence"]},
            matched_clusters,
        ),
        "older_history_to_ridge_return": regression(
            residuals["ridge_future_return"],
            {"prior_score": residuals["prior_score"]},
            matched_clusters,
        ),
    }
    return {
        "best_layers": metrics["best_layers"],
        "best_layer_metrics": {
            target: next(
                row["targets"][target]
                for row in metrics["layers"]
                if row["layer"] == metrics["best_layers"][target]
            )
            for target in ("future_return", "persistence")
        },
        "nonlinear_future_return": (
            None
            if nonlinear is None
            else next(
                row
                for row in nonlinear["layers"]
                if row["layer"] == nonlinear["best_layer"]
            )
        ),
        "persistence_models": {
            "history_only": regression(persistence, controls, clusters),
            "ridge_return_only": regression(
                persistence, {"ridge_future_return": ridge_return}, clusters
            ),
            "history_plus_ridge_return": regression(
                persistence,
                {**controls, "ridge_future_return": ridge_return},
                clusters,
            ),
            "direct_probe_only": regression(
                persistence, {"ridge_persistence": direct_persistence}, clusters
            ),
        },
        "future_return_models": {
            "history_only": regression(future, controls, clusters),
            "ridge_return_only": regression(
                future, {"ridge_future_return": ridge_return}, clusters
            ),
            "history_plus_ridge_return": regression(
                future, {**controls, "ridge_future_return": ridge_return}, clusters
            ),
        },
        "exact_matching": {
            key: value for key, value in matched.items() if key != "residuals"
        }
        | {"models": exact_models},
        "layer_overlap": [
            {
                "layer": row["layer"],
                **row["return_persistence_overlap"],
            }
            for row in metrics["layers"]
        ],
        "caveat": (
            "Layer and ridge alpha use validation episodes; all reported test "
            "metrics and exact-match diagnostics use untouched test episodes."
        ),
    }


def make_figure(metrics: dict, summary: dict, path: Path) -> None:
    svg = Svg(1500, 1030)
    svg.text(55, 45, "Linear decoding: return → advantage → persistence", "title")
    svg.text(55, 70, "Ridge probes with episode-held-out layer and regularization selection", "subtitle")
    panels = ((45, 95), (760, 95), (45, 565), (760, 565))
    layers = metrics["layers"]

    x, y = panels[0]
    series = (
        ("Return validation", [row["targets"]["future_return"]["validation"]["r_squared"] for row in layers], COLORS["observed"]),
        ("Return test", [row["targets"]["future_return"]["test"]["r_squared"] for row in layers], COLORS["model"]),
    )
    values = [value for _label, data, _color in series for value in data]
    lower, upper = min(-0.1, min(values) - 0.05), max(0.2, max(values) + 0.05)
    sx, sy, box = axes(svg, x, y, 690, 430, "A. Ridge future-return decoding", "Layer", "R²", [(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")], [(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")], (0,31), (lower,upper))
    for label, data, color in series:
        svg.polyline([(sx(i), sy(value)) for i, value in enumerate(data)], color, 2)
    legend(svg, box[0]+5, box[1]+18, [(label,color) for label,_data,color in series])

    x, y = panels[1]
    data = [row["targets"]["persistence"]["test"]["r_squared"] for row in layers]
    lower, upper = min(-0.05, min(data)-0.03), max(1.0, max(data)+0.03)
    sx, sy, box = axes(svg, x, y, 690, 430, "B. Where persistence becomes linearly decodable", "Layer", "Test R²", [(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")], [(lower,f"{lower:.1f}"),(0,"0"),(0.5,".5"),(upper,f"{upper:.1f}")], (0,31), (lower,upper))
    svg.polyline([(sx(i),sy(value)) for i,value in enumerate(data)], COLORS["both_positive"], 2)

    x, y = panels[2]
    exact = summary["exact_matching"]["models"]
    estimates = (
        ("Ridge return → return", exact["ridge_return_to_future"], "ridge_future_return", COLORS["observed"]),
        ("Ridge return → persist", exact["ridge_return_to_persistence"], "ridge_future_return", COLORS["model"]),
        ("Direct probe → persist", exact["direct_probe_to_persistence"], "ridge_persistence", COLORS["both_positive"]),
    )
    bounds=[]
    for _label,result,key,_color in estimates:
        c=result["coefficients"][key]; b=c["standardized_beta"]; e=c["cluster_robust_standard_error"] or 0; bounds.extend((b-1.96*e,b+1.96*e))
    lower,upper=min(-.3,min(bounds)-.05),max(1.0,max(bounds)+.05)
    sx,sy,box=axes(svg,x,y,690,430,"C. Exact matched-state discriminator","Standardized coefficient","Target",[(lower,f"{lower:.1f}"),(0,"0"),(upper,f"{upper:.1f}")],[(i,label) for i,(label,_r,_k,_c) in enumerate(estimates)],(lower,upper),(-.6,len(estimates)-.4))
    svg.line(sx(0),box[1],sx(0),box[3],COLORS["reference"],1.2,"5 5")
    for i,(_label,result,key,color) in enumerate(estimates):
        c=result["coefficients"][key]; b=c["standardized_beta"]; e=c["cluster_robust_standard_error"] or 0
        svg.line(sx(b-1.96*e),sy(i),sx(b+1.96*e),sy(i),color,2); svg.circle(sx(b),sy(i),5,color)

    x,y=panels[3]
    overlap=summary["layer_overlap"]
    cosine=[row["absolute_cosine_similarity"] for row in overlap]
    jaccard=[row["top_dimension_jaccard"] for row in overlap]
    sx,sy,box=axes(svg,x,y,690,430,"D. Return/persistence direction overlap","Layer","Overlap",[(0,"0"),(8,"8"),(16,"16"),(24,"24"),(31,"31")],[(0,"0"),(.5,".5"),(1,"1")],(0,31),(0,1))
    svg.polyline([(sx(i),sy(value)) for i,value in enumerate(cosine)],COLORS["observed"],2)
    svg.polyline([(sx(i),sy(value)) for i,value in enumerate(jaccard)],COLORS["model"],2)
    legend(svg,box[0]+5,box[1]+18,[("|Cosine|",COLORS["observed"]),("Top-1% Jaccard",COLORS["model"])])
    svg.save(path)


def write_report(summary: dict, path: Path) -> None:
    return_metrics = summary["best_layer_metrics"]["future_return"]
    persistence_metrics = summary["best_layer_metrics"]["persistence"]
    exact = summary["exact_matching"]
    models = exact["models"]
    return_exact = models["ridge_return_to_future"]["coefficients"]["ridge_future_return"]
    return_persist = models["ridge_return_to_persistence"]["coefficients"]["ridge_future_return"]
    direct = models["direct_probe_to_persistence"]["coefficients"]["ridge_persistence"]
    maximum_cosine = max(
        summary["layer_overlap"], key=lambda row: row["absolute_cosine_similarity"]
    )
    maximum_jaccard = max(
        summary["layer_overlap"], key=lambda row: row["top_dimension_jaccard"]
    )
    nonlinear = summary["nonlinear_future_return"]
    nonlinear_text = "not available"
    if nonlinear is not None:
        nonlinear_text = f"{nonlinear['full']['test']['r_squared']:.3f} full / {nonlinear['sparse']['test']['r_squared']:.3f} sparse"
    lines = [
        "# Ridge-linear future-return and persistence probes",
        "",
        "## Bottom line",
        "",
        f"Future return is linearly decodable at layer **{summary['best_layers']['future_return']}** "
        f"(validation/test R² **{return_metrics['validation']['r_squared']:.3f} / {return_metrics['test']['r_squared']:.3f}**). "
        f"The nonlinear benchmark test R² is {nonlinear_text}.",
        f"The direct persistence probe peaks at layer **{summary['best_layers']['persistence']}** "
        f"(validation/test R² **{persistence_metrics['validation']['r_squared']:.3f} / {persistence_metrics['test']['r_squared']:.3f}**).",
        "",
        "## Exact matching",
        "",
        f"Matching retained **{exact['states']} states in {exact['eligible_strata']} strata from {exact['episode_clusters']} episodes**.",
        f"Ridge return → actual future return: beta **{return_exact['standardized_beta']:.3f}**, p={return_exact['normal_approximation_p_value']:.3g}.",
        f"Ridge return → persistence: beta **{return_persist['standardized_beta']:.3f}**, p={return_persist['normal_approximation_p_value']:.3g}.",
        f"Direct persistence probe → persistence: beta **{direct['standardized_beta']:.3f}**, p={direct['normal_approximation_p_value']:.3g}.",
        "",
        "## Direction overlap",
        "",
        f"Maximum absolute return/persistence direction cosine was **{maximum_cosine['absolute_cosine_similarity']:.3f}** at layer **{maximum_cosine['layer']}**. "
        f"Maximum top-1% dimension Jaccard was **{maximum_jaccard['top_dimension_jaccard']:.3f}** at layer **{maximum_jaccard['layer']}**.",
        "",
        "The direct probe is a localization/control analysis, not evidence for value. Continuation advantage remains the substantive missing target.",
        "",
        summary["caveat"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="artifacts/linear_probes")
    parser.add_argument("--nonlinear-dir", default="artifacts/mc_value_probes")
    parser.add_argument("--output-dir", default="artifacts/linear_probes/publication")
    args = parser.parse_args()
    probe_dir, output = Path(args.probe_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((probe_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = read_rows(probe_dir / "test_predictions.csv")
    nonlinear_path = Path(args.nonlinear_dir) / "metrics.json"
    nonlinear = json.loads(nonlinear_path.read_text()) if nonlinear_path.exists() else None
    summary = summarize(metrics, rows, nonlinear)
    (output / "linear_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(metrics, summary, output / "linear_probe_results.svg")
    write_report(summary, output / "linear_probe_report.md")
    print(json.dumps({"best_layers": summary["best_layers"]}, indent=2))


if __name__ == "__main__":
    main()
