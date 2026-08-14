"""Held-out tests of integrated probe value versus recent-outcome heuristics."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


MECHANISM_ALPHA = 0.05
MIN_INCREMENTAL_R_SQUARED = 0.01


def _standardize(values):
    import torch

    tensor = torch.as_tensor(values, dtype=torch.float64)
    centered = tensor - tensor.mean()
    scale = centered.square().mean().sqrt()
    if float(scale) < 1e-12:
        return torch.zeros_like(tensor), 0.0
    return centered / scale, float(scale)


def _control_matrix(rows: list[dict], include_cumulative_score: bool = False):
    import torch

    columns = [torch.ones(len(rows), dtype=torch.float64)]
    names = ["intercept"]
    candidates = {
        "previous_outcome": [
            0.0 if row["previous_outcome"] is None else float(row["previous_outcome"])
            for row in rows
        ],
        "initial_state": [float(row["previous_outcome"] is None) for row in rows],
        "loss_streak": [float(row["loss_streak"]) for row in rows],
        "log_round": [math.log1p(float(row["round"])) for row in rows],
        "log_round_squared": [math.log1p(float(row["round"])) ** 2 for row in rows],
    }
    if include_cumulative_score:
        candidates["cumulative_score"] = [
            float(row["cumulative_score"]) for row in rows
        ]
    for name, values in candidates.items():
        standardized, scale = _standardize(values)
        if scale > 0:
            columns.append(standardized)
            names.append(name)
    return torch.stack(columns, dim=1), names


def _clustered_ols(outcome, design, clusters: list[str]) -> dict:
    import torch

    outcome = torch.as_tensor(outcome, dtype=torch.float64)
    design = torch.as_tensor(design, dtype=torch.float64)
    beta = torch.linalg.pinv(design) @ outcome
    fitted = design @ beta
    residual = outcome - fitted
    centered = outcome - outcome.mean()
    total = float(centered.square().sum())
    residual_sum = float(residual.square().sum())
    r_squared = 1.0 - residual_sum / total if total > 0 else 0.0

    bread = torch.linalg.pinv(design.T @ design)
    meat = torch.zeros_like(bread)
    grouped = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[cluster].append(index)
    for indices in grouped.values():
        selected = torch.tensor(indices, dtype=torch.long)
        score = design[selected].T @ residual[selected]
        meat += torch.outer(score, score)
    cluster_count = len(grouped)
    n, parameter_count = design.shape
    correction = 1.0
    if cluster_count > 1 and n > parameter_count:
        correction = (cluster_count / (cluster_count - 1)) * (
            (n - 1) / (n - parameter_count)
        )
    covariance = correction * bread @ meat @ bread
    standard_error = covariance.diag().clamp_min(0).sqrt()
    return {
        "beta": beta,
        "standard_error": standard_error,
        "fitted": fitted,
        "residual": residual,
        "r_squared": r_squared,
        "rank": int(torch.linalg.matrix_rank(design)),
        "clusters": cluster_count,
    }


def _normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def _incremental_regression(
    rows: list[dict],
    *,
    outcome_key: str,
    predictor_key: str,
    include_cumulative_score: bool = False,
) -> dict:
    import torch

    if len(rows) < 3:
        raise ValueError("at least three states are required for regression")
    clusters = [str(row["episode_id"]) for row in rows]
    outcome, outcome_scale = _standardize([row[outcome_key] for row in rows])
    predictor, predictor_scale = _standardize([row[predictor_key] for row in rows])
    controls, control_names = _control_matrix(
        rows, include_cumulative_score=include_cumulative_score
    )
    control_fit = _clustered_ols(outcome, controls, clusters)
    predictor_fit = _clustered_ols(predictor, controls, clusters)
    residual_predictor = predictor_fit["residual"]
    residual_scale = float(residual_predictor.square().mean().sqrt())

    if predictor_scale == 0 or residual_scale < 1e-10:
        return {
            "states": len(rows),
            "episode_clusters": len(set(clusters)),
            "controls": control_names,
            "control_r_squared": control_fit["r_squared"],
            "augmented_r_squared": control_fit["r_squared"],
            "delta_r_squared": 0.0,
            "predictor_standardized_beta": 0.0,
            "cluster_robust_standard_error": None,
            "z_value": 0.0,
            "normal_approximation_p_value": 1.0,
            "partial_correlation": 0.0,
            "outcome_scale": outcome_scale,
            "predictor_scale": predictor_scale,
        }

    augmented = torch.cat([controls, residual_predictor[:, None]], dim=1)
    augmented_fit = _clustered_ols(outcome, augmented, clusters)
    coefficient = float(augmented_fit["beta"][-1])
    standard_error = float(augmented_fit["standard_error"][-1])
    z_value = coefficient / standard_error if standard_error > 0 else float("inf")
    residual_outcome = control_fit["residual"]
    partial_correlation = float(
        torch.dot(residual_outcome, residual_predictor)
        / (
            residual_outcome.square().sum().sqrt()
            * residual_predictor.square().sum().sqrt()
        ).clamp_min(1e-12)
    )
    return {
        "states": len(rows),
        "episode_clusters": len(set(clusters)),
        "controls": control_names,
        "control_r_squared": control_fit["r_squared"],
        "augmented_r_squared": augmented_fit["r_squared"],
        "delta_r_squared": max(
            0.0, augmented_fit["r_squared"] - control_fit["r_squared"]
        ),
        "predictor_standardized_beta": coefficient,
        "cluster_robust_standard_error": standard_error,
        "z_value": z_value,
        "normal_approximation_p_value": _normal_two_sided_p(z_value),
        "partial_correlation": partial_correlation,
        "outcome_scale": outcome_scale,
        "predictor_scale": predictor_scale,
    }


def nested_probe_regression(
    rows: list[dict], *, include_cumulative_score: bool = False
) -> dict:
    """Test incremental probe association with persistence on held-out states."""
    result = _incremental_regression(
        rows,
        outcome_key="persistence_logit",
        predictor_key="probe_value",
        include_cumulative_score=include_cumulative_score,
    )
    result["probe_standardized_beta"] = result["predictor_standardized_beta"]
    return result


def _parse_history(value) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in json.loads(value)]
    return [int(item) for item in value]


def extract_probe_rows(
    shards: list,
    probe,
    layer: int,
    neuron_indices,
    episode_ids: Iterable[str],
    full_probe=None,
    probe_output_mean: float = 0.0,
    probe_output_std: float = 1.0,
) -> list[dict]:
    """Evaluate full and sparse probe outputs with behavioral covariates."""
    import torch

    selected_ids = set(episode_ids)
    mask = torch.zeros(probe.hidden.in_features, dtype=torch.float32)
    mask[neuron_indices.long()] = 1.0
    rows = []
    probe.eval()
    full_probe = probe if full_probe is None else full_probe
    full_probe.eval()
    with torch.no_grad():
        for shard in shards:
            if shard["episode_id"] not in selected_ids:
                continue
            activations = shard["activations"][:, layer, :].float()
            full_values = full_probe(activations)
            sparse_values = probe(activations, input_mask=mask)
            for record, full_value, sparse_value in zip(
                shard["records"], full_values, sparse_values
            ):
                reward_history = _parse_history(record["reward_history"])
                loss_streak = 0
                for reward in reversed(reward_history):
                    if reward != -2:
                        break
                    loss_streak += 1
                previous_outcome = record.get("previous_outcome")
                if previous_outcome in ("", None):
                    previous_outcome = None
                else:
                    previous_outcome = float(previous_outcome)
                rows.append(
                    {
                        "episode_id": str(record["episode_id"]),
                        "state_id": str(record["state_id"]),
                        "round": int(record["round"]),
                        "previous_outcome": previous_outcome,
                        "loss_streak": loss_streak,
                        "cumulative_score": float(record["cumulative_score"]),
                        "persistence_logit": float(record["persistence_logit"]),
                        "p_stop": float(record["p_stop"]),
                        "sampled_stop": int(record["sampled_action"] == "C"),
                        "probe_value": float(sparse_value) * probe_output_std
                        + probe_output_mean,
                        "probe_value_full": float(full_value) * probe_output_std
                        + probe_output_mean,
                    }
                )
    if not rows:
        raise ValueError("test split contains no probe-mechanism states")
    return rows


def _with_probe_key(rows: list[dict], key: str) -> list[dict]:
    return [{**row, "probe_value": row[key]} for row in rows]


def analyze_probe_rows(rows: list[dict]) -> dict:
    """Run the prespecified held-out integrated-value diagnostics."""
    last_loss = [row for row in rows if row["previous_outcome"] == -2]
    last_gain = [row for row in rows if row["previous_outcome"] == 3]
    primary = nested_probe_regression(rows)
    score_control = nested_probe_regression(rows, include_cumulative_score=True)
    within_loss = nested_probe_regression(last_loss)
    within_gain = nested_probe_regression(last_gain)
    full_probe = nested_probe_regression(_with_probe_key(rows, "probe_value_full"))
    history_encoding = {
        "all_states": _incremental_regression(
            rows,
            outcome_key="probe_value",
            predictor_key="cumulative_score",
        ),
        "last_loss": _incremental_regression(
            last_loss,
            outcome_key="probe_value",
            predictor_key="cumulative_score",
        ),
        "last_gain": _incremental_regression(
            last_gain,
            outcome_key="probe_value",
            predictor_key="cumulative_score",
        ),
    }
    evidence = {
        "probe_adds_beyond_recent_history": bool(
            primary["probe_standardized_beta"] > 0
            and primary["delta_r_squared"] >= MIN_INCREMENTAL_R_SQUARED
            and primary["normal_approximation_p_value"] < MECHANISM_ALPHA
        ),
        "probe_adds_within_last_loss_states": bool(
            within_loss["probe_standardized_beta"] > 0
            and within_loss["normal_approximation_p_value"] < MECHANISM_ALPHA
        ),
        "probe_encodes_history_within_last_loss_states": bool(
            history_encoding["last_loss"]["predictor_standardized_beta"] > 0
            and history_encoding["last_loss"]["normal_approximation_p_value"]
            < MECHANISM_ALPHA
        ),
    }
    evidence["integrated_value_pattern_supported"] = all(evidence.values())
    return {
        "analysis_split": "test episodes only",
        "states": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "primary_pruned_probe": primary,
        "pruned_probe_controlling_cumulative_score": score_control,
        "full_probe": full_probe,
        "within_previous_outcome": {
            "last_loss": within_loss,
            "last_gain": within_gain,
        },
        "probe_encodes_cumulative_history": history_encoding,
        "evidence_criteria": {
            "two_sided_alpha": MECHANISM_ALPHA,
            "minimum_primary_delta_r_squared": MIN_INCREMENTAL_R_SQUARED,
        },
        "evidence_flags": evidence,
        "caveat": (
            "These held-out associations distinguish integrated value from a "
            "recent-outcome heuristic but are not causal; activation steering "
            "provides the causal test."
        ),
    }


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(result: dict, path: Path) -> None:
    primary = result["primary_pruned_probe"]
    score = result["pruned_probe_controlling_cumulative_score"]
    loss = result["within_previous_outcome"]["last_loss"]
    gain = result["within_previous_outcome"]["last_gain"]
    lines = [
        "# Held-out probe mechanism diagnostic",
        "",
        f"Best layer: **{result['layer']}**; sparse neurons: **{result['neuron_count']}**",
        f"Test data: **{result['episodes']} episodes / {result['states']} states**",
        "",
        "| Analysis | States | Probe beta | Cluster SE | Delta R² | p (normal approx.) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, value in (
        ("Recent-history controls", primary),
        ("Controls + cumulative score", score),
        ("Previous outcome fixed at loss", loss),
        ("Previous outcome fixed at gain", gain),
    ):
        standard_error = value["cluster_robust_standard_error"]
        standard_error_text = (
            "NA" if standard_error is None else f"{standard_error:.3f}"
        )
        lines.append(
            f"| {label} | {value['states']} | "
            f"{value['probe_standardized_beta']:.3f} | "
            f"{standard_error_text} | {value['delta_r_squared']:.3f} | "
            f"{value['normal_approximation_p_value']:.3g} |"
        )
    lines.extend(
        [
            "",
            "Primary controls are previous outcome, an initial-state indicator, "
            "loss streak, and nonlinear round terms. Standard errors use episode "
            "clusters. Layer and sparse-neuron selection used validation data; "
            "this diagnostic uses test episodes only.",
            "",
            "## Evidence flags",
            "",
        ]
    )
    for key, value in result["evidence_flags"].items():
        lines.append(f"- {key}: **{value}**")
    lines.extend(["", result["caveat"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _partial_points(rows: list[dict], include_score: bool = False):
    controls, _ = _control_matrix(rows, include_cumulative_score=include_score)
    clusters = [row["episode_id"] for row in rows]
    persistence, _ = _standardize([row["persistence_logit"] for row in rows])
    probe, _ = _standardize([row["probe_value"] for row in rows])
    y_residual = _clustered_ols(persistence, controls, clusters)["residual"]
    x_residual = _clustered_ols(probe, controls, clusters)["residual"]
    pairs = sorted(zip(x_residual.tolist(), y_residual.tolist()))
    points = []
    for index in range(10):
        start = index * len(pairs) // 10
        end = (index + 1) * len(pairs) // 10
        selected = pairs[start:end]
        if selected:
            points.append(
                (
                    statistics.mean(value[0] for value in selected),
                    statistics.mean(value[1] for value in selected),
                )
            )
    return points


def _make_figure(rows: list[dict], result: dict, path: Path) -> None:
    from analysis.analyze_pilot_detailed import COLORS, Svg, axes

    svg = Svg(1500, 570)
    svg.text(55, 45, "Does the value probe integrate history?", "title")
    svg.text(
        55,
        70,
        "Best validation-selected sparse probe; untouched test episodes only",
        "subtitle",
    )
    panels = ((45, 100), (515, 100), (985, 100))

    x, y = panels[0]
    primary = result["primary_pruned_probe"]
    score = result["pruned_probe_controlling_cumulative_score"]
    values = (
        primary["control_r_squared"],
        primary["augmented_r_squared"],
        score["control_r_squared"],
        score["augmented_r_squared"],
    )
    upper = min(1.0, max(values) * 1.2 + 0.01)
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "A. Held-out persistence variance",
        "Nested model",
        "R²",
        [(0, "Recent"), (1, "+ probe"), (2, "+ score"), (3, "+ probe")],
        [(value, f"{value:.2f}") for value in (0, upper / 2, upper)],
        (-0.6, 3.6),
        (0, upper),
    )
    colors = ("#94A3B8", COLORS["observed"], "#64748B", COLORS["model"])
    for index, (value, color) in enumerate(zip(values, colors)):
        svg.rect(sx(index) - 32, sy(value), 64, box[3] - sy(value), color, 0.9)
        svg.text(sx(index), sy(value) - 8, f"{value:.3f}", anchor="middle")

    x, y = panels[1]
    points = _partial_points(rows)
    x_limit = max(1.0, max(abs(point[0]) for point in points) * 1.15)
    y_limit = max(1.0, max(abs(point[1]) for point in points) * 1.15)
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "B. Partial probe-persistence relation",
        "Residualized probe value",
        "Residualized persistence logit",
        [(-x_limit, f"{-x_limit:.1f}"), (0, "0"), (x_limit, f"{x_limit:.1f}")],
        [(-y_limit, f"{-y_limit:.1f}"), (0, "0"), (y_limit, f"{y_limit:.1f}")],
        (-x_limit, x_limit),
        (-y_limit, y_limit),
    )
    plotted = [(sx(px), sy(py)) for px, py in points]
    svg.polyline(plotted, COLORS["observed"])
    for px, py in plotted:
        svg.circle(px, py, 4.5, COLORS["observed"])
    svg.text(box[0] + 10, box[1] + 18, "Decile means after recent-history controls", "note")

    x, y = panels[2]
    groups = (
        ("All", result["primary_pruned_probe"]),
        ("Last loss", result["within_previous_outcome"]["last_loss"]),
        ("Last gain", result["within_previous_outcome"]["last_gain"]),
    )
    estimates = [group[1]["probe_standardized_beta"] for group in groups]
    errors = [group[1]["cluster_robust_standard_error"] or 0 for group in groups]
    limit = max(
        0.5,
        max(
            abs(value) + 2 * error
            for value, error in zip(estimates, errors)
        )
        * 1.15,
    )
    sx, sy, box = axes(
        svg,
        x,
        y,
        430,
        410,
        "C. Probe effect within the latest outcome",
        "Held-out state subset",
        "Standardized probe coefficient",
        [(index, label) for index, (label, _) in enumerate(groups)],
        [(-limit, f"{-limit:.1f}"), (0, "0"), (limit, f"{limit:.1f}")],
        (-0.6, 2.6),
        (-limit, limit),
    )
    svg.line(box[0], sy(0), box[2], sy(0), "#6B7280", 1.2, "5 5")
    for index, (estimate, error) in enumerate(zip(estimates, errors)):
        lower = sy(estimate - 1.96 * error)
        upper = sy(estimate + 1.96 * error)
        center = sx(index)
        svg.line(center, lower, center, upper, COLORS["model"], 2)
        svg.line(center - 7, lower, center + 7, lower, COLORS["model"], 2)
        svg.line(center - 7, upper, center + 7, upper, COLORS["model"], 2)
        svg.circle(sx(index), sy(estimate), 5, COLORS["model"])
    svg.text(
        1450,
        550,
        "Error bars: normal 95% intervals using episode-clustered sandwich standard errors.",
        "note",
        "end",
    )
    svg.save(path)


def run_probe_mechanism_analysis(
    shards: list,
    probe,
    layer: int,
    neuron_indices,
    test_episode_ids: Iterable[str],
    output_dir: str | Path,
    full_probe=None,
    probe_output_mean: float = 0.0,
    probe_output_std: float = 1.0,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = extract_probe_rows(
        shards,
        probe,
        layer,
        neuron_indices,
        test_episode_ids,
        full_probe=full_probe,
        probe_output_mean=probe_output_mean,
        probe_output_std=probe_output_std,
    )
    result = analyze_probe_rows(rows)
    result["layer"] = int(layer)
    result["neuron_count"] = int(len(neuron_indices))
    _write_rows(rows, output / "probe_mechanism_test_states.csv")
    (output / "probe_mechanism.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(result, output / "probe_mechanism_report.md")
    _make_figure(rows, result, output / "probe_mechanism.svg")
    return result
