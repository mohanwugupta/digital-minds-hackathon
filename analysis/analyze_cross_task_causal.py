"""Analyze causal transfer of the bandit persistence direction across tasks."""

import argparse
import csv
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.analyze_pilot_detailed import COLORS, Svg, axes
from experiments.runtime import run_metadata

def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                row = dict(source)
                for key in (
                    "alpha",
                    "p_positive",
                    "p_negative",
                    "choice_logit",
                    "probe_value_pre",
                    "probe_value_post",
                    "direction_l2_norm",
                    "intervention_relative_rms",
                    "baseline_replay_absolute_difference",
                ):
                    row[key] = float(row[key])
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no cross-task causal rows match {pattern}")
    return rows


def audit_rows(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    row_keys = []
    for row in rows:
        grouped[(row["state_id"], row["control_type"], row["control_id"])].append(row)
        row_keys.append(
            (
                row["state_id"],
                row["control_type"],
                row["control_id"],
                row["alpha"],
            )
        )
    incomplete = context_failures = probe_ordering_failures = 0
    for selected in grouped.values():
        by_alpha = {row["alpha"]: row for row in selected}
        if set(by_alpha) != {-1.0, 0.0, 1.0} or len(selected) != 3:
            incomplete += 1
            continue
        context_failures += len({row["context_hash"] for row in selected}) != 1
        if selected[0]["control_type"] == "target":
            probe_ordering_failures += not (
                by_alpha[-1.0]["probe_value_post"]
                < by_alpha[0.0]["probe_value_post"]
                < by_alpha[1.0]["probe_value_post"]
            )
    alpha_zero = defaultdict(list)
    for row in rows:
        if row["alpha"] == 0:
            alpha_zero[row["state_id"]].append(row)
    zero_failures = 0
    for selected in alpha_zero.values():
        signatures = {
            (
                row["p_positive"],
                row["p_negative"],
                row["choice_logit"],
                row.get("logit_X"),
                row.get("logit_Y"),
            )
            for row in selected
        }
        zero_failures += len(signatures) != 1
    return {
        "rows": len(rows),
        "unique_rows": len(set(row_keys)),
        "duplicate_rows": len(row_keys) - len(set(row_keys)),
        "states": len({row["state_id"] for row in rows}),
        "episodes": len({row["episode_id"] for row in rows}),
        "direction_state_groups": len(grouped),
        "incomplete_triplets": incomplete,
        "context_hash_failures": context_failures,
        "probe_ordering_failures": probe_ordering_failures,
        "alpha_zero_exact_across_directions": zero_failures == 0,
        "alpha_zero_failures": zero_failures,
        "maximum_collection_baseline_difference": max(
            row.get("baseline_replay_absolute_difference", 0.0) for row in rows
        ),
        "direction_l2_norm_values": sorted(
            {row["direction_l2_norm"] for row in rows}
        ),
        "intervention_relative_rms_values": sorted(
            {row["intervention_relative_rms"] for row in rows}
        ),
        "matched_direction_geometry": (
            len({row["direction_l2_norm"] for row in rows}) == 1
            and len({row["intervention_relative_rms"] for row in rows}) == 1
        ),
    }


def _effects(rows: list[dict], control_type: str, control_id: str) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        if row["control_type"] == control_type and row["control_id"] == control_id:
            grouped[row["state_id"]][row["alpha"]] = row
    result = []
    for state_id, by_alpha in grouped.items():
        if set(by_alpha) != {-1.0, 0.0, 1.0}:
            continue
        result.append(
            {
                "state_id": state_id,
                "episode_id": by_alpha[0.0]["episode_id"],
                "cluster_id": by_alpha[0.0].get(
                    "pair_id", by_alpha[0.0]["episode_id"]
                ),
                "mapping_id": by_alpha[0.0]["mapping_id"],
                "probability_difference": by_alpha[1.0]["p_positive"]
                - by_alpha[-1.0]["p_positive"],
                "logit_difference": by_alpha[1.0]["choice_logit"]
                - by_alpha[-1.0]["choice_logit"],
                "monotonic": by_alpha[-1.0]["p_positive"]
                < by_alpha[0.0]["p_positive"]
                < by_alpha[1.0]["p_positive"],
                "p_negative": by_alpha[-1.0]["p_positive"],
                "p_zero": by_alpha[0.0]["p_positive"],
                "p_positive": by_alpha[1.0]["p_positive"],
            }
        )
    return result


def _episode_bootstrap(effects: list[dict], samples: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in effects:
        grouped[row.get("cluster_id", row["episode_id"])].append(
            row["probability_difference"]
        )
    episodes = sorted(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [
            value
            for _episode in episodes
            for value in grouped[rng.choice(episodes)]
        ]
        draws.append(statistics.mean(selected))
    ordered = sorted(draws)
    return {
        "samples": samples,
        "cluster_unit": "counterbalanced_pair_when_available_else_episode",
        "lower_95": ordered[int(0.025 * samples)],
        "upper_95": ordered[max(0, int(0.975 * samples) - 1)],
        "bootstrap_mean": statistics.mean(draws),
    }


def summarize_task(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
    expected_random_directions: int | None = None,
    baseline_tolerance: float = 1e-4,
) -> dict:
    audit = audit_rows(rows)
    if (
        audit["duplicate_rows"]
        or audit["incomplete_triplets"]
        or audit["context_hash_failures"]
        or not audit["alpha_zero_exact_across_directions"]
        or not audit["matched_direction_geometry"]
        or audit["maximum_collection_baseline_difference"] > baseline_tolerance
    ):
        raise ValueError(f"causal replay audit failed: {audit}")
    target = _effects(rows, "target", "target")
    if not target:
        raise ValueError("causal replay has no target-direction effects")
    random_ids = sorted(
        {
            row["control_id"]
            for row in rows
            if row["control_type"] == "random"
        }
    )
    random_means = {
        control_id: statistics.mean(
            row["probability_difference"]
            for row in _effects(rows, "random", control_id)
        )
        for control_id in random_ids
    }
    target_states = {row["state_id"] for row in target}
    random_state_coverage = {
        control_id: {
            row["state_id"]
            for row in _effects(rows, "random", control_id)
        }
        for control_id in random_ids
    }
    coverage_failures = sum(
        selected != target_states for selected in random_state_coverage.values()
    )
    count_failure = (
        expected_random_directions is not None
        and len(random_ids) != int(expected_random_directions)
    )
    if count_failure or coverage_failures:
        raise ValueError(
            "causal random-control coverage failed: "
            f"expected={expected_random_directions}, observed={len(random_ids)}, "
            f"state_coverage_failures={coverage_failures}"
        )
    random_95 = (
        sorted(random_means.values())[
            max(0, int(0.95 * len(random_means) + 0.999999) - 1)
        ]
        if random_means
        else None
    )
    means_by_alpha = {
        key: statistics.mean(row[key] for row in target)
        for key in ("p_negative", "p_zero", "p_positive")
    }
    mapping_effects = {
        mapping_id: statistics.mean(
            row["probability_difference"]
            for row in target
            if row["mapping_id"] == mapping_id
        )
        for mapping_id in sorted({row["mapping_id"] for row in target})
    }
    mapping_bootstrap = {
        mapping_id: _episode_bootstrap(
            [row for row in target if row["mapping_id"] == mapping_id],
            bootstrap_samples,
            seed + 100 + index,
        )
        for index, mapping_id in enumerate(sorted(mapping_effects))
    }
    target_mean = statistics.mean(row["probability_difference"] for row in target)
    return {
        "audit": audit,
        "target": {
            "states": len(target),
            "mean_probability_difference": target_mean,
            "mean_logit_difference": statistics.mean(
                row["logit_difference"] for row in target
            ),
            "means_by_alpha": means_by_alpha,
            "state_monotonic_fraction": statistics.mean(
                float(row["monotonic"]) for row in target
            ),
            "mapping_probability_differences": mapping_effects,
            "mapping_episode_bootstrap": mapping_bootstrap,
            "episode_bootstrap": _episode_bootstrap(
                target, bootstrap_samples, seed
            ),
        },
        "random_controls": {
            "count": len(random_means),
            "expected_count": expected_random_directions,
            "state_coverage_failures": coverage_failures,
            "mean_probability_differences": random_means,
            "probability_difference_95th_percentile": random_95,
        },
    }


def analyze(
    foraging_rows: list[dict],
    control_rows: list[dict] | None = None,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 102026,
    expected_random_directions: int | None = None,
    baseline_tolerance: float = 1e-4,
    negative_control_max_absolute_effect: float = 0.05,
    negative_control_relative_effect_fraction: float = 0.50,
    require_negative_control: bool = False,
) -> dict:
    foraging = summarize_task(
        foraging_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        expected_random_directions=expected_random_directions,
        baseline_tolerance=baseline_tolerance,
    )
    negative_control = (
        summarize_task(
            control_rows,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
            expected_random_directions=expected_random_directions,
            baseline_tolerance=baseline_tolerance,
        )
        if control_rows
        else None
    )
    if require_negative_control and negative_control is None:
        raise ValueError("causal Track B requires the non-persistence control replay")
    target = foraging["target"]
    random_95 = foraging["random_controls"]["probability_difference_95th_percentile"]
    means = target["means_by_alpha"]
    target_effect = target["mean_probability_difference"]
    control_effect = (
        negative_control["target"]["mean_probability_difference"]
        if negative_control is not None
        else None
    )
    specificity = (
        control_effect is None
        or abs(control_effect) <= float(negative_control_max_absolute_effect)
        or abs(control_effect)
        <= float(negative_control_relative_effect_fraction) * abs(target_effect)
    )
    criteria = {
        "decoded_quantity_ordered": foraging["audit"]["probe_ordering_failures"] == 0,
        "zero_steering_reproduces_baseline": foraging["audit"][
            "alpha_zero_exact_across_directions"
        ]
        and foraging["audit"]["maximum_collection_baseline_difference"]
        <= baseline_tolerance,
        "behavior_monotonic": means["p_negative"] < means["p_zero"] < means["p_positive"],
        "episode_bootstrap_effect_positive": target["episode_bootstrap"]["lower_95"]
        > 0,
        "exceeds_random_directions": random_95 is not None
        and target_effect > random_95,
        "survives_label_reversal": all(
            value > 0
            and target["mapping_episode_bootstrap"][mapping_id]["lower_95"] > 0
            for mapping_id, value in target["mapping_probability_differences"].items()
        ),
        "negative_control_specificity": specificity,
    }
    calibration_valid = (
        criteria["decoded_quantity_ordered"]
        and criteria["zero_steering_reproduces_baseline"]
    )
    classification = (
        "causal_transfer"
        if all(criteria.values())
        else "invalid_or_inconclusive"
        if not calibration_valid
        else "no_convincing_causal_transfer"
    )
    return {
        "classification": classification,
        "criteria": criteria,
        "foraging": foraging,
        "negative_control": negative_control,
        "analysis_roles": {
            "causal_transfer": "confirmatory",
            "negative_control": "confirmatory",
        },
    }


def write_report(result: dict, path: Path) -> None:
    target = result["foraging"]["target"]
    interval = target["episode_bootstrap"]
    lines = [
        "# Cross-task causal transfer",
        "",
        f"Classification: **{result['classification'].replace('_', ' ')}**.",
        f"Decision-matrix outcome: **{result['decision_matrix_outcome'].replace('_', ' ')}**.",
        "",
        f"Bandit-direction +λ minus −λ changed semantic STAY probability by **{target['mean_probability_difference']:.4f}** on average.",
        f"Episode-bootstrap 95% interval: **{interval['lower_95']:.4f} to {interval['upper_95']:.4f}**.",
        f"State-level monotonic fraction: **{target['state_monotonic_fraction']:.3f}**.",
        "",
        "## Required checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in result["criteria"].items()
    )
    if result["negative_control"] is not None:
        lines.extend(
            [
                "",
                "## Non-persistence control",
                "",
                f"The same intervention changed the arbitrary control choice probability by **{result['negative_control']['target']['mean_probability_difference']:.4f}**.",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(result: dict, path: Path) -> None:
    target = result["foraging"]["target"]
    control = result["negative_control"]["target"]
    random_95 = result["foraging"]["random_controls"][
        "probability_difference_95th_percentile"
    ]
    entries = [
        ("Foraging target", target["mean_probability_difference"], COLORS["observed"]),
        ("Random 95th", random_95, COLORS["reference"]),
        ("Binary control", control["mean_probability_difference"], COLORS["both_negative"]),
    ]
    all_values = [value for _name, value, _color in entries]
    all_values.extend(
        [target["episode_bootstrap"]["lower_95"], target["episode_bootstrap"]["upper_95"], 0.0]
    )
    lower, upper = min(all_values) - 0.03, max(all_values) + 0.03
    if lower == upper:
        lower, upper = -0.1, 0.1
    svg = Svg(920, 620)
    svg.text(50, 42, "Track B causal-transfer checkpoint", "title")
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
        (-0.6, 2.6),
        (lower, upper),
    )
    svg.line(box[0], sy(0), box[2], sy(0), COLORS["reference"], 1.2, "5 5")
    for index, (_name, value, color) in enumerate(entries):
        svg.circle(sx(index), sy(value), 7, color)
        svg.text(sx(index), sy(value) - 12, f"{value:.3f}", "axis", "middle")
    interval = target["episode_bootstrap"]
    svg.line(
        sx(0), sy(interval["lower_95"]), sx(0), sy(interval["upper_95"]),
        COLORS["observed"], 2.5,
    )
    svg.text(
        870,
        585,
        f"Classification: {result['classification'].replace('_', ' ')}",
        "note",
        "end",
    )
    svg.save(path)


def _expected_test_state_ids(shards: list[dict], split: dict) -> set[str]:
    test_episodes = set(split["test"])
    return {
        str(record["state_id"])
        for shard in shards
        if shard["episode_id"] in test_episodes
        for record in shard["records"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foraging-input", default="artifacts/cross_task/causal/foraging_replays*.csv"
    )
    parser.add_argument(
        "--control-input", default="artifacts/cross_task/causal/control_replays*.csv"
    )
    parser.add_argument(
        "--foraging-bank", default="artifacts/cross_task/foraging_activation_bank"
    )
    parser.add_argument(
        "--control-bank", default="artifacts/cross_task/control_activation_bank"
    )
    parser.add_argument(
        "--foraging-split", default="artifacts/cross_task/foraging_episode_split.json"
    )
    parser.add_argument(
        "--control-split", default="artifacts/cross_task/control_episode_split.json"
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument(
        "--calibration", default="artifacts/cross_task/causal/calibration.json"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/cross_task/causal/publication"
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    if (
        calibration.get("status") != "valid"
        or calibration.get("selection_data") != "foraging validation states only"
        or calibration.get("test_states_inspected") is not False
    ):
        raise ValueError("causal analysis requires a valid foraging-validation calibration")
    causal = config["causal_transfer"]
    from experiments.cross_task_utils import load_activation_shards, make_or_validate_split

    foraging_shards = load_activation_shards(args.foraging_bank)
    control_shards = load_activation_shards(args.control_bank)
    foraging_split = make_or_validate_split(
        foraging_shards, args.foraging_split, seed=int(config["split_seed"])
    )
    control_split = make_or_validate_split(
        control_shards, args.control_split, seed=int(config["split_seed"])
    )
    foraging_rows = read_rows(args.foraging_input)
    control_rows = read_rows(args.control_input)
    expected_foraging = _expected_test_state_ids(foraging_shards, foraging_split)
    expected_control = _expected_test_state_ids(control_shards, control_split)
    observed_foraging = {row["state_id"] for row in foraging_rows}
    observed_control = {row["state_id"] for row in control_rows}
    if observed_foraging != expected_foraging or observed_control != expected_control:
        raise ValueError(
            "causal replay test-state coverage failed: "
            f"foraging missing/extra={len(expected_foraging - observed_foraging)}/"
            f"{len(observed_foraging - expected_foraging)}, control missing/extra="
            f"{len(expected_control - observed_control)}/"
            f"{len(observed_control - expected_control)}"
        )
    result = analyze(
        foraging_rows,
        control_rows,
        bootstrap_samples=int(
            causal["episode_bootstrap_samples"]
        ),
        seed=int(config["analysis_seed"]) + 20_000,
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
    if set(causal["required_checks"]) != set(result["criteria"]):
        raise ValueError(
            "causal preregistration and implemented criteria differ: "
            f"configured={sorted(causal['required_checks'])}, "
            f"implemented={sorted(result['criteria'])}"
        )
    result["calibration"] = calibration
    result["preregistered_config"] = str(Path(args.config).resolve())
    result["decision_matrix_outcome"] = (
        "outcome_a_task_general_persistence_related_causal_direction"
        if result["classification"] == "causal_transfer"
        else "outcome_b_shared_representation_without_shared_downstream_controller"
        if result["classification"] == "no_convincing_causal_transfer"
        else "causal_checkpoint_invalid_or_inconclusive"
    )
    result["provenance"] = run_metadata(
        {
            "model": foraging_shards[0].get("model_id", config["model"]),
            "analysis": "bandit_to_foraging_causal_transfer",
            "config": str(Path(args.config).resolve()),
        }
    )
    result["test_state_coverage"] = {
        "foraging_expected": len(expected_foraging),
        "foraging_observed": len(observed_foraging),
        "control_expected": len(expected_control),
        "control_observed": len(observed_control),
        "passed": True,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "causal_transfer_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(result, output / "causal_transfer_report.md")
    make_figure(result, output / "causal_transfer_summary.svg")
    print(json.dumps({"classification": result["classification"], "criteria": result["criteria"]}, indent=2))


if __name__ == "__main__":
    main()
