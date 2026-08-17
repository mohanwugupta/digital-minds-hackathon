"""Analyze causal transfer of the bandit persistence direction across tasks."""

import argparse
import csv
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


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
    for row in rows:
        grouped[(row["state_id"], row["control_type"], row["control_id"])].append(row)
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
        grouped[row["episode_id"]].append(row["probability_difference"])
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
        "lower_95": ordered[int(0.025 * samples)],
        "upper_95": ordered[max(0, int(0.975 * samples) - 1)],
        "bootstrap_mean": statistics.mean(draws),
    }


def summarize_task(rows: list[dict], *, bootstrap_samples: int, seed: int) -> dict:
    audit = audit_rows(rows)
    if audit["incomplete_triplets"] or audit["context_hash_failures"]:
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
            "episode_bootstrap": _episode_bootstrap(
                target, bootstrap_samples, seed
            ),
        },
        "random_controls": {
            "count": len(random_means),
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
) -> dict:
    foraging = summarize_task(
        foraging_rows, bootstrap_samples=bootstrap_samples, seed=seed
    )
    target = foraging["target"]
    random_95 = foraging["random_controls"]["probability_difference_95th_percentile"]
    means = target["means_by_alpha"]
    criteria = {
        "decoded_quantity_ordered": foraging["audit"]["probe_ordering_failures"] == 0,
        "behavior_monotonic": means["p_negative"] < means["p_zero"] < means["p_positive"],
        "exceeds_random_directions": random_95 is not None
        and target["mean_probability_difference"] > random_95,
        "survives_label_reversal": all(
            value > 0 for value in target["mapping_probability_differences"].values()
        ),
    }
    calibration_valid = criteria["decoded_quantity_ordered"]
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
        "negative_control": (
            summarize_task(
                control_rows,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1,
            )
            if control_rows
            else None
        ),
    }


def write_report(result: dict, path: Path) -> None:
    target = result["foraging"]["target"]
    lines = [
        "# Cross-task causal transfer",
        "",
        f"Classification: **{result['classification'].replace('_', ' ')}**.",
        "",
        f"Bandit-direction +λ minus −λ changed semantic STAY probability by **{target['mean_probability_difference']:.4f}** on average.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foraging-input", default="artifacts/cross_task/causal/foraging_replays*.csv"
    )
    parser.add_argument(
        "--control-input", default="artifacts/cross_task/causal/control_replays*.csv"
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument(
        "--output-dir", default="artifacts/cross_task/causal/publication"
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    control_paths = glob.glob(args.control_input)
    result = analyze(
        read_rows(args.foraging_input),
        read_rows(args.control_input) if control_paths else None,
        bootstrap_samples=int(
            config["causal_transfer"]["episode_bootstrap_samples"]
        ),
        seed=int(config["analysis_seed"]) + 20_000,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "causal_transfer_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(result, output / "causal_transfer_report.md")
    print(json.dumps({"classification": result["classification"], "criteria": result["criteria"]}, indent=2))


if __name__ == "__main__":
    main()
