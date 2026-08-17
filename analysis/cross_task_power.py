"""Dependency-free prospective power calculations for clustered correlations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def fisher_correlation_power(
    correlation: float, independent_clusters: int, *, alpha: float = 0.05
) -> float:
    """Approximate two-sided correlation-test power using Fisher's z."""
    correlation = float(correlation)
    clusters = int(independent_clusters)
    if not -1 < correlation < 1 or clusters <= 3 or alpha != 0.05:
        raise ValueError("power requires |r|<1, >3 clusters, and frozen alpha=.05")
    noncentrality = abs(math.atanh(correlation)) * math.sqrt(clusters - 3)
    critical = 1.959963984540054
    return 1.0 - _normal_cdf(critical - noncentrality) + _normal_cdf(
        -critical - noncentrality
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandit-metrics", default="artifacts/linear_probes/metrics.json")
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import yaml

    with open(args.bandit_metrics, encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    power_config = config["cross_task_power"]
    reference_layer = int(power_config["bandit_reference_layer"])
    reference = next(
        row["targets"]["persistence"]["test"]["correlation"]
        for row in metrics["layers"]
        if int(row["layer"]) == reference_layer
    )
    pairs = int(config["collection"]["foraging_episodes"]) // 2
    development_train_pairs = int(pairs * 0.70)
    development_validation_pairs = int(pairs * 0.15)
    heldout_test_pairs = pairs - development_train_pairs - development_validation_pairs
    fractions = [float(value) for value in power_config["attenuation_fractions"]]
    rows = [
        {
            "attenuation_fraction_of_bandit_reference": fraction,
            "assumed_heldout_correlation": reference * fraction,
            "independent_pairs": pairs,
            "heldout_test_independent_pairs": heldout_test_pairs,
            "approximate_power_if_all_pairs_were_confirmatory": fisher_correlation_power(
                reference * fraction, pairs
            ),
            "approximate_primary_test_power": fisher_correlation_power(
                reference * fraction, heldout_test_pairs
            ),
        }
        for fraction in fractions
    ]
    desired = float(power_config["desired_power"])
    minimum_pairs = int(power_config["minimum_underlying_pairs"])
    maximum_pairs = int(power_config["maximum_underlying_pairs"])
    result = {
        "passed": minimum_pairs <= pairs <= maximum_pairs,
        "analysis_role": "prospective_sample_size_check_not_an_observed_effect",
        "bandit_reference_layer": reference_layer,
        "bandit_reference_test_correlation": reference,
        "underlying_counterbalanced_pairs": pairs,
        "planned_primary_test_independent_pairs": heldout_test_pairs,
        "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "rows": rows,
        "desired_power": desired,
        "primary_test_power_meets_desired_by_attenuation": {
            str(row["attenuation_fraction_of_bandit_reference"]): (
                row["approximate_primary_test_power"] >= desired
            )
            for row in rows
        },
        "design_pair_preference": [minimum_pairs, maximum_pairs],
        "warning": (
            "Power is governed by held-out test pairs, not all collected pairs; "
            "the conservative attenuation scenarios may remain underpowered."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
