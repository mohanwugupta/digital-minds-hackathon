"""Check stored sprint results against a compact frozen regression manifest."""

import argparse
import hashlib
import json
import math


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def check_baseline(
    manifest: dict,
    probe_metrics: dict,
    split: dict,
    factorial: dict,
    causal: dict,
) -> dict:
    failures = []
    expected_probe = manifest["linear_probe_metrics"]
    if probe_metrics["best_layers"] != expected_probe["best_layers"]:
        failures.append("validation-selected probe layers changed")
    tolerance = float(manifest["absolute_tolerance"])
    for target, expected in expected_probe["selected_test_r_squared"].items():
        layer = int(probe_metrics["best_layers"][target])
        observed = float(probe_metrics["layers"][layer]["targets"][target]["test"]["r_squared"])
        if not math.isclose(observed, float(expected), abs_tol=tolerance, rel_tol=0):
            failures.append(f"held-out {target} R2 changed")

    expected_split = manifest["episode_split"]
    if {name: len(values) for name, values in split.items()} != expected_split["counts"]:
        failures.append("episode split dimensions changed")
    if canonical_hash(split) != expected_split["canonical_sha256"]:
        failures.append("exact episode split membership changed")

    expected_factorial = manifest["factorial"]
    audit = factorial["audit"]
    for key in ("episodes", "states", "rows", "complete_states"):
        if int(audit[key]) != int(expected_factorial[key]):
            failures.append(f"factorial {key} changed")
    coefficients = factorial["state_fixed_effects"]["persistence_logit"][
        "stop_and_continue"
    ]["coefficients"]
    observed_signs = {
        "stop_raw_slope_sign": -1 if coefficients["stop_payoff"]["raw_slope"] < 0 else 1,
        "continue_raw_slope_sign": -1
        if coefficients["continue_bonus"]["raw_slope"] < 0
        else 1,
    }
    for key, observed in observed_signs.items():
        if observed != expected_factorial[key]:
            failures.append(f"factorial {key} changed")

    expected_causal = manifest["causal"]
    if (
        causal["audit"]["alpha_zero_exact_across_conditions"]
        != expected_causal["alpha_zero_exact_across_conditions"]
    ):
        failures.append("alpha-zero baseline replay changed")
    if causal["positive_control_passed"] != expected_causal["positive_control_passed"]:
        failures.append("persistence steering positive-control status changed")
    return {"passed": not failures, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="config/baseline_regression.json")
    parser.add_argument("--probe-metrics", default="artifacts/linear_probes/metrics.json")
    parser.add_argument("--split", default="artifacts/value_probes/episode_split.json")
    parser.add_argument(
        "--factorial-summary",
        default="artifacts/value_dissociation/publication/value_dissociation_summary.json",
    )
    parser.add_argument(
        "--causal-summary",
        default="artifacts/causal_steering/publication/causal_steering_summary.json",
    )
    args = parser.parse_args()
    paths = (
        args.manifest,
        args.probe_metrics,
        args.split,
        args.factorial_summary,
        args.causal_summary,
    )
    loaded = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            loaded.append(json.load(handle))
    result = check_baseline(*loaded)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
