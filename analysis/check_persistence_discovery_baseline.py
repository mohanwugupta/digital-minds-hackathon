"""Protect frozen Track A/B outputs before Track C discovery code runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math


def _hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_discovery_baseline(manifest, track_a, track_b, bandit_split) -> dict:
    failures = []
    tolerance = float(manifest["absolute_tolerance"])
    expected_a = manifest["track_a"]
    for key in ("episodes", "states", "rows", "complete_states"):
        if int(track_a["audit"][key]) != int(expected_a[key]):
            failures.append(f"Track A factorial {key} changed")
    final = track_a["layers"][-1]
    comparisons = {
        "layer_31_stop_raw_slope": final["stop"]["raw_slope"],
        "layer_31_continue_raw_slope": final["continue"]["raw_slope"],
        "layer_31_relative_incentive_r_squared": final[
            "relative_incentive_r_squared"
        ],
    }
    for key, observed in comparisons.items():
        if not math.isclose(
            float(observed), float(expected_a[key]), abs_tol=tolerance, rel_tol=0
        ):
            failures.append(f"Track A {key} changed")
    trajectory = [
        {
            "layer": row["layer"],
            "stop": row["stop"]["raw_slope"],
            "continue": row["continue"]["raw_slope"],
            "r2": row["relative_incentive_r_squared"],
        }
        for row in track_a["layers"]
    ]
    if _hash(trajectory) != expected_a["trajectory_sha256"]:
        failures.append("Track A non-monotonic layer trajectory changed")
    growth_checks = {
        "stop": (
            expected_a["stop_largest_growth_from"],
            expected_a["stop_largest_growth_to"],
        ),
        "relative_incentive": (
            expected_a["relative_largest_growth_from"],
            expected_a["relative_largest_growth_to"],
        ),
    }
    for effect, expected in growth_checks.items():
        observed = track_a["detection_summary"][effect]["largest_absolute_growth_step"]
        if (int(observed["from_layer"]), int(observed["to_layer"])) != expected:
            failures.append(f"Track A {effect} transformation region changed")

    expected_b = manifest["track_b"]
    primary = track_b["primary_heldout_result"]
    if track_b["classification"] != expected_b["classification"]:
        failures.append("Track B classification changed")
    if set(track_b["primary_discovery_tasks"]) != set(
        expected_b["primary_discovery_tasks"]
    ) or track_b["primary_heldout_task"] != expected_b["primary_heldout_task"]:
        failures.append("Track B primary fold changed")
    track_b_values = {
        "selected_layer": primary["selected_layer"],
        "strict_zero_shot_correlation": primary["strict_zero_shot"]["correlation"],
        "arbitrary_choice_correlation": primary["non_persistence_binary_control"][
            "correlation"
        ],
        "terminality_correlation": primary["rule_determined_terminality_control"][
            "correlation"
        ],
    }
    if int(track_b_values.pop("selected_layer")) != int(expected_b["selected_layer"]):
        failures.append("Track B selected layer changed")
    for key, observed in track_b_values.items():
        if not math.isclose(
            float(observed), float(expected_b[key]), abs_tol=tolerance, rel_tol=0
        ):
            failures.append(f"Track B {key} changed")

    expected_split = manifest["bandit_split"]
    if {key: len(value) for key, value in bandit_split.items()} != expected_split[
        "counts"
    ]:
        failures.append("Bandit episode split dimensions changed")
    if _hash(bandit_split) != expected_split["canonical_sha256"]:
        failures.append("Bandit episode split membership changed")
    return {"passed": not failures, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="config/persistence_discovery_regression.json"
    )
    parser.add_argument(
        "--track-a",
        default="artifacts/value_dissociation/layerwise_publication_track_a_v1/factorial_layerwise_summary.json",
    )
    parser.add_argument(
        "--track-b",
        default="artifacts/cross_task/track_b_shared_v3/shared_transfer/shared_persistence_transfer_summary.json",
    )
    parser.add_argument(
        "--bandit-split", default="artifacts/value_probes/episode_split.json"
    )
    args = parser.parse_args()
    loaded = []
    for path in (args.manifest, args.track_a, args.track_b, args.bandit_split):
        with open(path, encoding="utf-8") as handle:
            loaded.append(json.load(handle))
    result = check_discovery_baseline(*loaded)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

