"""Paired matched-state persistence analysis and random-neuron comparison."""

import argparse
import json
import os


def analyze_frame(frame) -> dict:
    from scipy import stats

    value = frame[frame["intervention_type"] == "value"]
    pivot = value.pivot(index="state_id", columns="alpha", values="persistence_logit").dropna()
    required = {-1.0, 0.0, 1.0}
    if not required.issubset(set(pivot.columns)):
        raise ValueError("value intervention is missing one or more alpha conditions")
    positive_delta = pivot[1.0] - pivot[0.0]
    negative_delta = pivot[0.0] - pivot[-1.0]
    total_effect = pivot[1.0] - pivot[-1.0]
    primary_order = bool(
        pivot[1.0].mean() > pivot[0.0].mean() > pivot[-1.0].mean()
        and positive_delta.mean() > 0 and negative_delta.mean() > 0
    )

    random_effects = []
    random_frame = frame[frame["intervention_type"] == "random"]
    for neuron_set, group in random_frame.groupby("neuron_set"):
        random_pivot = group.pivot(index="state_id", columns="alpha", values="persistence_logit").dropna()
        if {-1.0, 1.0}.issubset(set(random_pivot.columns)):
            random_effects.append(float((random_pivot[1.0] - random_pivot[-1.0]).mean()))
    value_effect = float(total_effect.mean())
    # Directional empirical p: how often a random set is at least as effective.
    empirical_p = (
        (1 + sum(effect >= value_effect for effect in random_effects)) / (1 + len(random_effects))
        if random_effects else None
    )
    return {
        "n_matched_states": int(len(pivot)),
        "mean_persistence_logit": {
            "negative": float(pivot[-1.0].mean()),
            "control": float(pivot[0.0].mean()),
            "positive": float(pivot[1.0].mean()),
        },
        "mean_positive_vs_control": float(positive_delta.mean()),
        "mean_control_vs_negative": float(negative_delta.mean()),
        "mean_positive_vs_negative": value_effect,
        "paired_t_positive_vs_control": float(stats.ttest_rel(pivot[1.0], pivot[0.0]).statistic),
        "paired_t_control_vs_negative": float(stats.ttest_rel(pivot[0.0], pivot[-1.0]).statistic),
        "primary_ordering_passed": primary_order,
        "random_set_count": len(random_effects),
        "random_effects": random_effects,
        "random_control_empirical_p": empirical_p,
        "random_control_passed": bool(
            len(random_effects) >= 20 and empirical_p is not None and empirical_p <= 0.05
        ),
    }


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/matched_intervention.csv")
    parser.add_argument("--output", default="artifacts/matched_analysis.json")
    args = parser.parse_args()
    result = analyze_frame(pd.read_csv(args.input))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
