"""Figures and decision report for the persistence stay/switch pivot."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/digital_minds_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/digital_minds_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(axis, path):
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=160)
    axis.figure.clf()


def generate_figures(output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt = _plotting()

    hazard = pd.read_csv(output / "behavior/hazard_model_comparison.csv")
    axis = hazard.set_index("model").test_log_loss.plot(kind="bar", figsize=(8, 4))
    axis.set_ylabel("Held-out log loss (lower is better)")
    axis.set_title("Stay/switch hazard architectures")
    axis.tick_params(axis="x", rotation=25)
    _finish(axis, figures / "hazard_model_comparison.png")
    plt.close(axis.figure)

    history = pd.read_csv(output / "behavior/history_kernel_results.csv")
    labels = history.kernel + " " + history.parameter.astype(str)
    axis = history.assign(label=labels).set_index("label").test_log_loss.plot(
        kind="bar", figsize=(9, 4), color=np.where(history.validation_selected, "#d95f02", "#7570b3")
    )
    axis.set_ylabel("Held-out log loss")
    axis.set_title("Shared-history kernel comparison (orange: validation selected)")
    axis.tick_params(axis="x", rotation=35)
    _finish(axis, figures / "history_kernel.png")
    plt.close(axis.figure)

    memory = pd.read_csv(output / "gru/memory_ablation.csv")
    axis = memory.set_index("ablation").r_squared.plot(kind="bar", figsize=(10, 4))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Held-out persistence-logit R²")
    axis.set_title("GRU recurrence and information ablations")
    axis.tick_params(axis="x", rotation=35)
    _finish(axis, figures / "gru_memory_ablation.png")
    plt.close(axis.figure)

    bottleneck = pd.read_csv(output / "gru/bottleneck_performance.csv")
    axis = bottleneck.plot(x="hidden_size", y="r_squared", marker="o", legend=False, figsize=(7, 4))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Held-out persistence-logit R²")
    axis.set_title("Recurrent bottleneck capacity")
    _finish(axis, figures / "gru_bottleneck.png")
    plt.close(axis.figure)

    direction = pd.read_csv(output / "neural/layerwise_direction_similarity.csv")
    mean_direction = direction[direction.metric == "mean_pairwise_cosine"]
    axis = mean_direction.plot(x="layer", y="value", marker="o", legend=False, figsize=(8, 4))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Mean task-readout cosine")
    axis.set_title("Independent task readouts across depth")
    _finish(axis, figures / "layerwise_convergence.png")
    plt.close(axis.figure)

    profiles = pd.read_csv(output / "interventions/intervention_profiles.csv")
    test_profiles = profiles[profiles.split.astype(str) == "test"]
    if test_profiles.empty:
        test_profiles = profiles
    pivot = test_profiles.pivot_table(
        index="layer", columns=["task", "manipulation"], values="mean_functional_effect"
    )
    axis = pivot.plot(figsize=(10, 5), legend=False)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Effect on task's own persistence readout")
    axis.set_title("Functional profiles of existing persistence manipulations")
    _finish(axis, figures / "intervention_convergence.png")
    plt.close(axis.figure)

    controls = pd.read_csv(output / "controls/persistence_vs_generic_decision.csv")
    grouped = controls.groupby(["layer", "control"], as_index=False)[
        ["persistence_test_r_squared", "control_test_r_squared"]
    ].mean()
    axis = grouped.pivot(index="layer", columns="control", values="control_test_r_squared").plot(figsize=(9, 4))
    persistence = grouped.groupby("layer").persistence_test_r_squared.mean()
    axis.plot(persistence.index, persistence.values, color="black", linewidth=2.5, label="persistence")
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_ylabel("Held-out target-logit R²")
    axis.set_title("Persistence versus one-shot decision controls")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _finish(axis, figures / "persistence_vs_controls.png")
    plt.close(axis.figure)


def _row(frame, name):
    rows = frame[frame.model.astype(str) == str(name)]
    return rows.iloc[0] if len(rows) else None


def _fmt(value):
    return "NA" if not np.isfinite(float(value)) else f"{float(value):.3f}"


def generate_report(output, *, smoke=False):
    output = Path(output)
    hazard = pd.read_csv(output / "behavior/hazard_model_comparison.csv")
    history = pd.read_csv(output / "behavior/history_kernel_results.csv")
    memory = pd.read_csv(output / "gru/memory_ablation.csv")
    bottleneck = pd.read_csv(output / "gru/bottleneck_performance.csv")
    direction = pd.read_csv(output / "neural/layerwise_direction_similarity.csv")
    profiles = pd.read_csv(output / "interventions/profile_similarity.csv")
    controls = pd.read_csv(output / "controls/persistence_vs_generic_decision.csv")

    task_specific = _row(hazard, "task_specific")
    shared_history = _row(hazard, "shared_history")
    shared_rule = _row(hazard, "shared_stay_switch")
    fully_shared = _row(hazard, "fully_shared")
    baseline = _row(hazard, "baseline")
    selected_history = history[history.validation_selected.astype(bool)].iloc[0]
    full_gru = memory[memory.ablation == "full_recurrence"].iloc[0]
    mlp = memory[memory.ablation == "no_recurrence_mlp"].iloc[0]
    finite = memory[memory.ablation.str.startswith("limited_history_")]
    best_finite = finite.sort_values("r_squared", ascending=False).iloc[0]
    best_bottleneck = float(bottleneck.r_squared.max())
    sufficient = bottleneck[bottleneck.r_squared >= 0.95 * best_bottleneck]
    sufficient_size = int(sufficient.hidden_size.min()) if len(sufficient) else int(bottleneck.loc[bottleneck.r_squared.idxmax(), "hidden_size"])
    cosine = direction[direction.metric == "mean_pairwise_cosine"].sort_values("layer")
    span = max(1, min(8, len(cosine) // 2))
    early_cosine = float(cosine.head(span).value.mean())
    late_cosine = float(cosine.tail(span).value.mean())
    profile_mean = profiles[profiles.kind == "mean_profile_correlation"].value
    profile_mean = float(profile_mean.iloc[0]) if len(profile_mean) else float("nan")
    control_delta = controls.groupby("control").delta_r_squared_persistence_minus_control.mean()

    history_gain = float(baseline.test_log_loss - selected_history.test_log_loss)
    recurrence_gain = float(full_gru.r_squared - mlp.r_squared)
    finite_gap = float(full_gru.r_squared - best_finite.r_squared)
    shared_history_gap = float(shared_history.test_log_loss - task_specific.test_log_loss)
    rule_gap = float(shared_rule.test_log_loss - task_specific.test_log_loss)
    generic_deltas_positive = bool(len(control_delta) and (control_delta > 0).all())

    if smoke:
        outcome = "Smoke validation only — no scientific outcome assigned"
        conclusion = "All analysis phases completed against real persisted artifacts; run the full protocol before interpreting estimates."
    elif shared_history.performance_fraction >= 0.9 and recurrence_gain > 0 and late_cosine > early_cosine and profile_mean > 0.5 and generic_deltas_positive:
        outcome = "Outcome A — shared history-dependent stay/switch computation"
        conclusion = "The results support task-specific evidence feeding a shared history-sensitive policy-maintenance computation."
    elif shared_history.performance_fraction >= 0.8 and shared_history_gap < rule_gap:
        outcome = "Outcome B — shared history with task-specific current-state computation"
        conclusion = "History generalizes better than a fully shared evidence-to-decision mapping; current evidence remains task specific."
    elif not generic_deltas_positive:
        outcome = "Outcome C — common generic decision machinery"
        conclusion = "The control readouts are at least as strong as persistence, so the observed structure is not persistence-specific."
    else:
        outcome = "Outcome D — fully task-specific algorithms"
        conclusion = "The shared models and convergence tests do not approach the task-specific ceiling."

    lines = [
        "# Persistence as a shared history-dependent stay/switch computation",
        "",
    ]
    if smoke:
        lines.extend(
            [
                "This is a reduced smoke run for pipeline validation; estimates are not scientific results.",
                "",
            ]
        )
    lines += [
        "## Direct answers",
        "",
        f"1. **Is recent history shared?** The validation-selected {selected_history.kernel} kernel ({selected_history.parameter}) changes held-out log loss relative to the intercept-only hazard by {_fmt(-history_gain)} (negative is better). Shared-history ceiling fraction={_fmt(shared_history.performance_fraction)}.",
        f"2. **Cost of forcing evidence mappings to be shared:** fully shared adds {_fmt(fully_shared.test_log_loss - task_specific.test_log_loss)} log loss versus the task-specific model; the rank-one shared stay/switch rule adds {_fmt(rule_gap)}.",
        f"3. **Shared history plus task-specific evidence:** its held-out log-loss gap from the task-specific ceiling is {_fmt(shared_history_gap)}, retaining {_fmt(shared_history.performance_fraction)} of ceiling improvement.",
        f"4. **Does recurrence matter?** Full GRU minus non-recurrent MLP R²={_fmt(recurrence_gain)}; full GRU minus best finite window ({best_finite.ablation}) R²={_fmt(finite_gap)}.",
        f"5. **Sufficient recurrent dimensions:** the smallest tested bottleneck within 95% of the best R² is {sufficient_size} (best R²={_fmt(best_bottleneck)}).",
        f"6. **Neural convergence with depth:** mean independent-readout cosine changes from {_fmt(early_cosine)} in early layers to {_fmt(late_cosine)} in late layers.",
        f"7. **Functional intervention convergence:** mean cross-profile correlation={_fmt(profile_mean)}; every primary effect was measured with that task's own independently fitted readout.",
        "8. **Specificity versus controls:** mean persistence-minus-control R² is " + ", ".join(f"{name}={_fmt(value)}" for name, value in control_delta.items()) + ". One-shot controls were not assigned history or recurrence.",
        "",
        "## Decision",
        "",
        f"**{outcome}.** {conclusion}",
        "",
        "## Causal gate",
        "",
        "These analyses reuse existing trajectories and activations and do not establish causal mediation. Head localization, DAS, broad patching, or new steering should wait for reproducible behavioral sharing, a common functional stage, convergent manipulation profiles, and differentiation from generic-decision controls.",
        "",
        "## Reuse and leakage safeguards",
        "",
        "No Qwen trajectories were generated. Risk sets stop at the first termination event; episode and counterbalanced-pair splits remain intact; normalization and readout fitting use training data, ridge selection uses validation data, and test targets are evaluation-only. Activation memmaps are local caches excluded from Git.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
