"""Figures and direct-answer report for PRD 2.5."""

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


def _bar(frame, index, value, output, title, ylabel, *, horizontal=False):
    plt = _plotting()
    local = frame.dropna(subset=[value]).copy()
    axis = local.set_index(index)[value].plot(
        kind="barh" if horizontal else "bar",
        figsize=(11, max(5, 0.32 * len(local))),
    )
    axis.set_title(title)
    axis.set_xlabel(ylabel if horizontal else "")
    axis.set_ylabel("" if horizontal else ylabel)
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=160)
    plt.close(axis.figure)


def generate_figures(output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(output / "tasks/task_summary.csv")
    persistence = summary[summary.is_persistence_task.astype(bool)]
    _bar(
        persistence,
        "task",
        "episodes",
        figures / "task_battery.png",
        "Expanded persistence battery",
        "Independent recorded episodes",
    )

    hyper = pd.read_csv(output / "gru/hyperparameter_results.csv")
    selected = hyper[hyper.hyperparameter_selected_on_validation.astype(bool)]
    ceiling_plot = (
        selected.groupby(["input_variant", "hidden_size"], as_index=False)
        .validation_macro_log_loss.mean()
    )
    plt = _plotting()
    figure, axis = plt.subplots(figsize=(10, 6))
    for variant, part in ceiling_plot.groupby("input_variant"):
        part = part.sort_values("hidden_size")
        axis.plot(
            part.hidden_size,
            part.validation_macro_log_loss,
            marker="o",
            label=variant,
        )
    axis.set(
        xlabel="GRU hidden size",
        ylabel="Validation task-macro log loss",
        title="Large-GRU capacity ceiling",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "gru_ceiling.png", dpi=160)
    plt.close(figure)

    curves = pd.read_csv(output / "gru/training_curves.csv")
    ceiling = pd.read_csv(output / "gru/ceiling_comparison.csv")
    best = ceiling[ceiling.model == "gru"].iloc[0]
    chosen = curves[
        (curves.input_variant == best.input_variant)
        & (curves.hidden_size == best.hidden_size)
        & (curves.layers == best.layers)
        & np.isclose(curves.learning_rate, best.learning_rate)
        & np.isclose(curves.weight_decay, best.weight_decay)
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    for seed, part in chosen.groupby("seed"):
        axis.plot(
            part.epoch,
            part.validation_macro_log_loss,
            label=f"validation seed {seed}",
        )
    axis.set(
        xlabel="Epoch",
        ylabel="Task-macro log loss",
        title="Selected GRU training stability",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "gru_training.png", dpi=160)
    plt.close(figure)

    gain = pd.read_csv(output / "matched_control/history_gain.csv")
    gain = gain[gain.model == "joint_history"].copy()
    gain["label"] = gain.version + "\n" + gain.framing
    _bar(
        gain,
        "label",
        "history_gain",
        figures / "history_gain_control.png",
        "History gain under matched goal continuity",
        "Current-state minus joint-history log loss",
    )

    similarity = pd.read_csv(output / "models/history_kernel_similarity.csv")
    kernel_summary = (
        similarity.groupby("kernel_type", as_index=False).cosine_similarity.mean()
    )
    matched = pd.read_csv(output / "matched_control/history_kernels.csv")
    matched_summary = pd.DataFrame(
        {
            "kernel_type": ["matched_action", "matched_outcome"],
            "cosine_similarity": [
                matched.action_kernel_cosine.mean(),
                matched.outcome_kernel_cosine.mean(),
            ],
        }
    )
    kernel_summary = pd.concat((kernel_summary, matched_summary), ignore_index=True)
    _bar(
        kernel_summary,
        "kernel_type",
        "cosine_similarity",
        figures / "history_kernel_similarity.png",
        "Persistence-task versus matched-control kernel similarity",
        "Mean cosine similarity",
    )

    loto = pd.read_csv(output / "models/loto.csv")
    loto_summary = (
        loto.groupby(["model", "sharing"], as_index=False)
        .delta_log_loss_vs_null.mean()
        .sort_values("delta_log_loss_vs_null", ascending=False)
        .head(20)
    )
    loto_summary["label"] = loto_summary.model + "\n" + loto_summary.sharing
    _bar(
        loto_summary,
        "label",
        "delta_log_loss_vs_null",
        figures / "loto_expanded_battery.png",
        "Expanded-battery leave-one-task-out transfer",
        "Improvement over source null",
    )

    signatures = pd.read_csv(output / "signatures/human_animal_signatures.csv")
    available = signatures[signatures.available.astype(bool)].copy()
    _bar(
        available,
        "signature",
        "effect",
        figures / "human_animal_signatures.png",
        "Non-gating comparative-cognition signatures",
        "Prespecified effect",
        horizontal=True,
    )


def _truth(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def generate_report(output, config, *, smoke):
    output = Path(output)
    inclusion = pd.read_csv(output / "tasks/inclusion.csv")
    summary = pd.read_csv(output / "tasks/task_summary.csv")
    validation = pd.read_csv(output / "tasks/validation.csv")
    ceiling = pd.read_csv(output / "gru/ceiling_comparison.csv")
    macro = pd.read_csv(output / "models/task_macro.csv").sort_values("macro_log_loss")
    sharing = pd.read_csv(output / "models/sharing_comparison.csv")
    loto_summary = pd.read_csv(output / "models/loto_summary.csv").sort_values(
        "macro_log_loss"
    )
    kernels = pd.read_csv(output / "models/history_kernels.csv")
    similarity = pd.read_csv(output / "models/history_kernel_similarity.csv")
    matched_gain = pd.read_csv(output / "matched_control/history_gain.csv")
    matched_kernels = pd.read_csv(output / "matched_control/history_kernels.csv")
    signatures = pd.read_csv(output / "signatures/human_animal_signatures.csv")
    recovery = pd.read_csv(output / "synthetic/recovery_summary.csv")

    included = inclusion[inclusion.included.astype(bool)].task.tolist()
    exercised_persistence_tasks = summary[
        summary.is_persistence_task.astype(bool)
    ].task.tolist()
    scientifically_usable = []
    for row in summary[summary.is_persistence_task.astype(bool)].itertuples():
        inclusion_row = inclusion[inclusion.task == row.task].iloc[0]
        if not _truth(inclusion_row.included):
            continue
        if "smoke plumbing override" in str(inclusion_row.reason):
            continue
        if str(inclusion_row.source) == "prd2_5_extension":
            minimum_met = getattr(row, "extension_minimum_sample_met", False)
            if pd.isna(minimum_met) or not _truth(minimum_met):
                continue
        scientifically_usable.append(row.task)
    persistence_tasks = scientifically_usable
    usable_count = len(scientifically_usable)
    repair_tasks = config["task_breadth"]["repaired_tasks"]
    repair_rows = inclusion[inclusion.task.isin(repair_tasks)]
    repair_text = "; ".join(
        f"{row.task}={'passed' if _truth(row.included) and 'passed' in row.reason else 'not yet passed'}"
        for row in repair_rows.itertuples()
    )

    gru = ceiling[ceiling.model == "gru"].iloc[0]
    mlp = ceiling[ceiling.model == "mlp"].iloc[0]
    finite_ceiling = ceiling[ceiling.model == "finite_history"].iloc[0]
    interpretable = macro[
        ~macro.model.isin(["flexible_linear", "mlp", "large_gru"])
    ]
    best_interpretable = interpretable.iloc[0]
    finite_sharing = sharing[sharing.model == "finite_history"].iloc[0]
    finite_loto = loto_summary[loto_summary.model == "finite_history"].iloc[0]
    loto_low = float(finite_loto.delta_log_loss_vs_null_ci_low)

    persistence_action_cosine = float(
        similarity[similarity.kernel_type == "action"].cosine_similarity.mean()
    )
    persistence_outcome_cosine = float(
        similarity[similarity.kernel_type == "outcome"].cosine_similarity.mean()
    )
    primary_gain = matched_gain[
        (matched_gain.version == "absorbing_primary")
        & (matched_gain.model == "joint_history")
    ]
    secondary_gain = matched_gain[
        (matched_gain.version == "advancing_secondary")
        & (matched_gain.model == "joint_history")
    ]
    matched_delta = float(primary_gain.delta_history_gain.iloc[0])
    matched_low = float(primary_gain.delta_history_gain_ci_low.iloc[0])
    matched_high = float(primary_gain.delta_history_gain_ci_high.iloc[0])
    matched_action_cosine = float(matched_kernels.action_kernel_cosine.mean())
    matched_outcome_cosine = float(matched_kernels.outcome_kernel_cosine.mean())

    decomposition = pd.read_csv(output / "models/history_decomposition.csv")
    mean_gains = decomposition.groupby("model").history_gain.mean()
    action_gain = float(mean_gains.get("action_only", np.nan))
    outcome_gain = float(mean_gains.get("outcome_only", np.nan))
    joint_gain = float(mean_gains.get("joint_history", np.nan))
    if action_gain > 0 and outcome_gain > 0:
        history_source = "both action perseveration and outcome history"
    elif action_gain > outcome_gain and action_gain > 0:
        history_source = "primarily action perseveration"
    elif outcome_gain > 0:
        history_source = "primarily outcome history"
    else:
        history_source = "neither component reliably improved prediction"

    available_signatures = signatures[signatures.available.astype(bool)].signature.tolist()
    recovery_passed = bool((recovery.recovery_rate >= 0.8).all())
    task_breadth_passed = usable_count >= int(config["task_breadth"]["minimum_persistence_tasks"])
    gru_credible = _truth(gru.credible_recurrent_ceiling)
    finite_loto_positive = loto_low > 0
    matched_specific = matched_low > 0

    if matched_specific and finite_loto_positive and finite_sharing.best_sharing == "hierarchical":
        hypothesis = "H2/H3: shared history architecture with task-varying evaluation and maintenance-specific amplification"
    elif matched_low <= 0 <= matched_high and matched_action_cosine > 0.7:
        hypothesis = "H4: generic sequential history sensitivity remains the leading account"
    elif finite_loto_positive:
        hypothesis = "H3: shared ingredients with task-specific computation"
    else:
        hypothesis = "task-specific behavior; no broad H1–H4 account is established"

    target_earned = bool(
        task_breadth_passed
        and gru_credible
        and recovery_passed
        and finite_loto_positive
    )
    target = (
        "recent action/outcome integration during ongoing policy maintenance"
        if target_earned and matched_specific
        else "generic recent-history integration compared across persistence and independent decisions"
        if target_earned
        else "none; the behavioral robustness gate is not yet met"
    )
    if smoke:
        hypothesis = "not evaluated — this is a model-free/reduced plumbing run"
        target = "none — smoke output is non-scientific"

    lines = [
        "# PRD 2.5 — persistence robustness report",
        "",
        "This is a reduced plumbing run and not a scientific result."
        if smoke
        else "This exploratory report uses held-out behavioral data; it is not a preregistration.",
        "",
        "## Direct answers",
        "",
        f"1. **Usable persistence tasks:** {usable_count} ({', '.join(persistence_tasks) if persistence_tasks else 'none'})."
        + (
            f" The smoke run exercised {len(exercised_persistence_tasks)} candidate persistence tasks for plumbing only."
            if smoke
            else ""
        ),
        f"2. **Repaired-task validity:** {repair_text}.",
        f"3. **Best GRU:** `{gru.input_variant}`, hidden={int(gru.hidden_size)}, layers={int(gru.layers)}, lr={gru.learning_rate:g}, weight decay={gru.weight_decay:g}; test macro log loss={gru.macro_log_loss:.4f}.",
        f"4. **Capacity plateau:** {'yes' if _truth(gru.capacity_plateau) else 'no'} under the predefined {config['gru']['plateau_epsilon']:.3f} validation tolerance.",
        f"5. **GRU versus MLP:** GRU={gru.macro_log_loss:.4f}, MLP={mlp.macro_log_loss:.4f}; {'competitive' if _truth(gru.competitive_with_mlp) else 'not competitive—treat as optimization/model mismatch'} within epsilon={config['gru']['ceiling_epsilon']:.3f}.",
        f"6. **GRU versus finite five-step history:** improvement={finite_ceiling.macro_log_loss - gru.macro_log_loss:.4f} log-loss units.",
        f"7. **Best interpretable model:** `{best_interpretable.model}/{best_interpretable.sharing}` at macro log loss {best_interpretable.macro_log_loss:.4f}.",
        f"8. **Finite-history sharing:** `{finite_sharing.best_sharing}` is best; hierarchical minus task-specific={finite_sharing.get('hierarchical_minus_task_specific', np.nan):.4f}.",
        f"9. **Finite-history LOTO:** mean improvement over source null={finite_loto.macro_delta_log_loss:.4f}, 95% task bootstrap [{finite_loto.delta_log_loss_vs_null_ci_low:.4f}, {finite_loto.delta_log_loss_vs_null_ci_high:.4f}].",
        f"10. **Persistence-kernel similarity:** mean action cosine={persistence_action_cosine:.3f}; mean outcome cosine={persistence_outcome_cosine:.3f} across task pairs.",
        f"11. **Matched independent-goal control:** primary ΔHG={matched_delta:.4f}, 95% latent-sequence bootstrap [{matched_low:.4f}, {matched_high:.4f}]; action cosine={matched_action_cosine:.3f}, outcome cosine={matched_outcome_cosine:.3f}. Secondary matched ΔHG={secondary_gain.delta_history_gain.iloc[0]:.4f}.",
        f"12. **History decomposition:** {history_source} (mean action gain={action_gain:.4f}, outcome gain={outcome_gain:.4f}, joint gain={joint_gain:.4f}).",
        f"13. **Human/animal signatures available:** {', '.join(available_signatures) if available_signatures else 'none'}; these are outputs, never gates.",
        f"14. **Best-supported account:** {hypothesis}.",
        f"15. **Mechanistic target:** {target}.",
        "",
        "## Guardrails and status",
        "",
        f"Included modeling tasks: {', '.join(included)}.",
        "An extension task is scientifically usable only after pilot approval and at least 256 independent semantic histories; smoke overrides never count toward task breadth.",
        f"Task-breadth gate (≥{config['task_breadth']['minimum_persistence_tasks']}): {'passed' if task_breadth_passed else 'not met'}.",
        f"Synthetic recovery gate (all ≥80%): {'passed' if recovery_passed else 'not met'}.",
        f"Credible recurrent-ceiling gate: {'passed' if gru_credible else 'not met'}.",
        "Task-macro averaging is primary. GRU minibatches contain equal episode counts from every task. Hyperparameters and optional depth expansion use validation data only. The matched control uses identical exogenous action/outcome histories and models policy probabilities at those states; it is not inserted into the absorbing-hazard risk set.",
    ]
    if not validation.empty:
        failed = validation[~validation.approved_for_full_collection.astype(bool)].task.tolist()
        lines.extend(
            [
                "",
                f"Extension tasks not approved in the recorded pilot: {', '.join(failed) if failed else 'none'}.",
            ]
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
