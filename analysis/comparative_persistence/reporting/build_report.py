"""Required figures and direct-answer comparative-persistence report."""

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


def _save_bar(frame, index, value, output, title, ylabel, *, horizontal=False):
    plt = _plotting()
    local = frame.dropna(subset=[value]).copy()
    axis = local.set_index(index)[value].plot(
        kind="barh" if horizontal else "bar", figsize=(11, max(5, 0.3 * len(local)))
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel if not horizontal else "")
    axis.set_xlabel(ylabel if horizontal else "")
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=160)
    plt.close(axis.figure)


def generate_figures(output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    model = pd.read_csv(output / "model_comparison/macro_average.csv")
    model["label"] = model.model + "\n" + model.sharing
    _save_bar(model.head(25), "label", "macro_log_loss", figures / "model_zoo.png", "Within-task hazard model comparison", "Log loss")

    loto = pd.read_csv(output / "generalization/loto_summary.csv")
    loto["label"] = loto.model + "\n" + loto.sharing
    _save_bar(loto, "label", "macro_delta_log_loss", figures / "loto_performance.png", "Zero-shot held-out-task improvement", "Δ log loss vs source null")

    architecture = pd.read_csv(output / "generalization/architecture_transfer.csv")
    architecture["label"] = architecture.task + "\n" + architecture.model
    _save_bar(architecture, "label", "architecture_transfer", figures / "architecture_transfer.png", "Architecture-transfer fraction", "G", horizontal=True)

    few = pd.read_csv(output / "generalization/few_shot_curves.csv")
    plt = _plotting()
    figure, axis = plt.subplots(figsize=(10, 6))
    for (model_name, sharing), part in few.groupby(["model", "sharing"]):
        curve = part.groupby("adaptation_pairs").macro_log_loss.mean()
        axis.plot(curve.index, curve.values, marker="o", label=f"{model_name}/{sharing}")
    axis.set(xlabel="Target-task adaptation pairs", ylabel="Macro log loss", title="Few-shot adaptation")
    axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(figures / "few_shot_adaptation.png", dpi=160); plt.close(figure)

    kernels = pd.read_csv(output / "history/finite_kernels.csv")
    selected = kernels[kernels.validation_selected.astype(bool)]
    _save_bar(selected, "task", "log_loss", figures / "history_kernels.png", "Validation-selected finite history", "Test log loss")

    ablation = pd.read_csv(output / "features/family_ablation.csv")
    _save_bar(ablation, "feature_family", "delta_log_loss", figures / "feature_ablation.png", "Shared feature-family ablation", "Δ log loss", horizontal=True)

    control = pd.read_csv(output / "history/persistence_vs_control.csv")
    _save_bar(control, "task", "history_log_loss_gain", figures / "persistence_vs_control.png", "History gain: persistence vs sequential control", "Log-loss improvement")

    recovery = pd.read_csv(output / "synthetic/recovery.csv")
    recovered = recovery.assign(correct=lambda x: ((x.generating_model == "H1_latent_commitment") & (x.selected_model == "latent_commitment")) | ((x.generating_model == "H2_shared_rule") & (x.selected_model == "shared_rule")) | ((x.generating_model == "H3_task_specific_evaluation") & (x.selected_model == "task_specific")) | ((x.generating_model == "H4_generic_sequential_choice") & (x.selected_model == "finite_history"))).groupby("generating_model", as_index=False).correct.mean()
    _save_bar(recovered, "generating_model", "correct", figures / "model_recovery.png", "Synthetic model recovery", "Recovery rate")


def generate_report(output, inclusion, *, smoke):
    output = Path(output)
    macro = pd.read_csv(output / "model_comparison/macro_average.csv")
    loto = pd.read_csv(output / "generalization/loto_summary.csv")
    architecture = pd.read_csv(output / "generalization/architecture_transfer.csv")
    few = pd.read_csv(output / "generalization/few_shot_curves.csv")
    ablation = pd.read_csv(output / "features/family_ablation.csv")
    control = pd.read_csv(output / "history/persistence_vs_control.csv")
    kernel_similarity = pd.read_csv(output / "history/task_kernel_similarity.csv")
    signatures = pd.read_csv(output / "human_animal_signatures/signature_effects.csv")
    recovery = pd.read_csv(output / "synthetic/recovery.csv")
    finite = macro[(macro.model == "finite_history")]
    latent = macro[(macro.model == "latent_commitment")]
    mlp = macro[macro.model == "mlp"]
    gru = macro[macro.model == "gru"]
    interpretable = macro[(~macro.model.isin(["flexible_linear", "mlp", "gru"])) & (macro.information_set == "observable")]
    best = interpretable.iloc[0]
    best_loto = loto.iloc[0]
    finite_loss = float(finite.macro_log_loss.min()) if not finite.empty else float("nan")
    latent_loss = float(latent.macro_log_loss.min()) if not latent.empty else float("nan")
    mlp_loss = float(mlp.macro_log_loss.min()) if not mlp.empty else float("nan")
    gru_loss = float(gru.macro_log_loss.min()) if not gru.empty else float("nan")
    architecture_summary = (
        architecture.replace([np.inf, -np.inf], np.nan)
        .groupby(["model", "sharing"], as_index=False)
        .architecture_transfer.mean()
        .dropna(subset=["architecture_transfer"])
        .sort_values("architecture_transfer", ascending=False)
    )
    best_architecture = architecture_summary.iloc[0]
    mean_transfer = float(best_architecture.architecture_transfer)
    psh = float(control.psh.iloc[0])
    psh_low = float(control.psh_ci_low.iloc[0]) if "psh_ci_low" in control else float("nan")
    psh_high = float(control.psh_ci_high.iloc[0]) if "psh_ci_high" in control else float("nan")
    history_ablation = ablation[ablation.feature_family == "history"]
    history_delta = float(history_ablation.delta_log_loss.iloc[0]) if not history_ablation.empty else float("nan")
    if best_loto.macro_delta_log_loss > 0.01 and psh > 0.01 and mean_transfer > 0.4:
        hypothesis = "H2 — shared stay/switch computation"
    elif best_loto.macro_delta_log_loss > 0 and history_delta > 0:
        hypothesis = "H3 — shared ingredients with task-specific computation"
    elif abs(psh) <= 0.01:
        hypothesis = "H4 — generic sequential choice remains viable"
    else:
        hypothesis = "No single H1–H4 account is yet decisive"
    candidate = "finite recent-history integration" if history_delta > 0 and best_loto.macro_delta_log_loss > 0 else "none; PRD 3 gate not met"
    included_tasks = inclusion[inclusion.included.astype(bool)].task.tolist()
    excluded = inclusion[~inclusion.included.astype(bool)][["task", "reason"]]
    few_summary = (
        few.groupby(["model", "sharing", "adaptation_pairs"], as_index=False)
        .macro_log_loss.mean()
        .sort_values("macro_log_loss")
    )
    few_best = few_summary.iloc[0]
    best_feature = ablation.sort_values("delta_log_loss", ascending=False).iloc[0]
    reproduced = signatures[signatures.direction_reproduced == True].signature.tolist()  # noqa: E712
    expected_recovery = {
        "H1_latent_commitment": "latent_commitment",
        "H2_shared_rule": "shared_rule",
        "H3_task_specific_evaluation": "task_specific",
        "H4_generic_sequential_choice": "finite_history",
    }
    recovery["correct"] = recovery.apply(
        lambda row: row.selected_model
        == expected_recovery.get(row.generating_model, "__unknown__"),
        axis=1,
    )
    recovery_rates = recovery.groupby("generating_model").correct.mean()
    recovery_passed = bool(
        set(expected_recovery) <= set(recovery_rates.index)
        and (recovery_rates.loc[list(expected_recovery)] >= 0.8).all()
    )
    neural_diagnostic = (
        f"GRU={gru_loss:.4f}, nonrecurrent MLP={mlp_loss:.4f}"
        if np.isfinite(gru_loss) and np.isfinite(mlp_loss)
        else "GRU/MLP comparison was not included in this reduced run"
    )
    loto_interval = (
        f" (95% task bootstrap [{best_loto.delta_log_loss_vs_null_ci_low:.4f}, "
        f"{best_loto.delta_log_loss_vs_null_ci_high:.4f}])"
        if {
            "delta_log_loss_vs_null_ci_low",
            "delta_log_loss_vs_null_ci_high",
        } <= set(loto.columns)
        else ""
    )
    psh_interval = (
        f" (95% task bootstrap [{psh_low:.4f}, {psh_high:.4f}])"
        if np.isfinite(psh_low) and np.isfinite(psh_high)
        else ""
    )
    control_task = "independent_effort_control"
    control_pairs = kernel_similarity[
        (kernel_similarity.task_a == control_task)
        | (kernel_similarity.task_b == control_task)
    ]
    persistence_pairs = kernel_similarity[
        (kernel_similarity.task_a != control_task)
        & (kernel_similarity.task_b != control_task)
    ]
    control_cosine = float(control_pairs.cosine_similarity.mean())
    persistence_cosine = float(persistence_pairs.cosine_similarity.mean())
    kernel_comparison = (
        f"; mean kernel cosine is {control_cosine:.3f} for persistence–control "
        f"versus {persistence_cosine:.3f} among persistence tasks"
        if np.isfinite(control_cosine) and np.isfinite(persistence_cosine)
        else ""
    )
    if not recovery_passed:
        hypothesis = "Undetermined — mandatory synthetic recovery did not pass"
        candidate = "none; synthetic identifiability gate not met"
    lines = [
        "# Comparative computational models of LLM persistence",
        "",
        "This is a reduced plumbing run, not a scientific result." if smoke else "Primary target: absorbing discrete-time termination hazard with task-macro evaluation.",
        "",
        "## Direct answers",
        "",
        f"1. **Best interpretable model:** `{best.model}` under `{best.sharing}` sharing (macro log loss {best.macro_log_loss:.4f}).",
        f"2. **Latent state versus finite history:** best latent={latent_loss:.4f}; best finite-history={finite_loss:.4f}; {neural_diagnostic}. A selected rho near zero is reported as collapse, not motivation.",
        f"3. **Zero-shot task transfer:** `{best_loto.model}/{best_loto.sharing}` improves macro log loss by {best_loto.macro_delta_log_loss:.4f} over the source-trained null{loto_interval}.",
        f"4. **Architecture transfer:** best mean G={mean_transfer:.3f} for `{best_architecture.model}/{best_architecture.sharing}`; values are relative to separately fitted target-task ceilings.",
        f"5. **Few-shot adaptation:** the best task-macro point uses {int(few_best.adaptation_pairs)} target pairs for `{few_best.model}/{few_best.sharing}` with log loss {few_best.macro_log_loss:.4f}.",
        f"6. **Most general ingredient:** `{best_feature.feature_family}` has the largest ablation cost ({best_feature.delta_log_loss:.4f}).",
        f"7. **Persistence-specific history:** PSH={psh:.4f}{psh_interval}{kernel_comparison}; positive PSH means more history gain than independent repeated choice.",
        f"8. **Human/animal signatures reproduced directionally:** {', '.join(reproduced) if reproduced else 'none among evaluable signatures'}.",
        f"9. **Current hypothesis:** {hypothesis}.",
        f"10. **PRD 3 candidate:** {candidate}.",
        "",
        "## Frozen task inventory",
        "",
        f"Included: {', '.join(included_tasks)}.",
        "",
    ]
    if not excluded.empty:
        lines += ["Excluded before fitting:", ""] + [
            f"- `{row.task}` — {row.reason}." for row in excluded.itertuples()
        ] + [""]
    lines += [
        "## Guardrails",
        "",
        "All normalization uses frozen environment specifications. Missing constructs retain explicit availability masks. LOTO/LOFO target records never enter fitting, normalization, hyperparameter selection, or model selection. Sunk cost, PREE, waiting-context, effort-breakpoint, and controllability effects are scientific outputs rather than task-validity criteria.",
        "",
        "Task-level generalization uncertainty is necessarily substantial with this small number of task identities.",
        "",
        "Synthetic H1–H4 recovery passed the 80% identifiability gate."
        if recovery_passed
        else "Synthetic H1–H4 recovery did not pass the 80% identifiability gate; empirical rankings are descriptive only.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
