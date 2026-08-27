"""Generate fixed model-zoo figures and a cautious automatic report."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _configure_plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/digital_minds_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/digital_minds_cache")


def _plot_bar(frame, value, title, output, *, lower_is_better=False):
    _configure_plotting()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(value, ascending=lower_is_better).copy()
    figure, axis = plt.subplots(figsize=(10, max(4, 0.3 * len(ordered))))
    axis.barh(ordered["model"], ordered[value], color="#386cb0")
    axis.set_xlabel(value.replace("_", " "))
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _taskwise_plot(frame, output):
    _configure_plotting()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = frame.pivot_table(index="model", columns="task", values="r_squared")
    axis = pivot.plot(kind="barh", figsize=(10, max(4, 0.3 * len(pivot))))
    axis.set_title("Held-out persistence-logit performance by task")
    axis.set_xlabel("R²")
    axis.grid(axis="x", alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=160)
    plt.close(axis.figure)


def _recovery_plot(matrix_path, output):
    _configure_plotting()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = pd.read_csv(matrix_path, index_col=0)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("Recovered family")
    axis.set_ylabel("Generating family")
    figure.colorbar(image, ax=axis, label="Selection fraction")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _feature_ablation_plot(ablation, bootstrap, output):
    _configure_plotting()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = ablation[ablation.task == "macro"].sort_values("delta_r_squared")
    intervals = bootstrap[
        (bootstrap.task == "macro") & (bootstrap.metric == "delta_r_squared")
    ].set_index("feature_group")
    estimates = frame.delta_r_squared.to_numpy()
    low = np.asarray(
        [intervals.loc[group, "ci_low"] for group in frame.feature_group]
    )
    high = np.asarray(
        [intervals.loc[group, "ci_high"] for group in frame.feature_group]
    )
    error = np.vstack((np.maximum(0, estimates - low), np.maximum(0, high - estimates)))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(frame.feature_group, estimates, xerr=error, color="#386cb0", capsize=3)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Macro delta R² (full minus feature-ablated)")
    axis.set_title("Necessity of observable feature families")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _feature_by_task_plot(ablation, output):
    _configure_plotting()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = ablation[ablation.task != "macro"]
    pivot = frame.pivot(
        index="feature_group", columns="task", values="delta_r_squared"
    )
    order = (
        ablation[ablation.task == "macro"]
        .sort_values("delta_r_squared")
        .feature_group.tolist()
    )
    pivot = pivot.reindex(order)
    axis = pivot.plot(kind="barh", figsize=(10, 6))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Delta R² (full minus feature-ablated)")
    axis.set_title("Feature-family necessity by task")
    axis.grid(axis="x", alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=160)
    plt.close(axis.figure)


def _flexible_comparison_plot(comparison, output):
    _configure_plotting()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = comparison[comparison.information_set == "observable"].sort_values(
        "r_squared"
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    colors = ["#7fc97f" if value else "#386cb0" for value in frame.best_flexible_model]
    axis.barh(frame.model, frame.r_squared, color=colors)
    axis.set_xlabel("Macro held-out persistence-logit R²")
    axis.set_title("Flexible behavioral predictor comparison")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def generate_report(output_dir: str | Path) -> None:
    output = Path(output_dir)
    metrics = pd.read_csv(output / "model_metrics.csv")
    taskwise = pd.read_csv(output / "taskwise_metrics.csv")
    primary = metrics[
        (metrics.information_set == "observable")
        & (metrics.sharing == "shared_architecture_task_observation")
    ].copy()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_bar(primary, "r_squared", "Held-out persistence-logit R²", figures / "persistence_r2.png")
    _plot_bar(
        primary,
        "log_loss",
        "Held-out sampled-choice log loss",
        figures / "choice_log_loss.png",
        lower_is_better=True,
    )
    _taskwise_plot(
        taskwise[
            (taskwise.information_set == "observable")
            & (taskwise.sharing == "shared_architecture_task_observation")
        ],
        figures / "taskwise_performance.png",
    )
    recovery_matrix = output / "model_recovery" / "recovery_matrix.csv"
    if recovery_matrix.exists():
        _recovery_plot(recovery_matrix, figures / "recovery_matrix.png")

    cross_task = pd.read_csv(output / "cross_task_model_ranking.csv")
    task_specific = pd.read_csv(output / "task_specific_model_ranking.csv")
    flexible = pd.read_csv(output / "flexible_model_comparison.csv")
    sanity = pd.read_json(output / "neural_ceiling_sanity.json", typ="series")
    ablation = pd.read_csv(output / "feature_group_ablation.csv")
    group_only = pd.read_csv(output / "feature_group_only.csv")
    feature_bootstrap = pd.read_csv(output / "feature_group_bootstrap.csv")
    _feature_ablation_plot(
        ablation, feature_bootstrap, figures / "feature_group_ablation.png"
    )
    _feature_by_task_plot(ablation, figures / "feature_group_by_task.png")
    _flexible_comparison_plot(
        flexible, figures / "flexible_ceiling_comparison.png"
    )

    best_cross = cross_task[cross_task.best_cross_task_model].iloc[0]
    best_interpretable = cross_task[
        cross_task.best_cross_task_interpretable
    ].iloc[0]
    best_flexible = flexible[flexible.best_flexible_model].iloc[0]
    oracle = flexible[flexible.information_set == "oracle"].iloc[0]
    macro_ablation = ablation[ablation.task == "macro"].sort_values(
        "delta_r_squared", ascending=False
    )
    positive_groups = macro_ablation[macro_ablation.delta_r_squared > 0]
    leading_groups = positive_groups.head(3).feature_group.tolist()
    if not leading_groups:
        leading_groups = macro_ablation.head(3).feature_group.tolist()
    task_best = {
        task: part[part.best_task_specific_model].iloc[0]
        for task, part in task_specific.groupby("task")
    }
    task_support = feature_bootstrap[
        (feature_bootstrap.metric == "delta_r_squared")
        & (feature_bootstrap.task != "macro")
    ]
    supported_tasks = {
        group: sorted(part.loc[part.ci_low > 0, "task"].tolist())
        for group, part in task_support.groupby("feature_group")
    }
    all_tasks = {"bandit", "foraging", "solvability"}
    consistent_groups = sorted(
        group
        for group, tasks in supported_tasks.items()
        if set(tasks) == all_tasks
    )
    task_limited = {
        group: tasks
        for group, tasks in supported_tasks.items()
        if tasks and set(tasks) != all_tasks
    }
    unsupported = sorted(
        group for group, tasks in supported_tasks.items() if not tasks
    )
    oracle_delta = float(oracle.delta_r_squared_oracle_minus_observable)
    oracle_meaningful = oracle_delta >= 0.02
    lines = [
        "# Computational models of cross-task persistence",
        "",
        "This is an exploratory behavioral model comparison. Predictive superiority does not establish neural implementation.",
        "",
        "## Corrected model rankings",
        "",
        f"The best cross-task model was **{best_cross.model}** ({best_cross.sharing}; macro R²={best_cross.r_squared:.3f}). Only candidates evaluated on Bandit, Foraging, and Solvability were eligible.",
        f"The best cross-task interpretable model was **{best_interpretable.model}** (macro R²={best_interpretable.r_squared:.3f}, MSE={best_interpretable.mse:.3f}).",
        "",
        "Task-specific winners:",
        "",
        *[
            f"- **{task.title()}**: {task_best[task].model} (R²={task_best[task].r_squared:.3f}, MSE={task_best[task].mse:.3f})"
            for task in ("bandit", "foraging", "solvability")
        ],
        "",
        "## Flexible behavioral ceiling",
        "",
        f"The synthetic linear-recovery gate passed={bool(sanity['passed'])}. The best flexible predictor was **{best_flexible.model}** (R²={best_flexible.r_squared:.3f}, MSE={best_flexible.mse:.3f}, r={best_flexible.pearson_r:.3f}, sampled-choice log loss={best_flexible.sampled_choice_log_loss:.3f}). GRU is therefore not assumed to be the ceiling.",
        "",
        "## Feature-family ablations",
        "",
        "The largest macro leave-one-family-out contributions were "
        + ", ".join(
            f"**{row.feature_group}** (delta R²={row.delta_r_squared:.3f})"
            for row in macro_ablation.head(3).itertuples()
        )
        + ".",
        "Group-only performance is reported in `feature_group_only.csv`, separating standalone predictiveness from conditional necessity.",
        "Using taskwise 95% pair/episode-clustered intervals, positive contributions were supported in all three tasks for: "
        + (", ".join(consistent_groups) if consistent_groups else "none")
        + ".",
        "Task-limited contributions were: "
        + (
            "; ".join(
                f"{group} ({', '.join(tasks)})"
                for group, tasks in sorted(task_limited.items())
            )
            if task_limited
            else "none"
        )
        + ". No taskwise positive effect was supported for: "
        + (", ".join(unsupported) if unsupported else "none")
        + ".",
        "",
        "## Observable versus oracle state",
        "",
        f"Replacing observable value variables with oracle-state counterparts changed macro R² by {oracle_delta:+.3f}. This is "
        + ("a meaningful positive gain." if oracle_meaningful else "not a meaningful positive gain at the prespecified 0.02 descriptive threshold."),
        "",
        "## Decision and mechanistic hypothesis",
        "",
    ]
    criteria = bool(sanity["passed"] and len(cross_task) and len(ablation))
    ingredients = " + ".join(leading_groups)
    lines.append(f"The behavioral follow-up gates passed={criteria}. The strongest current computational ingredients are **{ingredients}**, which should be used as targets for any later L21/L22 analysis rather than assuming a named recurrent model is the mechanism.")
    lines += [
        "",
        "No L21/L22 activation analysis, steering, or neural-mechanistic claim is performed by this pipeline.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
