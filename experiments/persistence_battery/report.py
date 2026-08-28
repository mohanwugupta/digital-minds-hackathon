"""Pilot/full behavioral figures and the ten-question task report."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import TASKS


def _plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/digital_minds_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/digital_minds_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_figures(frames, manipulations, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt = _plotting()

    summary_rows = []
    for task, frame in frames.items():
        summary_rows.append(
            {
                "task": task,
                "mean_positive_probability": frame.p_positive_semantic.mean(),
                "semantic_positive_rate": (
                    frame.semantic_action == TASKS[task].positive_action
                ).mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("task")
    axis = summary.plot(kind="bar", figsize=(11, 5))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability / sampled action rate")
    axis.set_title("Behavior across the persistence battery")
    axis.tick_params(axis="x", rotation=30)
    axis.figure.tight_layout()
    axis.figure.savefig(output / "task_behavior_summary.png", dpi=160)
    plt.close(axis.figure)

    valid = manipulations[np.isfinite(manipulations.effect)].copy()
    valid["label"] = valid.task + "\n" + valid.check
    axis = valid.set_index("label").effect.plot(
        kind="barh", figsize=(11, max(5, 0.38 * len(valid)))
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Signed manipulation effect (positive is predicted direction)")
    axis.set_title("Validity gates and non-gating scientific predictions")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "manipulation_effects.png", dpi=160)
    plt.close(axis.figure)

    lengths = []
    for task, frame in frames.items():
        for _episode, length in frame.groupby("episode_id").size().items():
            lengths.append({"task": task, "states": int(length)})
    length_frame = pd.DataFrame(lengths)
    grouped = [
        length_frame[length_frame.task == task].states.to_numpy()
        for task in sorted(length_frame.task.unique())
    ]
    labels = sorted(length_frame.task.unique())
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(grouped, tick_labels=labels, showfliers=False)
    axis.set_ylabel("Decision states per episode")
    axis.set_title("Episode-length distributions")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output / "episode_length_distributions.png", dpi=160)
    plt.close(figure)


def generate_report(
    task_specs,
    frames,
    manipulations,
    label_bias,
    nondegeneracy,
    config,
    output,
    *,
    mode,
    smoke,
    model_free,
):
    output = Path(output)
    lines = [
        "# Literature-grounded persistence task battery",
        "",
        f"Collection stage: **{mode}**. "
        + (
            "This model-free smoke validates plumbing only and cannot approve tasks."
            if model_free
            else "This is a behavior-only run; no hidden activations were collected."
        ),
        "",
    ]
    if smoke:
        lines += [
            "The episode budget was reduced for smoke validation; behavioral estimates are not scientific results.",
            "",
        ]
    for task in frames:
        spec = task_specs[task]
        frame = frames[task]
        checks = manipulations[manipulations.task == task]
        validity = checks[checks.gate_role == "validity_gate"]
        bias = label_bias[label_bias.task == task].iloc[0]
        gate = nondegeneracy[nondegeneracy.task == task].iloc[0]
        manipulated = ", ".join(spec["manipulated_variables"])
        behavior = (
            f"{TASKS[task].positive_action} probability mean={frame.p_positive_semantic.mean():.3f}; "
            f"sampled rate={(frame.semantic_action == TASKS[task].positive_action).mean():.3f}."
        )
        strongest = validity.sort_values("effect", ascending=False).iloc[0]
        adjustments = config.get("pilot_adjustments", {}).get(task, [])
        lines += [
            f"## {task.replace('_', ' ').title()}",
            "",
            f"1. **Construct:** {spec['construct']}; adapted from {spec['source_paradigm']} ({spec['source_citation']}).",
            f"2. **Manipulated variables:** {manipulated}.",
            f"3. **Persistence behavior:** {behavior}",
            f"4. **Nondegenerate choices:** {'yes' if gate.probability_non_degenerate else 'no'} (positive-probability SD={gate.positive_probability_sd:.3f}; within-episode choice-logit SD={gate.within_episode_choice_logit_sd:.3f}; valid top-token rate={gate.top_token_action_rate:.3f}).",
            f"5. **Basic incentive manipulation:** {strongest['check']}; signed effect={strongest['effect']:.3f}; {'passed' if strongest['passed'] else 'did not pass'}.",
            f"6. **Label mappings balanced:** {'yes' if bias.passed else 'no'} (paired correlation={bias.mapping_probability_correlation:.3f}, mean absolute gap={bias.mean_absolute_mapping_gap:.3f}).",
            f"7. **Median episode length:** {gate.median_episode_length:.1f} decision states.",
            f"8. **Sufficient history depth:** {'yes' if gate.sufficient_history_depth else 'no'}.",
            f"9. **Approved for full collection:** {'yes' if gate.approved_for_full_collection else 'no'} — {gate.approval_note}.",
            f"10. **Parameters changed after pilot:** {', '.join(adjustments) if adjustments else 'none recorded'}.",
            "",
        ]
    lines += [
        "## Interpretation guardrail",
        "",
        "Sunk-cost sensitivity, partial-reinforcement extinction, controllability transfer, goal gradients, and human-like recency are scientific hypotheses, not validity gates. Their absence must be reported rather than designed away.",
        "",
        "All counterbalanced pairs share environmental seeds, semantic actions, outcomes, and histories. The independent-effort control is sequential but marks `same_goal_across_steps=false`; persistence probabilities are left null rather than fabricated.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
