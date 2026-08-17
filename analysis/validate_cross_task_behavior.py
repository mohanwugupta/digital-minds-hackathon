"""Validate Track B integrity and behavior before any held-out transfer test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.cross_task_integrity import (
    audit_cross_task_shards,
    evaluate_behavioral_gate,
    evaluate_solvability_behavioral_gate,
)
from experiments.cross_task_utils import load_activation_shards, make_or_validate_split
from experiments.runtime import run_metadata


def write_report(result: dict, path: Path) -> None:
    behavior = result["behavioral_validation"]
    solvability_behavior = result["solvability_behavioral_validation"]
    foraging = result["foraging_integrity"]
    control = result["control_integrity"]
    terminality = result["terminality_integrity"]
    solvability = result["solvability_integrity"]
    economics = behavior["economic_manipulation_logit_effects"]
    lines = [
        "# Track B behavioral and counterbalancing gate",
        "",
        f"Overall gate passed: **{result['passed']}**.",
        "",
        "This confirmatory gate uses only train and validation episodes; held-out test episodes were not used for behavioral tuning.",
        "",
        "## Integrity and label counterbalancing",
        "",
        f"- Foraging: {foraging['episodes']} episodes / {foraging['pairs']} counterbalanced pairs; passed **{foraging['passed']}**.",
        f"- Binary control: {control['episodes']} episodes / {control['pairs']} counterbalanced pairs; passed **{control['passed']}**.",
        f"- Solvability: {solvability['episodes']} episodes / {solvability['pairs']} counterbalanced pairs; passed **{solvability['passed']}**.",
        f"- Rule terminality control: {terminality['episodes']} episodes / {terminality['pairs']} counterbalanced pairs; passed **{terminality['passed']}**.",
        "- Every pair must use the same ecology/stimulus and exact inverse task-specific label mappings.",
        "",
        "## Development-set behavior",
        "",
        f"- Episodes/states: **{behavior['development_episodes']} / {behavior['development_states']}**.",
        f"- Semantic STAY rate: **{behavior['semantic_stay_choice_rate']:.3f}**.",
        f"- Mean decisions per episode: **{behavior['mean_episode_decisions']:.3f}**.",
        f"- Episodes ending by LEAVE: **{behavior['episodes_ending_by_leave_rate']:.3f}**.",
        f"- Persistence-logit SD: **{behavior['persistence_logit_standard_deviation']:.3f}**.",
        f"- STAY-probability P90−P10: **{behavior['p_stay']['interdecile_range']:.3f}**.",
        f"- Initial mapping gap: **{behavior['initial_mapping_p_stay_gap']:.3f}**.",
        f"- Higher-minus-lower outside-option logit effect: **{economics['higher_outside_option_minus_lower']:.3f}**.",
        f"- Higher-minus-lower stay-cost logit effect: **{economics['higher_stay_cost_minus_lower']:.3f}**.",
        "",
        "## Solvability development-set behavior",
        "",
        f"- Episodes/states: **{solvability_behavior['development_episodes']} / {solvability_behavior['development_states']}**.",
        f"- Semantic TRY-AGAIN rate: **{solvability_behavior['semantic_persistence_choice_rate']:.3f}**.",
        f"- Mean decisions per episode: **{solvability_behavior['mean_episode_decisions']:.3f}**.",
        f"- Persistence-logit SD: **{solvability_behavior['persistence_logit_standard_deviation']:.3f}**.",
        f"- Initial M/N semantic-persistence probability gap: **{solvability_behavior['initial_mapping_probability_gap']:.3f}** "
        f"(diagnostic threshold **{solvability_behavior['initial_mapping_gap_diagnostic']['threshold']:.3f}**; "
        f"passed **{solvability_behavior['initial_mapping_gap_diagnostic']['passed']}**; non-gating).",
        "- This M/N offset remains scientifically important: the held-out test must pass within each mapping and on exact matched semantic histories.",
        "",
        "### Mapping-stratified Solvability behavior",
        "",
    ]
    for mapping_id, metrics in solvability_behavior["mapping_stratified_behavior"][
        "metrics_by_mapping"
    ].items():
        lines.append(
            f"- {mapping_id}: {metrics['episodes']} episodes / {metrics['states']} states; "
            f"TRY-AGAIN rate **{metrics['semantic_persistence_choice_rate']:.3f}**; "
            f"mean decisions **{metrics['mean_episode_decisions']:.3f}**; "
            f"progress/cost/fallback logit effects **{metrics['higher_progress_evidence_minus_lower']:.3f} / "
            f"{metrics['higher_attempt_cost_minus_lower']:.3f} / "
            f"{metrics['higher_give_up_value_minus_lower']:.3f}**."
        )
    lines.extend([
        "",
        "## Development gate checks (v3.1 amendment disclosed)",
        "",
    ])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in behavior["criteria"].items()
    )
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — solvability: {name.replace('_', ' ')}"
        for name, passed in solvability_behavior["criteria"].items()
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foraging-bank", default="artifacts/cross_task/foraging_activation_bank"
    )
    parser.add_argument(
        "--control-bank", default="artifacts/cross_task/control_activation_bank"
    )
    parser.add_argument(
        "--solvability-bank",
        default="artifacts/cross_task/solvability_activation_bank",
    )
    parser.add_argument(
        "--terminality-bank",
        default="artifacts/cross_task/terminality_activation_bank",
    )
    parser.add_argument(
        "--foraging-split", default="artifacts/cross_task/foraging_episode_split.json"
    )
    parser.add_argument(
        "--solvability-split",
        default="artifacts/cross_task/solvability_episode_split.json",
    )
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument("--output-dir", default="artifacts/cross_task/behavioral")
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Smoke-only audit; never produces a behavioral-clearance artifact.",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    collection = config["collection"]
    foraging = load_activation_shards(args.foraging_bank)
    solvability = load_activation_shards(args.solvability_bank)
    control = load_activation_shards(args.control_bank)
    terminality = load_activation_shards(args.terminality_bank)
    foraging_audit = audit_cross_task_shards(
        foraging,
        "foraging",
        response_labels=tuple(collection["foraging_response_labels"]),
        expected_episodes=(
            None
            if args.integrity_only
            else int(config["collection"]["foraging_episodes"])
        ),
    )
    solvability_audit = audit_cross_task_shards(
        solvability,
        "solvability",
        response_labels=tuple(collection["solvability_response_labels"]),
        expected_episodes=(
            None
            if args.integrity_only
            else int(collection["solvability_episodes"])
        ),
    )
    control_audit = audit_cross_task_shards(
        control,
        "control",
        response_labels=tuple(collection["control_response_labels"]),
        expected_episodes=(
            None
            if args.integrity_only
            else int(config["collection"]["control_episodes"])
        ),
    )
    terminality_audit = audit_cross_task_shards(
        terminality,
        "terminality",
        response_labels=tuple(collection["terminality_response_labels"]),
        expected_episodes=(
            None
            if args.integrity_only
            else int(collection["terminality_episodes"])
        ),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.integrity_only:
        result = {
            "passed": foraging_audit["passed"]
            and solvability_audit["passed"]
            and control_audit["passed"]
            and terminality_audit["passed"],
            "analysis_role": "smoke_integrity_only_not_a_behavioral_gate",
            "foraging_integrity": foraging_audit,
            "solvability_integrity": solvability_audit,
            "control_integrity": control_audit,
            "terminality_integrity": terminality_audit,
        }
        (output / "cross_task_integrity_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(2)
        return
    split = make_or_validate_split(
        foraging, args.foraging_split, seed=int(config["split_seed"])
    )
    solvability_split = make_or_validate_split(
        solvability, args.solvability_split, seed=int(config["split_seed"])
    )
    behavior = evaluate_behavioral_gate(
        foraging, split, config["behavioral_validation"]
    )
    solvability_behavior = evaluate_solvability_behavioral_gate(
        solvability,
        solvability_split,
        config["solvability_behavioral_validation"],
    )
    result = {
        "passed": bool(
            foraging_audit["passed"]
            and solvability_audit["passed"]
            and control_audit["passed"]
            and terminality_audit["passed"]
            and behavior["passed"]
            and solvability_behavior["passed"]
        ),
        "analysis_role": "confirmatory_gate",
        "preregistered_config": str(Path(args.config).resolve()),
        "provenance": run_metadata(
            {
                "model": config["model"],
                "analysis": "cross_task_behavioral_validation",
                "config": str(Path(args.config).resolve()),
            }
        ),
        "foraging_integrity": foraging_audit,
        "solvability_integrity": solvability_audit,
        "control_integrity": control_audit,
        "terminality_integrity": terminality_audit,
        "behavioral_validation": behavior,
        "solvability_behavioral_validation": solvability_behavior,
        "protocol_amendments": [
            behavior["protocol_amendment"],
            solvability_behavior["protocol_amendment"],
        ],
        # Duplicated at top level so downstream gates cannot accidentally accept
        # an all-data descriptive summary in place of this development-only gate.
        "test_episodes_inspected": bool(
            behavior["test_episodes_inspected"]
            or solvability_behavior["test_episodes_inspected"]
        ),
    }
    summary_path = output / "behavioral_validation_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(result, output / "behavioral_validation_report.md")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "foraging_integrity": foraging_audit["passed"],
                "control_integrity": control_audit["passed"],
                "solvability_integrity": solvability_audit["passed"],
                "terminality_integrity": terminality_audit["passed"],
                "behavioral_criteria": behavior["criteria"],
                "solvability_behavioral_criteria": solvability_behavior["criteria"],
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
