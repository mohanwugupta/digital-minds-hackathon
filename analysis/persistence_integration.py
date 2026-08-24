"""Integrate contrast-derived and latent-state-derived exploratory candidates."""

from __future__ import annotations

import json
from pathlib import Path


def integrate_discovery_results(
    *, contrast_summary: dict, contrast_artifacts: dict, latent_summary: dict,
    latent_probe_artifacts: dict | None,
    contrasts: list[dict] | None = None,
    latent_by_state: dict[str, float] | None = None,
) -> dict:
    candidates = contrast_summary.get("layerwise_candidates", [])
    if not candidates:
        raise ValueError("integration requires contrast candidates")
    contrast_candidate = max(
        candidates,
        key=lambda row: (
            bool(row["decision"]["causal_gate_passed"]),
            row["cross_task_transfer"]["captured_energy_fraction"],
            row["cross_manipulation_transfer"]["captured_energy_fraction"],
        ),
    )
    overlap = []
    latent_internal = latent_summary.get("internal_representation", {})
    if (
        latent_probe_artifacts is not None
        and contrast_candidate["feature_type"] == "static"
    ):
        import torch

        contrast = contrast_artifacts["wide_candidates"][contrast_candidate["key"]]
        basis = contrast["basis"].float()
        for heldout, probe in latent_probe_artifacts.get("loto", {}).items():
            fold = next(
                row
                for row in latent_internal["leave_one_task_out"]
                if row["heldout_task"] == heldout
            )
            if int(fold["selected_layer"]) != int(contrast_candidate["layer"]):
                overlap.append(
                    {
                        "heldout_task": heldout,
                        "comparable": False,
                        "reason": "selected layers differ",
                    }
                )
                continue
            direction = probe.raw_activation_direction().float()
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
            overlap.append(
                {
                    "heldout_task": heldout,
                    "comparable": True,
                    "contrast_subspace_capture_of_latent_direction": float(
                        (basis.T @ direction).square().sum()
                    ),
                }
            )
    manipulation_to_latent = {}
    cross_decoding = None
    if contrasts is not None and latent_by_state is not None:
        paired = []
        for row in contrasts:
            if row.get("contrast_kind") != "persistence":
                continue
            positive = latent_by_state.get(str(row["positive_state_id"]))
            negative = latent_by_state.get(str(row["negative_state_id"]))
            if positive is None or negative is None:
                continue
            latent_delta = float(positive) - float(negative)
            manipulation_to_latent.setdefault(str(row["manipulation"]), []).append(
                latent_delta
            )
            paired.append((row, latent_delta))
        manipulation_to_latent = {
            key: {
                "matched_contrasts": len(values),
                "mean_latent_shift": sum(values) / len(values),
                "positive_shift_fraction": sum(value > 0 for value in values)
                / len(values),
            }
            for key, values in manipulation_to_latent.items()
        }
        if paired and contrast_candidate["feature_type"] == "static":
            import torch

            candidate = contrast_artifacts["wide_candidates"][contrast_candidate["key"]]
            direction = candidate["orientation_vector"].float()
            projection = torch.tensor(
                [
                    float(
                        torch.dot(
                            row["activation_delta"][contrast_candidate["layer"]].float(),
                            direction,
                        )
                    )
                    for row, _latent in paired
                ]
            )
            latent_delta = torch.tensor([value for _row, value in paired])
            correlation = 0.0
            if float(projection.std(unbiased=False)) > 0 and float(
                latent_delta.std(unbiased=False)
            ) > 0:
                correlation = float(
                    torch.corrcoef(torch.stack((projection, latent_delta)))[0, 1]
                )
            cross_decoding = {
                "matched_contrasts": len(paired),
                "contrast_projection_to_latent_shift_correlation": correlation,
            }

    contrast_gate = bool(contrast_summary.get("causal_gate_passed"))
    latent_behavior_gate = bool(latent_summary.get("behavioral_gate_passed"))
    latent_representation_gate = bool(
        latent_internal.get("all_loto_clustered_intervals_positive", False)
    )
    causal_gate = contrast_gate and latent_behavior_gate and latent_representation_gate
    return {
        "analysis_role": "exploratory_integration",
        "contrast_candidate": {
            "key": contrast_candidate["key"],
            "feature_type": contrast_candidate["feature_type"],
            "layer": contrast_candidate["layer"],
            "rank": contrast_candidate["rank"],
            "decision": contrast_candidate["decision"],
        },
        "latent_candidate_taxonomy": latent_summary.get("candidate_taxonomy"),
        "subspace_overlap": overlap,
        "persistence_manipulation_shift_of_latent_state": manipulation_to_latent,
        "contrast_to_latent_cross_decoding": cross_decoding,
        "gates": {
            "contrast_specificity": contrast_gate,
            "latent_future_behavior": latent_behavior_gate,
            "latent_cross_task_representation": latent_representation_gate,
        },
        "causal_gate_passed": causal_gate,
        "decision": (
            "eligible_for_targeted_existing_task_intervention"
            if causal_gate
            else "stop_causal_pipeline"
        ),
        "task4_status": (
            "freeze_candidate_then_design_untouched_task4"
            if causal_gate
            else "do_not_design_task4_yet"
        ),
    }


def save_integration_report(result: dict, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "persistence_integration_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Persistence discovery integration",
        "",
        f"Causal gate passed: **{result['causal_gate_passed']}**.",
        f"Decision: **{result['decision']}**.",
        f"Task 4 status: **{result['task4_status']}**.",
        "",
        "## Component gates",
        "",
    ]
    report.extend(f"- {key}: **{value}**." for key, value in result["gates"].items())
    (output / "persistence_integration_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
