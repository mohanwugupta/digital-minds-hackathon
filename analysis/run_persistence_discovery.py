"""Resumable Track C0/C1 discovery orchestrator over existing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.check_persistence_discovery_baseline import check_discovery_baseline
from analysis.persistence_contrast_bank import build_contrast_bank, save_contrast_bank
from analysis.persistence_cross_generalization import (
    run_contrast_search,
    save_search_outputs,
)
from analysis.persistence_integration import (
    integrate_discovery_results,
    save_integration_report,
)


def _json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _baseline_gate() -> dict:
    result = check_discovery_baseline(
        _json("config/persistence_discovery_regression.json"),
        _json(
            "artifacts/value_dissociation/layerwise_publication_track_a_v1/factorial_layerwise_summary.json"
        ),
        _json(
            "artifacts/cross_task/track_b_shared_v3/shared_transfer/shared_persistence_transfer_summary.json"
        ),
        _json("artifacts/value_probes/episode_split.json"),
    )
    if not result["passed"]:
        raise RuntimeError(f"Track A/B baseline regression failed: {result['failures']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_discovery.yaml")
    parser.add_argument(
        "--phase",
        choices=("baseline", "audit", "contrast", "search", "latent", "integration", "all"),
        default="all",
    )
    parser.add_argument(
        "--allow-missing-generic-value",
        action="store_true",
        help="audit/contrast smoke only; search always requires the value control",
    )
    args = parser.parse_args()
    baseline = _baseline_gate()
    if args.phase == "baseline":
        print(json.dumps(baseline, indent=2, sort_keys=True))
        return
    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    import torch
    paths = config["paths"]
    if args.phase in {"audit", "contrast", "all"}:
        contrasts, audit = build_contrast_bank(
            config,
            allow_missing_generic_value=args.allow_missing_generic_value
            and args.phase in {"audit", "contrast"},
        )
        save_contrast_bank(
            contrasts,
            bank_path=paths["contrast_bank"],
            inventory_path=paths["contrast_inventory"],
            audit_path=paths["contrast_audit"],
            audit=audit,
        )
        if args.phase == "audit":
            print(json.dumps(audit, indent=2, sort_keys=True))
            return
    if args.phase in {"search", "all"}:
        payload = torch.load(paths["contrast_bank"], map_location="cpu", weights_only=False)
        audit = payload["audit"]
        if config["search"]["require_behavioral_gate"] and not audit[
            "all_behavioral_gates_passed"
        ]:
            raise RuntimeError("one or more persistence manipulations failed the behavioral gate")
        if config["search"]["require_generic_value_control"] and not audit[
            "generic_value_control_available"
        ]:
            raise RuntimeError("generic-value specificity control is absent")
        summary, artifacts = run_contrast_search(payload["contrasts"], config)
        save_search_outputs(summary, artifacts, paths["search_output"])
        if args.phase == "search":
            return
    if args.phase in {"latent", "all"}:
        # Direct function invocation avoids a nested subprocess while using
        # the same frozen configs as the standalone cluster phase.
        from analysis.run_persistence_latent_state import (
            run_real_latent_analysis,
            run_synthetic_gates,
        )

        latent_config = yaml.safe_load(
            Path("config/persistence_latent_state.yaml").read_text(encoding="utf-8")
        )
        latent_output = Path("artifacts/persistence_discovery/latent_state")
        latent_output.mkdir(parents=True, exist_ok=True)
        synthetic = run_synthetic_gates(latent_config)
        (latent_output / "synthetic_recovery.json").write_text(
            json.dumps(synthetic, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not synthetic["passed"]:
            raise RuntimeError("synthetic latent-state recovery/confusion gate failed")
        run_real_latent_analysis(
            latent_config=latent_config,
            discovery_config=config,
            output_dir=str(latent_output),
        )
        if args.phase == "latent":
            return
    if args.phase in {"integration", "all"}:
        search_dir = Path(paths["search_output"])
        contrast_summary = _json(search_dir / "persistence_discovery_summary.json")
        contrast_artifacts = torch.load(
            search_dir / "persistence_candidate_subspaces.pt",
            map_location="cpu",
            weights_only=False,
        )
        latent_dir = Path("artifacts/persistence_discovery/latent_state")
        latent_summary = _json(latent_dir / "latent_state_summary.json")
        latent_probe_path = latent_dir / "latent_representation_probes.pt"
        latent_probes = (
            torch.load(latent_probe_path, map_location="cpu", weights_only=False)
            if latent_probe_path.exists()
            else None
        )
        contrast_payload = torch.load(
            paths["contrast_bank"], map_location="cpu", weights_only=False
        )
        latent_by_state = {}
        with (latent_dir / "inferred_latent_states.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                latent_by_state[str(row["state_id"])] = float(row["latent_state"])
        result = integrate_discovery_results(
            contrast_summary=contrast_summary,
            contrast_artifacts=contrast_artifacts,
            latent_summary=latent_summary,
            latent_probe_artifacts=latent_probes,
            contrasts=contrast_payload["contrasts"],
            latent_by_state=latent_by_state,
        )
        save_integration_report(result, "artifacts/persistence_discovery/integration")


if __name__ == "__main__":
    main()
