"""Run PRD 2.5 task breadth, matched control, and recurrent ceiling analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from analysis.comparative_persistence.semantic_features import (
    FORBIDDEN_FUTURE_FIELDS,
)
from analysis.persistence_robustness.build_dataset import build_robustness_dataset
from analysis.persistence_robustness.gru_ceiling import run_gru_ceiling
from analysis.persistence_robustness.history_decomposition import (
    run_history_decomposition,
)
from analysis.persistence_robustness.matched_control import (
    load_matched_records,
    run_matched_control_analysis,
)
from analysis.persistence_robustness.model_comparison import (
    run_reduced_model_comparison,
)
from analysis.persistence_robustness.reporting import generate_figures, generate_report
from analysis.persistence_robustness.signatures import run_robustness_signatures
from analysis.persistence_robustness.synthetic_recovery import (
    run_robustness_recovery,
)
from analysis.run_comparative_persistence import _read_dataset, _serialize_dataset, _smoke_subset
from computational_modeling.analysis.run_model_zoo import ProgressLogger
from experiments.runtime import save_run_metadata


def _load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _prepare(config, output, *, smoke, logger):
    with logger.section("prepare_dataset"):
        records, inclusion, validation, summary, model_config, stage = (
            build_robustness_dataset(config, output, smoke=smoke)
        )
        if smoke:
            records = _smoke_subset(
                records, config["smoke"]["maximum_episodes_per_task"]
            )
        _serialize_dataset(records, output / "modeling_dataset.csv.gz")
        _write(inclusion, output / "tasks/inclusion.csv")
        _write(validation, output / "tasks/validation.csv")
        local_summary = []
        for task, part in records.groupby("task"):
            original = summary[summary.task == task].iloc[0]
            local_summary.append(
                {
                    **original.to_dict(),
                    "states": len(part),
                    "episodes": int(part.episode_id.nunique()),
                    "semantic_pairs": int(part.pair_id.nunique()),
                }
            )
        summary = pd.DataFrame(local_summary).sort_values("task")
        _write(summary, output / "tasks/task_summary.csv")
        (output / "effective_model_config.yaml").write_text(
            yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "states": len(records),
            "episodes": int(records.episode_id.nunique()),
            "persistence_tasks": sorted(
                records[records.is_persistence_task.astype(bool)].task.unique()
            ),
            "control_tasks": sorted(
                records[~records.is_persistence_task.astype(bool)].task.unique()
            ),
            "extension_dataset_stage": stage,
            "task_macro_primary": True,
            "future_fields_forbidden": sorted(FORBIDDEN_FUTURE_FIELDS),
            "dataset_sha256": hashlib.sha256(
                (output / "modeling_dataset.csv.gz").read_bytes()
            ).hexdigest(),
        }
        (output / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.note(
            "prepare_dataset",
            f"wrote {len(records)} states across {len(manifest['persistence_tasks'])} persistence tasks; extension_stage={stage}",
        )
    return records, inclusion, model_config


def _model_config(config, output, *, smoke):
    model_config = _load(output / "effective_model_config.yaml")
    if smoke:
        model_config["ridge_penalties"] = config["smoke"]["ridge_penalties"]
        model_config["evaluation"]["few_shot_counts"] = config["smoke"][
            "few_shot_counts"
        ]
        model_config["evaluation"]["few_shot_models"] = config["smoke"][
            "few_shot_models"
        ]
        model_config["mlp"] = {
            **model_config["mlp"],
            "hidden_sizes": config["smoke"]["mlp_hidden_sizes"],
            "max_epochs": config["smoke"]["mlp_maximum_epochs"],
            "patience": config["smoke"]["mlp_patience"],
        }
    return model_config


def _read_or_prepare(config, output, *, smoke, logger, force=False):
    dataset = output / "modeling_dataset.csv.gz"
    if force or not dataset.exists():
        return _prepare(config, output, smoke=smoke, logger=logger)
    return (
        _read_dataset(dataset),
        pd.read_csv(output / "tasks/inclusion.csv"),
        _model_config(config, output, smoke=smoke),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_robustness_v1.yaml")
    parser.add_argument("--run-id", default="robustness_v1")
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "gru",
            "matched_control",
            "models",
            "signatures",
            "synthetic",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--models", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load(config_path)
    output = Path(config["output_root"]) / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"output exists: {output}; collection and analysis share a run ID, so pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "config.yaml").exists():
        shutil.copy2(config_path, output / "config.yaml")
    logger = ProgressLogger(output, label="persistence-robustness")
    selected_models = (
        [name.strip() for name in args.models.split(",") if name.strip()]
        if args.models
        else list(config["smoke"]["models"] if args.smoke else config["primary_models"])
    )
    save_run_metadata(
        str(output / "run_metadata.json"),
        {
            **vars(args),
            "protocol_version": config["protocol_version"],
            "models": selected_models,
            "behavior_only": True,
            "mechanistic_analysis": False,
            "task_macro_primary": True,
            "balanced_gru_minibatches": True,
        },
    )
    logger.note(
        "pipeline",
        f"run={args.run_id}; phase={args.phase}; smoke={args.smoke}; models={','.join(selected_models)}",
    )
    records, inclusion, model_config = _read_or_prepare(
        config,
        output,
        smoke=args.smoke,
        logger=logger,
        force=args.phase in {"prepare", "all"},
    )
    model_config = _model_config(config, output, smoke=args.smoke)
    if args.phase == "prepare":
        return

    if args.phase in {"gru", "all"}:
        with logger.section("gru_ceiling"):
            hyper, stability, ceiling, curves, taskwise = run_gru_ceiling(
                records,
                config,
                model_config,
                smoke=args.smoke,
                logger=logger,
            )
            _write(hyper, output / "gru/hyperparameter_results.csv")
            _write(stability, output / "gru/seed_stability.csv")
            _write(ceiling, output / "gru/ceiling_comparison.csv")
            _write(curves, output / "gru/training_curves.csv")
            _write(taskwise, output / "gru/within_task.csv")
        if args.phase == "gru":
            return

    if args.phase in {"matched_control", "all"}:
        with logger.section("matched_control"):
            matched = load_matched_records(output)
            losses, gains, kernels, predictions = run_matched_control_analysis(
                matched, config, smoke=args.smoke, logger=logger
            )
            _write(losses, output / "matched_control/model_losses.csv")
            _write(gains, output / "matched_control/history_gain.csv")
            _write(kernels, output / "matched_control/history_kernels.csv")
            _write(
                gains.merge(
                    losses,
                    on=["version", "framing", "model"],
                    how="left",
                    suffixes=("", "_fit"),
                ),
                output / "matched_control/action_vs_outcome_history.csv",
            )
            _write(predictions, output / "matched_control/heldout_predictions.csv.gz")
        if args.phase == "matched_control":
            return

    if args.phase in {"models", "all"}:
        with logger.section("model_comparison"):
            ceiling_path = output / "gru/ceiling_comparison.csv"
            taskwise_path = output / "gru/within_task.csv"
            ceiling = pd.read_csv(ceiling_path) if ceiling_path.exists() else None
            gru_taskwise = (
                pd.read_csv(taskwise_path) if taskwise_path.exists() else None
            )
            comparison = run_reduced_model_comparison(
                records,
                config,
                model_config,
                models=selected_models,
                gru_ceiling=ceiling,
                gru_taskwise=gru_taskwise,
                smoke=args.smoke,
                logger=logger,
            )
            for name, frame in comparison.items():
                _write(frame, output / f"models/{name}.csv")
            decomposition, kernels, similarity = run_history_decomposition(
                records, config, smoke=args.smoke, logger=logger
            )
            _write(decomposition, output / "models/history_decomposition.csv")
            _write(kernels, output / "models/history_kernels.csv")
            _write(similarity, output / "models/history_kernel_similarity.csv")
        if args.phase == "models":
            return

    if args.phase in {"signatures", "all"}:
        with logger.section("signatures"):
            kernels = pd.read_csv(output / "models/history_kernels.csv")
            signatures = run_robustness_signatures(records, inclusion, kernels)
            _write(signatures, output / "signatures/human_animal_signatures.csv")
        if args.phase == "signatures":
            return

    if args.phase in {"synthetic", "all"}:
        with logger.section("synthetic_recovery"):
            recovery, summary = run_robustness_recovery(
                config, smoke=args.smoke
            )
            _write(recovery, output / "synthetic/recovery.csv")
            _write(summary, output / "synthetic/recovery_summary.csv")
        if args.phase == "synthetic":
            return

    if args.phase in {"report", "all"}:
        with logger.section("report"):
            try:
                generate_figures(output)
            except ModuleNotFoundError as error:
                if error.name != "matplotlib":
                    raise
                logger.note(
                    "report",
                    "matplotlib unavailable; tables/report retained for local figure generation",
                )
            generate_report(output, config, smoke=args.smoke)
            logger.note("report", f"report written to {output / 'report.md'}")


if __name__ == "__main__":
    main()

