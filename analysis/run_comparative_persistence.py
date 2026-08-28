"""Run the leak-proof comparative computational persistence model zoo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from computational_modeling.analysis.run_model_zoo import ProgressLogger
from experiments.runtime import save_run_metadata

from analysis.comparative_persistence.build_modeling_dataset import load_modeling_dataset
from analysis.comparative_persistence.controls.sequential_choice import run_sequential_control
from analysis.comparative_persistence.evaluation.bootstrap import (
    add_episode_bootstrap_intervals,
    add_task_bootstrap_intervals,
)
from analysis.comparative_persistence.evaluation.feature_ablations import run_feature_ablations
from analysis.comparative_persistence.evaluation.generalization import (
    run_few_shot,
    run_lofo,
    run_loto,
)
from analysis.comparative_persistence.evaluation.human_animal_signatures import run_signature_analysis
from analysis.comparative_persistence.evaluation.within_task import run_within_task
from analysis.comparative_persistence.flexible.neural import run_gru_bottleneck
from analysis.comparative_persistence.hazard_models.baselines import MODEL_SPECS
from analysis.comparative_persistence.hazard_models.history_analysis import run_history_analysis
from analysis.comparative_persistence.reporting.build_report import generate_figures, generate_report
from analysis.comparative_persistence.semantic_features import (
    ALL_OBSERVABLE_FEATURES,
    FORBIDDEN_FUTURE_FIELDS,
)
from analysis.comparative_persistence.synthetic.recovery import run_recovery_experiment


def _load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _smoke_subset(records, maximum_episodes):
    pieces = []
    per_split = max(2, int(maximum_episodes) // 3)
    for (_task, _split), part in records.groupby(["task", "split"]):
        pairs = sorted(part.pair_id.unique())
        selected, episodes = [], set()
        for pair in pairs:
            pair_episodes = set(part[part.pair_id == pair].episode_id)
            if selected and len(episodes | pair_episodes) > per_split:
                break
            selected.append(pair)
            episodes |= pair_episodes
        pieces.append(part[part.pair_id.isin(selected)])
    return pd.concat(pieces, ignore_index=True)


def _serialize_dataset(records, path):
    frame = records.copy()
    for column in ("prehistory_actions", "prehistory_outcomes"):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: json.dumps(value) if isinstance(value, (list, tuple)) else "[]"
            )
    frame.to_csv(path, index=False, compression="gzip")


def _read_dataset(path):
    frame = pd.read_csv(path, compression="gzip")
    for column in ("prehistory_actions", "prehistory_outcomes"):
        if column in frame:
            frame[column] = frame[column].fillna("[]").map(json.loads)
    for column in ("continued", "is_persistence_task"):
        if frame[column].dtype != bool:
            frame[column] = frame[column].astype(str).str.lower().eq("true")
    return frame


def _write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _models(args, config):
    if args.models:
        selected = [value.strip() for value in args.models.split(",") if value.strip()]
    elif args.smoke:
        selected = list(config["smoke"]["models"])
    else:
        selected = list(MODEL_SPECS)
    if args.skip_neural:
        selected = [name for name in selected if name not in {"mlp", "gru"}]
    unknown = sorted(set(selected) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    return selected


def _prepare(config, output, *, smoke, logger):
    with logger.section("prepare_dataset"):
        records, inclusion = load_modeling_dataset(config, smoke=False)
        if smoke:
            records = _smoke_subset(
                records, config["smoke"]["maximum_episodes_per_task"]
            )
        _serialize_dataset(records, output / "modeling_dataset.csv.gz")
        _write(inclusion, output / "task_inclusion.csv")
        manifest = {
            "states": len(records),
            "episodes": int(records.episode_id.nunique()),
            "tasks": records.groupby("task").size().to_dict(),
            "persistence_tasks": sorted(records[records.is_persistence_task].task.unique()),
            "control_tasks": sorted(records[~records.is_persistence_task].task.unique()),
            "target": "hazard_event",
            "normalization": "frozen environment specifications",
            "observable_features": list(ALL_OBSERVABLE_FEATURES),
            "forbidden_future_fields": sorted(FORBIDDEN_FUTURE_FIELDS),
            "dataset_sha256": hashlib.sha256(
                (output / "modeling_dataset.csv.gz").read_bytes()
            ).hexdigest(),
        }
        (output / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.note("prepare_dataset", f"wrote {len(records)} at-risk states")
    return records, inclusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/comparative_persistence.yaml")
    parser.add_argument("--run-id", default="comparative_v1")
    parser.add_argument(
        "--phase",
        choices=("prepare", "models", "generalization", "history", "features", "synthetic", "report", "all"),
        default="all",
    )
    parser.add_argument("--models", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_yaml(config_path)
    if args.smoke:
        config["ridge_penalties"] = config["smoke"]["ridge_penalties"]
        config["evaluation"]["few_shot_counts"] = config["smoke"]["few_shot_counts"]
    output = Path(config["output_root"]) / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists: {output}; use --resume or a new run ID")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    logger = ProgressLogger(output, label="comparative-persistence")
    selected_models = _models(args, config)
    save_run_metadata(
        str(output / "run_metadata.json"),
        {**vars(args), "protocol_version": config["protocol_version"], "models": selected_models, "target": "termination_hazard", "target_task_normalization": False},
    )
    logger.note("pipeline", f"run={args.run_id}; phase={args.phase}; smoke={args.smoke}; models={','.join(selected_models)}")

    dataset_path = output / "modeling_dataset.csv.gz"
    if args.phase in {"prepare", "all"} or not dataset_path.exists():
        records, inclusion = _prepare(config, output, smoke=args.smoke, logger=logger)
    else:
        records = _read_dataset(dataset_path)
        inclusion = pd.read_csv(output / "task_inclusion.csv")
    if args.phase == "prepare":
        return

    bootstrap_samples = int(
        config["smoke"]["bootstrap_samples"] if args.smoke else config["bootstrap"]["samples"]
    )
    within = None
    if args.phase in {"models", "all"}:
        with logger.section("model_comparison"):
            within = run_within_task(records, config, models=selected_models, logger=logger)
            within["taskwise"] = add_episode_bootstrap_intervals(
                within["taskwise"],
                within["predictions"],
                samples=bootstrap_samples,
                seed=config["seed"] + 3,
            )
            within["macro"] = add_task_bootstrap_intervals(
                within["macro"], within["taskwise"],
                group_columns=("model", "sharing"), value="log_loss",
                samples=bootstrap_samples, seed=config["seed"],
            )
            _write(within["taskwise"], output / "model_comparison/within_task.csv")
            _write(within["macro"], output / "model_comparison/macro_average.csv")
            _write(within["macro"], output / "model_comparison/model_rankings.csv")
            _write(within["predictions"], output / "model_comparison/heldout_predictions.csv.gz")
        if args.phase == "models":
            return
    if within is None:
        within = {"taskwise": pd.read_csv(output / "model_comparison/within_task.csv")}

    if args.phase in {"generalization", "all"}:
        with logger.section("generalization"):
            loto, loto_summary, architecture, frozen = run_loto(
                records, config, within, models=selected_models, logger=logger
            )
            loto_taskwise = loto.rename(columns={"heldout_task": "task"})
            loto_summary = add_task_bootstrap_intervals(
                loto_summary, loto_taskwise,
                group_columns=("model", "sharing"), value="delta_log_loss_vs_null",
                samples=bootstrap_samples, seed=config["seed"] + 1,
            )
            _write(loto, output / "generalization/loto.csv")
            _write(loto_summary, output / "generalization/loto_summary.csv")
            _write(architecture, output / "generalization/architecture_transfer.csv")
            _write(run_lofo(records, config, models=selected_models, logger=logger), output / "generalization/lofo.csv")
            few_models = config["smoke"]["few_shot_models"] if args.smoke else config["evaluation"]["few_shot_models"]
            few_models = [name for name in few_models if name in selected_models]
            _write(run_few_shot(records, config, frozen, models=few_models, logger=logger), output / "generalization/few_shot_curves.csv")
        if args.phase == "generalization":
            return

    if args.phase in {"history", "all"}:
        with logger.section("history"):
            history_config = dict(config)
            history_config["bootstrap"] = {
                **config["bootstrap"],
                "samples": bootstrap_samples,
            }
            finite, exponential, similarity = run_history_analysis(records, config, logger=logger)
            _write(finite, output / "history/finite_kernels.csv")
            _write(exponential, output / "history/exponential_kernels.csv")
            _write(similarity, output / "history/task_kernel_similarity.csv")
            _write(
                run_sequential_control(
                    records,
                    history_config,
                    models=selected_models,
                    logger=logger,
                ),
                output / "history/persistence_vs_control.csv",
            )
            if "gru" in selected_models:
                _write(run_gru_bottleneck(records, config, logger=logger), output / "history/gru_bottleneck.csv")
        if args.phase == "history":
            return

    if args.phase in {"features", "all"}:
        with logger.section("features"):
            ablation, family_only, support = run_feature_ablations(records, config, logger=logger)
            _write(ablation, output / "features/family_ablation.csv")
            _write(family_only, output / "features/family_only.csv")
            _write(support, output / "features/cross_task_feature_support.csv")
            _write(run_signature_analysis(records, inclusion), output / "human_animal_signatures/signature_effects.csv")
        if args.phase == "features":
            return

    if args.phase in {"synthetic", "all"}:
        with logger.section("synthetic_recovery"):
            recovery, confusion = run_recovery_experiment(config, smoke=args.smoke, logger=logger)
            _write(recovery, output / "synthetic/recovery.csv")
            _write(confusion, output / "synthetic/confusion_matrix.csv")
        if args.phase == "synthetic":
            return

    if args.phase in {"report", "all"}:
        with logger.section("report"):
            generate_figures(output)
            generate_report(output, inclusion, smoke=args.smoke)
            logger.note("report", f"report written to {output / 'report.md'}")


if __name__ == "__main__":
    main()
