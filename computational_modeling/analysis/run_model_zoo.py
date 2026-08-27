"""Export, recover, fit, evaluate, and report the computational model zoo."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from computational_modeling.analysis.evaluate_models import (
    clustered_bootstrap_differences,
    clustered_bootstrap_intervals,
    evaluate_predictions,
    model_comparisons,
    normalized_performance,
)
from computational_modeling.analysis.followup_analysis import (
    run_followup_analysis,
    run_neural_linear_recovery_sanity,
)
from computational_modeling.analysis.model_fitting import fit_interpretable_model
from computational_modeling.analysis.model_recovery import (
    GENERATING_ARCHITECTURES,
    recover_architecture,
    simulate_architecture,
)
from computational_modeling.analysis.summarize_model_zoo import generate_report
from computational_modeling.data.build_cross_task_behavioral_dataset import (
    export_behavioral_dataset,
)
from computational_modeling.data.feature_schema import (
    FEATURE_SCHEMA,
    serialized_schema,
)
from computational_modeling.models.baselines import FLEXIBLE_FEATURES, enabled_definitions
from computational_modeling.models.gru import fit_gru_ceiling
from computational_modeling.models.mlp import fit_mlp_ceiling


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {whole_seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {whole_seconds:02d}s"
    return f"{seconds:.1f}s"


class ProgressLogger:
    """Flush progress to stdout and durable JSONL/timing sidecars."""

    def __init__(self, output: Path, *, label: str = "model-zoo"):
        self.output = Path(output)
        self.label = str(label)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.progress_path = self.output / "progress.jsonl"
        self.timings_path = self.output / "timings.json"
        self.timings = []
        if self.timings_path.exists():
            try:
                self.timings = json.loads(self.timings_path.read_text(encoding="utf-8"))[
                    "sections"
                ]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.timings = []

    def note(self, section: str, message: str, **details) -> None:
        elapsed = time.perf_counter() - self.started
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "section": str(section),
            "event": "progress",
            "message": str(message),
            **details,
        }
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(
            f"[{self.label} +{_format_duration(elapsed)}] [{section}] {message}",
            flush=True,
        )

    def _save_timings(self) -> None:
        self.timings_path.write_text(
            json.dumps(
                {
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "total_elapsed_seconds": time.perf_counter() - self.started,
                    "sections": self.timings,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @contextmanager
    def section(self, name: str, **details):
        started = time.perf_counter()
        self.note(name, "started", **details)
        try:
            yield
        except Exception as error:
            duration = time.perf_counter() - started
            row = {
                "section": name,
                "status": "failed",
                "duration_seconds": duration,
                "error": f"{type(error).__name__}: {error}",
                **details,
            }
            self.timings.append(row)
            self._save_timings()
            self.note(name, f"failed after {_format_duration(duration)}: {error}")
            raise
        duration = time.perf_counter() - started
        self.timings.append(
            {
                "section": name,
                "status": "completed",
                "duration_seconds": duration,
                **details,
            }
        )
        self._save_timings()
        self.note(name, f"completed in {_format_duration(duration)}")


def _note(logger, section: str, message: str, **details) -> None:
    if logger is not None:
        logger.note(section, message, **details)


def _load_config(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _read_records(records_dir: Path, tasks) -> list[dict]:
    records = []
    for task in tasks:
        frame = pd.read_csv(records_dir / f"{task}_records.csv")
        records.extend(frame.to_dict(orient="records"))
    return records


def _split(records):
    return {
        name: [dict(row) for row in records if row["split"] == name]
        for name in ("train", "validation", "test")
    }


def _flexible_features(information_set):
    features = list(FLEXIBLE_FEATURES)
    if information_set == "oracle":
        features.extend(
            (
                "oracle_continue_value",
                "oracle_outside_value",
                "oracle_termination_advantage",
                "oracle_relative_value",
            )
        )
    return features


def _fit_neural_model(name, splits, information_set, config, *, logger=None):
    features = _flexible_features(information_set)
    candidates = []
    if name == "mlp":
        grid = [
            {"learning_rate": float(rate), "dropout": float(dropout)}
            for rate in config["mlp"]["learning_rates"]
            for dropout in config["mlp"]["dropout"]
        ]
        fitter = fit_mlp_ceiling
    else:
        grid = [
            {
                "hidden_size": int(hidden),
                "learning_rate": float(rate),
                "dropout": float(dropout),
            }
            for hidden in config["gru"]["hidden_sizes"]
            for rate in config["gru"]["learning_rates"]
            for dropout in config["gru"]["dropout"]
        ]
        fitter = fit_gru_ceiling
    for candidate_index, parameters in enumerate(grid, start=1):
        candidate_started = time.perf_counter()
        _note(
            logger,
            "neural_hyperparameters",
            f"{name}/{information_set}: candidate {candidate_index}/{len(grid)} "
            f"{parameters} started",
        )
        result = fitter(
            splits["train"],
            splits["validation"],
            splits["validation"],
            features,
            max_epochs=int(config[name]["max_epochs"]),
            patience=int(config[name]["early_stopping_patience"]),
            seed=int(config["seed"]),
            **parameters,
        )
        candidates.append((float(result["validation_mse"]), parameters))
        _note(
            logger,
            "neural_hyperparameters",
            f"{name}/{information_set}: candidate {candidate_index}/{len(grid)} "
            f"validation MSE={result['validation_mse']:.6f}; "
            f"{_format_duration(time.perf_counter() - candidate_started)}",
        )
    best_score = min(item[0] for item in candidates)
    tolerance = max(1e-8, 0.01 * best_score)
    validation_mse, selected = min(
        (item for item in candidates if item[0] <= best_score + tolerance),
        key=lambda item: (
            int(item[1].get("hidden_size", 0)),
            sorted(item[1].items()),
        ),
    )
    _note(
        logger,
        "neural_hyperparameters",
        f"{name}/{information_set}: selected {selected}; refitting frozen test model",
    )
    refit_started = time.perf_counter()
    result = fitter(
        splits["train"],
        splits["validation"],
        splits["test"],
        features,
        max_epochs=int(config[name]["max_epochs"]),
        patience=int(config[name]["early_stopping_patience"]),
        seed=int(config["seed"]),
        **selected,
    )
    _note(
        logger,
        "neural_hyperparameters",
        f"{name}/{information_set}: frozen test predictions complete in "
        f"{_format_duration(time.perf_counter() - refit_started)}",
    )
    return {
        "model": name,
        "code": "M17" if name == "mlp" else "M18",
        "information_set": information_set,
        "sharing": "shared_architecture_task_observation",
        "prediction": result["prediction"],
        "test_records": splits["test"],
        "selected_hyperparameters": {**selected, "epochs": result["selected_epochs"]},
        "validation_macro_mse": validation_mse,
        "fitted_parameters": {},
        "feature_names": features,
        "state_columns": {},
        "parameter_count": result["parameter_count"],
        "hyperparameter_candidates": [
            {"validation_macro_mse": score, **parameters}
            for score, parameters in candidates
        ],
    }


def _oracle_fit(splits, information_set, sharing):
    return {
        "model": "oracle_policy",
        "code": "CEILING",
        "information_set": information_set,
        "sharing": sharing,
        "prediction": np.asarray([row["persistence_logit"] for row in splits["test"]]),
        "test_records": splits["test"],
        "selected_hyperparameters": {},
        "validation_macro_mse": 0.0,
        "fitted_parameters": {},
        "feature_names": ["recorded_llm_policy"],
        "state_columns": {},
        "parameter_count": 0,
        "hyperparameter_candidates": [],
    }


def _write_latent_states(fit, output):
    if fit["model"] not in {
        "termination_advantage",
        "sticky_termination",
        "disengagement_accumulator",
        "latent_commitment",
        "generic_latent_value",
    }:
        return
    destination = output / "latent_states"
    destination.mkdir(parents=True, exist_ok=True)
    state_columns = fit["state_columns"]
    with (destination / f"{fit['model']}_{fit['information_set']}_{fit['sharing']}_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        base_fields = [
            "episode_id",
            "state_id",
            "task",
            "round",
            "observed_persistence_logit",
            "predicted_persistence_logit",
            "termination_advantage",
            "estimated_continue_value",
            "estimated_outside_value",
            "cost_component",
            "progress_component",
        ]
        extra_fields = []
        for name, values in state_columns.items():
            array = np.asarray(values)
            if array.ndim == 1:
                extra_fields.append(name)
            else:
                extra_fields.extend(f"{name}_{index}" for index in range(array.shape[1]))
        writer = csv.DictWriter(handle, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        for index, record in enumerate(fit["test_records"]):
            row = {
                "episode_id": record["episode_id"],
                "state_id": record["state_id"],
                "task": record["task"],
                "round": record["round"],
                "observed_persistence_logit": record["persistence_logit"],
                "predicted_persistence_logit": fit["prediction"][index],
                "termination_advantage": record["termination_advantage"],
                "estimated_continue_value": record["estimated_continue_value"],
                "estimated_outside_value": record["estimated_outside_value"],
                "cost_component": record["cost_pressure"],
                "progress_component": record["progress_evidence"],
            }
            for name, values in state_columns.items():
                array = np.asarray(values)
                if array.ndim == 1:
                    row[name] = array[index]
                else:
                    for column in range(array.shape[1]):
                        row[f"{name}_{column}"] = array[index, column]
            writer.writerow(row)


def run_recovery(config, output, *, logger=None):
    recovery_dir = output / "model_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    repetitions = int(config["model_recovery"]["repetitions"])
    counts = pd.DataFrame(0, index=GENERATING_ARCHITECTURES, columns=GENERATING_ARCHITECTURES)
    details = []
    for generating in GENERATING_ARCHITECTURES:
        architecture_started = time.perf_counter()
        _note(
            logger,
            "model_recovery",
            f"{generating}: starting {repetitions} blinded recovery repetitions",
        )
        for repetition in range(repetitions):
            synthetic = simulate_architecture(
                generating,
                episodes=int(config["model_recovery"]["episodes"]),
                decisions=int(config["model_recovery"]["decisions"]),
                seed=int(config["seed"]) + repetition,
            )
            result = recover_architecture(synthetic)
            counts.loc[generating, result["selected_family"]] += 1
            details.append({"generating": generating, "repetition": repetition, **result})
            progress_every = max(1, repetitions // 4)
            if (repetition + 1) % progress_every == 0 or repetition + 1 == repetitions:
                _note(
                    logger,
                    "model_recovery",
                    f"{generating}: {repetition + 1}/{repetitions} repetitions complete",
                )
        _note(
            logger,
            "model_recovery",
            f"{generating}: completed in "
            f"{_format_duration(time.perf_counter() - architecture_started)}; "
            f"direct recovery={counts.loc[generating, generating] / repetitions:.1%}",
        )
    matrix = counts / repetitions
    matrix.to_csv(recovery_dir / "recovery_matrix.csv")
    summary = {
        "repetitions": repetitions,
        "architectures": list(GENERATING_ARCHITECTURES),
        "diagonal_recovery": {
            architecture: float(matrix.loc[architecture, architecture])
            for architecture in GENERATING_ARCHITECTURES
        },
        "details": details,
    }
    (recovery_dir / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _reproduce_existing_bandit() -> dict:
    from computational_modeling.analysis.compare_behavioral_models import (
        compare_models,
        summarize_predictions,
    )

    frame = pd.read_csv("artifacts/bandit_pilot.csv")
    predictions, selections = compare_models(frame)
    observed = summarize_predictions(predictions)
    frozen = json.loads(Path("computational_modeling/results/summary.json").read_text())
    expected = pd.DataFrame(frozen["models"]).set_index("model")
    actual = observed.set_index("model")
    maximum_error_by_metric = {
        metric: max(
            abs(float(actual.loc[model, metric]) - float(expected.loc[model, metric]))
            for model in expected.index
        )
        for metric in ("log_loss", "auc", "brier")
    }
    # Rank-based AUC can move at tied probabilities across SciPy versions;
    # likelihood and calibration metrics must reproduce much more tightly.
    tolerances = {"log_loss": 1e-10, "brier": 1e-10, "auc": 1e-4}
    return {
        "passed": all(
            maximum_error_by_metric[metric] <= tolerance
            for metric, tolerance in tolerances.items()
        ),
        "maximum_absolute_metric_error": maximum_error_by_metric,
        "tolerance": tolerances,
        "episodes": int(frame.episode_id.nunique()),
        "states": len(frame),
        "selected_alphas": selections.alpha.tolist(),
    }


def write_checkpoint(config, manifest, output, *, logger=None):
    reproduction_started = time.perf_counter()
    _note(logger, "checkpoint", "reproducing frozen Bandit behavioral-model metrics")
    reproduction = _reproduce_existing_bandit()
    _note(
        logger,
        "checkpoint",
        "Bandit regression complete in "
        f"{_format_duration(time.perf_counter() - reproduction_started)}; "
        f"passed={reproduction['passed']}",
    )
    if not reproduction["passed"]:
        raise RuntimeError("existing Bandit behavioral-model results did not reproduce")
    features = _flexible_features("observable")
    records = _read_records(Path(config["records_output"]), config["tasks"])
    gru_dimensions = {}
    for split_name in ("train", "validation", "test"):
        selected = [row for row in records if row["split"] == split_name]
        episode_lengths: dict[str, int] = {}
        for row in selected:
            episode = str(row["episode_id"])
            episode_lengths[episode] = episode_lengths.get(episode, 0) + 1
        gru_dimensions[split_name] = {
            "episodes": len(episode_lengths),
            "states": len(selected),
            "maximum_sequence_length": max(episode_lengths.values()),
            "feature_count": len(features),
            "padded_shape": [
                len(episode_lengths),
                max(episode_lengths.values()),
                len(features),
            ],
        }
    checkpoint = {
        "files_changed": {
            "data": [
                "computational_modeling/data/build_cross_task_behavioral_dataset.py",
                "computational_modeling/data/feature_schema.py",
            ],
            "models": [
                "computational_modeling/models/base.py",
                "computational_modeling/models/baselines.py",
                "computational_modeling/models/history.py",
                "computational_modeling/models/rw.py",
                "computational_modeling/models/termination.py",
                "computational_modeling/models/accumulator.py",
                "computational_modeling/models/latent_commitment.py",
                "computational_modeling/models/mlp.py",
                "computational_modeling/models/gru.py",
            ],
            "analysis": [
                "computational_modeling/analysis/model_fitting.py",
                "computational_modeling/analysis/evaluate_models.py",
                "computational_modeling/analysis/model_recovery.py",
                "computational_modeling/analysis/run_model_zoo.py",
                "computational_modeling/analysis/summarize_model_zoo.py",
            ],
            "integration": [
                "config/computational_model_zoo.yaml",
                "scripts/submit_model_zoo_checkpoint.sh",
                "run_qwen35_bandit.sh",
                "pyproject.toml",
                "environment.yml",
                "computational_modeling/README.md",
                "computational_modeling/data/README.md",
            ],
        },
        "tests_added": sorted(
            str(path)
            for path in Path("computational_modeling/tests").glob("test_*.py")
            if path.name != "test_behavioral_models.py"
        ),
        "existing_model_results_reproduced": reproduction,
        "behavioral_counts": manifest["tasks"],
        "observable_features": {
            task: FEATURE_SCHEMA[task]["observable"] for task in config["tasks"]
        },
        "oracle_only_features": {
            task: FEATURE_SCHEMA[task]["oracle"] for task in config["tasks"]
        },
        "missing_data_issues": [],
        "gru_input_dimensions": {
            "states_by_features": [manifest["total_states"], len(features)],
            "features": features,
            "by_split": gru_dimensions,
        },
        "full_run_command": (
            "python -m computational_modeling.analysis.run_model_zoo "
            "--config config/computational_model_zoo.yaml --phase all "
            "--run-id model_zoo_mac_v2"
        ),
    }
    (output / "implementation_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint


def run_full(config, records, output, *, final_bootstrap=False, logger=None):
    splits = _split(records)
    fits = []
    definitions = enabled_definitions(config)
    information_set = config["information_sets"]["primary"]
    total_interpretable = len(config["sharing"]["architectures"]) * len(definitions)
    completed_interpretable = 0
    with logger.section("neural_linear_sanity") if logger is not None else nullcontext():
        run_neural_linear_recovery_sanity(config, output, logger=logger)
    _note(
        logger,
        "interpretable_models",
        f"starting {total_interpretable} observable sharing/model fits",
    )
    for sharing in config["sharing"]["architectures"]:
        for definition in definitions:
            fit_started = time.perf_counter()
            _note(
                logger,
                "interpretable_models",
                f"{definition.code} {definition.name} | {information_set} | "
                f"{sharing} started",
            )
            fit = fit_interpretable_model(
                splits["train"],
                splits["validation"],
                splits["test"],
                definition,
                information_set=information_set,
                sharing=sharing,
                config=config,
            )
            fits.append(fit)
            completed_interpretable += 1
            _note(
                logger,
                "interpretable_models",
                f"{completed_interpretable}/{total_interpretable}: "
                f"{definition.name} complete in "
                f"{_format_duration(time.perf_counter() - fit_started)}; "
                f"validation macro MSE={fit['validation_macro_mse']:.6f}; "
                f"selected={fit['selected_hyperparameters']}",
            )
        fits.append(_oracle_fit(splits, information_set, sharing))
        _note(
            logger,
            "interpretable_models",
            f"oracle-policy ceiling registered for {information_set} | {sharing}",
        )
    if config["models"].get("mlp", True):
        fits.append(
            _fit_neural_model(
                "mlp", splits, information_set, config, logger=logger
            )
        )
    if config["models"].get("gru", True):
        fits.append(
            _fit_neural_model(
                "gru", splits, information_set, config, logger=logger
            )
        )

    # The follow-up uses oracle state only for the full flexible linear model.
    # Running every scientific ablation under both information sets would be a
    # different, much larger analysis than the mini PRD requests.
    flexible_definition = next(
        definition for definition in definitions if definition.name == "flexible_linear"
    )
    for oracle_information_set in config["information_sets"].get("secondary", []):
        for sharing in config["sharing"]["architectures"]:
            fits.append(
                fit_interpretable_model(
                    splits["train"],
                    splits["validation"],
                    splits["test"],
                    flexible_definition,
                    information_set=oracle_information_set,
                    sharing=sharing,
                    config=config,
                )
            )
    _note(logger, "evaluation", f"evaluating {len(fits)} frozen model fits")
    evaluation_started = time.perf_counter()
    metric_rows, taskwise_rows, selected = [], [], {}
    for fit_index, fit in enumerate(fits, start=1):
        aggregate, taskwise = evaluate_predictions(fit)
        metric_rows.append(aggregate)
        taskwise_rows.extend(taskwise)
        key = f"{fit['model']}::{fit['information_set']}::{fit['sharing']}"
        selected[key] = {
            "selected": fit["selected_hyperparameters"],
            "validation_macro_mse": fit["validation_macro_mse"],
            "feature_names": fit["feature_names"],
            "candidates": fit.get("hyperparameter_candidates", []),
        }
        _write_latent_states(fit, output)
        if fit_index % 10 == 0 or fit_index == len(fits):
            _note(
                logger,
                "evaluation",
                f"evaluated {fit_index}/{len(fits)} fits",
            )
    _note(
        logger,
        "evaluation",
        f"metrics and latent-state evaluation completed in "
        f"{_format_duration(time.perf_counter() - evaluation_started)}",
    )
    metrics = pd.DataFrame(metric_rows)
    taskwise = pd.DataFrame(taskwise_rows)
    normalized = normalized_performance(metrics)
    comparisons = model_comparisons(metrics)
    bootstrap_samples = int(
        config["bootstrap"]["final_samples" if final_bootstrap else "development_samples"]
    )
    bootstrap_started = time.perf_counter()
    _note(
        logger,
        "bootstrap",
        f"starting {bootstrap_samples} pair/episode-cluster samples for "
        f"{len(fits)} fits",
    )
    intervals = clustered_bootstrap_intervals(
        fits, samples=bootstrap_samples, seed=int(config["seed"])
    )
    differences = clustered_bootstrap_differences(
        fits, samples=bootstrap_samples, seed=int(config["seed"])
    )
    intervals = pd.concat((intervals, differences), ignore_index=True)
    _note(
        logger,
        "bootstrap",
        f"wrote {len(intervals)} metric/difference intervals in "
        f"{_format_duration(time.perf_counter() - bootstrap_started)}",
    )
    metrics.to_csv(output / "model_metrics.csv", index=False)
    taskwise.to_csv(output / "taskwise_metrics.csv", index=False)
    normalized.to_csv(output / "normalized_performance.csv", index=False)
    comparisons.to_csv(output / "model_comparisons.csv", index=False)
    intervals.to_csv(output / "bootstrap_intervals.csv", index=False)
    with logger.section("followup_analysis") if logger is not None else nullcontext():
        _note(
            logger,
            "followup_analysis",
            "building coverage-safe rankings, flexible comparison, and feature-family analyses",
        )
        followup = run_followup_analysis(
            config,
            splits,
            fits,
            metrics,
            taskwise,
            output,
            bootstrap_samples=bootstrap_samples,
            logger=logger,
        )
        _note(
            logger,
            "followup_analysis",
            f"best flexible model={followup['best_flexible_model']}",
        )
    (output / "selected_hyperparameters.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _note(
        logger,
        "outputs",
        f"saved {len(metrics)} aggregate metrics, {len(taskwise)} taskwise metrics, "
        f"and {len(selected)} hyperparameter selections",
    )
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/computational_model_zoo.yaml")
    parser.add_argument(
        "--phase", choices=("checkpoint", "export", "recovery", "fit", "report", "all"), default="checkpoint"
    )
    parser.add_argument("--run-id", default="model_zoo_mac_v2")
    parser.add_argument("--final-bootstrap", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    output = Path(config["output_root"]) / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"run output already exists: {output}; choose a new --run-id or pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(output)
    logger.note(
        "pipeline",
        f"run {args.run_id!r} started; phase={args.phase}; "
        f"final_bootstrap={args.final_bootstrap}",
    )
    records_dir = Path(config["records_output"])
    if args.phase in {"checkpoint", "export", "all"} or not (records_dir / "dataset_manifest.json").exists():
        with logger.section("records_export"):
            manifest = export_behavioral_dataset(
                config,
                records_dir,
                progress=lambda message: logger.note("records_export", message),
            )
    else:
        logger.note(
            "records_export",
            f"reusing existing records manifest at {records_dir / 'dataset_manifest.json'}",
        )
        manifest = json.loads((records_dir / "dataset_manifest.json").read_text())
    with logger.section("provenance"):
        shutil.copy2(args.config, output / "config.yaml")
        shutil.copy2(records_dir / "dataset_manifest.json", output / "dataset_manifest.json")
        shutil.copy2(records_dir / "feature_schema.json", output / "feature_schema.json")
        from experiments.runtime import run_metadata

        metadata = run_metadata(
            {
                "analysis": "computational_model_zoo",
                "protocol_version": config["protocol_version"],
                "analysis_role": config["analysis_role"],
                "run_id": args.run_id,
                "seed": config["seed"],
                "tasks": config["tasks"],
                "model": config["model"],
                "split_hashes": {
                    task: manifest["tasks"][task]["split_sha256"]
                    for task in config["tasks"]
                },
                "input_artifact_hashes": {
                    task: manifest["tasks"][task]["behavioral_payload_sha256"]
                    for task in config["tasks"]
                },
                "feature_schema": serialized_schema(),
            }
        )
        (output / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.phase == "checkpoint":
        with logger.section("checkpoint"):
            checkpoint = write_checkpoint(
                config, manifest, output, logger=logger
            )
        logger.note("pipeline", "checkpoint phase finished successfully")
        print(json.dumps(checkpoint, indent=2, sort_keys=True), flush=True)
        return
    if args.phase == "export":
        logger.note("pipeline", "export phase finished successfully")
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return
    if args.phase in {"recovery", "all"} and config["model_recovery"]["enabled"]:
        with logger.section("model_recovery"):
            run_recovery(config, output, logger=logger)
        if args.phase == "recovery":
            logger.note("pipeline", "recovery phase finished successfully")
            return
    records = _read_records(records_dir, config["tasks"])
    if args.phase in {"fit", "all"}:
        with logger.section("heldout_model_zoo"):
            run_full(
                config,
                records,
                output,
                final_bootstrap=args.final_bootstrap,
                logger=logger,
            )
        if args.phase == "fit":
            logger.note("pipeline", "fit phase finished successfully")
            return
    if args.phase in {"report", "all"}:
        with logger.section("report_generation"):
            logger.note(
                "report_generation",
                "rendering performance figures and the recovery matrix",
            )
            generate_report(output)
            logger.note(
                "report_generation",
                f"report available at {output / 'report.md'}",
            )
    logger.note(
        "pipeline",
        f"phase {args.phase!r} finished successfully in "
        f"{_format_duration(time.perf_counter() - logger.started)}",
    )


if __name__ == "__main__":
    main()
