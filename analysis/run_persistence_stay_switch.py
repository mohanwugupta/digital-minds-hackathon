"""Run the history-dependent stay/switch pivot using only existing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from analysis.persistence_convergence.activation_cache import build_activation_cache
from analysis.persistence_convergence.functional_convergence import run_task_specific_readouts
from analysis.persistence_convergence.report_convergence import (
    generate_figures,
    generate_report,
)
from analysis.persistence_convergence.run_controls import run_controls
from analysis.persistence_convergence.run_intervention_profiles import (
    run_intervention_profiles,
)
from analysis.persistence_gru.bottleneck_analysis import run_bottleneck_analysis
from analysis.persistence_gru.distill_gru import run_gru_distillation
from analysis.persistence_gru.memory_ablations import run_memory_ablations
from analysis.persistence_hazard.build_risk_sets import read_behavior_records
from analysis.persistence_hazard.compare_shared_architectures import (
    compare_history_kernels,
)
from analysis.persistence_hazard.fit_hazard_models import fit_hazard_architectures
from computational_modeling.analysis.run_model_zoo import ProgressLogger


PHASES = ("behavior", "gru", "neural", "interventions", "controls", "report", "all")


def _load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _smoke_records(records, maximum_episodes_per_task_split):
    """Take complete pair groups, preserving whole episodes and risk histories."""

    frame = pd.DataFrame(records)
    selected = []
    for (_task, _split), part in frame.groupby(["task", "split"], sort=False):
        episodes = 0
        for _pair, group in part.groupby("pair_id", sort=False):
            selected.extend(group.index.tolist())
            episodes += group.episode_id.nunique()
            if episodes >= int(maximum_episodes_per_task_split):
                break
    return frame.loc[sorted(set(selected))].to_dict(orient="records")


def _gru_configuration(config, smoke):
    selected_path = Path(config["paths"]["model_zoo_output"]) / "selected_hyperparameters.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))[
        "gru::observable::shared_architecture_task_observation"
    ]
    parameters = selected["selected"]
    settings = {
        "hidden_size": int(parameters["hidden_size"]),
        "learning_rate": float(parameters["learning_rate"]),
        "dropout": float(parameters["dropout"]),
        "max_epochs": int(config["gru"]["max_epochs"]),
        "patience": int(config["gru"]["patience"]),
        "seed": int(config["seed"]),
        "history_windows": list(config["gru"]["history_windows"]),
        "bottleneck_sizes": list(config["gru"]["bottleneck_sizes"]),
        "ridge_grid": list(config["gru"]["ridge_grid"]),
    }
    if smoke:
        settings.update(
            {
                "hidden_size": int(config["smoke"]["gru_hidden_size"]),
                "max_epochs": int(config["smoke"]["gru_max_epochs"]),
                "patience": int(config["smoke"]["gru_patience"]),
                "bottleneck_sizes": list(config["smoke"]["gru_bottleneck_sizes"]),
            }
        )
    return list(selected["feature_names"]), settings, selected_path


def _neural_config(config, smoke):
    local = {
        **config["neural"],
        "seed": int(config["seed"]),
    }
    if smoke:
        local["projection_dimensions"] = int(
            config["smoke"]["projection_dimensions"]
        )
    return local


def _activation_datasets(records, config, output, smoke, resume, logger):
    maximum = (
        int(config["smoke"]["max_activation_states_per_task"])
        if smoke
        else None
    )
    datasets = {}
    for task in config["tasks"]:
        task_records = [row for row in records if str(row["task"]) == str(task)]
        datasets[task] = build_activation_cache(
            task,
            task_records,
            config["paths"]["activation_banks"][task],
            output / "cache",
            maximum_states=maximum,
            resume=resume,
            logger=logger,
        )
    return datasets


def _write_behavior(records, config, output, logger):
    with logger.section("behavior_hazards"):
        result = fit_hazard_architectures(records, config["behavior"])
        comparison = result["comparison"]
        comparison.to_csv(output / "behavior/hazard_model_comparison.csv", index=False)
        result["taskwise"].to_csv(output / "behavior/taskwise_hazard_metrics.csv", index=False)
        result["calibration"].to_csv(output / "behavior/hazard_calibration.csv", index=False)
        ceiling = comparison[comparison.model == "task_specific"].iloc[0]
        shared = comparison[comparison.model.isin(config["behavior"]["hazard_models"])].copy()
        shared["delta_test_log_loss_vs_task_specific"] = (
            shared.test_log_loss - float(ceiling.test_log_loss)
        )
        shared["delta_test_brier_vs_task_specific"] = (
            shared.test_brier - float(ceiling.test_brier)
        )
        shared.to_csv(output / "behavior/shared_architecture_metrics.csv", index=False)
        logger.note(
            "behavior_hazards",
            "held-out log loss: "
            + ", ".join(
                f"{row.model}={row.test_log_loss:.3f}"
                for row in comparison.itertuples()
            ),
        )
    with logger.section("history_kernels"):
        kernels = compare_history_kernels(records, config["behavior"])
        kernels.to_csv(output / "behavior/history_kernel_results.csv", index=False)
        selected = kernels[kernels.validation_selected].iloc[0]
        logger.note(
            "history_kernels",
            f"validation selected {selected.kernel} kernel={selected.parameter}; test log loss={selected.test_log_loss:.3f}",
        )


def _write_gru(records, config, output, smoke, logger):
    features, settings, _selected_path = _gru_configuration(config, smoke)
    with logger.section("gru_memory_ablations"):
        memory = run_memory_ablations(records, features, settings, logger=logger)
        memory.to_csv(output / "gru/memory_ablation.csv", index=False)
        logger.note(
            "gru_memory_ablations",
            f"full recurrence R2={memory[memory.ablation == 'full_recurrence'].r_squared.iloc[0]:.3f}",
        )
    with logger.section("gru_bottlenecks"):
        bottleneck = run_bottleneck_analysis(records, features, settings, logger=logger)
        bottleneck.to_csv(output / "gru/bottleneck_performance.csv", index=False)
        best = bottleneck.sort_values("r_squared", ascending=False).iloc[0]
        logger.note(
            "gru_bottlenecks",
            f"best tested hidden size={int(best.hidden_size)}; R2={best.r_squared:.3f}",
        )
    with logger.section("gru_distillation"):
        distilled = run_gru_distillation(records, features, settings, logger=logger)
        distilled.to_csv(output / "gru/distilled_models.csv", index=False)


def _fit_and_write_neural(records, config, output, smoke, resume, logger):
    with logger.section("activation_caches"):
        datasets = _activation_datasets(
            records, config, output, smoke, resume, logger
        )
    with logger.section("task_specific_readouts"):
        result = run_task_specific_readouts(
            datasets, _neural_config(config, smoke), logger=logger
        )
        result["metrics"].to_csv(
            output / "neural/task_specific_readout_metrics.csv", index=False
        )
        result["direction_similarity"].to_csv(
            output / "neural/layerwise_direction_similarity.csv", index=False
        )
        result["subspace_similarity"].to_csv(
            output / "neural/subspace_similarity.csv", index=False
        )
        result["functional_convergence"].to_csv(
            output / "neural/functional_convergence.csv", index=False
        )
        final = result["metrics"][result["metrics"].layer == result["metrics"].layer.max()]
        logger.note(
            "task_specific_readouts",
            "final-layer test R2: "
            + ", ".join(f"{row.task}={row.test_r_squared:.3f}" for row in final.itertuples()),
        )
    return datasets, result


def _write_interventions(config, output, datasets, readout_result, logger):
    with logger.section("intervention_profiles"):
        inventory = pd.read_csv(config["paths"]["contrast_inventory"])
        intervention_config = dict(config["interventions"])
        if config.get("_smoke", False):
            intervention_config["max_pairs_per_group"] = int(
                config["smoke"]["max_intervention_pairs_per_group"]
            )
        profiles, similarity = run_intervention_profiles(
            inventory,
            datasets,
            readout_result,
            intervention_config,
            bandit_factorial={
                "rows": config["paths"]["bandit_factorial"],
                "activations": config["paths"]["bandit_factorial_activations"],
            },
            logger=logger,
        )
        if profiles.empty:
            raise ValueError(
                "no intervention pairs overlap the selected activation states; increase the smoke activation limit"
            )
        profiles.to_csv(output / "interventions/intervention_profiles.csv", index=False)
        similarity.to_csv(output / "interventions/profile_similarity.csv", index=False)
        mean = similarity[similarity.kind == "mean_profile_correlation"].value
        logger.note(
            "intervention_profiles",
            f"mean profile correlation={float(mean.iloc[0]):.3f}" if len(mean) else "profile correlation unavailable",
        )


def _write_controls(config, output, persistence_metrics, smoke, resume, logger):
    with logger.section("generic_decision_controls"):
        inventory = pd.read_csv(config["paths"]["contrast_inventory"])
        maximum = (
            int(config["smoke"]["max_activation_states_per_task"])
            if smoke
            else None
        )
        comparison = run_controls(
            inventory,
            config["paths"]["control_banks"],
            output / "cache",
            persistence_metrics,
            _neural_config(config, smoke),
            maximum_states=maximum,
            resume=resume,
            logger=logger,
        )
        comparison.to_csv(
            output / "controls/persistence_vs_generic_decision.csv", index=False
        )
        delta = comparison.groupby("control").delta_r_squared_persistence_minus_control.mean()
        logger.note(
            "generic_decision_controls",
            "mean persistence-minus-control R2: "
            + ", ".join(f"{name}={value:+.3f}" for name, value in delta.items()),
        )


def _metadata(config, config_path, records, selected_path, smoke):
    return {
        "protocol_version": config["protocol_version"],
        "analysis_role": config["analysis_role"],
        "model": config["model"],
        "smoke": bool(smoke),
        "tasks": list(config["tasks"]),
        "states": len(records),
        "episodes": len({(row["task"], row["episode_id"]) for row in records}),
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "selected_gru_hyperparameters": str(selected_path),
        "selected_gru_hyperparameters_sha256": _sha256(selected_path),
        "data_reuse": {
            "new_qwen_trajectories_generated": False,
            "behavior_records": str(config["paths"]["behavior_records"]),
            "activation_banks": config["paths"]["activation_banks"],
            "control_banks": config["paths"]["control_banks"],
            "contrast_inventory": str(config["paths"]["contrast_inventory"]),
            "bandit_factorial_rows": str(config["paths"]["bandit_factorial"]),
            "bandit_factorial_activations": str(
                config["paths"]["bandit_factorial_activations"]
            ),
        },
        "leakage_guards": {
            "absorbing_risk_sets": True,
            "episode_and_pair_splits_preserved": True,
            "train_only_normalization": True,
            "validation_only_hyperparameter_selection": True,
            "test_target_fitting": False,
            "one_shot_control_history_fabricated": False,
        },
        "cache_policy": "local float16 memmaps; excluded by .gitignore",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_stay_switch.yaml")
    parser.add_argument("--run-id", default="stay_switch_v1")
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_yaml(config_path)
    config["_smoke"] = bool(args.smoke)
    output = Path(config["output_root"]) / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"run output already exists: {output}; choose a new --run-id or pass --resume"
        )
    for directory in ("behavior", "gru", "neural", "interventions", "controls"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    logger = ProgressLogger(output, label="stay-switch")
    logger.note("pipeline", f"run={args.run_id}; phase={args.phase}; smoke={args.smoke}")

    records = read_behavior_records(config["paths"]["behavior_records"], config["tasks"])
    if args.smoke:
        records = _smoke_records(
            records, config["smoke"]["max_episodes_per_task_split"]
        )
    features, _settings, selected_path = _gru_configuration(config, args.smoke)
    del features, _settings
    (output / "run_metadata.json").write_text(
        json.dumps(
            _metadata(config, config_path, records, selected_path, args.smoke),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.note(
        "data_reuse",
        f"loaded {len(records)} existing at-risk states; no model inference or trajectory collection",
    )

    if args.phase in {"behavior", "all"}:
        _write_behavior(records, config, output, logger)
    if args.phase in {"gru", "all"}:
        _write_gru(records, config, output, args.smoke, logger)

    datasets = readout_result = None
    needs_neural = args.phase in {"neural", "interventions", "controls", "all"}
    if needs_neural:
        datasets, readout_result = _fit_and_write_neural(
            records, config, output, args.smoke, args.resume, logger
        )
    if args.phase in {"interventions", "all"}:
        _write_interventions(config, output, datasets, readout_result, logger)
    if args.phase in {"controls", "all"}:
        _write_controls(
            config,
            output,
            readout_result["metrics"],
            args.smoke,
            args.resume,
            logger,
        )
    if args.phase in {"report", "all"}:
        with logger.section("report_generation"):
            generate_figures(output)
            generate_report(output, smoke=args.smoke)
            logger.note("report_generation", f"report available at {output / 'report.md'}")
    logger.note("pipeline", "persistence stay/switch analysis complete")


if __name__ == "__main__":
    main()
