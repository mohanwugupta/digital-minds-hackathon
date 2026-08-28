"""Pilot and collect the literature-grounded behavior-only persistence battery."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pandas as pd

from computational_modeling.analysis.run_model_zoo import ProgressLogger
from experiments.persistence_battery.collection import (
    DeterministicSmokeModel,
    build_specs,
    collect_pair,
    read_pair,
    write_pair,
)
from experiments.persistence_battery.manifests import write_manifests
from experiments.persistence_battery.registry import TASKS, enabled_tasks
from experiments.persistence_battery.report import generate_figures, generate_report
from experiments.persistence_battery.storage import read_records_frame, write_records_frame
from experiments.persistence_battery.validation import validate_records
from experiments.runtime import save_run_metadata


def _load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _selected_tasks(config, value):
    available = enabled_tasks(config)
    if value in {None, "all"}:
        return available
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"tasks are unavailable or disabled: {unknown}; enabled={list(available)}"
        )
    return selected


def _locations(output, mode):
    base = output / "pilot" if mode == "pilot" else output
    return {
        "base": base,
        "records": base / "records",
        "pairs": output / "cache" / f"{mode}_pairs",
    }


def _spec_groups(specs):
    groups = defaultdict(list)
    for spec in specs:
        groups[spec.pair_id].append(spec)
    if any(len(group) != 2 for group in groups.values()):
        raise ValueError("every semantic pair must contain exactly two label mappings")
    return dict(groups)


def _cached_collection_complete(
    output, tasks, specs_by_task, *, mode, expected_model_id
):
    """Validate a complete raw cache without constructing/loading the model."""

    paths = _locations(Path(output), mode)
    for task in tasks:
        groups = _spec_groups(specs_by_task[task])
        for pair_id, pair_specs in groups.items():
            path = paths["pairs"] / task / f"{pair_id}.json"
            if not path.exists():
                return False
            existing = read_pair(path)
            expected_condition = json.dumps(
                asdict(pair_specs[0].condition),
                sort_keys=True,
                separators=(",", ":"),
            )
            if (
                not existing
                or existing[0].get("condition") != expected_condition
                or existing[0].get("model_id") != expected_model_id
            ):
                raise RuntimeError(
                    f"resume cache does not match the current condition/model: {path}; use a new run ID"
                )
    return True


def _collect(
    model,
    config,
    output,
    tasks,
    *,
    mode,
    smoke,
    num_shards,
    shard_index,
    resume,
    logger,
):
    specs_by_task = {
        task: build_specs(config, task, mode=mode, smoke=smoke) for task in tasks
    }
    write_manifests(output, config, specs_by_task, mode=mode, smoke=smoke)
    paths = _locations(output, mode)
    for task in tasks:
        groups = _spec_groups(specs_by_task[task])
        with logger.section(f"collect_{task}", pairs=len(groups), mode=mode):
            for pair_index, pair_id in enumerate(sorted(groups)):
                if pair_index % int(num_shards) != int(shard_index):
                    continue
                path = paths["pairs"] / task / f"{pair_id}.json"
                if resume and path.exists():
                    existing = read_pair(path)
                    expected_condition = json.dumps(
                        asdict(groups[pair_id][0].condition),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if (
                        not existing
                        or existing[0].get("condition") != expected_condition
                        or existing[0].get("model_id")
                        != getattr(model, "model_id", "unknown")
                    ):
                        raise RuntimeError(
                            f"resume cache does not match the current condition/model: {path}; use a new run ID"
                        )
                    continue
                records = collect_pair(
                    model,
                    TASKS[task],
                    groups[pair_id],
                    config["tasks"][task],
                )
                write_pair(path, records)
                if pair_index == len(groups) - 1 or (pair_index + 1) % 10 == 0:
                    logger.note(
                        f"collect_{task}",
                        f"shard {shard_index}/{num_shards}: pair {pair_index + 1}/{len(groups)}; {len(records)} decision records",
                    )
    return specs_by_task


def _finalize(config, output, tasks, *, mode, smoke, logger):
    specs_by_task = {
        task: build_specs(config, task, mode=mode, smoke=smoke) for task in tasks
    }
    write_manifests(output, config, specs_by_task, mode=mode, smoke=smoke)
    paths = _locations(output, mode)
    paths["records"].mkdir(parents=True, exist_ok=True)
    complete = True
    record_manifest = {}
    for task in tasks:
        groups = _spec_groups(specs_by_task[task])
        expected = [paths["pairs"] / task / f"{pair_id}.json" for pair_id in groups]
        missing = [path for path in expected if not path.exists()]
        if missing:
            complete = False
            logger.note(
                "finalize",
                f"{task}: waiting for {len(missing)}/{len(expected)} semantic-pair files",
            )
            continue
        records = [row for path in expected for row in read_pair(path)]
        frame = pd.DataFrame(records).sort_values(
            ["pair_id", "mapping_id", "step"]
        )
        write_result = write_records_frame(frame, paths["records"], task)
        path = write_result.path
        record_manifest[task] = {
            "path": str(path.relative_to(output)),
            "format": write_result.format,
            "states": len(frame),
            "episodes": int(frame.episode_id.nunique()),
        }
        if write_result.parquet_error:
            logger.note(
                "finalize",
                f"{task}: Parquet engine unavailable; wrote portable compressed CSV instead",
            )
        logger.note(
            "finalize",
            f"{task}: wrote {len(frame)} states from {frame.episode_id.nunique()} episodes to {path}",
        )
    if record_manifest:
        (paths["records"] / "records_manifest.json").write_text(
            json.dumps(record_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return complete, specs_by_task, paths["records"]


def _load_specs(output):
    return json.loads(
        (Path(output) / "manifests/task_specs.json").read_text(encoding="utf-8")
    )


def _validate_and_report(
    config,
    output,
    tasks,
    record_directory,
    *,
    mode,
    smoke,
    model_free,
    logger,
):
    with logger.section("behavioral_validation", mode=mode):
        frames, manipulations, label_bias, nondegeneracy, approval = validate_records(
            record_directory,
            output / "validation",
            config,
            tasks,
            model_free=model_free,
        )
        logger.note(
            "behavioral_validation",
            f"approved {sum(approval['tasks'].values())}/{len(tasks)} tasks for full collection",
        )
    with logger.section("behavioral_report", mode=mode):
        try:
            generate_figures(frames, manipulations, output / "figures")
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            logger.note(
                "behavioral_report",
                "matplotlib is unavailable; records, validation, and report will be kept and figures can be regenerated locally",
            )
        generate_report(
            _load_specs(output),
            frames,
            manipulations,
            label_bias,
            nondegeneracy,
            config,
            output,
            mode=mode,
            smoke=smoke,
            model_free=model_free,
        )
        logger.note("behavioral_report", f"report available at {output / 'report.md'}")


def _require_pilot_approval(output, tasks):
    path = output / "validation/pilot_approval.json"
    if not path.exists():
        raise RuntimeError("full collection requires a completed pilot and approval file")
    approval = json.loads(path.read_text(encoding="utf-8"))
    rejected = [task for task in tasks if not approval.get("tasks", {}).get(task, False)]
    if approval.get("model_free") or rejected:
        raise RuntimeError(
            f"full collection is blocked by pilot approval for tasks: {rejected or list(tasks)}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_battery.yaml")
    parser.add_argument("--run-id", default="battery_pilot_v1")
    parser.add_argument(
        "--phase",
        choices=("inventory", "pilot", "full", "finalize", "validate", "report"),
        default="pilot",
    )
    parser.add_argument("--dataset", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-free", action="store_true")
    parser.add_argument(
        "--skip-pilot-approval",
        action="store_true",
        help="allow a full exploratory collection before the functional pilot gate",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.model_free and not args.smoke:
        raise ValueError("--model-free is restricted to --smoke plumbing validation")
    if args.skip_pilot_approval and args.phase != "full":
        raise ValueError("--skip-pilot-approval is valid only with --phase full")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must lie in [0, num-shards)")
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    tasks = _selected_tasks(config, args.tasks)
    output = Path(config["output_root"]) / args.run_id
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"run output already exists: {output}; choose a new --run-id or pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    logger = ProgressLogger(output, label="persistence-battery")
    logger.note(
        "pipeline",
        f"run={args.run_id}; phase={args.phase}; tasks={','.join(tasks)}; smoke={args.smoke}; model_free={args.model_free}",
    )
    save_run_metadata(
        str(output / "run_metadata.json"),
        {
            **vars(args),
            "model": args.model or config["model"],
            "protocol_version": config["protocol_version"],
            "tasks": list(tasks),
            "behavior_only": True,
            "capture_hidden_states": False,
            "full_activation_banks_collected": False,
        },
    )

    inventory_mode = "pilot" if args.phase in {"inventory", "pilot"} else args.dataset
    if args.phase == "full":
        inventory_mode = "full"
    specs_by_task = {
        task: build_specs(config, task, mode=inventory_mode, smoke=args.smoke)
        for task in tasks
    }
    write_manifests(
        output, config, specs_by_task, mode=inventory_mode, smoke=args.smoke
    )
    if args.phase == "inventory":
        logger.note("pipeline", "condition and split manifests complete; no model loaded")
        return

    if args.phase in {"pilot", "full"}:
        mode = args.phase
        if mode == "full" and config["collection"].get(
            "require_pilot_approval_for_full", True
        ) and not args.skip_pilot_approval:
            _require_pilot_approval(output, tasks)
        if mode == "full" and args.skip_pilot_approval:
            logger.note(
                "pilot_gate",
                "pilot approval explicitly bypassed for exploratory full collection",
            )
        expected_model_id = (
            DeterministicSmokeModel.model_id
            if args.model_free
            else args.model or config["model"]
        )
        cache_complete = bool(
            args.resume
            and _cached_collection_complete(
                output,
                tasks,
                specs_by_task,
                mode=mode,
                expected_model_id=expected_model_id,
            )
        )
        if cache_complete:
            logger.note(
                "collection_cache",
                "all semantic-pair files are complete and validated; skipping model load/inference",
            )
        else:
            if args.model_free:
                model = DeterministicSmokeModel()
            else:
                from models.hooked_qwen import HookedQwen

                model = HookedQwen.from_pretrained(
                    args.model or config["model"],
                    revision=args.revision,
                    local_files_only=not args.online,
                )
            _collect(
                model,
                config,
                output,
                tasks,
                mode=mode,
                smoke=args.smoke,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
                resume=args.resume,
                logger=logger,
            )
        complete, _specs, record_directory = _finalize(
            config, output, tasks, mode=mode, smoke=args.smoke, logger=logger
        )
        if complete:
            _validate_and_report(
                config,
                output,
                tasks,
                record_directory,
                mode=mode,
                smoke=args.smoke,
                model_free=args.model_free,
                logger=logger,
            )
        else:
            logger.note("pipeline", "collection shard complete; run finalize after all shards")
        logger.note("pipeline", f"{mode} battery phase complete")
        return

    mode = args.dataset
    locations = _locations(output, mode)
    if args.phase == "finalize":
        complete, _specs, record_directory = _finalize(
            config, output, tasks, mode=mode, smoke=args.smoke, logger=logger
        )
        if not complete:
            raise RuntimeError("cannot finalize until all semantic-pair shards exist")
        _validate_and_report(
            config,
            output,
            tasks,
            record_directory,
            mode=mode,
            smoke=args.smoke,
            model_free=args.model_free,
            logger=logger,
        )
        return
    if args.phase == "validate":
        _validate_and_report(
            config,
            output,
            tasks,
            locations["records"],
            mode=mode,
            smoke=args.smoke,
            model_free=args.model_free,
            logger=logger,
        )
        return
    if args.phase == "report":
        frames = {
            task: read_records_frame(locations["records"], task)
            for task in tasks
        }
        validation = output / "validation"
        manipulations = pd.read_csv(validation / "manipulation_checks.csv")
        label_bias = pd.read_csv(validation / "label_bias.csv")
        nondegeneracy = pd.read_csv(validation / "behavioral_non_degeneracy.csv")
        try:
            generate_figures(frames, manipulations, output / "figures")
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            logger.note(
                "behavioral_report",
                "matplotlib is unavailable; regenerating the text report only",
            )
        generate_report(
            _load_specs(output),
            frames,
            manipulations,
            label_bias,
            nondegeneracy,
            config,
            output,
            mode=mode,
            smoke=args.smoke,
            model_free=args.model_free,
        )
        logger.note("pipeline", f"report regenerated at {output / 'report.md'}")


if __name__ == "__main__":
    main()
