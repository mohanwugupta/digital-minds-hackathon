"""Collect the PRD 2.5 yoked goal-continuity behavioral control."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil

import pandas as pd
import yaml

from computational_modeling.analysis.run_model_zoo import ProgressLogger
from experiments.persistence_battery.base_environment import assign_pair_splits
from experiments.persistence_battery.collection import DeterministicSmokeModel
from experiments.persistence_battery.storage import write_records_frame
from experiments.persistence_robustness.matched_goal_control import (
    collect_matched_sequence,
    factorial_conditions,
)
from experiments.runtime import save_run_metadata


def _load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _inventory(config, mode, smoke):
    matched = config["matched_control"]
    count = int(
        config["smoke"]["matched_semantic_pairs"]
        if smoke
        else matched["pilot_semantic_pairs"]
        if mode == "pilot"
        else matched["semantic_pairs"]
    )
    conditions = factorial_conditions(matched["conditions"])
    shuffled = list(conditions)
    random.Random(int(config["base_seed"]) + 90_000).shuffle(shuffled)
    pairs = []
    for index in range(count):
        pair_id = f"matched-goal-pair-{int(config['base_seed']) + index:07d}"
        pairs.append(
            {
                "pair_id": pair_id,
                "condition": shuffled[index % len(shuffled)],
                "seed": int(config["base_seed"]) + 90_000 + index,
                "action_seed": int(config["base_seed"]) + 190_000 + index,
            }
        )
    splits = assign_pair_splits(
        [pair["pair_id"] for pair in pairs], int(config["split_seed"]) + 90_000
    )
    for pair in pairs:
        pair["split"] = splits[pair["pair_id"]]
    return pairs


def _cache_path(root, mode, pair_id):
    return root / "cache" / f"matched_goal_{mode}" / f"{pair_id}.json"


def _write_cache(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_cache(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_cache(payload, pair, model_id):
    expected = asdict(pair["condition"])
    if payload.get("condition") != expected or payload.get("model_id") != model_id:
        raise RuntimeError(
            "matched-control cache differs from the current condition/model; use a new run ID"
        )
    return payload["records"]


def _write_inventory(root, pairs, mode, smoke):
    directory = root / "matched_control"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "pair_id": pair["pair_id"],
            "split": pair["split"],
            "seed": pair["seed"],
            "action_seed": pair["action_seed"],
            **asdict(pair["condition"]),
        }
        for pair in pairs
    ]
    pd.DataFrame(rows).to_csv(directory / "condition_inventory.csv", index=False)
    (directory / "collection_manifest.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "smoke": bool(smoke),
                "semantic_pairs": len(pairs),
                "label_mappings": 2,
                "goal_framings": 2,
                "versions": ["absorbing_primary", "advancing_secondary"],
                "history_source": "exogenous_yoked_replay",
                "behavioral_target": "p_engage_at_matched_state",
                "future_fields_are_model_invisible": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _finalize(root, pairs, mode, model_id, logger):
    records, missing = [], []
    for pair in pairs:
        path = _cache_path(root, mode, pair["pair_id"])
        if not path.exists():
            missing.append(path)
            continue
        records.extend(_validate_cache(_read_cache(path), pair, model_id))
    if missing:
        logger.note("matched_finalize", f"waiting for {len(missing)}/{len(pairs)} pair caches")
        return False
    frame = pd.DataFrame(records).sort_values(
        ["pair_id", "version", "mapping_id", "step", "framing"]
    )
    result = write_records_frame(frame, root / "matched_control", "paired_records")
    manifest = {
        "path": str(result.path.relative_to(root)),
        "format": result.format,
        "states": len(frame),
        "semantic_pairs": len(pairs),
        "comparison_states": int(frame.comparison_pair_id.nunique()),
        "framing_rows_balanced": bool(frame.groupby("framing").size().nunique() == 1),
        "all_histories_equivalent": bool(frame.history_equivalent.all()),
    }
    (root / "matched_control/records_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result.parquet_error:
        logger.note("matched_finalize", "Parquet unavailable; wrote compressed CSV")
    logger.note("matched_finalize", f"wrote {len(frame)} matched framing rows to {result.path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/persistence_robustness_v1.yaml")
    parser.add_argument("--run-id", default="robustness_v1")
    parser.add_argument("--phase", choices=("inventory", "collect", "finalize"), default="collect")
    parser.add_argument("--dataset", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-free", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.model_free and not args.smoke:
        raise ValueError("--model-free is restricted to --smoke")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must lie in [0, num-shards)")

    config_path = Path(args.config)
    config = _load(config_path)
    root = Path(config["output_root"]) / args.run_id
    if root.exists() and not args.resume:
        raise FileExistsError(f"output exists: {root}; use --resume or a new run ID")
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "config.yaml").exists():
        shutil.copy2(config_path, root / "config.yaml")
    logger = ProgressLogger(root, label="matched-goal-control")
    pairs = _inventory(config, args.dataset, args.smoke)
    _write_inventory(root, pairs, args.dataset, args.smoke)
    model_id = (
        DeterministicSmokeModel.model_id
        if args.model_free
        else args.model or config["model"]
    )
    save_run_metadata(
        str(root / "matched_control/run_metadata.json"),
        {
            **vars(args),
            "protocol_version": config["protocol_version"],
            "model": model_id,
            "behavior_only": True,
            "capture_hidden_states": False,
        },
    )
    logger.note(
        "matched_pipeline",
        f"phase={args.phase}; dataset={args.dataset}; pairs={len(pairs)}; shard={args.shard_index}/{args.num_shards}",
    )
    if args.phase == "inventory":
        return
    if args.phase == "finalize":
        if not _finalize(root, pairs, args.dataset, model_id, logger):
            raise RuntimeError("matched-control shards are incomplete")
        return

    model = None
    for index, pair in enumerate(pairs):
        if index % args.num_shards != args.shard_index:
            continue
        path = _cache_path(root, args.dataset, pair["pair_id"])
        if args.resume and path.exists():
            _validate_cache(_read_cache(path), pair, model_id)
            continue
        if model is None:
            if args.model_free:
                model = DeterministicSmokeModel()
            else:
                from models.hooked_qwen import HookedQwen

                model = HookedQwen.from_pretrained(
                    model_id,
                    revision=args.revision,
                    local_files_only=not args.online,
                )
        records = collect_matched_sequence(
            model,
            pair["condition"],
            pair_id=pair["pair_id"],
            seed=pair["seed"],
            action_seed=pair["action_seed"],
            split=pair["split"],
            labels=tuple(config["response_labels"]),
        )
        for row in records:
            row["model_id"] = model_id
        _write_cache(
            path,
            {
                "condition": asdict(pair["condition"]),
                "model_id": model_id,
                "records": records,
            },
        )
        if index + 1 == len(pairs) or (index + 1) % 10 == 0:
            logger.note("matched_collect", f"pair {index + 1}/{len(pairs)}")
    if args.num_shards == 1:
        _finalize(root, pairs, args.dataset, model_id, logger)
    else:
        logger.note("matched_pipeline", "shard complete; run --phase finalize after all shards")


if __name__ == "__main__":
    main()

