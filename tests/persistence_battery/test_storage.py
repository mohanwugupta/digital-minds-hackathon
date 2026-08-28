from pathlib import Path
from dataclasses import asdict
import json

import pandas as pd
import pytest
import yaml

from analysis.run_persistence_battery import _cached_collection_complete, _locations
from experiments.persistence_battery.collection import build_specs
from experiments.persistence_battery.storage import (
    read_records_frame,
    resolve_records_path,
    write_records_frame,
)


def test_parquet_engine_failure_falls_back_to_compressed_csv(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            {"episode_id": "episode-1", "step": 0, "continued": True},
            {"episode_id": "episode-1", "step": 1, "continued": False},
        ]
    )

    def unavailable(*_args, **_kwargs):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", unavailable)
    result = write_records_frame(frame, tmp_path, "voluntary_waiting")

    assert result.format == "csv.gz"
    assert result.path == tmp_path / "voluntary_waiting.csv.gz"
    assert not (tmp_path / "voluntary_waiting.parquet").exists()
    restored = read_records_frame(tmp_path, "voluntary_waiting")
    pd.testing.assert_frame_equal(restored, frame)


def test_record_resolution_prefers_parquet_when_both_formats_exist(tmp_path):
    parquet = tmp_path / "sunk_cost.parquet"
    compressed_csv = tmp_path / "sunk_cost.csv.gz"
    parquet.touch()
    compressed_csv.touch()
    assert resolve_records_path(tmp_path, "sunk_cost") == parquet


def test_missing_record_file_lists_both_supported_locations(tmp_path):
    try:
        resolve_records_path(tmp_path, "information_sampling")
    except FileNotFoundError as error:
        message = str(error)
        assert "information_sampling.parquet" in message
        assert "information_sampling.csv.gz" in message
    else:
        raise AssertionError("missing records should fail")


def test_complete_validated_raw_cache_can_skip_model_loading(tmp_path):
    config = yaml.safe_load(Path("config/persistence_battery.yaml").read_text())
    task = "voluntary_waiting"
    specs = build_specs(config, task, mode="pilot", smoke=True)
    groups = {}
    for spec in specs:
        groups.setdefault(spec.pair_id, []).append(spec)
    assert not _cached_collection_complete(
        tmp_path,
        [task],
        {task: specs},
        mode="pilot",
        expected_model_id="model/revision",
    )

    pair_directory = _locations(tmp_path, "pilot")["pairs"] / task
    pair_directory.mkdir(parents=True)
    for pair_id, pair_specs in groups.items():
        condition = json.dumps(
            asdict(pair_specs[0].condition), sort_keys=True, separators=(",", ":")
        )
        (pair_directory / f"{pair_id}.json").write_text(
            json.dumps([{"condition": condition, "model_id": "model/revision"}])
        )
    assert _cached_collection_complete(
        tmp_path,
        [task],
        {task: specs},
        mode="pilot",
        expected_model_id="model/revision",
    )
    with pytest.raises(RuntimeError, match="condition/model"):
        _cached_collection_complete(
            tmp_path,
            [task],
            {task: specs},
            mode="pilot",
            expected_model_id="different/model",
        )
