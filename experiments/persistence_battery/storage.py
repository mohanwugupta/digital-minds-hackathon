"""Durable behavior-record storage with a dependency-light fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RecordWriteResult:
    path: Path
    format: str
    parquet_error: str | None = None


def _locations(directory, task):
    directory = Path(directory)
    return (
        directory / f"{task}.parquet",
        directory / f"{task}.csv.gz",
    )


def resolve_records_path(directory, task):
    """Resolve preferred Parquet or the compressed-CSV portability fallback."""

    parquet, compressed_csv = _locations(directory, task)
    if parquet.exists():
        return parquet
    if compressed_csv.exists():
        return compressed_csv
    raise FileNotFoundError(
        f"no records found for {task!r}; expected {parquet} or {compressed_csv}"
    )


def read_records_frame(directory, task):
    path = resolve_records_path(directory, task)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="gzip")


def write_records_frame(frame, directory, task):
    """Write Parquet atomically, falling back only when its engine is absent.

    CSV is compressed and lossless for the scalar/JSON-string behavioral
    schema.  A genuine Parquet serialization error is not hidden: only the
    optional-engine ImportError activates the fallback.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parquet, compressed_csv = _locations(directory, task)
    parquet_tmp = parquet.with_suffix(".parquet.tmp")
    try:
        frame.to_parquet(parquet_tmp, index=False)
    except ImportError as error:
        if parquet_tmp.exists():
            parquet_tmp.unlink()
        csv_tmp = compressed_csv.with_suffix(".csv.gz.tmp")
        frame.to_csv(csv_tmp, index=False, compression="gzip")
        csv_tmp.replace(compressed_csv)
        if parquet.exists():
            parquet.unlink()
        return RecordWriteResult(
            compressed_csv,
            "csv.gz",
            parquet_error=f"{type(error).__name__}: {error}",
        )
    parquet_tmp.replace(parquet)
    if compressed_csv.exists():
        compressed_csv.unlink()
    return RecordWriteResult(parquet, "parquet")
