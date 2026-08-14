"""Download the primary and fallback Qwen checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_MODEL_ROOT = Path("/scratch/gpfs/JORDANAT/mg9965/models")


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    repo_id: str
    destination: Path


_CHECKPOINTS = (
    ("primary", "Qwen/Qwen3.5-4B", "Qwen--Qwen3.5-4B"),
    (
        "fallback",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen--Qwen3-4B-Instruct-2507",
    ),
)


def build_download_plan(
    model_root: Path, selection: str = "all"
) -> list[DownloadSpec]:
    """Return the requested checkpoints and their deterministic local paths."""
    valid_selections = {"all", "primary", "fallback"}
    if selection not in valid_selections:
        choices = ", ".join(sorted(valid_selections))
        raise ValueError(f"Unknown selection {selection!r}; expected one of: {choices}")

    return [
        DownloadSpec(key=key, repo_id=repo_id, destination=model_root / dirname)
        for key, repo_id, dirname in _CHECKPOINTS
        if selection in {"all", key}
    ]


def _verify_checkpoint(destination: Path) -> None:
    required_files = ("config.json", "tokenizer_config.json")
    missing = [
        name for name in required_files if not (destination / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Incomplete checkpoint at {destination}: missing {', '.join(missing)}"
        )
    if not any(destination.rglob("*.safetensors")):
        raise RuntimeError(
            f"Incomplete checkpoint at {destination}: no safetensors found"
        )


def _write_manifest(spec: DownloadSpec, requested_revision: str, commit: str) -> None:
    manifest = {
        "repo_id": spec.repo_id,
        "requested_revision": requested_revision,
        "resolved_commit": commit,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = spec.destination / ".download_manifest.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def download_checkpoint(
    spec: DownloadSpec,
    revision: str,
    max_workers: int,
) -> None:
    """Resolve, download, verify, and document one checkpoint."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; activate the project Conda environment "
            "or run: pip install huggingface_hub"
        ) from exc

    token = os.environ.get("HF_TOKEN")
    info = HfApi().model_info(spec.repo_id, revision=revision, token=token)
    commit = info.sha
    if not commit:
        raise RuntimeError(f"Hugging Face did not return a commit for {spec.repo_id}")

    spec.destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {spec.repo_id}@{commit}", flush=True)
    print(f"Destination: {spec.destination}", flush=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=commit,
        local_dir=spec.destination,
        token=token,
        max_workers=max_workers,
    )
    _verify_checkpoint(spec.destination)
    _write_manifest(spec, revision, commit)
    print(f"Verified {spec.repo_id}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=f"parent directory for model checkpoints (default: {DEFAULT_MODEL_ROOT})",
    )
    parser.add_argument(
        "--selection",
        choices=("all", "primary", "fallback"),
        default="all",
        help="download both checkpoints or only one of them",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch, tag, or commit to resolve (default: main)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="number of concurrent file downloads per checkpoint (default: 4)",
    )
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.model_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.model_root).free / (1024**3)
    print(f"Model root: {args.model_root}", flush=True)
    print(f"Free space before download: {free_gib:.1f} GiB", flush=True)

    for spec in build_download_plan(args.model_root, args.selection):
        download_checkpoint(spec, args.revision, args.max_workers)

    print("All requested checkpoints are ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
