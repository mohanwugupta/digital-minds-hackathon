"""Shared artifact, provenance, and resumability helpers."""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(config: Dict[str, Any], model=None) -> Dict[str, Any]:
    model_config = getattr(getattr(model, "model", None), "config", None)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "model_identifier": getattr(model, "model_id", config.get("model", "unknown")),
        "model_revision": getattr(model_config, "_commit_hash", None),
        "transformers_version": _version("transformers"),
        "torch_version": _version("torch"),
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
    }


def save_run_metadata(path: str, config: Dict[str, Any], model=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(run_metadata(config, model), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(value, path: str) -> None:
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)
