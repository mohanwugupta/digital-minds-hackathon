import json
import sys

from experiments.smoke_persistence_discovery import main


def test_persistence_discovery_smoke_writes_complete_outputs(tmp_path, monkeypatch):
    output_dir = tmp_path / "search"
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke_persistence_discovery", "--output-dir", str(output_dir)],
    )

    main()

    summary = json.loads(
        (output_dir / "persistence_discovery_summary.json").read_text(
            encoding="utf-8"
        )
    )
    provenance_config = summary["provenance"]["config"]
    assert provenance_config["protocol_version"] == (
        "task_general_persistence_discovery_smoke_v1"
    )
    assert provenance_config["model"] == "synthetic"
