import csv

import pytest

from analysis.analyze_factorial_layerwise import audit_source_coverage


def _write_source(path):
    rows = [
        {
            "state_id": "state-1",
            "stop_payoff": stop,
            "continue_bonus": continued,
        }
        for stop in (-10, 0, 10, 20)
        for continued in (-10, 0, 10)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_source_coverage_requires_every_retained_factorial_cell(tmp_path):
    source = tmp_path / "factorial.csv"
    rows = _write_source(source)
    parsed = [
        {
            **row,
            "stop_payoff": int(row["stop_payoff"]),
            "continue_bonus": int(row["continue_bonus"]),
        }
        for row in rows
    ]

    audit = audit_source_coverage(parsed, str(source))

    assert audit["expected_cells"] == 12
    assert audit["observed_cells"] == 12
    assert audit["missing_cells"] == 0


def test_source_coverage_rejects_a_missing_replay_cell(tmp_path):
    source = tmp_path / "factorial.csv"
    rows = _write_source(source)

    with pytest.raises(ValueError, match="missing=1"):
        audit_source_coverage(rows[:-1], str(source))
