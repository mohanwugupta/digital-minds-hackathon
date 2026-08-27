import json

from computational_modeling.analysis.run_model_zoo import ProgressLogger


def test_progress_logger_flushes_events_and_section_timings(tmp_path):
    logger = ProgressLogger(tmp_path)
    logger.note("setup", "loading records", completed=3, total=10)
    with logger.section("fit", model="time"):
        logger.note("fit", "validation complete")

    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    timings = json.loads((tmp_path / "timings.json").read_text(encoding="utf-8"))
    assert any(event["message"] == "loading records" for event in events)
    assert timings["sections"][-1]["section"] == "fit"
    assert timings["sections"][-1]["status"] == "completed"
    assert timings["sections"][-1]["duration_seconds"] >= 0
