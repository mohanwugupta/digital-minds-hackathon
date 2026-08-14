import json

from bandit.schemas import DecisionRecord, split_episode_ids


def test_episode_split_has_no_overlap_and_is_reproducible():
    ids = [f"episode-{i}" for i in range(20)]
    first = split_episode_ids(ids, seed=7)
    second = split_episode_ids(ids, seed=7)
    assert first == second
    assert not (set(first.train) & set(first.validation))
    assert not (set(first.train) & set(first.test))
    assert not (set(first.validation) & set(first.test))
    assert set(first.train + first.validation + first.test) == set(ids)


def test_decision_record_serializes_nested_fields_for_csv():
    record = DecisionRecord.minimal(episode_id="e1", state_id="e1:0", seed=1)
    row = record.to_row()
    assert json.loads(row["choice_history"]) == []
    assert json.loads(row["conversation"])[0]["role"] == "user"

