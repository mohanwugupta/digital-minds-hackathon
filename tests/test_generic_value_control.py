from cross_task.generic_value import (
    LEFT_VOUCHER,
    RIGHT_VOUCHER,
    episode_conditions,
    voucher_prompt,
)


def test_generic_value_prompt_is_one_shot_and_has_no_persistence_semantics():
    condition = list(episode_conditions(2, 17, labels=("U", "V")))[0]
    _pair, left, right, mapping, _seed = condition
    prompt = voucher_prompt(left, right, mapping)
    assert "single one-shot choice" in prompt
    assert "continue" not in prompt.lower()
    assert "stop" not in prompt.lower()
    assert mapping.label_for(LEFT_VOUCHER) in prompt
    assert mapping.label_for(RIGHT_VOUCHER) in prompt


def test_generic_value_conditions_are_exactly_label_counterbalanced():
    conditions = list(episode_conditions(12, 29, labels=("U", "V")))
    assert len(conditions) == 12
    by_pair = {}
    for pair_id, left, right, mapping, seed in conditions:
        by_pair.setdefault(pair_id, []).append((left, right, mapping, seed))
        assert left != right
    assert all(len(rows) == 2 for rows in by_pair.values())
    for rows in by_pair.values():
        assert rows[0][0:2] == rows[1][0:2]
        assert rows[0][3] == rows[1][3]
        assert {
            row[2].label_for(LEFT_VOUCHER) for row in rows
        } == {"U", "V"}

