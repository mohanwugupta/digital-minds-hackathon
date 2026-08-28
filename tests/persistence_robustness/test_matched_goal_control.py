import pytest

from cross_task.common import counterbalanced_mappings
from experiments.persistence_battery.collection import DeterministicSmokeModel
from experiments.persistence_robustness.matched_goal_control import (
    ENGAGE,
    INDEPENDENT,
    NUMERICAL_MATCH_FIELDS,
    PERSISTENT,
    PRIMARY,
    SECONDARY,
    MatchedGoalCondition,
    MatchedGoalEnvironment,
    collect_matched_sequence,
    yoked_action_schedule,
)


def _condition():
    return MatchedGoalCondition(8, 2, 0.6, 1, 6)


@pytest.mark.parametrize("version", [PRIMARY, SECONDARY])
def test_matched_conditions_receive_identical_latent_sequences_and_histories(version):
    persistent = MatchedGoalEnvironment(
        _condition(), 17, framing=PERSISTENT, version=version
    )
    independent = MatchedGoalEnvironment(
        _condition(), 17, framing=INDEPENDENT, version=version
    )
    assert persistent.success_uniforms == independent.success_uniforms
    for action in yoked_action_schedule(_condition(), 99, version):
        left, right = persistent.current_state(), independent.current_state()
        assert all(left[name] == right[name] for name in NUMERICAL_MATCH_FIELDS)
        assert persistent.step(action).outcome == independent.step(action).outcome
        if persistent.terminated:
            break


def test_goal_ids_are_constant_only_for_persistence():
    persistent = MatchedGoalEnvironment(
        _condition(), 3, framing=PERSISTENT, version=SECONDARY
    )
    independent = MatchedGoalEnvironment(
        _condition(), 3, framing=INDEPENDENT, version=SECONDARY
    )
    persistent_ids, independent_ids = [], []
    for _ in range(3):
        persistent_ids.append(persistent.current_state()["goal_id"])
        independent_ids.append(independent.current_state()["goal_id"])
        persistent.step(ENGAGE)
        independent.step(ENGAGE)
    assert len(set(persistent_ids)) == 1
    assert len(set(independent_ids)) == 3


def test_collection_replays_labels_and_never_exposes_future_fields_as_observables():
    rows = collect_matched_sequence(
        DeterministicSmokeModel(),
        _condition(),
        pair_id="matched-1",
        seed=8,
        action_seed=9,
        split="train",
    )
    assert {row["mapping_id"] for row in rows} == {
        mapping.mapping_id
        for mapping in counterbalanced_mappings(ENGAGE, "SKIP", ("X", "Y"))
    }
    grouped = {}
    for row in rows:
        grouped.setdefault(row["comparison_pair_id"], []).append(row)
    assert all(len(group) == 2 for group in grouped.values())
    assert all(group[0]["latent_sequence_id"] == group[1]["latent_sequence_id"] for group in grouped.values())
    assert all(row["history_equivalent"] for row in rows)
    assert all("subsequent" not in row["conversation_json"] for row in rows)

