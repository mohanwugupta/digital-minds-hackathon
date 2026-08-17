import pytest

torch = pytest.importorskip("torch")

from cross_task.common import counterbalanced_mappings
from cross_task.control import LEFT_GREATER, RIGHT_GREATER
from cross_task.foraging import LEAVE, STAY, ForagingCondition
from cross_task.solvability import GIVE_UP, TRY_AGAIN, SolvabilityCondition
from cross_task.terminality import END, PROCEED
from experiments.collect_cross_task_activations import (
    collect_control_episode,
    collect_foraging_episode,
    collect_solvability_episode,
    collect_terminality_episode,
)
from models.hooked_qwen import binary_choice_metrics


class LeaveImmediately:
    model_id = "mock/binary"

    def binary_decision(
        self,
        _messages,
        labels,
        *,
        positive_label,
        capture_hidden_states=False,
        **_kwargs,
    ):
        negative = next(label for label in labels if label != positive_label)
        result = binary_choice_metrics(
            {positive_label: -10.0, negative: 10.0},
            positive_label=positive_label,
        )
        result["p_action_mass_raw"] = 1.0
        result["top_token_is_action"] = True
        if capture_hidden_states:
            result["hidden_states"] = [torch.tensor([1.0, -1.0]) for _ in range(2)]
        return result


def test_foraging_collection_keeps_semantic_and_raw_choices_separate():
    mapping = counterbalanced_mappings(STAY, LEAVE)[1]
    artifact = collect_foraging_episode(
        LeaveImmediately(),
        "pair-1",
        ForagingCondition(0.5, 0.1, 2, 1),
        mapping,
        seed=4,
        action_seed=9,
    )
    record = artifact["records"][0]

    assert record["semantic_choice"] == LEAVE
    assert record["raw_label"] == mapping.label_for(LEAVE)
    assert record["positive_semantic"] == STAY
    assert record["persistence_logit"] < 0
    assert artifact["activations"].shape == (1, 2, 2)


def test_control_collection_has_no_persistence_semantics():
    mapping = counterbalanced_mappings(LEFT_GREATER, RIGHT_GREATER)[0]
    artifact = collect_control_episode(
        LeaveImmediately(), "control-pair", 8, 3, mapping, seed=5
    )
    record = artifact["records"][0]

    assert artifact["task"] == "binary_control"
    assert "persistence_logit" not in record
    assert record["correct_choice"] == LEFT_GREATER
    assert record["target_logit"] == record["choice_logit"]


def test_solvability_collection_uses_distinct_labels_and_semantic_target():
    mapping = counterbalanced_mappings(TRY_AGAIN, GIVE_UP, ("M", "N"))[1]
    artifact = collect_solvability_episode(
        LeaveImmediately(),
        "solvability-pair",
        SolvabilityCondition(0.5, 1, 2),
        mapping,
        seed=4,
        action_seed=9,
    )
    record = artifact["records"][0]

    assert artifact["task"] == "solvability"
    assert record["semantic_choice"] == GIVE_UP
    assert record["raw_label"] == mapping.label_for(GIVE_UP)
    assert record["positive_semantic"] == TRY_AGAIN
    assert record["persistence_logit"] < 0
    assert set(mapping.labels) == {"M", "N"}


def test_terminality_collection_is_rule_determined_not_persistence_labeled():
    mapping = counterbalanced_mappings(PROCEED, END, ("M", "N"))[0]
    artifact = collect_terminality_episode(
        LeaveImmediately(), "terminality-pair", 8, mapping, seed=5
    )
    record = artifact["records"][0]

    assert artifact["task"] == "terminality_control"
    assert "persistence_logit" not in record
    assert record["correct_choice"] == PROCEED
    assert record["terminality_logit"] == record["choice_logit"]
