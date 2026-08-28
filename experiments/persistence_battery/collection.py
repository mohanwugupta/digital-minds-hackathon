"""Behavior-only Qwen collection with exact semantic label-map replays."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from cross_task.common import LabelMapping

from .base_environment import (
    BatteryConversation,
    COMMON_RECORD_FIELDS,
    task_episode_specs,
)
from .information_sampling import ANSWER_A, ANSWER_B
from .registry import TASKS


class DeterministicSmokeModel:
    """Model-free plumbing check; never a source of scientific results."""

    model_id = "deterministic/model-free-smoke"

    def binary_decision(self, messages, labels, *, positive_label, **_kwargs):
        payload = json.dumps(messages, sort_keys=True).encode()
        unit = int(hashlib.sha256(payload).hexdigest()[:8], 16) / 0xFFFFFFFF
        logit = -1.5 + 3.0 * unit
        probability = 1.0 / (1.0 + math.exp(-logit))
        negative_label = next(label for label in labels if label != positive_label)
        return {
            "positive_label": positive_label,
            "negative_label": negative_label,
            "p_positive": probability,
            "p_negative": 1.0 - probability,
            "choice_logit": logit,
            f"logit_{positive_label}": logit,
            f"logit_{negative_label}": 0.0,
            f"p_{positive_label}": probability,
            f"p_{negative_label}": 1.0 - probability,
            "p_action_mass_raw": 1.0,
            "top_token_is_action": True,
        }


def _score(model, messages, mapping):
    result = model.binary_decision(
        messages,
        mapping.labels,
        positive_label=mapping.positive_label,
        capture_hidden_states=False,
    )
    required = {"p_positive", "p_negative", "choice_logit"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"binary decision lacks fields: {sorted(missing)}")
    if "hidden_states" in result:
        raise ValueError("behavior-only battery must not retain hidden states")
    return dict(result)


def _final_answer(model, conversation, environment):
    mapping = LabelMapping(ANSWER_A, ANSWER_B, "A", "B")
    messages = conversation.snapshot() + [
        {"role": "user", "content": environment.final_answer_prompt()}
    ]
    metrics = _score(model, messages, mapping)
    answer = ANSWER_A if metrics["p_positive"] >= metrics["p_negative"] else ANSWER_B
    return {
        "final_answer": "A" if answer == ANSWER_A else "B",
        "final_answer_correct": answer.endswith(environment.condition.true_state),
        "p_answer_a": float(metrics["p_positive"]),
        "p_answer_b": float(metrics["p_negative"]),
    }


def collect_episode(
    model,
    definition,
    spec,
    task_config,
    *,
    replay_actions=None,
    expected_transitions=None,
):
    environment = definition.environment(spec.condition, spec.seed, task_config)
    conversation = BatteryConversation(environment, spec.mapping)
    rng = random.Random(spec.action_seed)
    records, actions, transitions = [], [], []
    while not environment.terminated:
        step = environment.step_index
        visible = conversation.snapshot()
        metrics = _score(model, visible, spec.mapping)
        if replay_actions is None:
            semantic_action = (
                definition.positive_action
                if rng.random() < float(metrics["p_positive"])
                else definition.negative_action
            )
            action_source = "sampled_primary_mapping"
        else:
            if step >= len(replay_actions):
                raise ValueError("semantic replay ended before the environment")
            semantic_action = replay_actions[step]
            action_source = "matched_semantic_replay"
        state = environment.current_state()
        transition = environment.step(semantic_action)
        conversation.record(semantic_action, transition)
        transition_signature = {
            "outcome": transition.outcome,
            "reward": transition.reward,
            "effort": transition.effort,
            "success": transition.success,
            "terminated": transition.terminated,
            "reason": transition.reason,
            "task_values": transition.task_values,
        }
        if expected_transitions is not None and transition_signature != expected_transitions[step]:
            raise ValueError("label reversal changed the semantic environment replay")
        is_persistence = definition.persistence
        excluded = {
            "p_positive",
            "p_negative",
            "choice_logit",
            "positive_label",
            "negative_label",
        }
        record = {
            "task": definition.name,
            "model_id": getattr(model, "model_id", "unknown"),
            "episode_id": spec.episode_id,
            "pair_id": spec.pair_id,
            "state_id": f"{spec.episode_id}:{step}",
            "step": step,
            "semantic_action": semantic_action,
            "continued": (
                semantic_action == definition.positive_action
                if is_persistence
                else None
            ),
            "terminated": transition.terminated,
            "p_continue": float(metrics["p_positive"]) if is_persistence else None,
            "p_disengage": float(metrics["p_negative"]) if is_persistence else None,
            "persistence_logit": float(metrics["choice_logit"]) if is_persistence else None,
            "label_mapping": spec.mapping.to_json(),
            "mapping_id": spec.mapping.mapping_id,
            "positive_semantic": definition.positive_action,
            "negative_semantic": definition.negative_action,
            "positive_label": spec.mapping.positive_label,
            "negative_label": spec.mapping.negative_label,
            "raw_label": spec.mapping.label_for(semantic_action),
            "condition": environment.condition_json(),
            "seed": spec.seed,
            "action_seed": spec.action_seed,
            "split": spec.split,
            "same_goal_across_steps": environment.same_goal_across_steps,
            "is_persistence_task": is_persistence,
            "action_source": action_source,
            "choice_logit": float(metrics["choice_logit"]),
            "p_positive_semantic": float(metrics["p_positive"]),
            "p_negative_semantic": float(metrics["p_negative"]),
            "conversation_json": json.dumps(visible, separators=(",", ":")),
            **environment.condition_dict(),
            **state,
            "subsequent_outcome": transition.outcome,
            "subsequent_reward": transition.reward,
            "subsequent_effort": transition.effort,
            "subsequent_success": transition.success,
            "termination_reason": transition.reason,
            **transition.task_values,
            **{key: value for key, value in metrics.items() if key not in excluded},
        }
        if definition.name == "information_sampling" and transition.terminated:
            record.update(_final_answer(model, conversation, environment))
        missing = set(COMMON_RECORD_FIELDS) - set(record)
        if missing:
            raise RuntimeError(f"common battery schema fields missing: {sorted(missing)}")
        records.append(record)
        actions.append(semantic_action)
        transitions.append(transition_signature)
    if replay_actions is not None and len(actions) != len(replay_actions):
        raise ValueError("semantic replay did not consume the full primary trajectory")
    return records, actions, transitions


def collect_pair(model, definition, pair_specs, task_config):
    pair_specs = sorted(pair_specs, key=lambda spec: spec.mapping.positive_label)
    if len(pair_specs) != 2 or pair_specs[0].pair_id != pair_specs[1].pair_id:
        raise ValueError("collection requires exactly two mappings for one semantic pair")
    primary, actions, transitions = collect_episode(
        model, definition, pair_specs[0], task_config
    )
    replay, replayed_actions, _replayed_transitions = collect_episode(
        model,
        definition,
        pair_specs[1],
        task_config,
        replay_actions=actions,
        expected_transitions=transitions,
    )
    if replayed_actions != actions:
        raise RuntimeError("counterbalanced episode changed semantic actions")
    return primary + replay


def build_specs(config, task, *, mode, smoke=False):
    definition = TASKS[task]
    task_config = config["tasks"][task]
    conditions = definition.conditions(task_config)
    task_offset = list(TASKS).index(task) * 10_000
    if smoke:
        recorded = 2 * int(config["smoke"]["semantic_pairs_per_task"])
    elif mode == "pilot":
        recorded = (
            2
            * len(conditions)
            * int(config["collection"]["pilot_semantic_pairs_per_cell"])
        )
    elif mode == "full":
        recorded = int(config["collection"]["recorded_episodes_per_task"])
    else:
        raise ValueError(f"unknown collection mode: {mode}")
    return list(
        task_episode_specs(
            task,
            conditions,
            definition.positive_action,
            definition.negative_action,
            recorded_episodes=recorded,
            base_seed=int(config["base_seed"]) + task_offset,
            split_seed=int(config["split_seed"]) + task_offset,
            labels=tuple(config["response_labels"]),
        )
    )


def write_pair(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_pair(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
