"""Shared deterministic state, history, prompting, and split infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Any, Iterable

from cross_task.common import LabelMapping, counterbalanced_mappings


COMMON_RECORD_FIELDS = (
    "task",
    "episode_id",
    "pair_id",
    "state_id",
    "step",
    "semantic_action",
    "continued",
    "terminated",
    "p_continue",
    "p_disengage",
    "persistence_logit",
    "previous_action",
    "previous_outcome",
    "success_streak",
    "failure_streak",
    "elapsed_steps",
    "cumulative_effort",
    "cumulative_reward",
    "current_continue_cost",
    "current_outside_option",
    "current_progress",
    "current_success_evidence",
    "label_mapping",
    "condition",
    "seed",
    "split",
)


@dataclass(frozen=True)
class Transition:
    action: str
    outcome: float | None
    reward: float
    effort: float
    success: bool | None
    terminated: bool
    reason: str | None
    task_values: dict[str, Any]


@dataclass(frozen=True)
class EpisodeSpec:
    task: str
    pair_id: str
    condition: Any
    mapping: LabelMapping
    seed: int
    action_seed: int
    split: str

    @property
    def episode_id(self) -> str:
        return f"{self.pair_id}-{self.mapping.mapping_id}"


class SemanticHistory:
    """Track only quantities that genuinely exist for an environment."""

    def __init__(self, *, elapsed_steps=0, cumulative_effort=0.0, cumulative_reward=0.0):
        self.actions: list[str] = []
        self.outcomes: list[float | None] = []
        self.elapsed_steps = int(elapsed_steps)
        self.cumulative_effort = float(cumulative_effort)
        self.cumulative_reward = float(cumulative_reward)
        self.progress: float | None = None

    @staticmethod
    def _streak(outcomes, positive):
        length = 0
        for value in reversed(outcomes):
            if value is None or (float(value) > 0) != bool(positive):
                break
            length += 1
        return length

    def record(self, action, *, outcome, effort, reward, progress=None):
        self.actions.append(str(action))
        self.outcomes.append(None if outcome is None else float(outcome))
        self.elapsed_steps += 1
        self.cumulative_effort += float(effort)
        self.cumulative_reward += float(reward)
        if progress is not None:
            self.progress = float(progress)

    def state(self):
        return {
            "previous_action": self.actions[-1] if self.actions else None,
            "previous_outcome": self.outcomes[-1] if self.outcomes else None,
            "success_streak": self._streak(self.outcomes, True),
            "failure_streak": self._streak(self.outcomes, False),
            "elapsed_steps": self.elapsed_steps,
            "cumulative_effort": self.cumulative_effort,
            "cumulative_reward": self.cumulative_reward,
            "current_progress": self.progress,
        }


class BasePersistenceEnvironment:
    task = "abstract"
    continue_action = "CONTINUE"
    disengage_action = "DISENGAGE"
    same_goal_across_steps = True
    is_persistence_task = True

    def __init__(self, condition, seed):
        self.condition = condition
        self.seed = int(seed)
        self.history = SemanticHistory()
        self._decision_count = 0
        self.terminated = False
        self.termination_reason: str | None = None

    @property
    def step_index(self):
        return self._decision_count

    def _ensure_active(self):
        if self.terminated:
            raise RuntimeError("episode has already terminated")

    def _finish_transition(
        self,
        action,
        *,
        outcome,
        reward,
        effort,
        success,
        terminated,
        reason,
        progress=None,
        task_values=None,
    ):
        self.history.record(
            action,
            outcome=outcome,
            effort=effort,
            reward=reward,
            progress=progress,
        )
        self._decision_count += 1
        self.terminated = bool(terminated)
        self.termination_reason = reason if terminated else None
        return Transition(
            action=str(action),
            outcome=None if outcome is None else float(outcome),
            reward=float(reward),
            effort=float(effort),
            success=success,
            terminated=self.terminated,
            reason=self.termination_reason,
            task_values={} if task_values is None else dict(task_values),
        )

    def condition_dict(self):
        return asdict(self.condition)

    def condition_json(self):
        return json.dumps(self.condition_dict(), sort_keys=True, separators=(",", ":"))

    def current_state(self):
        raise NotImplementedError

    def initial_prompt(self, mapping):
        raise NotImplementedError

    def feedback_prompt(self, transition, mapping):
        raise NotImplementedError


class BatteryConversation:
    def __init__(self, environment, mapping):
        self.environment = environment
        self.mapping = mapping
        self.messages = [
            {"role": "user", "content": environment.initial_prompt(mapping)}
        ]

    def snapshot(self):
        return [dict(message) for message in self.messages]

    def record(self, semantic_action, transition):
        self.messages.append(
            {
                "role": "assistant",
                "content": self.mapping.label_for(semantic_action),
            }
        )
        if not transition.terminated:
            self.messages.append(
                {
                    "role": "user",
                    "content": self.environment.feedback_prompt(
                        transition, self.mapping
                    ),
                }
            )


def choice_block(mapping, positive, negative, positive_text, negative_text):
    return (
        "Choose one:\n"
        f"{mapping.label_for(positive)} = {positive_text}\n"
        f"{mapping.label_for(negative)} = {negative_text}\n\n"
        f"Respond with only {mapping.labels[0]} or {mapping.labels[1]}."
    )


def condition_id(task, condition):
    payload = json.dumps(asdict(condition), sort_keys=True, separators=(",", ":"))
    return f"{task}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def assign_pair_splits(pair_ids: Iterable[str], seed: int):
    pair_ids = sorted(set(str(value) for value in pair_ids))
    if len(pair_ids) < 3:
        raise ValueError("at least three pairs are required for train/validation/test")
    random.Random(int(seed)).shuffle(pair_ids)
    # Reserve at least one whole semantic pair for every split, including the
    # three-pair laptop smoke.  Larger runs retain the requested 70/15/15-ish
    # allocation without ever allowing train to consume validation or test.
    train_end = min(len(pair_ids) - 2, max(1, int(len(pair_ids) * 0.70)))
    validation_count = min(
        len(pair_ids) - train_end - 1,
        max(1, int(len(pair_ids) * 0.15)),
    )
    validation_end = train_end + validation_count
    return {
        pair_id: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, pair_id in enumerate(pair_ids)
    }


def build_episode_specs(
    task,
    conditions,
    *,
    recorded_episodes,
    base_seed,
    split_seed,
    labels=("X", "Y"),
):
    recorded_episodes = int(recorded_episodes)
    if recorded_episodes < 6 or recorded_episodes % 2:
        raise ValueError("recorded episodes must be even and at least six")
    conditions = list(conditions)
    if not conditions:
        raise ValueError("at least one factorial condition is required")
    ordered = list(conditions)
    random.Random(int(base_seed)).shuffle(ordered)
    pairs = []
    for pair_index in range(recorded_episodes // 2):
        condition = ordered[pair_index % len(ordered)]
        pair_id = f"{task}-pair-{int(base_seed) + pair_index:07d}"
        pairs.append((pair_id, condition, int(base_seed) + pair_index))
    split = assign_pair_splits([pair_id for pair_id, _condition, _seed in pairs], split_seed)
    mappings = counterbalanced_mappings(
        "PLACEHOLDER_POSITIVE", "PLACEHOLDER_NEGATIVE", tuple(labels)
    )
    return pairs, split, mappings


def task_episode_specs(
    task,
    conditions,
    positive,
    negative,
    *,
    recorded_episodes,
    base_seed,
    split_seed,
    labels=("X", "Y"),
):
    pairs, split, _unused = build_episode_specs(
        task,
        conditions,
        recorded_episodes=recorded_episodes,
        base_seed=base_seed,
        split_seed=split_seed,
        labels=labels,
    )
    mappings = counterbalanced_mappings(positive, negative, tuple(labels))
    for pair_id, condition, seed in pairs:
        for mapping_index, mapping in enumerate(mappings):
            yield EpisodeSpec(
                task=task,
                pair_id=pair_id,
                condition=condition,
                mapping=mapping,
                seed=seed,
                action_seed=seed + 1_000_000,
                split=split[pair_id],
            )
