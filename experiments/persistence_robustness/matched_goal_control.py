"""Yoked ongoing-goal versus independent-goal sequential decisions.

The action/outcome history is assigned exogenously and replayed in both
framings.  The scientific response is the model's probability of ENGAGE at
each matched state, not the action used to advance the yoked history.  This
keeps numerical histories identical without fabricating post-termination
states in an absorbing behavioral trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import product
import random

from cross_task.common import counterbalanced_mappings
from experiments.persistence_battery.base_environment import (
    BatteryConversation,
    BasePersistenceEnvironment,
    SemanticHistory,
    choice_block,
)


ENGAGE = "ENGAGE"
SKIP = "SKIP"
PERSISTENT = "persistent_goal"
INDEPENDENT = "independent_goals"
PRIMARY = "absorbing_primary"
SECONDARY = "advancing_secondary"

NUMERICAL_MATCH_FIELDS = (
    "step",
    "current_continue_cost",
    "current_outside_option",
    "current_progress",
    "current_success_evidence",
    "previous_history_action",
    "previous_outcome",
    "success_streak",
    "failure_streak",
    "elapsed_steps",
    "cumulative_effort",
    "cumulative_reward",
)

OBSERVABLE_MATCHED_FIELDS = (
    *NUMERICAL_MATCH_FIELDS,
    "goal_continuity",
    "goal_id",
    "same_goal_across_steps",
)


@dataclass(frozen=True)
class MatchedGoalCondition:
    reward_magnitude: int
    effort_cost: int
    success_probability: float
    outside_option: int
    horizon: int

    def __post_init__(self):
        if self.reward_magnitude <= 0 or self.effort_cost < 0:
            raise ValueError("invalid reward/cost")
        if not 0 <= self.success_probability <= 1:
            raise ValueError("success probability must lie in [0, 1]")
        if self.horizon < 3:
            raise ValueError("matched sequence requires at least three rounds")


def factorial_conditions(config):
    return [
        MatchedGoalCondition(
            int(reward),
            int(cost),
            float(probability),
            int(outside),
            int(config["horizon"]),
        )
        for reward, cost, probability, outside in product(
            config["reward_magnitudes"],
            config["effort_costs"],
            config["success_probabilities"],
            config["outside_options"],
        )
    ]


def latent_sequence_id(condition, seed):
    payload = json.dumps(
        {"condition": asdict(condition), "seed": int(seed)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


class MatchedGoalEnvironment(BasePersistenceEnvironment):
    """One numerical process rendered with or without goal continuity."""

    task = "matched_goal_control"
    continue_action = ENGAGE
    disengage_action = SKIP

    def __init__(self, condition, seed, *, framing, version):
        super().__init__(condition, seed)
        if framing not in {PERSISTENT, INDEPENDENT}:
            raise ValueError(f"unknown goal framing: {framing}")
        if version not in {PRIMARY, SECONDARY}:
            raise ValueError(f"unknown matched-control version: {version}")
        self.framing = framing
        self.version = version
        self.same_goal_across_steps = framing == PERSISTENT
        self.is_persistence_task = framing == PERSISTENT
        self.history = SemanticHistory()
        rng = random.Random(int(seed) * 2 + 25_001)
        self.success_uniforms = [rng.random() for _ in range(condition.horizon)]

    @property
    def goal_id(self):
        return "project-A" if self.same_goal_across_steps else f"project-{self.step_index + 1}"

    def current_state(self):
        history = self.history.state()
        return {
            "step": self.step_index,
            "current_continue_cost": float(self.condition.effort_cost),
            "current_outside_option": float(self.condition.outside_option),
            "current_progress": self.step_index / self.condition.horizon,
            "current_success_evidence": float(self.condition.success_probability),
            "previous_history_action": history["previous_action"],
            "previous_outcome": history["previous_outcome"],
            "success_streak": history["success_streak"],
            "failure_streak": history["failure_streak"],
            "elapsed_steps": history["elapsed_steps"],
            "cumulative_effort": history["cumulative_effort"],
            "cumulative_reward": history["cumulative_reward"],
            "goal_continuity": self.framing,
            "goal_id": self.goal_id,
            "same_goal_across_steps": self.same_goal_across_steps,
        }

    def step(self, action):
        self._ensure_active()
        action = str(action).upper()
        if action not in {ENGAGE, SKIP}:
            raise ValueError(f"invalid matched-control action: {action}")
        if action == ENGAGE:
            success = (
                self.success_uniforms[self.step_index]
                < self.condition.success_probability
            )
            effort = self.condition.effort_cost
            reward = (
                self.condition.reward_magnitude if success else 0
            ) - effort
            outcome = reward
        else:
            success = None
            effort = 0
            reward = float(self.condition.outside_option)
            outcome = reward
        reaches_horizon = self.step_index + 1 >= self.condition.horizon
        absorbing_skip = (
            self.version == PRIMARY
            and self.framing == PERSISTENT
            and action == SKIP
        )
        terminated = reaches_horizon or absorbing_skip
        return self._finish_transition(
            action,
            outcome=outcome,
            reward=reward,
            effort=effort,
            success=success,
            terminated=terminated,
            reason=(
                "abandoned_goal"
                if absorbing_skip
                else "sequence_complete"
                if reaches_horizon
                else None
            ),
            progress=(self.step_index + 1) / self.condition.horizon,
            task_values={"latent_sequence_id": latent_sequence_id(self.condition, self.seed)},
        )

    def _choice(self, mapping):
        if self.framing == PERSISTENT:
            positive = "ENGAGE with the same ongoing project"
            negative = "SKIP further work on the ongoing project"
        else:
            positive = "ENGAGE with this new independent project"
            negative = "SKIP this new independent project"
        return choice_block(mapping, ENGAGE, SKIP, positive, negative)

    def initial_prompt(self, mapping):
        continuity = (
            "All decisions concern the same ongoing project."
            if self.same_goal_across_steps
            else "Every decision concerns a new unrelated project."
        )
        terminality = (
            "Skipping ends the ongoing project."
            if self.version == PRIMARY and self.same_goal_across_steps
            else "Either choice advances to the next scheduled decision."
        )
        return (
            f"{continuity} {terminality} There are at most {self.condition.horizon} decisions. "
            f"Engaging costs {self.condition.effort_cost} points and succeeds with probability "
            f"{self.condition.success_probability:.0%}, paying {self.condition.reward_magnitude} points. "
            f"Skipping pays {self.condition.outside_option} points.\n\n"
            + self._choice(mapping)
        )

    def feedback_prompt(self, transition, mapping):
        outcome = (
            "The scheduled engagement succeeded."
            if transition.success is True
            else "The scheduled engagement failed."
            if transition.success is False
            else "The scheduled opportunity was skipped."
        )
        continuity = (
            "The same ongoing project remains."
            if self.same_goal_across_steps
            else "A new unrelated project is now available."
        )
        return f"{outcome} {continuity}\n\n{self._choice(mapping)}"


def yoked_action_schedule(condition, action_seed, version):
    """Create a reproducible history with action variation and usable depth."""

    rng = random.Random(int(action_seed))
    if version == PRIMARY:
        stop = rng.randint(2, condition.horizon - 1)
        return (ENGAGE,) * stop + (SKIP,)
    actions = [ENGAGE if rng.random() < 0.65 else SKIP for _ in range(condition.horizon)]
    actions[0] = ENGAGE
    if SKIP not in actions:
        actions[-1] = SKIP
    return tuple(actions)


def assert_numerical_history_match(left, right):
    mismatched = [name for name in NUMERICAL_MATCH_FIELDS if left.get(name) != right.get(name)]
    if mismatched:
        raise RuntimeError(f"matched histories diverged: {mismatched}")


def _score(model, messages, mapping):
    result = model.binary_decision(
        messages,
        mapping.labels,
        positive_label=mapping.positive_label,
        capture_hidden_states=False,
    )
    if "hidden_states" in result:
        raise ValueError("matched behavioral control must not retain hidden states")
    return result


def collect_matched_sequence(
    model,
    condition,
    *,
    pair_id,
    seed,
    action_seed,
    split,
    labels=("X", "Y"),
):
    """Collect both framings, versions, and label mappings for one sequence."""

    rows = []
    for version in (PRIMARY, SECONDARY):
        schedule = yoked_action_schedule(condition, action_seed, version)
        for mapping in counterbalanced_mappings(ENGAGE, SKIP, tuple(labels)):
            environments = {
                framing: MatchedGoalEnvironment(
                    condition, seed, framing=framing, version=version
                )
                for framing in (PERSISTENT, INDEPENDENT)
            }
            conversations = {
                framing: BatteryConversation(environment, mapping)
                for framing, environment in environments.items()
            }
            for step, history_action in enumerate(schedule):
                states = {
                    framing: environment.current_state()
                    for framing, environment in environments.items()
                }
                assert_numerical_history_match(states[PERSISTENT], states[INDEPENDENT])
                transitions = {}
                for framing in (PERSISTENT, INDEPENDENT):
                    environment = environments[framing]
                    conversation = conversations[framing]
                    visible = conversation.snapshot()
                    metrics = _score(model, visible, mapping)
                    transition = environment.step(history_action)
                    transitions[framing] = transition
                    conversation.record(history_action, transition)
                    task = f"matched_goal_{'persistence' if framing == PERSISTENT else 'independent'}"
                    episode_id = f"{pair_id}-{version}-{framing}-{mapping.mapping_id}"
                    rows.append(
                        {
                            "task": task,
                            "version": version,
                            "framing": framing,
                            "episode_id": episode_id,
                            "pair_id": f"{pair_id}-{version}-{framing}",
                            "comparison_pair_id": f"{pair_id}-{version}-{mapping.mapping_id}:{step}",
                            "state_id": f"{episode_id}:{step}",
                            "step": step,
                            "split": split,
                            "seed": int(seed),
                            "action_seed": int(action_seed),
                            "latent_sequence_id": latent_sequence_id(condition, seed),
                            "mapping_id": mapping.mapping_id,
                            "label_mapping": mapping.to_json(),
                            "positive_label": mapping.positive_label,
                            "negative_label": mapping.negative_label,
                            "positive_semantic": ENGAGE,
                            "negative_semantic": SKIP,
                            "history_action": history_action,
                            "semantic_action": history_action,
                            "p_positive_semantic": float(metrics["p_positive"]),
                            "p_negative_semantic": float(metrics["p_negative"]),
                            "choice_logit": float(metrics["choice_logit"]),
                            "behavior_target_probability": float(metrics["p_positive"]),
                            "conversation_json": json.dumps(visible, separators=(",", ":")),
                            "prompt_character_count": sum(len(message["content"]) for message in visible),
                            "history_equivalent": True,
                            "target_kind": "matched_policy_probability",
                            **asdict(condition),
                            **states[framing],
                            "subsequent_outcome": transition.outcome,
                            "subsequent_reward": transition.reward,
                            "subsequent_effort": transition.effort,
                            "subsequent_success": transition.success,
                            "terminated": transition.terminated,
                        }
                    )
                if transitions[PERSISTENT].outcome != transitions[INDEPENDENT].outcome:
                    raise RuntimeError("matched latent outcomes diverged")
                if transitions[PERSISTENT].terminated:
                    break
    return rows

