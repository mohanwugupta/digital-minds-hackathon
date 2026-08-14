"""Artifact schemas and episode-level split utilities."""

from dataclasses import asdict, dataclass, field
import json
import random
from typing import Any, Dict, List, Optional

from .prompts import initial_prompt


@dataclass(frozen=True)
class EpisodeSplit:
    train: List[str]
    validation: List[str]
    test: List[str]


def split_episode_ids(episode_ids: List[str], seed: int = 0) -> EpisodeSplit:
    unique = sorted(set(episode_ids))
    if len(unique) != len(episode_ids):
        raise ValueError("episode_ids must be unique")
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * 0.70)
    n_validation = int(n * 0.15)
    return EpisodeSplit(
        train=unique[:n_train],
        validation=unique[n_train : n_train + n_validation],
        test=unique[n_train + n_validation :],
    )


@dataclass
class DecisionRecord:
    episode_id: str
    state_id: str
    seed: int
    action_seed: int = 0
    round: int = 0
    p_A_true: float = 0.0
    p_B_true: float = 0.0
    cumulative_score: int = 0
    choice_history: List[str] = field(default_factory=list)
    reward_history: List[int] = field(default_factory=list)
    conversation: List[Dict[str, str]] = field(default_factory=list)
    previous_outcome: Optional[int] = None
    layer: Optional[int] = None
    neuron_set: str = "none"
    intervention_type: str = "none"
    alpha: float = 0.0
    probe_value_pre: Optional[float] = None
    probe_value_post: Optional[float] = None
    logit_A: Optional[float] = None
    logit_B: Optional[float] = None
    logit_C: Optional[float] = None
    p_A: Optional[float] = None
    p_B: Optional[float] = None
    p_stop: Optional[float] = None
    p_continue: Optional[float] = None
    persistence_logit: Optional[float] = None
    p_action_mass_raw: Optional[float] = None
    top_token_is_action: Optional[bool] = None
    sampled_action: Optional[str] = None
    subsequent_reward: Optional[int] = None
    future_cumulative_return: Optional[float] = None
    terminated: bool = False

    @classmethod
    def minimal(
        cls, episode_id: str, state_id: str, seed: int, action_labels: str = "ABC"
    ) -> "DecisionRecord":
        return cls(
            episode_id=episode_id,
            state_id=state_id,
            seed=seed,
            conversation=[{"role": "user", "content": initial_prompt(action_labels)}],
        )

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        for key in ("choice_history", "reward_history", "conversation"):
            row[key] = json.dumps(row[key], separators=(",", ":"), ensure_ascii=False)
        return row
