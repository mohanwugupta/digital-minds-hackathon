"""Deterministic construction of matched conversation-state replays."""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class MatchedReplay:
    """One intervention strength applied to an otherwise identical state."""

    state_id: str
    alpha: float
    conversation: list[dict]
    context_bytes: bytes
    context_hash: str


def canonical_context(conversation: list[dict]) -> bytes:
    """Serialize model-visible messages deterministically for matching/audits."""

    return json.dumps(
        conversation, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def build_matched_replays(
    state_id: str, conversation: list[dict]
) -> list[MatchedReplay]:
    """Construct the frozen -1/0/+1 intervention triplet for one state."""

    context = canonical_context(conversation)
    digest = hashlib.sha256(context).hexdigest()
    return [
        MatchedReplay(
            state_id=state_id,
            alpha=alpha,
            conversation=json.loads(context.decode("utf-8")),
            context_bytes=context,
            context_hash=digest,
        )
        for alpha in (-1.0, 0.0, 1.0)
    ]
