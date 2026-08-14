"""Conversation-only state; private bandit state never enters this object."""

from dataclasses import dataclass, field
from typing import Dict, List

from .prompts import feedback_prompt, initial_prompt


@dataclass
class BanditConversation:
    messages: List[Dict[str, str]] = field(default_factory=list)
    action_labels: str = "ABC"

    @classmethod
    def start(cls, action_labels: str = "ABC") -> "BanditConversation":
        return cls(
            messages=[{"role": "user", "content": initial_prompt(action_labels)}],
            action_labels=action_labels,
        )

    def record_action(self, action: str) -> None:
        normalized = action.strip().upper()
        if normalized not in {"A", "B", "C"}:
            raise ValueError(f"invalid action: {action!r}")
        if not self.messages or self.messages[-1]["role"] != "user":
            raise RuntimeError("an action must follow a user message")
        display = dict(zip("ABC", self.action_labels))[normalized]
        self.messages.append({"role": "assistant", "content": display})

    def record_feedback(self, reward: int) -> None:
        if not self.messages or self.messages[-1]["role"] != "assistant":
            raise RuntimeError("feedback must follow an assistant action")
        if self.messages[-1]["content"] == self.action_labels[2]:
            raise RuntimeError("cannot append feedback after STOP")
        self.messages.append({"role": "user", "content": feedback_prompt(reward, self.action_labels)})

    def snapshot(self) -> List[Dict[str, str]]:
        return [dict(message) for message in self.messages]
