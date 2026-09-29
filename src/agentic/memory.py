"""Conversation memory with token-ish budget and optional summarisation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationMemory:
    """Bounded rolling window of messages, with an optional pinned system prompt."""

    system: str | None = None
    max_messages: int = 20
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))
        # keep the system prompt implicit; trim oldest non-system messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def history(self) -> list[dict]:
        out = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.extend({"role": m.role, "content": m.content} for m in self.messages)
        return out

    def clear(self) -> None:
        self.messages.clear()


class SummaryMemory(ConversationMemory):
    """Memory that folds evicted messages into a running summary string."""

    def __init__(self, system: str | None = None, max_messages: int = 6):
        super().__init__(system=system, max_messages=max_messages)
        self.summary: str = ""

    def add(self, role: str, content: str) -> None:
        super().add(role, content)
        # naive extractive summary: keep the newest evicted snippet
        if len(self.messages) >= self.max_messages:
            self.summary = (self.summary + " " + content)[-500:].strip()

    def history(self) -> list[dict]:
        hist = []
        if self.summary:
            hist.append({"role": "system", "content": f"Conversation so far: {self.summary}"})
        return hist + super().history()
