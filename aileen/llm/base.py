"""The contract every LLM "brain" must satisfy.

Keeping this tiny is deliberate: swapping OpenAI for Claude (or adding a new
provider later) means implementing a single ``respond`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Message:
    """One turn in the conversation. ``role`` is "user" or "assistant"."""

    role: str
    content: str


class LLMProvider(ABC):
    """A text-in / text-out reasoning engine."""

    @abstractmethod
    def respond(self, system_prompt: str, messages: list[Message]) -> str:
        """Return the assistant's reply given the system prompt and history."""
        raise NotImplementedError

    def stream(self, system_prompt: str, messages: list[Message]) -> Iterator[str]:
        """Yield the reply as text deltas as they're generated.

        Streaming lets a front-end start speaking the first sentence before the
        whole reply is written. The default yields the full reply in one chunk;
        providers override for real token streaming.
        """
        yield self.respond(system_prompt, messages)
