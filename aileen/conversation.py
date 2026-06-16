"""The dialogue engine — front-end-agnostic.

It knows nothing about microphones, phones, or speakers. It takes user text in
and produces reply text out, pulling in relevant knowledge and keeping history.
The terminal app today and a telephony front-end tomorrow both drive this same
class.
"""

from __future__ import annotations

from collections.abc import Iterator

from .config import Config
from .knowledge.base import KnowledgeBase
from .llm.base import LLMProvider, Message
from .prompts import build_system_prompt


class Conversation:
    def __init__(self, llm: LLMProvider, knowledge: KnowledgeBase, config: Config):
        self._llm = llm
        self._knowledge = knowledge
        self._config = config
        self._history: list[Message] = []

    def greeting(self) -> str:
        """The opening line the bot says when an interaction starts."""
        return (
            f"Hi, thanks for calling. I'm {self._config.bot_name}, "
            "a virtual assistant. How can I help you today?"
        )

    def handle(self, user_text: str) -> str:
        """Process one user turn and return the bot's spoken reply."""
        context = self._knowledge.context_for(user_text)
        system_prompt = build_system_prompt(self._config.bot_name, context)

        self._history.append(Message("user", user_text))
        reply = self._llm.respond(system_prompt, self._history)
        self._history.append(Message("assistant", reply))
        return reply

    def handle_stream(self, user_text: str) -> Iterator[str]:
        """Like :meth:`handle`, but yield the reply in deltas as it's generated.

        History is updated with the user turn up front and the full assistant
        reply once the stream is exhausted, so it stays consistent with
        :meth:`handle`.
        """
        context = self._knowledge.context_for(user_text)
        system_prompt = build_system_prompt(self._config.bot_name, context)

        self._history.append(Message("user", user_text))
        parts: list[str] = []
        for delta in self._llm.stream(system_prompt, self._history):
            parts.append(delta)
            yield delta
        self._history.append(Message("assistant", "".join(parts).strip()))
