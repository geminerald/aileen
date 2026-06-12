"""Anthropic (Claude) brain — the swappable alternative.

Set ``AILEEN_LLM_PROVIDER=anthropic`` to use this instead of OpenAI.
"""

from __future__ import annotations

from anthropic import Anthropic

from .base import LLMProvider, Message


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ):
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def respond(self, system_prompt: str, messages: list[Message]) -> str:
        # Claude takes the system prompt as a separate argument, not a message.
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        parts = [block.text for block in response.content if block.type == "text"]
        return "".join(parts).strip()
