"""OpenAI (ChatGPT) brain — the default provider."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from .base import LLMProvider, Message


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.4,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._temperature = temperature

    def _payload(self, system_prompt: str, messages: list[Message]) -> list[dict]:
        payload = [{"role": "system", "content": system_prompt}]
        payload += [{"role": m.role, "content": m.content} for m in messages]
        return payload

    def respond(self, system_prompt: str, messages: list[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._payload(system_prompt, messages),
            temperature=self._temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def stream(self, system_prompt: str, messages: list[Message]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._payload(system_prompt, messages),
            temperature=self._temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
