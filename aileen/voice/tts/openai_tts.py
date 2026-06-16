"""OpenAI text-to-speech.

Handy when you want the whole bot to run on a single OpenAI key (brain +
speech-to-text + voice), e.g. before signing up for an ElevenLabs paid plan.
Swap via ``AILEEN_TTS_PROVIDER=openai`` (the default) or ``=elevenlabs``.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from .base import TTSProvider


class OpenAITTS(TTSProvider):
    # OpenAI's "pcm" response format is raw 16-bit mono at 24 kHz.
    SAMPLE_RATE = 24000

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        voice: str = "alloy",
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        # A bounded timeout + retries means a stalled request fails fast (and is
        # retried) instead of hanging the caller for the SDK's 10-minute default.
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._voice = voice

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def synthesize(self, text: str) -> tuple[bytes, int]:
        with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",
        ) as response:
            pcm = response.read()
        return pcm, self.SAMPLE_RATE

    def stream(self, text: str) -> Iterator[bytes]:
        with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",
        ) as response:
            for chunk in response.iter_bytes():
                if chunk:
                    yield chunk
