"""OpenAI (Whisper) speech-to-text."""

from __future__ import annotations

from openai import OpenAI

from .base import STTProvider


class OpenAISTT(STTProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        # Transcription can take longer than chat, so a more generous timeout.
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model

    def transcribe(self, wav_path: str) -> str:
        with open(wav_path, "rb") as audio_file:
            result = self._client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
            )
        return result.text.strip()
