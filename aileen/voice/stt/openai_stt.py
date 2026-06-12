"""OpenAI (Whisper) speech-to-text."""

from __future__ import annotations

from openai import OpenAI

from .base import STTProvider


class OpenAISTT(STTProvider):
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def transcribe(self, wav_path: str) -> str:
        with open(wav_path, "rb") as audio_file:
            result = self._client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
            )
        return result.text.strip()
