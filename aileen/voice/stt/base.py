"""Contract for turning recorded speech into text."""

from __future__ import annotations

from abc import ABC, abstractmethod


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, wav_path: str) -> str:
        """Return the transcript of the audio file at ``wav_path``."""
        raise NotImplementedError
