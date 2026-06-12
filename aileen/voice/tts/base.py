"""Contract for turning text into speakable audio."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Return ``(pcm_s16le_bytes, sample_rate)`` for the given text.

        Returning raw little-endian 16-bit PCM (rather than MP3) means the
        player can stream it straight to the speakers with no ffmpeg/codec
        dependency.
        """
        raise NotImplementedError
