"""Contract for turning text into speakable audio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class TTSProvider(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate (Hz) of the PCM this provider produces."""
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Return ``(pcm_s16le_bytes, sample_rate)`` for the given text.

        Returning raw little-endian 16-bit PCM (rather than MP3) means the
        player can stream it straight to the speakers with no ffmpeg/codec
        dependency.
        """
        raise NotImplementedError

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM (at :attr:`sample_rate`) as it's produced.

        Streaming lets playback start on the first chunk instead of waiting for
        the whole clip. The default falls back to one chunk from
        :meth:`synthesize`; providers override for real streaming.
        """
        yield self.synthesize(text)[0]
