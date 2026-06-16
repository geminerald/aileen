"""Speaker playback for raw 16-bit PCM audio (as produced by the TTS layer)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import sounddevice as sd

# Windows output devices drop the first few milliseconds of audio while the
# stream spins up, which clips the start of speech. Writing a short pad of
# silence first lets the warm-up eat the silence instead of real audio.
_WARMUP_SILENCE_MS = 120


class PcmSpeaker:
    """An open output stream you can feed PCM to in chunks, gaplessly.

    Used to play a whole reply as it streams in sentence-by-sentence: a single
    stream stays open for the reply, so there's no fresh warm-up clip between
    sentences. ``close()`` drains buffered audio before stopping (PortAudio's
    stop waits for pending buffers), so the tail isn't cut off either.
    """

    def __init__(self, sample_rate: int):
        self._stream = sd.RawOutputStream(
            samplerate=sample_rate, channels=1, dtype="int16"
        )
        self._stream.start()
        self._tail = b""  # carries an odd trailing byte between chunk writes
        pad_frames = int(sample_rate * _WARMUP_SILENCE_MS / 1000)
        self._stream.write(np.zeros(pad_frames, dtype=np.int16).tobytes())

    def write(self, pcm_bytes: bytes) -> None:
        """Write PCM, keeping writes frame-aligned (int16 = 2 bytes/frame)."""
        if not pcm_bytes:
            return
        data = self._tail + pcm_bytes
        usable = len(data) - (len(data) % 2)
        self._tail = data[usable:]
        if usable:
            self._stream.write(data[:usable])

    def close(self) -> None:
        if self._tail:  # flush a dangling byte so we don't lose a sample
            self._stream.write(self._tail + b"\x00")
            self._tail = b""
        self._stream.stop()
        self._stream.close()

    def __enter__(self) -> "PcmSpeaker":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def play_pcm16(pcm_bytes: bytes, sample_rate: int) -> None:
    """Play little-endian signed 16-bit PCM mono audio and block until done."""
    if not pcm_bytes:
        return
    with PcmSpeaker(sample_rate) as speaker:
        speaker.write(pcm_bytes)


def play_pcm16_stream(chunks: Iterable[bytes], sample_rate: int) -> None:
    """Play PCM chunks as they arrive, starting on the first one."""
    with PcmSpeaker(sample_rate) as speaker:
        for chunk in chunks:
            speaker.write(chunk)
