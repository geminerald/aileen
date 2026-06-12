"""Speaker playback for raw 16-bit PCM audio (as produced by the TTS layer)."""

from __future__ import annotations

import numpy as np
import sounddevice as sd


def play_pcm16(pcm_bytes: bytes, sample_rate: int) -> None:
    """Play little-endian signed 16-bit PCM mono audio and block until done."""
    if not pcm_bytes:
        return
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(samples, samplerate=sample_rate)
    sd.wait()
