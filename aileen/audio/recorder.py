"""Microphone capture.

Exposes ``start()`` / ``stop()`` so any front-end can drive recording on its
own schedule — the terminal uses Enter presses, the GUI uses a button. When we
move to telephony, this is replaced by the call's audio stream and neither the
conversation engine nor the front-ends' logic changes.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd


class MicRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self._sample_rate = sample_rate
        self._channels = channels
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _on_audio(self, indata, _frames, _time, _status):
        with self._lock:
            self._chunks.append(indata.copy())

    def start(self) -> None:
        """Begin capturing from the default microphone (no-op if already on)."""
        if self._stream is not None:
            return
        with self._lock:
            self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop capturing and return int16 samples, shape (n, channels)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros((0, self._channels), dtype=np.int16)
        return np.concatenate(chunks, axis=0)

    def record_until_enter(self) -> np.ndarray:
        """Terminal helper: press Enter to start, Enter again to stop."""
        input("🎙️  Press Enter to start speaking… ")
        self.start()
        input("🔴 Recording… press Enter to stop. ")
        return self.stop()
