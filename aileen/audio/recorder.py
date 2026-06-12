"""Microphone capture for the terminal voice loop.

Uses a simple press-Enter-to-start / press-Enter-to-stop scheme, which is
reliable across platforms and needs no voice-activity detection. When we move
to telephony, this whole module is replaced by the call's audio stream — the
conversation engine doesn't change.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd


class MicRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self._sample_rate = sample_rate
        self._channels = channels

    def record_until_enter(self) -> np.ndarray:
        """Record from the default mic. Returns int16 samples, shape (n, channels)."""
        input("🎙️  Press Enter to start speaking… ")

        chunks: list[np.ndarray] = []
        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        stop = threading.Event()

        def on_audio(indata, _frames, _time, _status):
            audio_q.put(indata.copy())

        def wait_for_stop():
            input("🔴 Recording… press Enter to stop. ")
            stop.set()

        threading.Thread(target=wait_for_stop, daemon=True).start()

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            callback=on_audio,
        ):
            while not stop.is_set():
                try:
                    chunks.append(audio_q.get(timeout=0.1))
                except queue.Empty:
                    continue
            # Drain anything still buffered after stop was signalled.
            while not audio_q.empty():
                chunks.append(audio_q.get_nowait())

        if not chunks:
            return np.zeros((0, self._channels), dtype=np.int16)
        return np.concatenate(chunks, axis=0)
