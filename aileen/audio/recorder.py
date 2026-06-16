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


def resolve_input_device(device: int | str | None) -> int | None:
    """Turn a device index or name fragment into a PortAudio device index.

    ``None`` (or an empty string) means "use the system default". A name
    fragment is matched case-insensitively against input devices, so
    ``MIC_DEVICE=C920`` finds the webcam mic without needing its exact label.
    Returns ``None`` for the default; raises ``ValueError`` if a requested
    device can't be found.
    """
    if device is None:
        return None
    if isinstance(device, str):
        device = device.strip()
        if not device:
            return None
        if device.isdigit():
            return int(device)
        needle = device.lower()
        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0 and needle in info["name"].lower():
                return index
        raise ValueError(
            f"No input device matching {device!r}. "
            "Run `aileen --list-mics` to see available microphones."
        )
    return int(device)


class MicRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = resolve_input_device(device)
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
            device=self._device,
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


def list_input_devices() -> str:
    """Return a human-readable list of input devices, marking the default.

    Handy for picking a value for ``MIC_DEVICE`` when the system default is a
    virtual mic (e.g. Steam's streaming microphone) that records silence.
    """
    default_index = sd.default.device[0]
    lines = ["Available microphones (set MIC_DEVICE to an index or name fragment):"]
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] <= 0:
            continue
        mark = "  <-- system default" if index == default_index else ""
        lines.append(
            f"  [{index}] {info['name']} "
            f"(channels={info['max_input_channels']}){mark}"
        )
    return "\n".join(lines)
