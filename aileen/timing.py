"""Records when Aileen's replies start and finish, for latency debugging.

Each spoken reply logs a line with the wall-clock start/end plus the two
latencies that matter: time to the first *sound* (the filler acknowledgement)
and time to the first *answer* audio (the real generate→synthesize pipeline).
Reviewing ``logs/aileen-timing.log`` after a chat shows where the lag is.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ReplyTiming:
    """Timestamps captured across one reply. Offsets are seconds from start."""

    started: datetime
    ended: datetime
    first_audio: datetime | None = None  # first sound out (filler or answer)
    first_answer: datetime | None = None  # first real-answer audio
    mode: str = "voice"

    def _offset(self, moment: datetime | None) -> float | None:
        return None if moment is None else (moment - self.started).total_seconds()

    @property
    def time_to_first_audio(self) -> float | None:
        return self._offset(self.first_audio)

    @property
    def time_to_answer(self) -> float | None:
        return self._offset(self.first_answer)

    @property
    def total(self) -> float:
        return (self.ended - self.started).total_seconds()


class ReplyTimingLog:
    """Appends one human-readable line per reply to a log file (thread-safe)."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fmt(offset: float | None) -> str:
        return "n/a" if offset is None else f"+{offset:.2f}s"

    def format_line(self, timing: ReplyTiming, text: str) -> str:
        preview = " ".join(text.split())
        if len(preview) > 80:
            preview = preview[:77] + "..."
        return (
            f"{timing.started:%Y-%m-%d %H:%M:%S} → {timing.ended:%H:%M:%S} | "
            f"mode={timing.mode} | "
            f"first-sound {self._fmt(timing.time_to_first_audio)} | "
            f"answer {self._fmt(timing.time_to_answer)} | "
            f"total {timing.total:.2f}s | "
            f'"{preview}"'
        )

    def record(self, timing: ReplyTiming, text: str) -> str:
        """Write the timing line and return it (handy for echoing to a UI)."""
        line = self.format_line(timing, text)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return line
