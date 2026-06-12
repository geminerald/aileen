"""Helpers for moving recorded audio to/from disk."""

from __future__ import annotations

import os
import tempfile

import numpy as np


def write_temp_wav(samples: np.ndarray, sample_rate: int) -> str:
    """Write int16 samples to a temp WAV file and return its path.

    The caller is responsible for deleting the file when done.
    """
    import soundfile as sf

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return path
