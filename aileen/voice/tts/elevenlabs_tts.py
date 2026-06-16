"""ElevenLabs text-to-speech."""

from __future__ import annotations

from collections.abc import Iterator

from elevenlabs.client import ElevenLabs

from .base import TTSProvider

# ElevenLabs only offers PCM at these sample rates.
_SUPPORTED_PCM_RATES = {16000, 22050, 24000, 44100}


class ElevenLabsTTS(TTSProvider):
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_turbo_v2_5",
        sample_rate: int = 24000,
    ):
        if sample_rate not in _SUPPORTED_PCM_RATES:
            raise ValueError(
                f"ElevenLabs PCM sample rate must be one of {sorted(_SUPPORTED_PCM_RATES)}, "
                f"got {sample_rate}."
            )
        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._model_id = model_id
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> tuple[bytes, int]:
        # convert() yields chunks of bytes; collect them into one buffer.
        pcm = b"".join(self.stream(text))
        return pcm, self._sample_rate

    def stream(self, text: str) -> Iterator[bytes]:
        chunks = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            model_id=self._model_id,
            text=text,
            output_format=f"pcm_{self._sample_rate}",
        )
        for chunk in chunks:
            if chunk:
                yield chunk
