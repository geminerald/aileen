"""Runtime configuration, loaded from environment variables / a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _split_phrases(raw: str | None) -> tuple[str, ...] | None:
    """Parse pipe-separated filler phrases; None if unset (use built-ins)."""
    if not raw:
        return None
    phrases = tuple(p.strip() for p in raw.split("|") if p.strip())
    return phrases or None


@dataclass
class Config:
    llm_provider: str = "openai"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    stt_model: str = "whisper-1"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # Which voice engine to use: "openai" (default) or "elevenlabs".
    tts_provider: str = "openai"

    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "nova"

    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str = "eleven_turbo_v2_5"

    bot_name: str = "Aileen"

    mic_sample_rate: int = 16000
    tts_sample_rate: int = 24000

    # Which microphone to record from: a device index or a name fragment
    # (e.g. "C920"). Empty/None uses the system default input device.
    mic_device: str | None = None

    # Speak a short random acknowledgement ("Sure.", "One moment.") at the
    # start of each reply to cover synthesis latency and sound conversational.
    speak_fillers: bool = True
    # Optional custom filler phrases (pipe-separated in the env); None = built-in.
    filler_phrases: tuple[str, ...] | None = None

    knowledge_dir: str = "data/knowledge"

    # Where to append per-reply latency timings (for debugging). Empty disables.
    timing_log_path: str = "logs/aileen-timing.log"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from the environment, reading .env if present."""
        load_dotenv()
        return cls(
            llm_provider=os.getenv("AILEEN_LLM_PROVIDER", "openai").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            stt_model=os.getenv("STT_MODEL", "whisper-1"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            tts_provider=os.getenv("AILEEN_TTS_PROVIDER", "openai").lower(),
            openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
            openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5"),
            bot_name=os.getenv("BOT_NAME", "Aileen"),
            mic_sample_rate=int(os.getenv("MIC_SAMPLE_RATE", "16000")),
            tts_sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
            mic_device=os.getenv("MIC_DEVICE") or None,
            speak_fillers=os.getenv("AILEEN_SPEAK_FILLERS", "true").strip().lower()
            not in ("0", "false", "no", "off"),
            filler_phrases=_split_phrases(os.getenv("AILEEN_FILLER_PHRASES")),
            knowledge_dir=os.getenv("KNOWLEDGE_DIR", "data/knowledge"),
            timing_log_path=os.getenv("AILEEN_TIMING_LOG", "logs/aileen-timing.log"),
        )
