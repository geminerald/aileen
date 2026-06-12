"""Runtime configuration, loaded from environment variables / a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    llm_provider: str = "openai"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    stt_model: str = "whisper-1"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str = "eleven_turbo_v2_5"

    bot_name: str = "Aileen"

    mic_sample_rate: int = 16000
    tts_sample_rate: int = 24000

    knowledge_dir: str = "data/knowledge"

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
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5"),
            bot_name=os.getenv("BOT_NAME", "Aileen"),
            mic_sample_rate=int(os.getenv("MIC_SAMPLE_RATE", "16000")),
            tts_sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
            knowledge_dir=os.getenv("KNOWLEDGE_DIR", "data/knowledge"),
        )
