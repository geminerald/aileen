"""Builds concrete providers from a :class:`~aileen.config.Config`.

This is the one place that knows which concrete classes exist, so the rest of
the app depends only on the interfaces. Adding a new provider means editing
here and nowhere else.
"""

from __future__ import annotations

from .config import Config
from .knowledge.base import KnowledgeBase
from .knowledge.static import StaticFileKnowledge
from .llm.anthropic_provider import AnthropicProvider
from .llm.base import LLMProvider
from .llm.openai_provider import OpenAIProvider
from .voice.stt.base import STTProvider
from .voice.stt.openai_stt import OpenAISTT
from .voice.tts.base import TTSProvider
from .voice.tts.elevenlabs_tts import ElevenLabsTTS
from .voice.tts.openai_tts import OpenAITTS


class ConfigError(RuntimeError):
    """Raised when configuration is missing or inconsistent."""


def build_llm(config: Config) -> LLMProvider:
    if config.llm_provider == "openai":
        if not config.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set (needed for the OpenAI brain).")
        return OpenAIProvider(config.openai_api_key, config.openai_model)
    if config.llm_provider == "anthropic":
        if not config.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set (needed for the Claude brain).")
        return AnthropicProvider(config.anthropic_api_key, config.anthropic_model)
    raise ConfigError(
        f"Unknown AILEEN_LLM_PROVIDER '{config.llm_provider}'. Use 'openai' or 'anthropic'."
    )


def build_tts(config: Config) -> TTSProvider:
    if config.tts_provider == "openai":
        if not config.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set (needed for the OpenAI voice).")
        return OpenAITTS(config.openai_api_key, config.openai_tts_model, config.openai_tts_voice)
    if config.tts_provider == "elevenlabs":
        if not config.elevenlabs_api_key:
            raise ConfigError("ELEVENLABS_API_KEY is not set (needed for the ElevenLabs voice).")
        if not config.elevenlabs_voice_id:
            raise ConfigError("ELEVENLABS_VOICE_ID is not set (pick a voice in ElevenLabs).")
        return ElevenLabsTTS(
            config.elevenlabs_api_key,
            config.elevenlabs_voice_id,
            config.elevenlabs_model_id,
            config.tts_sample_rate,
        )
    raise ConfigError(
        f"Unknown AILEEN_TTS_PROVIDER '{config.tts_provider}'. Use 'openai' or 'elevenlabs'."
    )


def build_stt(config: Config) -> STTProvider:
    # Speech-to-text uses OpenAI regardless of which LLM brain is selected.
    if not config.openai_api_key:
        raise ConfigError("OPENAI_API_KEY is not set (needed for speech-to-text).")
    return OpenAISTT(config.openai_api_key, config.stt_model)


def build_knowledge(config: Config) -> KnowledgeBase:
    return StaticFileKnowledge(config.knowledge_dir)
