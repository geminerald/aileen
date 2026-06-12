"""Lightweight tests that need no API keys, audio devices, or network."""

from pathlib import Path

import pytest

from aileen.config import Config
from aileen.conversation import Conversation
from aileen.factory import ConfigError, build_tts
from aileen.knowledge.static import StaticFileKnowledge
from aileen.llm.base import LLMProvider, Message
from aileen.prompts import build_system_prompt
from aileen.voice.tts.elevenlabs_tts import ElevenLabsTTS
from aileen.voice.tts.openai_tts import OpenAITTS


class FakeLLM(LLMProvider):
    """Records what it was asked and echoes a canned reply."""

    def __init__(self):
        self.last_system_prompt = None
        self.last_messages = None

    def respond(self, system_prompt, messages):
        self.last_system_prompt = system_prompt
        self.last_messages = list(messages)
        return "This is a test reply."


def test_static_knowledge_matches_relevant_section():
    kb = StaticFileKnowledge("data/knowledge")
    context = kb.context_for("What are your opening hours?")
    assert "9am to 5pm" in context


def test_static_knowledge_returns_empty_for_irrelevant_query():
    kb = StaticFileKnowledge("data/knowledge")
    assert kb.context_for("xyzzy plugh nothing relevant") == ""


def test_static_knowledge_empty_when_dir_missing(tmp_path: Path):
    kb = StaticFileKnowledge(tmp_path / "does-not-exist")
    assert kb.is_empty
    assert kb.context_for("anything") == ""


def test_build_system_prompt_includes_knowledge():
    prompt = build_system_prompt("Aileen", "We are open 9 to 5.")
    assert "Aileen" in prompt
    assert "We are open 9 to 5." in prompt


def test_conversation_threads_knowledge_into_prompt_and_keeps_history():
    llm = FakeLLM()
    kb = StaticFileKnowledge("data/knowledge")
    convo = Conversation(llm, kb, Config(bot_name="Aileen"))

    reply = convo.handle("What is your refund policy?")

    assert reply == "This is a test reply."
    assert "refund" in llm.last_system_prompt.lower()
    # One user message went in; user + assistant are now in history.
    assert llm.last_messages[-1] == Message("user", "What is your refund policy?")


def test_greeting_uses_bot_name():
    convo = Conversation(FakeLLM(), StaticFileKnowledge("data/knowledge"), Config(bot_name="Aileen"))
    assert "Aileen" in convo.greeting()


def test_build_tts_selects_openai_by_default():
    config = Config(tts_provider="openai", openai_api_key="sk-dummy")
    assert isinstance(build_tts(config), OpenAITTS)


def test_build_tts_selects_elevenlabs_when_configured():
    config = Config(
        tts_provider="elevenlabs",
        elevenlabs_api_key="el-dummy",
        elevenlabs_voice_id="voice123",
    )
    assert isinstance(build_tts(config), ElevenLabsTTS)


def test_build_tts_rejects_unknown_provider():
    with pytest.raises(ConfigError):
        build_tts(Config(tts_provider="bogus"))
