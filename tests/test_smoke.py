"""Lightweight tests that need no API keys, audio devices, or network."""

from pathlib import Path

import pytest

from aileen.config import Config
from aileen.conversation import Conversation
from aileen.factory import ConfigError, build_tts
from aileen.knowledge.static import StaticFileKnowledge
from aileen.llm.base import LLMProvider, Message
from aileen.prompts import build_system_prompt
from datetime import datetime, timedelta

from aileen.speech import FillerBank, speak_stream, split_sentences
from aileen.timing import ReplyTiming, ReplyTimingLog
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


def test_split_sentences_groups_streamed_deltas():
    # Deltas split mid-word, like real token streaming.
    deltas = ["Hello ", "there. How can ", "I help", " you? Visit 9.5", " now.\n", "Bye"]
    assert list(split_sentences(deltas)) == [
        "Hello there.",
        "How can I help you?",
        "Visit 9.5 now.",
        "Bye",
    ]


class _FakeTTS:
    """Captures each sentence it's asked to speak; emits dummy PCM."""

    sample_rate = 24000

    def __init__(self):
        self.spoken = []

    def synthesize(self, text):
        return (b"\x00\x00", self.sample_rate)

    def stream(self, text):
        self.spoken.append(text)
        yield b"\x00\x00" * 4


def test_speak_stream_speaks_each_sentence(monkeypatch):
    writes = []

    class FakeSpeaker:
        def __init__(self, sample_rate):
            assert sample_rate == 24000

        def write(self, pcm):
            writes.append(pcm)

        def close(self):
            pass

    monkeypatch.setattr("aileen.speech.PcmSpeaker", FakeSpeaker)
    tts = _FakeTTS()
    seen = []

    full = speak_stream(iter(["One. ", "Two. ", "Three."]), tts, on_sentence=seen.append)

    assert full == "One. Two. Three."
    # Text is revealed / played back in order...
    assert seen == ["One.", "Two.", "Three."]
    assert len(writes) == 3
    # ...even though synthesis runs concurrently (so call order isn't fixed).
    assert sorted(tts.spoken) == ["One.", "Three.", "Two."]


def test_filler_bank_prewarms_and_serves_ready_phrase():
    tts = _FakeTTS()
    bank = FillerBank(tts, phrases=["Sure.", "One moment."])
    # Nothing rendered yet -> no prelude available.
    assert bank.random_prelude() is None
    bank.prewarm()
    prelude = bank.random_prelude()
    assert prelude is not None
    text, pcm = prelude
    assert text in {"Sure.", "One moment."}
    assert pcm  # non-empty rendered audio


def test_speak_stream_plays_prelude_first(monkeypatch):
    order = []

    class FakeSpeaker:
        def __init__(self, sample_rate):
            pass

        def write(self, pcm):
            order.append(pcm)

        def close(self):
            pass

    monkeypatch.setattr("aileen.speech.PcmSpeaker", FakeSpeaker)
    tts = _FakeTTS()
    seen = []

    speak_stream(
        iter(["Hello there."]),
        tts,
        on_sentence=seen.append,
        prelude=("Sure.", b"PRELUDE"),
    )

    # Prelude text is revealed first, and its audio is written before the reply.
    assert seen[0] == "Sure."
    assert order[0] == b"PRELUDE"


def test_speak_stream_fires_timing_hooks(monkeypatch):
    monkeypatch.setattr("aileen.speech.PcmSpeaker", lambda sample_rate: _NullSpeaker())
    tts = _FakeTTS()
    events = []

    speak_stream(
        iter(["Hi there. All good."]),
        tts,
        on_first_audio=lambda: events.append("audio"),
        on_first_answer=lambda: events.append("answer"),
    )

    # Each hook fires exactly once; without a prelude they fire together.
    assert events.count("audio") == 1
    assert events.count("answer") == 1


class _NullSpeaker:
    def write(self, pcm):
        pass

    def close(self):
        pass


def test_reply_timing_offsets():
    start = datetime(2026, 6, 16, 14, 0, 0)
    timing = ReplyTiming(
        started=start,
        ended=start + timedelta(seconds=8),
        first_audio=start + timedelta(seconds=0.2),
        first_answer=start + timedelta(seconds=1.9),
    )
    assert timing.time_to_first_audio == pytest.approx(0.2)
    assert timing.time_to_answer == pytest.approx(1.9)
    assert timing.total == pytest.approx(8.0)


def test_reply_timing_log_appends_line(tmp_path):
    log = ReplyTimingLog(tmp_path / "logs" / "timing.log")
    start = datetime(2026, 6, 16, 14, 0, 0)
    timing = ReplyTiming(
        started=start,
        ended=start + timedelta(seconds=5),
        first_audio=start + timedelta(seconds=0.3),
        first_answer=start + timedelta(seconds=2.0),
    )
    line = log.record(timing, "We are open 9am to 5pm.")

    written = (tmp_path / "logs" / "timing.log").read_text(encoding="utf-8")
    assert line + "\n" == written
    assert "first-sound +0.30s" in line
    assert "answer +2.00s" in line
    assert "total 5.00s" in line
