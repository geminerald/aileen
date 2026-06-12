"""Terminal front-end: wires the pieces together and runs the conversation.

Two modes:
  * voice (default): speak into the mic, hear Aileen's reply.
  * --text:          type your message; reply is still spoken unless --mute.

This is just one front-end over :class:`~aileen.conversation.Conversation`.
The telephony front-end (later) will reuse the same engine.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from .config import Config
from .conversation import Conversation
from .factory import ConfigError, build_knowledge, build_llm, build_stt, build_tts

# Phrases that end the call.
_GOODBYE = {"goodbye", "bye", "quit", "exit", "stop", "that's all", "no thanks", "nothing else"}
_SIGN_OFF = "Thanks for calling. Goodbye!"


def _looks_like_goodbye(text: str) -> bool:
    cleaned = text.strip().lower().strip(".!? ")
    return cleaned in _GOODBYE


def _speak(tts, text: str) -> None:
    from .audio.player import play_pcm16  # imported lazily so --text needs no audio libs

    pcm, sample_rate = tts.synthesize(text)
    play_pcm16(pcm, sample_rate)


def _emit(text: str, bot_name: str, tts) -> None:
    print(f"{bot_name}: {text}")
    if tts is not None:
        _speak(tts, text)


def _write_temp_wav(samples, sample_rate: int) -> str:
    import soundfile as sf

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return path


def run_voice(config: Config, speak: bool = True) -> None:
    from .audio.recorder import MicRecorder

    llm = build_llm(config)
    knowledge = build_knowledge(config)
    stt = build_stt(config)
    tts = build_tts(config) if speak else None

    convo = Conversation(llm, knowledge, config)
    recorder = MicRecorder(sample_rate=config.mic_sample_rate)

    _emit(convo.greeting(), config.bot_name, tts)

    while True:
        samples = recorder.record_until_enter()
        if samples.shape[0] == 0:
            print("(didn't catch any audio — try again)")
            continue

        wav_path = _write_temp_wav(samples, config.mic_sample_rate)
        try:
            user_text = stt.transcribe(wav_path)
        finally:
            os.remove(wav_path)

        if not user_text:
            print("(couldn't make that out — try again)")
            continue

        print(f"You: {user_text}")
        if _looks_like_goodbye(user_text):
            _emit(_SIGN_OFF, config.bot_name, tts)
            break

        _emit(convo.handle(user_text), config.bot_name, tts)


def run_text(config: Config, speak: bool = True) -> None:
    llm = build_llm(config)
    knowledge = build_knowledge(config)
    tts = build_tts(config) if speak else None

    convo = Conversation(llm, knowledge, config)
    _emit(convo.greeting(), config.bot_name, tts)

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if _looks_like_goodbye(user_text):
            _emit(_SIGN_OFF, config.bot_name, tts)
            break
        _emit(convo.handle(user_text), config.bot_name, tts)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aileen",
        description="Aileen — a customer-service voice bot.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Type your messages instead of speaking (no microphone needed).",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Don't speak replies aloud (skips ElevenLabs; text output only).",
    )
    args = parser.parse_args()

    config = Config.from_env()
    speak = not args.mute

    try:
        if args.text:
            run_text(config, speak=speak)
        else:
            run_voice(config, speak=speak)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print("Copy .env.example to .env and fill in your keys.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
