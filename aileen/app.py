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
from datetime import datetime

from .config import Config
from .conversation import Conversation
from .factory import ConfigError, build_knowledge, build_llm, build_stt, build_tts
from .timing import ReplyTiming, ReplyTimingLog

# Phrases that end the call.
_GOODBYE = {"goodbye", "bye", "quit", "exit", "stop", "that's all", "no thanks", "nothing else"}
_SIGN_OFF = "Thanks for calling. Goodbye!"


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _looks_like_goodbye(text: str) -> bool:
    cleaned = text.strip().lower().strip(".!? ")
    return cleaned in _GOODBYE


def _speak(tts, text: str) -> None:
    from .audio.player import play_pcm16  # imported lazily so --text needs no audio libs

    pcm, sample_rate = tts.synthesize(text)
    play_pcm16(pcm, sample_rate)


def _emit(text: str, bot_name: str, tts) -> None:
    print(f"[{_stamp()}] {bot_name}: {text}")
    if tts is not None:
        _speak(tts, text)


def _build_fillers(config: Config, tts):
    """Pre-render filler acknowledgements in the background, if enabled."""
    if tts is None or not config.speak_fillers:
        return None
    import threading

    from .speech import DEFAULT_FILLERS, FillerBank

    bank = FillerBank(tts, config.filler_phrases or DEFAULT_FILLERS)
    threading.Thread(target=bank.prewarm, daemon=True).start()
    return bank


def _respond(convo, user_text: str, bot_name: str, tts, fillers=None, timing_log=None) -> str:
    """Produce a reply; when speaking, stream it sentence-by-sentence aloud."""
    started = datetime.now()
    if tts is None:
        reply = convo.handle(user_text)
        print(f"[{_stamp()}] {bot_name}: {reply}")
        if timing_log is not None:
            timing_log.record(
                ReplyTiming(started=started, ended=datetime.now(), mode="text"), reply
            )
        return reply

    from .speech import speak_stream

    marks: dict[str, datetime] = {}
    prelude = fillers.random_prelude() if fillers else None
    print(f"[{_stamp()}] {bot_name}: ", end="", flush=True)
    reply = speak_stream(
        convo.handle_stream(user_text),
        tts,
        on_sentence=lambda s: print(s, end=" ", flush=True),
        prelude=prelude,
        on_first_audio=lambda: marks.setdefault("audio", datetime.now()),
        on_first_answer=lambda: marks.setdefault("answer", datetime.now()),
    )
    print()
    if timing_log is not None:
        timing = ReplyTiming(
            started=started,
            ended=datetime.now(),
            first_audio=marks.get("audio"),
            first_answer=marks.get("answer"),
            mode="voice",
        )
        line = timing_log.record(timing, reply)
        print(f"  ⏱  {line.split('| mode=', 1)[-1]}")
    return reply


def run_voice(config: Config, speak: bool = True) -> None:
    from .audio.files import write_temp_wav
    from .audio.recorder import MicRecorder

    llm = build_llm(config)
    knowledge = build_knowledge(config)
    stt = build_stt(config)
    tts = build_tts(config) if speak else None
    fillers = _build_fillers(config, tts)
    timing_log = ReplyTimingLog(config.timing_log_path) if config.timing_log_path else None

    convo = Conversation(llm, knowledge, config)
    recorder = MicRecorder(sample_rate=config.mic_sample_rate, device=config.mic_device)

    _emit(convo.greeting(), config.bot_name, tts)

    while True:
        samples = recorder.record_until_enter()
        if samples.shape[0] == 0:
            print("(didn't catch any audio — try again)")
            continue

        wav_path = write_temp_wav(samples, config.mic_sample_rate)
        try:
            user_text = stt.transcribe(wav_path)
        finally:
            os.remove(wav_path)

        if not user_text:
            print("(couldn't make that out — try again)")
            continue

        print(f"[{_stamp()}] You: {user_text}")
        if _looks_like_goodbye(user_text):
            _emit(_SIGN_OFF, config.bot_name, tts)
            break

        _respond(convo, user_text, config.bot_name, tts, fillers, timing_log)


def run_text(config: Config, speak: bool = True) -> None:
    llm = build_llm(config)
    knowledge = build_knowledge(config)
    tts = build_tts(config) if speak else None
    fillers = _build_fillers(config, tts)
    timing_log = ReplyTimingLog(config.timing_log_path) if config.timing_log_path else None

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
        _respond(convo, user_text, config.bot_name, tts, fillers, timing_log)


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
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available microphones (and their indices) then exit.",
    )
    args = parser.parse_args()

    if args.list_mics:
        from .audio.recorder import list_input_devices

        print(list_input_devices())
        return 0

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
