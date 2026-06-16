"""Turn a streaming text reply into speech, sentence by sentence.

Speaking each sentence the moment it's complete — instead of waiting for the
whole reply, then the whole audio clip — is what kills the awkward pause:
Aileen starts talking while she's still composing the rest of her answer.
"""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from .audio.player import PcmSpeaker
from .voice.tts.base import TTSProvider

_SENTENCE_ENDERS = ".!?…"

# How many sentences to synthesize concurrently / look ahead of playback. A few
# workers give playback a comfortable lead so one slow TTS request can't starve
# the speaker, without hammering the API.
_TTS_LOOKAHEAD = 3

# Short, answer-agnostic acknowledgements spoken at the very start of a reply.
# Pre-rendered once, they play instantly while the real answer is still being
# generated and synthesized — covering the gap and sounding conversational.
DEFAULT_FILLERS = (
    "Sure.",
    "Of course.",
    "Okay.",
    "Right.",
    "Let me see.",
    "Let me check that.",
    "One moment.",
    "I understand.",
)


class FillerBank:
    """Pre-renders filler phrases so one can be played instantly before a reply.

    Synthesizing is done once (in :meth:`prewarm`, meant for a background
    thread) and cached as PCM, so :meth:`random_prelude` hands back ready audio
    with no network wait. A phrase that fails to synthesize is simply skipped —
    fillers are a nicety, never required.
    """

    def __init__(self, tts: TTSProvider, phrases: Iterable[str] = DEFAULT_FILLERS):
        self._tts = tts
        self._phrases = [p.strip() for p in phrases if p.strip()]
        self._cache: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def prewarm(self) -> None:
        for phrase in self._phrases:
            try:
                pcm = b"".join(self._tts.stream(phrase))
            except Exception:  # noqa: BLE001 - a missing filler is non-fatal
                continue
            with self._lock:
                self._cache[phrase] = pcm

    def random_prelude(self) -> tuple[str, bytes] | None:
        """Return ``(text, pcm)`` for a random ready phrase, or ``None``."""
        with self._lock:
            ready = [(p, self._cache[p]) for p in self._phrases if p in self._cache]
        return random.choice(ready) if ready else None


def split_sentences(deltas: Iterable[str]) -> Iterator[str]:
    """Group a stream of text deltas into whole sentences.

    A sentence boundary is end punctuation followed by whitespace (so "9.5" or
    "www.site.com" don't split mid-token), or a newline. Whatever's left when
    the stream ends is emitted as a final sentence.
    """
    buf: list[str] = []
    pending_end = False
    for delta in deltas:
        for ch in delta:
            if pending_end and ch.isspace():
                text = "".join(buf).strip()
                if text:
                    yield text
                buf = []
                pending_end = False
                if ch == "\n":
                    continue
            buf.append(ch)
            if ch == "\n":
                text = "".join(buf).strip()
                if text:
                    yield text
                buf = []
                pending_end = False
            else:
                pending_end = ch in _SENTENCE_ENDERS
    tail = "".join(buf).strip()
    if tail:
        yield tail


def speak_stream(
    deltas: Iterable[str],
    tts: TTSProvider,
    on_sentence: Callable[[str], None] | None = None,
    prelude: tuple[str, bytes] | None = None,
    on_first_audio: Callable[[], None] | None = None,
    on_first_answer: Callable[[], None] | None = None,
) -> str:
    """Speak a streaming reply sentence-by-sentence; return the full text.

    ``prelude`` is an optional ``(text, pcm)`` acknowledgement (see
    :class:`FillerBank`) played first, through the same speaker, while the real
    answer is still being generated behind it.

    ``on_first_audio`` fires once when the first sound (prelude or answer) is
    about to play; ``on_first_answer`` fires once when the first real-answer
    audio is ready. Both are for latency timing (see :mod:`aileen.timing`).

    Synthesis and playback are decoupled and synthesis runs *ahead* of, and
    concurrently with, playback: a background thread submits each sentence to a
    small thread pool the moment its text arrives from the LLM, so the next few
    sentences are being synthesized while the current one is still playing. This
    thread plays the finished audio back in order through a single
    :class:`PcmSpeaker`. The lead this builds means a single slow TTS request
    can't starve the speaker — which is what removes both the between-sentence
    lag and the clipped onset a dry stream causes on resume. One speaker for the
    whole reply means a single warm-up pad, not one per sentence.

    ``on_sentence`` (if given) is called with each sentence right as its audio
    begins playing, so on-screen text stays in step with the voice.
    """
    # Producer submits sentences to the pool and hands the consumer ordered
    # (sentence, future-of-PCM) jobs via this queue. ``_DONE`` ends the stream.
    jobs_q: queue.Queue = queue.Queue()
    spoken: list[str] = []
    error: list[BaseException] = []
    _DONE = object()

    def synth(sentence: str) -> bytes:
        return b"".join(tts.stream(sentence))

    def produce(pool: ThreadPoolExecutor) -> None:
        try:
            for sentence in split_sentences(deltas):
                spoken.append(sentence)
                future: Future = pool.submit(synth, sentence)
                jobs_q.put((sentence, future))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            error.append(exc)
        finally:
            jobs_q.put(_DONE)

    speaker: PcmSpeaker | None = None
    fired = {"audio": False, "answer": False}

    def fire(key: str, hook: Callable[[], None] | None) -> None:
        if not fired[key]:
            fired[key] = True
            if hook is not None:
                hook()

    with ThreadPoolExecutor(max_workers=_TTS_LOOKAHEAD) as pool:
        producer = threading.Thread(target=produce, args=(pool,), daemon=True)
        producer.start()
        try:
            if prelude is not None:
                # Play the pre-rendered acknowledgement immediately; the real
                # sentences are already synthesizing in the pool behind it.
                prelude_text, prelude_pcm = prelude
                if on_sentence is not None:
                    on_sentence(prelude_text)
                if prelude_pcm:
                    fire("audio", on_first_audio)
                    speaker = PcmSpeaker(tts.sample_rate)
                    speaker.write(prelude_pcm)
            while True:
                item = jobs_q.get()
                if item is _DONE:
                    break
                sentence, future = item
                pcm = future.result()  # waits only if synthesis hasn't caught up
                if on_sentence is not None:
                    on_sentence(sentence)
                if pcm:
                    fire("answer", on_first_answer)
                    fire("audio", on_first_audio)  # no-op if the prelude played
                    if speaker is None:
                        speaker = PcmSpeaker(tts.sample_rate)
                    speaker.write(pcm)
        finally:
            if speaker is not None:
                speaker.close()
            producer.join()
    if error:
        raise error[0]
    return " ".join(spoken)
