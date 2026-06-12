# Aileen — Customer Service Voice Bot

Aileen is a customer-service bot that can start a conversation and answer
questions about what it knows. Right now it runs as a terminal **voice loop**:
you speak, it transcribes your speech, an LLM reasons over the bot's knowledge,
and [ElevenLabs](https://elevenlabs.io) speaks the reply back to you.

The conversation engine is deliberately front-end-agnostic. Today's front-end
is your terminal; the planned one is a **phone line** (so out-of-hours callers
can talk to Aileen or leave a voicemail). Moving to phones means adding a new
front-end — the brain, voice, and knowledge layers don't change.

## How it works

```
            ┌──────────── front-end (terminal today, phone later) ────────────┐
  you speak │  mic ──► speech-to-text ──► [ Conversation ] ──► text-to-speech  │ you hear
            └─────────────────────────────────┬───────────────────────────────┘
                                               │
                       ┌───────────────────────┼───────────────────────┐
                       ▼                        ▼                       ▼
                  LLM "brain"            knowledge base            (voice = ElevenLabs)
              OpenAI  /  Claude        data/knowledge/*.md
```

Every layer sits behind a small interface, so each is swappable:

| Layer            | Interface                          | Default            | Swap to            |
|------------------|------------------------------------|--------------------|--------------------|
| Brain (LLM)      | `aileen.llm.LLMProvider`           | OpenAI (ChatGPT)   | Anthropic (Claude) |
| Voice (TTS)      | `aileen.voice.tts.TTSProvider`     | ElevenLabs         | —                  |
| Hearing (STT)    | `aileen.voice.stt.STTProvider`     | OpenAI (Whisper)   | —                  |
| Knowledge        | `aileen.knowledge.KnowledgeBase`   | static files       | embeddings / RAG   |

## Setup

Requires **Python 3.10+** and a working microphone & speakers for voice mode.

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -e .            # add ".[dev]" to also install pytest
```

Then configure your keys:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Fill in `.env`:

- `OPENAI_API_KEY` — required (used as the default brain **and** for speech-to-text)
- `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` — required for the voice
- `ANTHROPIC_API_KEY` — only if you set `AILEEN_LLM_PROVIDER=anthropic`

## Run

```bash
aileen                 # full voice loop: press Enter to talk, Enter to stop
aileen --text          # type instead of speaking (reply still spoken aloud)
aileen --text --mute   # pure text chat, no audio at all (no ElevenLabs needed)
```

(`python -m aileen` works too.) Say or type "goodbye" to end the call.

### Desktop GUI (easiest way to test on a PC)

```bash
aileen-gui             # opens a window: type or "Hold to Talk", replies spoken aloud
```

(`python -m aileen.gui` works too.) The GUI drives the same conversation engine
as the CLI. Untick **Speak replies** for a quiet, text-only test (no ElevenLabs
key needed); voice input still needs `OPENAI_API_KEY` for transcription.

### Switching the brain

```bash
# in .env
AILEEN_LLM_PROVIDER=anthropic   # or "openai" (default)
```

## What the bot knows

Aileen only answers from the files in [`data/knowledge/`](data/knowledge/)
(`.md` and `.txt`). Edit [faq.md](data/knowledge/faq.md) or drop in more files —
they're loaded on startup. If nothing matches a question, Aileen says it doesn't
know rather than guessing.

The current retriever is a simple keyword match (zero dependencies). When the
knowledge grows, swap `StaticFileKnowledge` for an embedding-based retriever;
nothing else changes.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use a fake LLM and need no API keys, network, or audio hardware.

## Project layout

```
aileen/
  app.py            # CLI / terminal front-end (voice + text loops)
  gui.py            # desktop GUI front-end (Tkinter)
  conversation.py   # the front-end-agnostic dialogue engine
  config.py         # settings from environment / .env
  factory.py        # builds providers from config
  prompts.py        # system-prompt / persona
  llm/              # LLMProvider + OpenAI and Anthropic adapters
  voice/tts/        # TTSProvider + ElevenLabs
  voice/stt/        # STTProvider + OpenAI Whisper
  audio/            # microphone capture + speaker playback
  knowledge/        # KnowledgeBase + static-file retriever
data/knowledge/     # what Aileen knows (edit these)
tests/
```

## Roadmap

- [x] Terminal voice loop (speak ↔ listen)
- [ ] Telephony front-end: inbound calls, after-hours "talk to Aileen vs. leave a voicemail"
- [ ] Streaming audio for lower latency (speak while the reply is generated)
- [ ] Embedding-based knowledge retrieval (RAG) as the knowledge grows
- [ ] Call transfer / message-taking when Aileen can't help
