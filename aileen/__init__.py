"""Aileen — a customer-service voice bot.

The package is organised around small, swappable pieces:

* ``aileen.llm``        — the "brain" (OpenAI or Claude, behind one interface)
* ``aileen.voice``      — text-to-speech (ElevenLabs) and speech-to-text (OpenAI)
* ``aileen.audio``      — microphone capture and speaker playback
* ``aileen.knowledge``  — what the bot knows (loaded from data/knowledge)
* ``aileen.conversation`` — the front-end-agnostic dialogue engine
* ``aileen.app``        — the CLI that wires everything into a voice loop

Front-ends (terminal today, telephony later) sit on top of ``Conversation``.
"""

__version__ = "0.1.0"
