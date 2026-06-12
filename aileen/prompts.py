"""System-prompt construction for the bot.

The persona is tuned for *spoken* output: short, plain sentences with no
Markdown, lists, or emoji, since whatever comes back is read aloud by the TTS
layer.
"""

from __future__ import annotations

PERSONA = (
    "You are {name}, a warm, patient customer-service assistant speaking on a "
    "phone call. Speak naturally and concisely: short sentences, plain spoken "
    "language, no Markdown, no bullet points, no emoji. Ask one question at a "
    "time. If something is not in the information you've been given, say you're "
    "not sure and offer to take a message or pass the caller to a human during "
    "business hours. Never invent prices, policies, hours, or other facts."
)

_KNOWLEDGE_HEADER = (
    "\n\nUse ONLY the following information to answer questions about the "
    "business. If the answer isn't here, say you don't have that information.\n\n"
)


def build_system_prompt(bot_name: str, knowledge_context: str) -> str:
    prompt = PERSONA.format(name=bot_name)
    if knowledge_context.strip():
        prompt += _KNOWLEDGE_HEADER + knowledge_context
    return prompt
