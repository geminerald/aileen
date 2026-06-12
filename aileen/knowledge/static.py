"""A simple file-backed knowledge source.

Loads every ``.md`` / ``.txt`` file under a directory, splits each into
sections (on Markdown headings), and returns the sections that best match a
question by keyword overlap.

This is deliberately dependency-free so the framework runs out of the box.
When the knowledge grows, swap this implementation for an embedding-based
retriever (RAG) — nothing else in the app needs to change, because callers
only depend on :class:`~aileen.knowledge.base.KnowledgeBase`.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import KnowledgeBase

_WORD = re.compile(r"[a-z0-9]+")
# Words too common to be useful for matching.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "is", "are", "do", "does",
    "i", "you", "it", "for", "on", "in", "what", "how", "can", "my", "me",
    "your", "with", "be", "this", "that", "have", "we", "us",
}


class StaticFileKnowledge(KnowledgeBase):
    def __init__(self, directory: str | Path, max_sections: int = 4):
        self._directory = Path(directory)
        self._max_sections = max_sections
        self._sections = self._load()

    @property
    def is_empty(self) -> bool:
        return not self._sections

    def _load(self) -> list[str]:
        sections: list[str] = []
        if not self._directory.exists():
            return sections
        for path in sorted(self._directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                sections.extend(self._split(path.read_text(encoding="utf-8")))
        return sections

    @staticmethod
    def _split(text: str) -> list[str]:
        # Break on Markdown headings; keep each heading with its body.
        chunks = re.split(r"\n(?=#{1,6}\s)", text)
        return [c.strip() for c in chunks if c.strip()]

    @staticmethod
    def _keywords(text: str) -> set[str]:
        return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}

    def context_for(self, query: str) -> str:
        if not self._sections:
            return ""
        terms = self._keywords(query)
        if not terms:
            return ""
        scored = sorted(
            ((len(terms & self._keywords(s)), s) for s in self._sections),
            key=lambda pair: pair[0],
            reverse=True,
        )
        hits = [section for score, section in scored if score > 0]
        return "\n\n".join(hits[: self._max_sections])
