"""Contract for the bot's knowledge source."""

from __future__ import annotations

from abc import ABC, abstractmethod


class KnowledgeBase(ABC):
    @abstractmethod
    def context_for(self, query: str) -> str:
        """Return relevant background text for ``query``, or "" if nothing fits.

        Returning "" is meaningful: it tells the bot it has nothing to go on,
        so it should say it doesn't know rather than guess.
        """
        raise NotImplementedError
