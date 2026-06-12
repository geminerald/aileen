"""What the bot knows. Sources implement :class:`KnowledgeBase`."""

from .base import KnowledgeBase
from .static import StaticFileKnowledge

__all__ = ["KnowledgeBase", "StaticFileKnowledge"]
