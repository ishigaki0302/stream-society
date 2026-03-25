"""Abstract base class for comment selection policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..schemas import CommentCandidate


class SelectionPolicy(ABC):
    """Abstract base for comment selection policies."""

    name: str = "base"

    @abstractmethod
    def select(
        self,
        candidates: List[CommentCandidate],
        context: Dict,
    ) -> Optional[CommentCandidate]:
        """Select a comment from candidates given context.

        Args:
            candidates: List of comment candidates for this turn.
            context: Additional context (turn number, streamer state, etc.).

        Returns:
            The selected CommentCandidate, or None if candidates is empty.
        """
        ...

    def update(self, selected: CommentCandidate, reward: float) -> None:
        """Update policy based on received reward (default no-op).

        Args:
            selected: The comment that was selected.
            reward: The reward signal received after selection.
        """
        pass
