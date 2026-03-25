"""Random comment selection policy."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..schemas import CommentCandidate
from .base import SelectionPolicy


class RandomPolicy(SelectionPolicy):
    """Picks a comment uniformly at random from candidates."""

    name: str = "random"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def select(
        self,
        candidates: List[CommentCandidate],
        context: Dict,
    ) -> Optional[CommentCandidate]:
        """Select uniformly at random."""
        if not candidates:
            return None
        return self._rng.choice(candidates)
