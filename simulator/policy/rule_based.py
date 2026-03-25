"""Rule-based comment selection policy."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..schemas import CommentCandidate
from .base import SelectionPolicy


class RuleBasedPolicy(SelectionPolicy):
    """Prioritizes questions, then high sentiment, then low toxicity."""

    name: str = "rule_based"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def select(
        self,
        candidates: List[CommentCandidate],
        context: Dict,
    ) -> Optional[CommentCandidate]:
        """Select based on rules: questions > sentiment > safety."""
        if not candidates:
            return None

        # Priority 1: questions with low toxicity
        questions = [c for c in candidates if c.question_flag and c.toxicity_score < 0.3]
        if questions:
            # Among questions, pick highest sentiment
            return max(questions, key=lambda c: c.sentiment)

        # Priority 2: filter out toxic comments
        safe = [c for c in candidates if c.toxicity_score < 0.3]
        if not safe:
            safe = candidates  # fallback: use all if all are toxic

        # Priority 3: pick highest sentiment among safe comments
        return max(safe, key=lambda c: c.sentiment)
