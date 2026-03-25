"""Score-based comment selection policy."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..schemas import CommentCandidate
from .base import SelectionPolicy


class ScoreBasedPolicy(SelectionPolicy):
    """Weighted score policy: 0.3*sentiment + 0.2*(1-toxicity) + 0.3*novelty + 0.2*question."""

    name: str = "score_based"

    def __init__(
        self,
        w_sentiment: float = 0.3,
        w_safety: float = 0.2,
        w_novelty: float = 0.3,
        w_question: float = 0.2,
        **kwargs,
    ) -> None:
        self.w_sentiment = w_sentiment
        self.w_safety = w_safety
        self.w_novelty = w_novelty
        self.w_question = w_question

    def _score(self, candidate: CommentCandidate) -> float:
        """Compute weighted score for a candidate."""
        # Normalize sentiment from [-1, 1] to [0, 1]
        norm_sentiment = (candidate.sentiment + 1.0) / 2.0
        safety = 1.0 - candidate.toxicity_score
        question_val = 1.0 if candidate.question_flag else 0.0
        return (
            self.w_sentiment * norm_sentiment
            + self.w_safety * safety
            + self.w_novelty * candidate.novelty_score
            + self.w_question * question_val
        )

    def select(
        self,
        candidates: List[CommentCandidate],
        context: Dict,
    ) -> Optional[CommentCandidate]:
        """Select candidate with highest weighted score."""
        if not candidates:
            return None
        return max(candidates, key=self._score)
