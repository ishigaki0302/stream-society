"""Contextual Bandit (LinUCB-style) comment selection policy stub."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..schemas import CommentCandidate
from .base import SelectionPolicy

# Feature vector indices:
# [0] sentiment (normalized 0-1)
# [1] novelty_score
# [2] safety (1 - toxicity)
# [3] question_flag (0 or 1)
# [4] bias term (always 1.0)
_FEATURE_DIM = 5


def _extract_features(candidate: CommentCandidate) -> list[float]:
    """Extract feature vector from a comment candidate."""
    return [
        (candidate.sentiment + 1.0) / 2.0,  # normalize to [0,1]
        candidate.novelty_score,
        1.0 - candidate.toxicity_score,
        1.0 if candidate.question_flag else 0.0,
        1.0,  # bias
    ]


class ContextualBanditPolicy(SelectionPolicy):
    """LinUCB-style contextual bandit for comment selection.

    This is a stub implementation that has the correct interface
    but uses simplified weight updates. See docs/contextual_bandit_todo.md
    for the full implementation plan.
    """

    name: str = "contextual_bandit"

    def __init__(self, alpha: float = 1.0, seed: int = 42) -> None:
        """Initialize the contextual bandit.

        Args:
            alpha: Exploration parameter. Higher = more exploration.
            seed: Random seed (unused in stub but kept for interface compatibility).
        """
        self.alpha = alpha
        self._seed = seed

        # Stub: weight vector (theta) initialized to equal weights
        # In full LinUCB: maintain A matrix (d x d) and b vector (d x 1) per arm
        self._theta: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2]

        # Simplified covariance approximation (diagonal only in stub)
        # Full impl would use A_inv = (X^T X + lambda*I)^{-1}
        self._A_diag: list[float] = [1.0] * _FEATURE_DIM
        self._b: list[float] = [0.0] * _FEATURE_DIM

        self._t = 0  # step counter

    def _dot(self, a: list[float], b: list[float]) -> float:
        """Dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))

    def _ucb_score(self, features: list[float]) -> float:
        """Compute UCB score for a feature vector.

        Full LinUCB: score = theta^T x + alpha * sqrt(x^T A^{-1} x)
        Stub uses diagonal approximation for A^{-1}.
        """
        # Expected reward estimate
        exploit = self._dot(self._theta, features)

        # Uncertainty estimate (diagonal approximation)
        variance = sum(
            features[i] ** 2 / max(self._A_diag[i], 1e-6)
            for i in range(_FEATURE_DIM)
        )
        explore = self.alpha * math.sqrt(variance)

        return exploit + explore

    def select(
        self,
        candidates: List[CommentCandidate],
        context: Dict,
    ) -> Optional[CommentCandidate]:
        """Select candidate with highest UCB score."""
        if not candidates:
            return None

        scored = [
            (self._ucb_score(_extract_features(c)), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def update(self, selected: CommentCandidate, reward: float) -> None:
        """Update weights based on observed reward.

        Full LinUCB update:
            A <- A + x x^T
            b <- b + r x
            theta <- A^{-1} b

        Stub uses simplified online gradient update.
        """
        features = _extract_features(selected)
        self._t += 1

        # Simplified update: gradient step on squared loss
        # Full impl: Sherman-Morrison rank-1 update on A_inv
        predicted = self._dot(self._theta, features)
        error = reward - predicted
        lr = 0.1 / math.sqrt(self._t)

        for i in range(_FEATURE_DIM):
            self._theta[i] += lr * error * features[i]
            self._A_diag[i] += features[i] ** 2
            self._b[i] += reward * features[i]

        # TODO: Full implementation should:
        # 1. Maintain A matrix (d x d) initialized as identity * lambda
        # 2. Use Sherman-Morrison formula for O(d^2) rank-1 update of A_inv
        # 3. Recompute theta = A_inv @ b after each update
        # See docs/contextual_bandit_todo.md for details
