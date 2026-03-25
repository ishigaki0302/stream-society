"""Contextual Bandit (LinUCB) comment selection policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..schemas import CommentCandidate
from .base import SelectionPolicy

# Feature vector:
# [0] sentiment  (normalized to [0,1])
# [1] novelty_score
# [2] safety     (1 - toxicity)
# [3] question_flag (0 or 1)
# [4] bias term  (always 1.0)
_FEATURE_DIM = 5
_LAMBDA = 1.0  # regularization (初期 A = λI)


def _extract_features(candidate: CommentCandidate) -> np.ndarray:
    """Extract feature vector from a comment candidate."""
    return np.array(
        [
            (candidate.sentiment + 1.0) / 2.0,  # normalize [-1,1] → [0,1]
            candidate.novelty_score,
            1.0 - candidate.toxicity_score,
            1.0 if candidate.question_flag else 0.0,
            1.0,  # bias
        ],
        dtype=float,
    )


class ContextualBanditPolicy(SelectionPolicy):
    """Disjoint LinUCB による contextual bandit コメント選択方策。

    Li et al. (2010) "A Contextual-Bandit Approach to Personalized News Article
    Recommendation" の Disjoint LinUCB アルゴリズムを実装。
    全候補に共通の (A, b) を持つ shared パラメータモデルを採用。

    UCB score = θᵀx + α * sqrt(xᵀ A⁻¹ x)
    更新:  A ← A + xxᵀ,  b ← b + r·x,  θ = A⁻¹b
    """

    name: str = "contextual_bandit"

    def __init__(self, alpha: float = 1.0, seed: int = 42) -> None:
        """Initialize the LinUCB policy.

        Args:
            alpha: 探索パラメータ。大きいほど不確実性を重視して探索。
            seed: 乱数シード（再現性のため保持するが本実装では未使用）。
        """
        self.alpha = alpha
        self._seed = seed

        # A: d×d 正定値行列 (初期値 = λI)
        self._A: np.ndarray = _LAMBDA * np.eye(_FEATURE_DIM)
        # b: d 次元ベクトル
        self._b: np.ndarray = np.zeros(_FEATURE_DIM)
        # A⁻¹ のキャッシュ (更新のたびに再計算)
        self._A_inv: np.ndarray = (1.0 / _LAMBDA) * np.eye(_FEATURE_DIM)
        # θ = A⁻¹ b
        self._theta: np.ndarray = np.zeros(_FEATURE_DIM)

        self._t = 0  # update step counter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ucb_score(self, x: np.ndarray) -> float:
        """UCB スコアを計算する。

        score = θᵀx + α * sqrt(xᵀ A⁻¹ x)
        """
        exploit = float(self._theta @ x)
        variance = float(x @ self._A_inv @ x)
        # 数値誤差で負になった場合のガード
        explore = self.alpha * float(np.sqrt(max(variance, 0.0)))
        return exploit + explore

    def _recompute(self) -> None:
        """A_inv と θ を再計算する。"""
        # np.linalg.solve(A, I) は A⁻¹ より数値的に安定
        self._A_inv = np.linalg.solve(self._A, np.eye(_FEATURE_DIM))
        self._theta = self._A_inv @ self._b

    # ------------------------------------------------------------------
    # Public interface (SelectionPolicy)
    # ------------------------------------------------------------------

    def select(
        self,
        candidates: List[CommentCandidate],
        context: Dict,
    ) -> Optional[CommentCandidate]:
        """UCB スコアが最大の候補を選択する。"""
        if not candidates:
            return None

        best_score = float("-inf")
        best = candidates[0]
        for c in candidates:
            x = _extract_features(c)
            score = self._ucb_score(x)
            if score > best_score:
                best_score = score
                best = c
        return best

    def update(self, selected: CommentCandidate, reward: float) -> None:
        """観測した報酬で (A, b, θ) を更新する。

        A ← A + xxᵀ
        b ← b + r·x
        θ = A⁻¹b  (再計算)
        """
        x = _extract_features(selected)
        self._A += np.outer(x, x)
        self._b += reward * x
        self._recompute()
        self._t += 1

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path) -> None:
        """学習済みパラメータを JSON で保存する（run をまたぐ継続学習用）。"""
        state = {
            "alpha": self.alpha,
            "t": self._t,
            "A": self._A.tolist(),
            "b": self._b.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def load_state(self, path: Path) -> None:
        """保存済みパラメータを読み込む。"""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        self.alpha = state["alpha"]
        self._t = state["t"]
        self._A = np.array(state["A"])
        self._b = np.array(state["b"])
        self._recompute()
