"""Tests for simulator.policy modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.schemas import CommentCandidate
from simulator.policy.random_policy import RandomPolicy
from simulator.policy.rule_based import RuleBasedPolicy
from simulator.policy.score_based import ScoreBasedPolicy
from simulator.policy.contextual_bandit_stub import ContextualBanditPolicy
from simulator.policy.factory import create_policy, POLICIES


def make_comment(
    comment_id: str = "c1",
    viewer_id: str = "v1",
    text: str = "テスト",
    sentiment: float = 0.5,
    question_flag: bool = False,
    toxicity_score: float = 0.01,
    novelty_score: float = 0.5,
    topic: str = "gaming",
) -> CommentCandidate:
    return CommentCandidate(
        comment_id=comment_id,
        viewer_id=viewer_id,
        persona_id="p1",
        persona_group="学生",
        text=text,
        timestamp_turn=0,
        topic=topic,
        sentiment=sentiment,
        question_flag=question_flag,
        toxicity_score=toxicity_score,
        novelty_score=novelty_score,
    )


# --- RandomPolicy ---


def test_random_returns_none_for_empty():
    policy = RandomPolicy(seed=42)
    assert policy.select([], {}) is None


def test_random_returns_candidate():
    policy = RandomPolicy(seed=42)
    candidates = [make_comment(f"c{i}") for i in range(5)]
    result = policy.select(candidates, {})
    assert result is not None
    assert result in candidates


def test_random_single_candidate():
    policy = RandomPolicy(seed=42)
    c = make_comment()
    assert policy.select([c], {}) == c


# --- RuleBasedPolicy ---


def test_rule_based_returns_none_for_empty():
    policy = RuleBasedPolicy(seed=42)
    assert policy.select([], {}) is None


def test_rule_based_returns_candidate():
    policy = RuleBasedPolicy(seed=42)
    candidates = [make_comment(f"c{i}") for i in range(3)]
    result = policy.select(candidates, {})
    assert result is not None
    assert result in candidates


def test_rule_based_prefers_questions():
    """RuleBasedPolicy should prefer questions over non-questions."""
    policy = RuleBasedPolicy(seed=42)
    non_question = make_comment("c1", sentiment=0.9, question_flag=False)
    question = make_comment("c2", sentiment=0.1, question_flag=True, toxicity_score=0.01)
    result = policy.select([non_question, question], {})
    assert result == question


def test_rule_based_avoids_toxic():
    """RuleBasedPolicy should avoid toxic comments."""
    policy = RuleBasedPolicy(seed=42)
    toxic = make_comment("c1", sentiment=1.0, toxicity_score=0.9)
    safe = make_comment("c2", sentiment=0.0, toxicity_score=0.01)
    result = policy.select([toxic, safe], {})
    assert result == safe


def test_rule_based_high_sentiment_among_safe():
    """RuleBasedPolicy should pick highest sentiment among safe non-questions."""
    policy = RuleBasedPolicy(seed=42)
    low = make_comment("c1", sentiment=0.1, toxicity_score=0.0)
    high = make_comment("c2", sentiment=0.9, toxicity_score=0.0)
    mid = make_comment("c3", sentiment=0.5, toxicity_score=0.0)
    result = policy.select([low, high, mid], {})
    assert result == high


# --- ScoreBasedPolicy ---


def test_score_based_returns_none_for_empty():
    policy = ScoreBasedPolicy()
    assert policy.select([], {}) is None


def test_score_based_returns_candidate():
    policy = ScoreBasedPolicy()
    candidates = [make_comment(f"c{i}") for i in range(3)]
    result = policy.select(candidates, {})
    assert result is not None
    assert result in candidates


def test_score_based_picks_highest_score():
    """ScoreBasedPolicy should pick the comment with the highest weighted score."""
    policy = ScoreBasedPolicy()
    low = make_comment(
        "c1", sentiment=-0.5, novelty_score=0.1, toxicity_score=0.5, question_flag=False
    )
    high = make_comment(
        "c2", sentiment=1.0, novelty_score=1.0, toxicity_score=0.0, question_flag=True
    )
    result = policy.select([low, high], {})
    assert result == high


def test_score_based_question_bonus():
    """ScoreBasedPolicy should give bonus to questions."""
    policy = ScoreBasedPolicy()
    no_q = make_comment(
        "c1", sentiment=0.5, novelty_score=0.5, toxicity_score=0.0, question_flag=False
    )
    q = make_comment("c2", sentiment=0.5, novelty_score=0.5, toxicity_score=0.0, question_flag=True)
    result = policy.select([no_q, q], {})
    assert result == q  # question gets bonus 0.2


# --- ContextualBanditPolicy ---


def test_bandit_returns_none_for_empty():
    policy = ContextualBanditPolicy(alpha=1.0, seed=42)
    assert policy.select([], {}) is None


def test_bandit_returns_candidate():
    policy = ContextualBanditPolicy(alpha=1.0, seed=42)
    candidates = [make_comment(f"c{i}") for i in range(3)]
    result = policy.select(candidates, {})
    assert result is not None
    assert result in candidates


def test_bandit_update_does_not_crash():
    """update() should not raise any exceptions."""
    policy = ContextualBanditPolicy(alpha=1.0, seed=42)
    c = make_comment()
    policy.update(c, reward=0.8)  # Should not raise


# --- Factory ---


def test_create_policy_random():
    policy = create_policy("random", seed=42)
    assert policy.name == "random"


def test_create_policy_rule_based():
    policy = create_policy("rule_based")
    assert policy.name == "rule_based"


def test_create_policy_score_based():
    policy = create_policy("score_based")
    assert policy.name == "score_based"


def test_create_policy_bandit():
    policy = create_policy("contextual_bandit")
    assert policy.name == "contextual_bandit"


def test_create_policy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown policy"):
        create_policy("nonexistent_policy")


def test_all_policies_in_registry():
    expected = {"random", "rule_based", "score_based", "contextual_bandit"}
    assert set(POLICIES.keys()) == expected


def test_bandit_learns_to_prefer_questions():
    """LinUCB は質問コメントに高報酬を与え続けると質問を優先するようになる。"""
    from simulator.policy.contextual_bandit_stub import ContextualBanditPolicy

    policy = ContextualBanditPolicy(alpha=0.1, seed=0)  # 低alpha=搾取優先

    # 質問コメント: question_flag=True に高報酬を繰り返す
    q_comment = make_comment(
        comment_id="q1", sentiment=0.0, novelty_score=0.5, toxicity_score=0.05, question_flag=True
    )
    nq_comment = make_comment(
        comment_id="nq1", sentiment=0.5, novelty_score=0.5, toxicity_score=0.05, question_flag=False
    )

    for _ in range(50):
        policy.update(q_comment, reward=1.0)
        policy.update(nq_comment, reward=0.0)

    # 学習後は質問コメントが選ばれるべき
    selected = policy.select([nq_comment, q_comment], {})
    assert selected is not None
    assert selected.question_flag is True, "学習後は質問コメントを優先するはず"


def test_bandit_save_load_state(tmp_path):
    """save_state / load_state でパラメータが正しく復元される。"""
    import numpy as np
    from simulator.policy.contextual_bandit_stub import ContextualBanditPolicy

    policy = ContextualBanditPolicy(alpha=1.0)
    c = make_comment(
        comment_id="c1", sentiment=0.5, novelty_score=0.8, toxicity_score=0.0, question_flag=True
    )
    policy.update(c, reward=1.0)

    state_path = tmp_path / "bandit_state.json"
    policy.save_state(state_path)

    policy2 = ContextualBanditPolicy(alpha=1.0)
    policy2.load_state(state_path)

    np.testing.assert_allclose(policy._A, policy2._A)
    np.testing.assert_allclose(policy._b, policy2._b)
    np.testing.assert_allclose(policy._theta, policy2._theta)
    assert policy2._t == 1
