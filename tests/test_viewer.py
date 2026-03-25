"""Tests for simulator.viewer module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.schemas import CommentCandidate, Persona
from simulator.viewer import ViewerAgent


def make_persona(persona_id: str = "p001", style: str = "friendly") -> Persona:
    return Persona(
        persona_id=persona_id,
        name="テスト太郎",
        age=25,
        occupation="大学生",
        interests=["gaming", "anime"],
        persona_group="学生",
        communication_style=style,
        base_activity_level=0.8,
        language="ja",
    )


def test_viewer_initializes_correctly():
    """ViewerAgent should initialize with correct viewer_id and persona."""
    persona = make_persona()
    agent = ViewerAgent(persona=persona, viewer_id="viewer_001", seed=42)
    assert agent.viewer_id == "viewer_001"
    assert agent.persona.persona_id == "p001"
    assert agent.state.viewer_id == "viewer_001"
    assert 0.0 <= agent.state.affinity_to_streamer <= 1.0
    assert 0.0 <= agent.state.activity_level <= 1.0


def test_decide_comment_returns_none_or_candidate():
    """decide_comment should return None or CommentCandidate."""
    persona = make_persona()
    agent = ViewerAgent(persona=persona, viewer_id="viewer_001", seed=42)

    result = agent.decide_comment(turn=0, streamer_topic="gaming", recent_response=None)
    assert result is None or isinstance(result, CommentCandidate)


def test_decide_comment_candidate_fields():
    """Generated CommentCandidate should have all required fields."""
    # Use high activity to ensure comment is generated
    persona = Persona(
        persona_id="p001",
        name="テスト太郎",
        age=25,
        occupation="大学生",
        interests=["gaming"],
        persona_group="学生",
        communication_style="enthusiastic",
        base_activity_level=1.0,
        language="ja",
    )
    agent = ViewerAgent(persona=persona, viewer_id="viewer_001", seed=42)

    # Try multiple turns to get at least one comment
    candidate = None
    for turn in range(20):
        candidate = agent.decide_comment(turn=turn, streamer_topic="gaming", recent_response=None)
        if candidate is not None:
            break

    assert candidate is not None, "Expected at least one comment with activity_level=1.0"
    assert candidate.viewer_id == "viewer_001"
    assert candidate.persona_id == "p001"
    assert candidate.text
    assert isinstance(candidate.question_flag, bool)
    assert -1.0 <= candidate.sentiment <= 1.0
    assert 0.0 <= candidate.toxicity_score <= 1.0
    assert 0.0 <= candidate.novelty_score <= 1.0


def test_decide_comment_reproducible():
    """Same seed and turn should produce same result."""
    persona = make_persona()
    agent1 = ViewerAgent(persona=persona, viewer_id="v1", seed=42)
    agent2 = ViewerAgent(persona=persona, viewer_id="v1", seed=42)

    result1 = agent1.decide_comment(turn=5, streamer_topic="anime", recent_response=None)
    result2 = agent2.decide_comment(turn=5, streamer_topic="anime", recent_response=None)

    if result1 is None:
        assert result2 is None
    else:
        assert result2 is not None
        assert result1.text == result2.text
        assert result1.sentiment == result2.sentiment


def test_update_state_affinity_boost_on_selection():
    """update_state should boost affinity when viewer was selected."""
    persona = make_persona()
    agent = ViewerAgent(persona=persona, viewer_id="v1", seed=42)

    initial_affinity = agent.state.affinity_to_streamer
    agent.update_state(was_selected=True, streamer_response="ありがとう！")

    assert agent.state.affinity_to_streamer >= initial_affinity
    assert agent.state.emotion_state == "happy"


def test_update_state_affinity_slight_decay_not_selected():
    """update_state should slightly decrease affinity when not selected."""
    persona = make_persona()
    agent = ViewerAgent(persona=persona, viewer_id="v1", seed=42)
    agent.state.affinity_to_streamer = 0.5

    agent.update_state(was_selected=False, streamer_response=None)

    assert agent.state.affinity_to_streamer <= 0.5


def test_all_communication_styles():
    """All communication styles should produce valid comments."""
    styles = ["friendly", "analytical", "enthusiastic", "quiet", "talkative", "critical"]
    for style in styles:
        persona = Persona(
            persona_id=f"p_{style}",
            name="テスト",
            age=25,
            occupation="テスト",
            interests=["gaming"],
            persona_group="テスト",
            communication_style=style,
            base_activity_level=1.0,
            language="ja",
        )
        agent = ViewerAgent(persona=persona, viewer_id="v_test", seed=1)
        candidate = None
        for turn in range(10):
            candidate = agent.decide_comment(
                turn=turn, streamer_topic="gaming", recent_response=None
            )
            if candidate is not None:
                break
        # Should have produced at least one comment
        assert candidate is not None, f"No comment for style={style}"


def test_topic_diversity_across_turns():
    """Viewer with diverse interests should produce multiple topics."""
    persona = Persona(
        persona_id="p_div",
        name="多様子",
        age=22,
        occupation="フリーター",
        interests=["ゲーム", "音楽", "アニメ", "料理", "旅行", "スポーツ"],
        persona_group="フリーター",
        communication_style="talkative",
        base_activity_level=0.95,
        language="ja",
    )
    agent = ViewerAgent(persona=persona, viewer_id="v_div", seed=0)
    topics = set()
    for turn in range(30):
        c = agent.decide_comment(turn, "gaming", None)
        if c:
            topics.add(c.topic)
    # 多様なインタレストがあるので複数トピックが出るはず
    assert len(topics) > 1, f"Expected >1 topics, got {topics}"
