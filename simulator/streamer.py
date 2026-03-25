"""Streamer agent for the stream-society simulator."""
from __future__ import annotations

import itertools
import uuid
from typing import Dict, List, Optional

from .adapters.base import LLMAdapter
from .policy.base import SelectionPolicy
from .schemas import CommentCandidate, StreamerResponse

_DEFAULT_TOPICS = ["gaming", "music", "anime", "technology", "lifestyle"]


class StreamerAgent:
    """Simulates the AI streamer that responds to viewer comments."""

    def __init__(
        self,
        config: Dict,
        policy: SelectionPolicy,
        llm_adapter: LLMAdapter,
        seed: int = 42,
    ) -> None:
        """Initialize the streamer agent.

        Args:
            config: Streamer configuration dict (name, topics, response_style, etc.).
            policy: Comment selection policy.
            llm_adapter: LLM adapter for generating responses.
            seed: Random seed.
        """
        self.config = config
        self.policy = policy
        self.llm_adapter = llm_adapter
        self.seed = seed

        self._name = config.get("name", "Aoi")
        self._response_style = config.get("response_style", "friendly")
        topics = config.get("topics", _DEFAULT_TOPICS)
        self._topic_cycle = itertools.cycle(topics)
        self._current_topic: str = config.get("current_topic", topics[0] if topics else "gaming")
        self._response_count = 0

    def get_current_topic(self) -> str:
        """Get the current streaming topic."""
        return self._current_topic

    def select_and_respond(
        self,
        candidates: List[CommentCandidate],
        turn: int,
        context: Dict,
    ) -> Optional[StreamerResponse]:
        """Select a comment and generate a response.

        Args:
            candidates: List of comment candidates for this turn.
            turn: Current turn number.
            context: Additional context dict.

        Returns:
            A StreamerResponse if a comment was selected, else None.
        """
        if not candidates:
            # Advance topic even if no comments
            self._advance_topic()
            return None

        # Select a comment using the policy
        selected = self.policy.select(candidates, context)
        if selected is None:
            self._advance_topic()
            return None

        # Adapt topic to selected comment
        self._current_topic = selected.topic

        # Build prompt and context for LLM
        llm_context = {
            "topic": selected.topic,
            "is_question": selected.question_flag,
            "viewer_comment": selected.text,
            "response_style": self._response_style,
            "streamer_name": self._name,
            "turn": turn,
        }
        prompt = (
            f"視聴者からのコメント: 「{selected.text}」\n"
            f"トピック: {selected.topic}\n"
            f"ストリーマー{self._name}として返答してください。"
        )

        response_text = self.llm_adapter.generate_response(prompt, llm_context)

        # Compute reward for policy update
        reward = self._compute_reward(selected)
        self.policy.update(selected, reward)

        self._response_count += 1
        response_id = str(uuid.uuid4())[:8]

        return StreamerResponse(
            response_id=response_id,
            selected_comment_id=selected.comment_id,
            selected_viewer_id=selected.viewer_id,
            response_text=response_text,
            policy_used=self.policy.name,
            timestamp_turn=turn,
            metadata={
                "topic": selected.topic,
                "is_question": selected.question_flag,
                "streamer_name": self._name,
            },
        )

    def _advance_topic(self) -> None:
        """Advance to the next topic in the cycle."""
        self._current_topic = next(self._topic_cycle)

    def _compute_reward(self, selected: CommentCandidate) -> float:
        """Compute reward signal for the selected comment.

        Higher reward for:
        - Questions (encourage engagement)
        - High sentiment (positive community)
        - Low toxicity (safety)
        - High novelty (interesting content)
        """
        reward = 0.0
        reward += 0.3 * (selected.sentiment + 1.0) / 2.0  # normalize to [0,1]
        reward += 0.2 * (1.0 - selected.toxicity_score)
        reward += 0.3 * selected.novelty_score
        reward += 0.2 * (1.0 if selected.question_flag else 0.0)
        return reward
