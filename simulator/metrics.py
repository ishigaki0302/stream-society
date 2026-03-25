"""Metrics computation for simulation runs."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

from .schemas import RunConfig, RunSummary, TurnLog


def compute_turn_metrics(turn_log: TurnLog, all_turns: List[TurnLog]) -> Dict:
    """Compute metrics for a single turn.

    Args:
        turn_log: The current turn log.
        all_turns: All turn logs up to this point (for context).

    Returns:
        Dictionary of turn-level metrics.
    """
    candidates = turn_log.comment_candidates
    num_candidates = len(candidates)

    avg_sentiment = 0.0
    avg_toxicity = 0.0
    num_questions = 0

    if candidates:
        avg_sentiment = sum(c.sentiment for c in candidates) / num_candidates
        avg_toxicity = sum(c.toxicity_score for c in candidates) / num_candidates
        num_questions = sum(1 for c in candidates if c.question_flag)

    selected_sentiment = None
    if turn_log.selected_comment:
        selected_sentiment = turn_log.selected_comment.sentiment

    return {
        "num_candidates": num_candidates,
        "avg_sentiment": round(avg_sentiment, 4),
        "avg_toxicity": round(avg_toxicity, 4),
        "num_questions": num_questions,
        "selected_sentiment": selected_sentiment,
        "response_latency_simulated": 0.0,
    }


def _topic_entropy(turn_logs: List[TurnLog]) -> float:
    """Compute Shannon entropy of topic distribution across all comments."""
    topic_counts: Counter = Counter()
    for turn in turn_logs:
        for c in turn.comment_candidates:
            topic_counts[c.topic] += 1

    total = sum(topic_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in topic_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return round(entropy, 4)


def _sentiment_shift(turn_logs: List[TurnLog]) -> float:
    """Compute sentiment shift as delta between last and first turn avg sentiment."""
    turn_sentiments = []
    for turn in turn_logs:
        if turn.comment_candidates:
            avg = sum(c.sentiment for c in turn.comment_candidates) / len(turn.comment_candidates)
            turn_sentiments.append(avg)

    if len(turn_sentiments) < 2:
        return 0.0

    # Simple first/last delta (positive = improving sentiment)
    return round(turn_sentiments[-1] - turn_sentiments[0], 4)


def compute_run_summary(
    run_config: RunConfig,
    turn_logs: List[TurnLog],
) -> RunSummary:
    """Compute summary metrics for a completed simulation run.

    Args:
        run_config: The run configuration.
        turn_logs: All turn logs from the run.

    Returns:
        RunSummary with aggregated metrics.
    """
    total_comments = sum(len(t.comment_candidates) for t in turn_logs)
    num_turns = len(turn_logs)

    # Engagement proxy: average comments per turn, normalized by viewer count
    avg_comments_per_turn = total_comments / num_turns if num_turns > 0 else 0.0
    engagement_proxy = (
        avg_comments_per_turn / run_config.num_viewers if run_config.num_viewers > 0 else 0.0
    )

    # Unique participants
    unique_viewers: set = set()
    for turn in turn_logs:
        for c in turn.comment_candidates:
            unique_viewers.add(c.viewer_id)
    unique_participant_rate = (
        len(unique_viewers) / run_config.num_viewers if run_config.num_viewers > 0 else 0.0
    )

    # Topic diversity
    topic_diversity = _topic_entropy(turn_logs)

    # Safety rate: 1 - (toxic / total)
    total_toxic = sum(1 for t in turn_logs for c in t.comment_candidates if c.toxicity_score > 0.3)
    safety_rate = 1.0 - (total_toxic / total_comments) if total_comments > 0 else 1.0

    # Sentiment shift
    sentiment_shift = _sentiment_shift(turn_logs)

    return RunSummary(
        run_id=run_config.run_id,
        experiment_name=run_config.experiment_name,
        policy=run_config.policy,
        num_turns=num_turns,
        num_viewers=run_config.num_viewers,
        total_comments=total_comments,
        engagement_proxy=round(engagement_proxy, 4),
        unique_participant_rate=round(unique_participant_rate, 4),
        topic_diversity=topic_diversity,
        safety_rate=round(safety_rate, 4),
        sentiment_shift=sentiment_shift,
    )
