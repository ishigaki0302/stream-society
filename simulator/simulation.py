"""Main simulation loop for stream-society."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .adapters.base import LLMAdapter
from .adapters.mock_adapter import MockLLMAdapter
from .metrics import compute_run_summary, compute_turn_metrics
from .persona import sample_personas
from .policy.base import SelectionPolicy
from .policy.factory import create_policy
from .schemas import Persona, RunConfig, RunSummary, TurnLog
from .streamer import StreamerAgent
from .viewer import ViewerAgent

logger = logging.getLogger(__name__)


class Simulation:
    """Orchestrates a full simulation run."""

    def __init__(
        self,
        config: RunConfig,
        personas: List[Persona],
        llm_adapter: Optional[LLMAdapter] = None,
    ) -> None:
        """Initialize the simulation.

        Args:
            config: Run configuration.
            personas: Pool of personas to sample from.
            llm_adapter: LLM adapter (defaults to MockLLMAdapter).
        """
        self.config = config
        self._turn_logs: List[TurnLog] = []

        # Set up LLM adapter
        self._llm = llm_adapter or MockLLMAdapter(seed=config.seed)

        # Sample viewers from persona pool
        sampled = sample_personas(personas, config.num_viewers, seed=config.seed)
        self._viewers: List[ViewerAgent] = []
        for i, persona in enumerate(sampled):
            viewer_id = f"viewer_{i:03d}"
            self._viewers.append(
                ViewerAgent(persona=persona, viewer_id=viewer_id, seed=config.seed + i)
            )

        # Create policy
        self._policy: SelectionPolicy = create_policy(config.policy, seed=config.seed)

        # Create streamer
        self._streamer = StreamerAgent(
            config=config.streamer_config,
            policy=self._policy,
            llm_adapter=self._llm,
            seed=config.seed,
        )

        logger.info(
            "Simulation initialized: run_id=%s, viewers=%d, turns=%d, policy=%s",
            config.run_id,
            config.num_viewers,
            config.num_turns,
            config.policy,
        )

    def run(self) -> List[TurnLog]:
        """Execute the simulation for num_turns turns.

        Returns:
            List of TurnLog objects, one per turn.
        """
        self._turn_logs = []

        for turn in range(self.config.num_turns):
            turn_log = self._run_turn(turn)
            self._turn_logs.append(turn_log)

            logger.debug(
                "Turn %d: %d candidates, selected=%s",
                turn,
                len(turn_log.comment_candidates),
                turn_log.selected_comment.comment_id if turn_log.selected_comment else "none",
            )

        return self._turn_logs

    def _run_turn(self, turn: int) -> TurnLog:
        """Execute a single simulation turn.

        Args:
            turn: Turn index (0-based).

        Returns:
            TurnLog for this turn.
        """
        current_topic = self._streamer.get_current_topic()

        # Step 1: Collect comment candidates from all viewers
        candidates = []
        recent_response = None
        if self._turn_logs:
            last = self._turn_logs[-1]
            if last.streamer_response:
                recent_response = last.streamer_response.response_text

        for viewer in self._viewers:
            candidate = viewer.decide_comment(
                turn=turn,
                streamer_topic=current_topic,
                recent_response=recent_response,
            )
            if candidate is not None:
                candidates.append(candidate)

        # Step 2: Streamer selects one and responds
        context = {
            "turn": turn,
            "topic": current_topic,
            "num_candidates": len(candidates),
        }
        response = self._streamer.select_and_respond(candidates, turn, context)

        # Step 3: Update viewer states
        selected_viewer_id = response.selected_viewer_id if response else None
        for viewer in self._viewers:
            was_selected = viewer.viewer_id == selected_viewer_id
            viewer.update_state(
                was_selected=was_selected,
                streamer_response=response.response_text if response else None,
            )

        # Step 4: Build turn log
        selected_comment = None
        if response:
            selected_comment = next(
                (c for c in candidates if c.comment_id == response.selected_comment_id),
                None,
            )

        turn_log = TurnLog(
            turn_id=turn,
            timestamp_turn=turn,
            comment_candidates=candidates,
            selected_comment=selected_comment,
            streamer_response=response,
            active_viewers=len(self._viewers),
            metrics={},
        )

        # Step 5: Compute turn metrics
        metrics = compute_turn_metrics(turn_log, self._turn_logs)
        turn_log.metrics = metrics

        return turn_log

    def get_summary(self) -> RunSummary:
        """Compute run summary from completed turn logs.

        Returns:
            RunSummary with aggregated metrics.
        """
        return compute_run_summary(self.config, self._turn_logs)

    def save(self, output_dir: Path) -> None:
        """Save simulation results to disk.

        Args:
            output_dir: Directory to save results in. A subdirectory named
                        by run_id will be created.
        """
        run_dir = output_dir / self.config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save turn logs as JSONL
        turns_path = run_dir / "turns.jsonl"
        with open(turns_path, "w", encoding="utf-8") as f:
            for turn_log in self._turn_logs:
                f.write(turn_log.model_dump_json() + "\n")

        # Save config
        config_path = run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self.config.model_dump_json(indent=2))

        # Save summary
        summary = self.get_summary()
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        logger.info("Results saved to %s", run_dir)
