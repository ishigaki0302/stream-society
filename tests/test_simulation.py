"""Integration tests for simulator.simulation module."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.persona import load_personas
from simulator.schemas import RunConfig, TurnLog
from simulator.simulation import Simulation

SAMPLE_DATA = Path(__file__).parent.parent / "data" / "personas" / "sample_personas.jsonl"


def make_config(
    policy: str = "random",
    num_viewers: int = 3,
    num_turns: int = 5,
) -> RunConfig:
    return RunConfig(
        run_id=f"test_{policy}_{num_turns}",
        experiment_name="test_run",
        seed=42,
        num_viewers=num_viewers,
        num_turns=num_turns,
        policy=policy,
        streamer_config={
            "name": "TestStreamer",
            "topics": ["gaming", "music"],
            "response_style": "friendly",
        },
        description="Test run",
    )


def test_simulation_run_returns_correct_length():
    """run() should return exactly num_turns TurnLog objects."""
    personas = load_personas(SAMPLE_DATA)
    config = make_config(num_turns=5, num_viewers=3)
    sim = Simulation(config=config, personas=personas)
    logs = sim.run()

    assert isinstance(logs, list)
    assert len(logs) == 5


def test_simulation_turn_log_structure():
    """Each TurnLog should have correct structure."""
    personas = load_personas(SAMPLE_DATA)
    config = make_config(num_turns=3, num_viewers=3)
    sim = Simulation(config=config, personas=personas)
    logs = sim.run()

    for i, log in enumerate(logs):
        assert isinstance(log, TurnLog)
        assert log.turn_id == i
        assert log.timestamp_turn == i
        assert isinstance(log.comment_candidates, list)
        assert isinstance(log.active_viewers, int)
        assert isinstance(log.metrics, dict)


def test_simulation_active_viewers():
    """active_viewers should match num_viewers."""
    personas = load_personas(SAMPLE_DATA)
    config = make_config(num_turns=3, num_viewers=3)
    sim = Simulation(config=config, personas=personas)
    logs = sim.run()

    for log in logs:
        assert log.active_viewers == 3


def test_simulation_run_summary_computed():
    """get_summary() should return a valid RunSummary."""
    personas = load_personas(SAMPLE_DATA)
    config = make_config(num_turns=5, num_viewers=3)
    sim = Simulation(config=config, personas=personas)
    sim.run()
    summary = sim.get_summary()

    assert summary.run_id == config.run_id
    assert summary.policy == "random"
    assert summary.num_turns == 5
    assert summary.num_viewers == 3
    assert 0.0 <= summary.safety_rate <= 1.0
    assert 0.0 <= summary.unique_participant_rate <= 1.0
    assert summary.topic_diversity >= 0.0


def test_simulation_all_policies():
    """All available policies should run without error."""
    personas = load_personas(SAMPLE_DATA)
    policies = ["random", "rule_based", "score_based", "contextual_bandit"]

    for policy in policies:
        config = make_config(policy=policy, num_turns=3, num_viewers=3)
        sim = Simulation(config=config, personas=personas)
        logs = sim.run()
        assert len(logs) == 3, f"Policy {policy} did not produce 3 turns"


def test_simulation_reproducibility():
    """Same seed should produce same turn count and same first candidates."""
    personas = load_personas(SAMPLE_DATA)
    config1 = make_config(num_turns=5, num_viewers=3)
    config2 = make_config(num_turns=5, num_viewers=3)

    sim1 = Simulation(config=config1, personas=personas)
    logs1 = sim1.run()

    sim2 = Simulation(config=config2, personas=personas)
    logs2 = sim2.run()

    assert len(logs1) == len(logs2)

    # Check first turn has same number of candidates
    assert len(logs1[0].comment_candidates) == len(logs2[0].comment_candidates)


def test_simulation_save(tmp_path):
    """save() should create expected files in output directory."""
    personas = load_personas(SAMPLE_DATA)
    config = make_config(num_turns=3, num_viewers=3)
    sim = Simulation(config=config, personas=personas)
    sim.run()
    sim.save(tmp_path)

    run_dir = tmp_path / config.run_id
    assert run_dir.exists()
    assert (run_dir / "turns.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config.json").exists()


def test_simulation_turn_metrics_populated():
    """Turn metrics dict should be populated after run."""
    personas = load_personas(SAMPLE_DATA)
    config = make_config(num_turns=5, num_viewers=5)
    sim = Simulation(config=config, personas=personas)
    logs = sim.run()

    for log in logs:
        assert "num_candidates" in log.metrics
        assert "avg_sentiment" in log.metrics
        assert "response_latency_simulated" in log.metrics
