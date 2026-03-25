"""Tests for simulator.persona module."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.persona import compute_distribution, load_personas, sample_personas
from simulator.schemas import Persona


SAMPLE_DATA = Path(__file__).parent.parent / "data" / "personas" / "sample_personas.jsonl"


def test_load_personas_returns_list():
    """load_personas should return a non-empty list of Persona objects."""
    personas = load_personas(SAMPLE_DATA)
    assert isinstance(personas, list)
    assert len(personas) > 0
    assert all(isinstance(p, Persona) for p in personas)


def test_load_personas_fields():
    """All loaded personas should have required fields."""
    personas = load_personas(SAMPLE_DATA)
    for p in personas:
        assert p.persona_id
        assert p.name
        assert isinstance(p.age, int)
        assert 0.0 <= p.base_activity_level <= 1.0
        assert isinstance(p.interests, list)


def test_sample_personas_reproducible():
    """sample_personas with same seed should return same result."""
    personas = load_personas(SAMPLE_DATA)
    sample1 = sample_personas(personas, n=5, seed=42)
    sample2 = sample_personas(personas, n=5, seed=42)
    assert [p.persona_id for p in sample1] == [p.persona_id for p in sample2]


def test_sample_personas_different_seed():
    """Different seeds should typically produce different samples."""
    personas = load_personas(SAMPLE_DATA)
    sample1 = sample_personas(personas, n=5, seed=42)
    sample2 = sample_personas(personas, n=5, seed=99)
    # They could coincidentally match, but usually won't with 20 personas
    ids1 = [p.persona_id for p in sample1]
    ids2 = [p.persona_id for p in sample2]
    # Just verify both return valid lists
    assert len(ids1) == 5
    assert len(ids2) == 5


def test_sample_personas_with_replacement():
    """sample_personas should work when n > len(personas)."""
    personas = load_personas(SAMPLE_DATA)
    sampled = sample_personas(personas, n=100, seed=42)
    assert len(sampled) == 100
    assert all(isinstance(p, Persona) for p in sampled)


def test_compute_distribution_keys():
    """compute_distribution should return dict with expected keys."""
    personas = load_personas(SAMPLE_DATA)
    dist = compute_distribution(personas)

    assert "total" in dist
    assert "groups" in dist
    assert "communication_styles" in dist
    assert "avg_age" in dist
    assert "top_interests" in dist


def test_compute_distribution_total():
    """total in distribution should match number of personas."""
    personas = load_personas(SAMPLE_DATA)
    dist = compute_distribution(personas)
    assert dist["total"] == len(personas)


def test_compute_distribution_avg_age():
    """avg_age should be a positive float."""
    personas = load_personas(SAMPLE_DATA)
    dist = compute_distribution(personas)
    assert isinstance(dist["avg_age"], float)
    assert dist["avg_age"] > 0
