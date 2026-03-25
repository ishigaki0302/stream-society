"""Persona management for viewer agents."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from .schemas import Persona

SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "personas" / "sample_personas.jsonl"


def load_personas(path: Optional[Path] = None) -> list[Persona]:
    """Load personas from JSONL file."""
    p = path or SAMPLE_DATA_PATH
    personas = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                personas.append(Persona(**json.loads(line)))
    return personas


def sample_personas(personas: list[Persona], n: int, seed: int = 42) -> list[Persona]:
    """Sample n personas with replacement if needed."""
    rng = random.Random(seed)
    if n <= len(personas):
        return rng.sample(personas, n)
    # Sample with replacement
    return [rng.choice(personas) for _ in range(n)]


def compute_distribution(personas: list[Persona]) -> dict:
    """Compute attribute distribution report."""
    groups = Counter(p.persona_group for p in personas)
    styles = Counter(p.communication_style for p in personas)
    avg_age = sum(p.age for p in personas) / len(personas) if personas else 0
    all_interests: list[str] = []
    for p in personas:
        all_interests.extend(p.interests)
    top_interests = Counter(all_interests).most_common(10)
    return {
        "total": len(personas),
        "groups": dict(groups),
        "communication_styles": dict(styles),
        "avg_age": round(avg_age, 1),
        "top_interests": dict(top_interests),
    }
