"""Persona ingestion pipeline for stream-society."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Import inside functions to avoid circular import issues at module level
# and to allow this module to be used standalone


def load_from_file(path: Path) -> List:
    """Load personas from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of Persona objects.
    """
    from simulator.schemas import Persona

    personas = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                persona = Persona(**raw)
                personas.append(persona)
            except Exception as e:
                logger.warning("Failed to parse line %d: %s", i, e)
    logger.info("Loaded %d personas from %s", len(personas), path)
    return personas


def normalize_persona(raw: Dict) -> object:
    """Map Nemotron-Personas-Japan fields to our Persona schema.

    This function handles field name mapping and normalization from
    the Nemotron-Personas-Japan dataset format to our internal schema.

    Args:
        raw: Raw dictionary from Nemotron-Personas-Japan dataset.

    Returns:
        Normalized Persona object.
    """
    from simulator.schemas import Persona

    # Field mapping: Nemotron-Personas-Japan -> our schema
    # Adjust these mappings as the actual dataset format becomes known
    normalized = {
        "persona_id": raw.get("id", raw.get("persona_id", f"auto_{id(raw)}")),
        "name": raw.get("name", raw.get("username", "Unknown")),
        "age": int(raw.get("age", 25)),
        "occupation": raw.get("occupation", raw.get("job", "不明")),
        "interests": _parse_interests(raw.get("interests", raw.get("hobbies", []))),
        "persona_group": raw.get("persona_group", raw.get("group", "社会人")),
        "communication_style": _normalize_style(
            raw.get("communication_style", raw.get("style", "friendly"))
        ),
        "base_activity_level": float(raw.get("base_activity_level", raw.get("activity", 0.5))),
        "language": raw.get("language", "ja"),
    }
    return Persona(**normalized)


def _parse_interests(interests_raw) -> List[str]:
    """Parse interests from various formats."""
    if isinstance(interests_raw, list):
        return [str(i) for i in interests_raw]
    if isinstance(interests_raw, str):
        # Try comma-separated
        return [i.strip() for i in interests_raw.split(",") if i.strip()]
    return []


def _normalize_style(style: str) -> str:
    """Normalize communication style to known values."""
    valid_styles = {"friendly", "analytical", "enthusiastic", "quiet", "talkative", "critical"}
    style_lower = style.lower()
    if style_lower in valid_styles:
        return style_lower
    # Common mappings
    mapping = {
        "positive": "friendly",
        "logical": "analytical",
        "energetic": "enthusiastic",
        "reserved": "quiet",
        "chatty": "talkative",
        "skeptical": "critical",
    }
    return mapping.get(style_lower, "friendly")


def save_to_parquet(personas: List, output: Path) -> None:
    """Save personas to Parquet format using polars.

    Args:
        personas: List of Persona objects.
        output: Output file path (.parquet).
    """
    import polars as pl

    records = [p.model_dump() for p in personas]
    # Flatten interests list to string for Parquet compatibility
    for r in records:
        r["interests"] = ",".join(r["interests"])
    df = pl.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)
    logger.info("Saved %d personas to %s", len(personas), output)


def save_to_jsonl(personas: List, output: Path) -> None:
    """Save personas to JSONL format.

    Args:
        personas: List of Persona objects.
        output: Output file path (.jsonl).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p in personas:
            f.write(p.model_dump_json() + "\n")
    logger.info("Saved %d personas to %s", len(personas), output)


def compute_distribution_report(personas: List) -> Dict:
    """Compute distribution report for a set of personas.

    Args:
        personas: List of Persona objects.

    Returns:
        Dictionary with distribution statistics.
    """
    from simulator.persona import compute_distribution

    return compute_distribution(personas)


def main(input_path: str, output_dir: str = "outputs/personas") -> None:
    """CLI entry point for persona ingestion.

    Args:
        input_path: Path to input JSONL file.
        output_dir: Output directory for processed personas.
    """
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    input_p = Path(input_path)
    output_p = Path(output_dir)

    personas = load_from_file(input_p)
    if not personas:
        logger.error("No personas loaded from %s", input_p)
        return

    report = compute_distribution_report(personas)
    logger.info("Distribution report: %s", json.dumps(report, ensure_ascii=False, indent=2))

    # Save outputs
    output_p.mkdir(parents=True, exist_ok=True)
    save_to_jsonl(personas, output_p / "personas_normalized.jsonl")
    save_to_parquet(personas, output_p / "personas_normalized.parquet")

    logger.info("Ingestion complete. Output: %s", output_p)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python persona_ingestion.py <input_path> [output_dir]")
        sys.exit(1)
    main(*sys.argv[1:])
