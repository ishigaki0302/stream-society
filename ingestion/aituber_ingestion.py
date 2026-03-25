"""AItuber-Personas-Japan dataset ingestion."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import here to avoid hard dependency
try:
    from datasets import load_dataset as hf_load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

DATASET_NAME = "DataPilot/AItuber-Personas-Japan"
DEFAULT_OUTPUT = Path("data/personas/aituber_personas.jsonl")
SAMPLE_FALLBACK = Path(__file__).parent.parent / "data" / "personas" / "aituber_sample.jsonl"

# Topic label mapping from Japanese keywords to english
_TOPIC_KEYWORD_MAP = {
    "ゲーム": "gaming",
    "game": "gaming",
    "Gaming": "gaming",
    "音楽": "music",
    "歌": "music",
    "ライブ": "music",
    "アニメ": "anime",
    "マンガ": "anime",
    "漫画": "anime",
    "技術": "technology",
    "プログラミング": "technology",
    "IT": "technology",
    "AI": "technology",
    "料理": "cooking",
    "グルメ": "cooking",
    "食": "cooking",
    "旅行": "travel",
    "旅": "travel",
    "スポーツ": "sports",
    "格闘": "sports",
    "武道": "sports",
    "ライフスタイル": "lifestyle",
    "日常": "lifestyle",
    "雑談": "lifestyle",
    "癒し": "lifestyle",
}

# Genre inference keyword list
_GENRE_RULES = [
    ("ASMR", "ASMR系"),
    ("asmr", "ASMR系"),
    ("料理", "料理・グルメ系"),
    ("グルメ", "料理・グルメ系"),
    ("食", "料理・グルメ系"),
    ("ゲーム", "ゲーム系"),
    ("game", "ゲーム系"),
    ("音楽", "音楽・歌系"),
    ("歌", "音楽・歌系"),
    ("ライブ", "音楽・歌系"),
    ("プログラミング", "技術・プログラミング系"),
    ("技術", "技術・プログラミング系"),
    ("IT", "技術・プログラミング系"),
    ("オカルト", "オカルト系"),
    ("タロット", "オカルト系"),
    ("占い", "オカルト系"),
    ("霊", "オカルト系"),
    ("学術", "学術・教育系"),
    ("教育", "学術・教育系"),
    ("勉強", "学術・教育系"),
    ("ラップ", "音楽・歌系"),
]

# Personality keyword → communication style mapping
_STYLE_MAP = {
    "ツンデレ": "critical",
    "毒舌": "critical",
    "批評": "critical",
    "クール": "quiet",
    "省エネ": "quiet",
    "おとなしい": "quiet",
    "無口": "quiet",
    "ダウナー": "quiet",
    "知的": "analytical",
    "分析": "analytical",
    "論理": "analytical",
    "学術": "analytical",
    "元気": "enthusiastic",
    "明るい": "enthusiastic",
    "テンション": "enthusiastic",
    "エネルギー": "enthusiastic",
    "おしゃべり": "talkative",
    "話好き": "talkative",
    "陽気": "talkative",
    "ギャル": "talkative",
    "親切": "friendly",
    "優しい": "friendly",
    "癒し": "friendly",
    "穏やか": "friendly",
}

# Response style mapping from personality keywords
_RESPONSE_STYLE_MAP = {
    "ツンデレ": "tsundere",
    "クール": "cool",
    "元気": "energetic",
    "毒舌": "sarcastic",
    "ゴスロリ": "gothic",
    "癒し": "soothing",
    "おしゃべり": "talkative",
    "ギャル": "gyaru",
}


def extract_name(concept: str) -> Tuple[str, str]:
    """Extract character name and reading (furigana) from concept document.

    Args:
        concept: Markdown concept document text.

    Returns:
        Tuple of (name, reading). reading may be empty string.
    """
    # Try table row pattern: | 名前（フリガナ） | 潮凪碧（しおなぎあお）|
    m = re.search(r"\|\s*名前(?:（フリガナ）)?\s*\|\s*(.+?)\s*\|", concept)
    if m:
        raw = m.group(1).strip()
        # Separate name from reading: 潮凪碧（しおなぎあお）
        name_m = re.match(r"^(.+?)(?:（(.+?)）)?$", raw)
        if name_m:
            name = name_m.group(1).strip()
            reading = name_m.group(2).strip() if name_m.group(2) else ""
            return name, reading

    # Fallback: first H1 header "# XXXコンセプト設計書"
    h1 = re.search(r"^#\s+(.+?)(?:コンセプト設計書)?$", concept, re.MULTILINE)
    if h1:
        name = h1.group(1).strip()
        return name, ""

    return "Unknown", ""


def extract_personality_keywords(concept: str) -> List[str]:
    """Extract personality keywords from concept document.

    Looks for a 性格キーワード section and extracts bold items.

    Args:
        concept: Markdown concept document text.

    Returns:
        List of keyword strings.
    """
    # Find the section
    section_m = re.search(r"性格キーワード[^\n]*\n(.*?)(?=\n#{1,3}\s|\Z)", concept, re.DOTALL)
    if not section_m:
        return []

    section = section_m.group(1)
    # Extract bold items **keyword**
    raw_keywords = re.findall(r"\*\*([^*]+)\*\*", section)
    keywords = []
    for kw in raw_keywords:
        # Take first part before : or ：
        kw = re.split(r"[：:]", kw)[0].strip()
        if kw:
            keywords.append(kw)
    return keywords


def extract_speech_style(system_prompt: str) -> str:
    """Extract speech / speaking style from system prompt.

    Args:
        system_prompt: LLM system prompt text.

    Returns:
        Speech style string (up to 50 chars), or "".
    """
    m = re.search(r"(?:口調|話し方)[：:]\s*(.+?)[\n。]", system_prompt)
    if m:
        return m.group(1).strip()[:50]
    return ""


def extract_pronoun(concept: str, system_prompt: str) -> str:
    """Extract first-person pronoun (一人称) from system prompt or concept.

    Args:
        concept: Markdown concept document text.
        system_prompt: LLM system prompt text.

    Returns:
        Pronoun string, e.g. "俺", "私", "ボク", or "".
    """
    pattern = r"一人称[：:]\s*(.+?)[\n,，（「。]"
    m = re.search(pattern, system_prompt)
    if m:
        return m.group(1).strip()
    m = re.search(pattern, concept)
    if m:
        return m.group(1).strip()
    return ""


def extract_fan_name(concept: str) -> str:
    """Extract fan name (ファンネーム) from concept document.

    Args:
        concept: Markdown concept document text.

    Returns:
        Fan name string, or "".
    """
    m = re.search(r"ファンネーム\s*\|\s*(.+?)\s*\|", concept)
    if m:
        return m.group(1).strip()
    return ""


def extract_gender_presentation(concept: str, system_prompt: str) -> str:
    """Extract gender presentation hint from concept or system prompt.

    Args:
        concept: Markdown concept document text.
        system_prompt: LLM system prompt text.

    Returns:
        Gender presentation string, or "".
    """
    for text in (concept, system_prompt):
        m = re.search(r"性別[：:]\s*(.+?)[\n,，]", text)
        if m:
            return m.group(1).strip()
    return ""


def infer_genre_from_themes(themes: List[dict]) -> str:
    """Infer genre label from broadcast themes via keyword matching.

    Args:
        themes: List of {"title": str, "content": str} dicts.

    Returns:
        Genre string from the fixed genre list.
    """
    # Combine all theme titles and content for matching
    combined = " ".join(t.get("title", "") + " " + t.get("content", "") for t in themes)
    for keyword, genre in _GENRE_RULES:
        if keyword in combined:
            return genre
    return "雑談・癒し系"


def _map_themes_to_topics(themes: List[dict]) -> List[str]:
    """Map theme titles to English topic labels.

    Args:
        themes: List of theme dicts.

    Returns:
        Deduplicated list of English topic labels.
    """
    topics: List[str] = []
    for theme in themes:
        title = theme.get("title", "")
        content = theme.get("content", "")
        combined = title + " " + content
        for jp_kw, en_topic in _TOPIC_KEYWORD_MAP.items():
            if jp_kw in combined and en_topic not in topics:
                topics.append(en_topic)
    if not topics:
        topics = ["lifestyle"]
    return topics


def _infer_communication_style(keywords: List[str]) -> str:
    """Map personality keywords to communication style.

    Args:
        keywords: List of personality keyword strings.

    Returns:
        Communication style string.
    """
    for kw in keywords:
        for pattern, style in _STYLE_MAP.items():
            if pattern in kw:
                return style
    return "friendly"


def _infer_response_style(keywords: List[str]) -> str:
    """Map personality keywords to response style label.

    Args:
        keywords: List of personality keyword strings.

    Returns:
        Response style string.
    """
    for kw in keywords:
        for pattern, style in _RESPONSE_STYLE_MAP.items():
            if pattern in kw:
                return style
    return "friendly"


def parse_aituber_record(raw: dict) -> "AItuberPersona":
    """Parse a raw dataset record into an AItuberPersona.

    Args:
        raw: Dict with keys: concept, system_prompt, thema, is_valid, quality_notes,
             and optionally _row_index.

    Returns:
        AItuberPersona instance.
    """
    from simulator.schemas import AItuberPersona

    row_index = raw.get("_row_index", 0)
    persona_id = f"aituber_{row_index:03d}"

    concept = raw.get("concept", "")
    system_prompt = raw.get("system_prompt", "")

    name, name_reading = extract_name(concept)

    # Parse thema JSON (may be string or already list)
    thema_raw = raw.get("thema", "[]")
    if isinstance(thema_raw, str):
        try:
            themes = json.loads(thema_raw)
        except json.JSONDecodeError:
            themes = []
    else:
        themes = list(thema_raw)

    personality_keywords = extract_personality_keywords(concept)
    speech_style = extract_speech_style(system_prompt)
    pronoun = extract_pronoun(concept, system_prompt)
    fan_name = extract_fan_name(concept)
    genre_hint = infer_genre_from_themes(themes)
    gender_presentation = extract_gender_presentation(concept, system_prompt)

    is_valid_raw = raw.get("is_valid", "true")
    if isinstance(is_valid_raw, bool):
        is_valid = is_valid_raw
    else:
        is_valid = str(is_valid_raw).lower() == "true"

    return AItuberPersona(
        persona_id=persona_id,
        name=name,
        name_reading=name_reading,
        fan_name=fan_name,
        system_prompt=system_prompt,
        concept=concept,
        themes=themes,
        genre_hint=genre_hint,
        personality_keywords=personality_keywords,
        speech_style=speech_style,
        gender_presentation=gender_presentation,
        pronoun=pronoun,
        quality_notes=raw.get("quality_notes", ""),
        is_valid=is_valid,
    )


def load_from_huggingface(split: str = "train") -> List[dict]:
    """Load raw records from HuggingFace dataset.

    Args:
        split: Dataset split to load (default: "train").

    Returns:
        List of raw record dicts with _row_index added.

    Raises:
        ImportError: If the datasets library is not installed.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        )
    # Import inside to avoid hard dependency at module level
    from datasets import load_dataset as hf_load_dataset  # noqa: F811

    ds = hf_load_dataset(DATASET_NAME, split=split)
    records = []
    for i, row in enumerate(ds):
        record = dict(row)
        record["_row_index"] = i
        records.append(record)
    return records


def load_from_jsonl(path: Path) -> List["AItuberPersona"]:
    """Load AItuberPersona objects from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of AItuberPersona instances.
    """
    from simulator.schemas import AItuberPersona

    personas = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                persona = AItuberPersona(**data)
                personas.append(persona)
            except Exception as e:
                logger.warning("Failed to parse line %d in %s: %s", i, path, e)
    logger.info("Loaded %d AItuberPersona records from %s", len(personas), path)
    return personas


def save_to_jsonl(personas: List["AItuberPersona"], output: Path) -> None:
    """Save AItuberPersona list to JSONL file.

    Args:
        personas: List of AItuberPersona instances.
        output: Output file path.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p in personas:
            f.write(p.model_dump_json() + "\n")
    logger.info("Saved %d AItuberPersona records to %s", len(personas), output)


def ingest_from_huggingface(
    output_path: Path = DEFAULT_OUTPUT,
) -> List["AItuberPersona"]:
    """Download from HuggingFace, parse, filter, and save personas.

    Args:
        output_path: Path to write the output JSONL file.

    Returns:
        List of valid AItuberPersona instances.
    """
    raw_records = load_from_huggingface()
    logger.info("Downloaded %d raw records from %s", len(raw_records), DATASET_NAME)

    personas = []
    for record in raw_records:
        try:
            persona = parse_aituber_record(record)
            if persona.is_valid:
                personas.append(persona)
        except Exception as e:
            logger.warning("Failed to parse record %s: %s", record.get("_row_index"), e)

    logger.info("Parsed %d valid personas", len(personas))
    save_to_jsonl(personas, output_path)
    return personas


def to_viewer_persona(aituber: "AItuberPersona", seed: int = 42) -> "Persona":
    """Convert AItuberPersona to viewer Persona schema.

    Args:
        aituber: Source AItuberPersona instance.
        seed: Random seed (unused, kept for API consistency).

    Returns:
        Persona instance suitable for viewer agent use.
    """
    from simulator.schemas import Persona

    # Derive interests from themes
    topics = _map_themes_to_topics(aituber.themes)
    interests = topics[:5] if len(topics) >= 5 else (topics + ["lifestyle"] * 5)[:5]

    communication_style = _infer_communication_style(aituber.personality_keywords)
    persona_group = aituber.genre_hint if aituber.genre_hint else "雑談・癒し系"

    return Persona(
        persona_id=aituber.persona_id,
        name=aituber.name,
        age=20,
        occupation="配信者",
        interests=interests,
        persona_group=persona_group,
        communication_style=communication_style,
        base_activity_level=0.7,
        language="ja",
    )


def to_streamer_config(aituber: "AItuberPersona") -> dict:
    """Convert AItuberPersona to streamer config dict.

    Compatible with configs/streamer.yaml structure and StreamerAgent.__init__.

    Args:
        aituber: Source AItuberPersona instance.

    Returns:
        Dict with name, persona, system_prompt, topics, current_topic, response_style.
    """
    topics = _map_themes_to_topics(aituber.themes)
    if not topics:
        topics = ["lifestyle"]

    response_style = _infer_response_style(aituber.personality_keywords)

    return {
        "name": aituber.name,
        "persona": aituber.system_prompt[:200],
        "system_prompt": aituber.system_prompt,
        "topics": topics,
        "current_topic": topics[0],
        "response_style": response_style,
    }
