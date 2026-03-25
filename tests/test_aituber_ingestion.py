"""Tests for ingestion.aituber_ingestion module."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.aituber_ingestion import (
    SAMPLE_FALLBACK,
    extract_fan_name,
    extract_name,
    extract_personality_keywords,
    extract_pronoun,
    extract_speech_style,
    infer_genre_from_themes,
    load_from_jsonl,
    parse_aituber_record,
    save_to_jsonl,
    to_streamer_config,
    to_viewer_persona,
)
from simulator.schemas import AItuberPersona, Persona

SAMPLE_DATA = Path(__file__).parent.parent / "data" / "personas" / "aituber_sample.jsonl"


# ---------------------------------------------------------------------------
# Unit tests for extraction helpers
# ---------------------------------------------------------------------------


def test_extract_name_table_format():
    """extract_name should parse table row with furigana."""
    concept = "| 名前（フリガナ） | 潮凪碧（しおなぎあお） |\n"
    name, reading = extract_name(concept)
    assert name == "潮凪碧"
    assert reading == "しおなぎあお"


def test_extract_name_no_reading():
    """extract_name should return empty reading if no furigana present."""
    concept = "| 名前 | テスト太郎 |\n"
    name, reading = extract_name(concept)
    assert name == "テスト太郎"
    assert reading == ""


def test_extract_name_h1_fallback():
    """extract_name falls back to H1 header when table row is absent."""
    concept = "# テストキャラコンセプト設計書\n\nsome content"
    name, reading = extract_name(concept)
    assert "テストキャラ" in name


def test_extract_personality_keywords():
    """extract_personality_keywords should parse bold items in 性格キーワード section."""
    concept = (
        "## 性格キーワード\n\n"
        "- **省エネ**：エネルギーを節約する\n"
        "- **観察者**：世界を見ている\n\n"
        "## 次のセクション\n"
    )
    kws = extract_personality_keywords(concept)
    assert "省エネ" in kws
    assert "観察者" in kws


def test_extract_speech_style():
    """extract_speech_style should find 口調 pattern."""
    system_prompt = "あなたはXです。口調：ダウナー系・省エネ。視聴者に話しかけます。"
    style = extract_speech_style(system_prompt)
    assert style == "ダウナー系・省エネ"


def test_extract_speech_style_missing():
    """extract_speech_style returns empty string when not found."""
    style = extract_speech_style("ここには口調情報がありません")
    assert style == ""


def test_extract_pronoun_system_prompt():
    """extract_pronoun should find 一人称 in system_prompt."""
    system_prompt = "一人称：私。話し方は丁寧です。"
    pronoun = extract_pronoun("", system_prompt)
    assert pronoun == "私"


def test_extract_fan_name():
    """extract_fan_name should parse ファンネーム table row."""
    concept = "| ファンネーム | 潮待ち組 |\n"
    fan_name = extract_fan_name(concept)
    assert fan_name == "潮待ち組"


def test_infer_genre_asmr():
    """infer_genre_from_themes detects ASMR keyword."""
    themes = [{"title": "ASMR配信", "content": "リラックス"}]
    assert infer_genre_from_themes(themes) == "ASMR系"


def test_infer_genre_game():
    """infer_genre_from_themes detects ゲーム keyword."""
    themes = [{"title": "ゲーム実況", "content": ""}]
    assert infer_genre_from_themes(themes) == "ゲーム系"


def test_infer_genre_default():
    """infer_genre_from_themes returns default for unknown themes."""
    themes = [{"title": "雑談", "content": "日常の話"}]
    assert infer_genre_from_themes(themes) == "雑談・癒し系"


# ---------------------------------------------------------------------------
# Integration tests using sample JSONL
# ---------------------------------------------------------------------------


def test_parse_aituber_record_from_sample():
    """load sample JSONL lines, parse each as AItuberPersona and verify fields."""
    assert SAMPLE_DATA.exists(), f"Sample file not found: {SAMPLE_DATA}"

    with open(SAMPLE_DATA, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 5, "Expected at least 5 sample records"

    for i, line in enumerate(lines):
        raw = json.loads(line)
        # parse_aituber_record expects _row_index
        raw["_row_index"] = i
        persona = parse_aituber_record(raw)

        assert isinstance(persona, AItuberPersona)
        assert persona.persona_id.startswith("aituber_")
        assert persona.name
        assert persona.system_prompt
        assert isinstance(persona.themes, list)
        assert persona.is_valid is True


def test_load_from_jsonl():
    """load_from_jsonl should round-trip through save_to_jsonl."""
    assert SAMPLE_DATA.exists(), f"Sample file not found: {SAMPLE_DATA}"
    original = load_from_jsonl(SAMPLE_DATA)
    assert len(original) >= 5

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_to_jsonl(original, tmp_path)
        reloaded = load_from_jsonl(tmp_path)

        assert len(reloaded) == len(original)
        for orig, reload in zip(original, reloaded):
            assert orig.persona_id == reload.persona_id
            assert orig.name == reload.name
            assert orig.system_prompt == reload.system_prompt
    finally:
        tmp_path.unlink(missing_ok=True)


def test_to_viewer_persona():
    """to_viewer_persona should produce a valid Persona instance."""
    personas = load_from_jsonl(SAMPLE_DATA)
    assert personas

    for aituber in personas:
        viewer = to_viewer_persona(aituber)

        assert isinstance(viewer, Persona)
        assert viewer.persona_id == aituber.persona_id
        assert viewer.name == aituber.name
        assert viewer.occupation == "配信者"
        assert viewer.age == 20
        assert viewer.language == "ja"
        assert 0.0 <= viewer.base_activity_level <= 1.0
        assert isinstance(viewer.interests, list)
        assert len(viewer.interests) == 5
        valid_styles = {"friendly", "analytical", "enthusiastic", "quiet", "talkative", "critical"}
        assert viewer.communication_style in valid_styles


def test_to_streamer_config():
    """to_streamer_config should produce a dict compatible with StreamerAgent."""
    personas = load_from_jsonl(SAMPLE_DATA)
    assert personas

    for aituber in personas:
        cfg = to_streamer_config(aituber)

        assert isinstance(cfg, dict)
        assert cfg["name"] == aituber.name
        assert "persona" in cfg
        assert isinstance(cfg["persona"], str)
        assert len(cfg["persona"]) <= 200
        assert "system_prompt" in cfg
        assert cfg["system_prompt"] == aituber.system_prompt
        assert isinstance(cfg["topics"], list)
        assert len(cfg["topics"]) >= 1
        assert "current_topic" in cfg
        assert cfg["current_topic"] == cfg["topics"][0]
        assert "response_style" in cfg


def test_sample_fallback_path_exists():
    """SAMPLE_FALLBACK constant should point to an existing file."""
    assert SAMPLE_FALLBACK.exists(), f"SAMPLE_FALLBACK not found: {SAMPLE_FALLBACK}"


def test_aituber_persona_schema_fields():
    """AItuberPersona schema should accept all defined fields."""
    persona = AItuberPersona(
        persona_id="test_001",
        name="テスト",
        system_prompt="テスト用プロンプト",
        concept="# テストコンセプト",
        themes=[{"title": "テーマ1", "content": "内容1"}],
    )
    assert persona.persona_id == "test_001"
    assert persona.name_reading == ""
    assert persona.fan_name == ""
    assert persona.genre_hint == ""
    assert persona.personality_keywords == []
    assert persona.is_valid is True
