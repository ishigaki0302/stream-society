"""Tests for web app persona routes."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from web.app import app

client = TestClient(app)

# First valid persona_id from the sample data
_SAMPLE_PERSONA_ID = "aituber_000"


def test_personas_page_returns_200() -> None:
    """GET /personas should return HTTP 200."""
    response = client.get("/personas")
    assert response.status_code == 200
    assert "AItuber" in response.text


def test_persona_detail_returns_200() -> None:
    """GET /personas/<valid_id> should return HTTP 200."""
    response = client.get(f"/personas/{_SAMPLE_PERSONA_ID}")
    assert response.status_code == 200
    assert _SAMPLE_PERSONA_ID in response.text or "潮凪碧" in response.text


def test_personas_data_api() -> None:
    """GET /personas/<valid_id>/data should return JSON with persona fields."""
    response = client.get(f"/personas/{_SAMPLE_PERSONA_ID}/data")
    assert response.status_code == 200
    data = response.json()
    assert data["persona_id"] == _SAMPLE_PERSONA_ID
    assert "name" in data
    assert "system_prompt" in data
    assert "themes" in data


def test_personas_filter_by_genre() -> None:
    """GET /personas?genre=ゲーム should filter results."""
    response = client.get("/personas?genre=%E3%82%B2%E3%83%BC%E3%83%A0%E7%B3%BB")
    assert response.status_code == 200


def test_personas_filter_by_personality() -> None:
    """GET /personas?personality=省エネ should filter results."""
    response = client.get("/personas?personality=%E7%9C%81%E3%82%A8%E3%83%8D")
    assert response.status_code == 200


def test_persona_detail_not_found() -> None:
    """GET /personas/<invalid_id> should return 404."""
    response = client.get("/personas/nonexistent_id_xyz")
    assert response.status_code == 404


def test_personas_data_api_not_found() -> None:
    """GET /personas/<invalid_id>/data should return 404."""
    response = client.get("/personas/nonexistent_id_xyz/data")
    assert response.status_code == 404
