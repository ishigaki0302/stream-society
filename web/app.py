"""FastAPI web application for StreamSociety."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# Paths
_PROJECT_ROOT = Path(__file__).parent.parent
_OUTPUTS_DIR = _PROJECT_ROOT / Path(os.environ.get("OUTPUT_DIR", "outputs")) / "runs"
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
_PERSONAS_FULL = _PROJECT_ROOT / "data" / "personas" / "aituber_personas.jsonl"
_PERSONAS_SAMPLE = _PROJECT_ROOT / "data" / "personas" / "aituber_sample.jsonl"

app = FastAPI(
    title="StreamSociety",
    description="AI Livestream Simulation Platform",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _personas_path() -> Optional[Path]:
    """Return path to persona JSONL file, preferring full dataset over sample."""
    if _PERSONAS_FULL.exists():
        return _PERSONAS_FULL
    if _PERSONAS_SAMPLE.exists():
        return _PERSONAS_SAMPLE
    return None


def _load_personas() -> List[Dict[str, Any]]:
    """Load all AItuber personas from JSONL file."""
    path = _personas_path()
    if path is None:
        return []
    personas: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    personas.append(json.loads(line))
    except Exception as e:
        logger.warning("Failed to load personas from %s: %s", path, e)
    return personas


def _load_persona_by_id(persona_id: str) -> Optional[Dict[str, Any]]:
    """Load a single persona by persona_id."""
    for persona in _load_personas():
        if persona.get("persona_id") == persona_id:
            return persona
    return None


def _runs_for_persona(persona_id: str) -> List[Dict[str, Any]]:
    """Find runs whose summary.json references streamer_persona_id."""
    runs: List[Dict[str, Any]] = []
    if not _OUTPUTS_DIR.exists():
        return runs
    for run_dir in sorted(_OUTPUTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            if summary.get("streamer_persona_id") == persona_id:
                runs.append(summary)
        except Exception as e:
            logger.warning("Failed to load summary from %s: %s", run_dir, e)
    return runs


def _list_runs() -> List[Dict[str, Any]]:
    """List all available runs in the outputs directory."""
    runs = []
    if not _OUTPUTS_DIR.exists():
        return runs

    for run_dir in sorted(_OUTPUTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            summary["run_dir"] = str(run_dir)
            # Get modification time as timestamp
            summary["timestamp"] = summary_path.stat().st_mtime
            runs.append(summary)
        except Exception as e:
            logger.warning("Failed to load summary from %s: %s", run_dir, e)
    return runs


def _load_run_data(run_id: str) -> Optional[Dict[str, Any]]:
    """Load complete run data including turn logs."""
    run_dir = _OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return None

    summary_path = run_dir / "summary.json"
    turns_path = run_dir / "turns.jsonl"

    if not summary_path.exists() or not turns_path.exists():
        return None

    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        turns = []
        with open(turns_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))

        return {"summary": summary, "turns": turns}
    except Exception as e:
        logger.error("Failed to load run %s: %s", run_id, e)
        return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Index page showing recent runs."""
    runs = _list_runs()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"runs": runs},
    )


@app.get("/runs", response_class=JSONResponse)
async def list_runs() -> JSONResponse:
    """API endpoint to list all runs."""
    runs = _list_runs()
    return JSONResponse(content={"runs": runs})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str) -> HTMLResponse:
    """View a specific run with timeline scrubber."""
    data = _load_run_data(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return templates.TemplateResponse(
        request=request,
        name="run_viewer.html",
        context={"run_id": run_id, "summary": data["summary"]},
    )


@app.get("/runs/{run_id}/data", response_class=JSONResponse)
async def get_run_data(run_id: str) -> JSONResponse:
    """API endpoint returning run data as JSON."""
    data = _load_run_data(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return JSONResponse(content=data)


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request) -> HTMLResponse:
    """Comparison page with run selector."""
    runs = _list_runs()
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={"runs": runs},
    )


@app.get("/compare/data", response_class=JSONResponse)
async def get_compare_data(run_ids: str = "") -> JSONResponse:
    """API endpoint for comparison data.

    Args:
        run_ids: Comma-separated list of run IDs to compare.
    """
    if not run_ids:
        return JSONResponse(content={"runs": []})

    id_list = [r.strip() for r in run_ids.split(",") if r.strip()]
    results = []
    for run_id in id_list:
        data = _load_run_data(run_id)
        if data:
            results.append(data["summary"])

    return JSONResponse(content={"runs": results})


@app.get("/personas", response_class=HTMLResponse)
async def personas_list(
    request: Request,
    genre: Optional[str] = None,
    personality: Optional[str] = None,
) -> HTMLResponse:
    """AItuber ペルソナ一覧ページ。"""
    personas = _load_personas()

    if genre:
        personas = [p for p in personas if genre.lower() in p.get("genre_hint", "").lower()]

    if personality:
        personas = [
            p
            for p in personas
            if any(personality.lower() in kw.lower() for kw in p.get("personality_keywords", []))
        ]

    # Collect unique genre hints for filter dropdown
    all_genres: List[str] = []
    seen_genres: set = set()
    for p in _load_personas():
        g = p.get("genre_hint", "")
        if g and g not in seen_genres:
            all_genres.append(g)
            seen_genres.add(g)

    return templates.TemplateResponse(
        request=request,
        name="personas.html",
        context={
            "personas": personas,
            "all_genres": all_genres,
            "selected_genre": genre or "",
            "selected_personality": personality or "",
        },
    )


@app.get("/personas/{persona_id}/data", response_class=JSONResponse)
async def get_persona_data(persona_id: str) -> JSONResponse:
    """API endpoint returning persona data as JSON."""
    persona = _load_persona_by_id(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
    return JSONResponse(content=persona)


@app.get("/personas/{persona_id}", response_class=HTMLResponse)
async def persona_detail(request: Request, persona_id: str) -> HTMLResponse:
    """キャラクター詳細ページ。"""
    persona = _load_persona_by_id(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")

    runs = _runs_for_persona(persona_id)

    return templates.TemplateResponse(
        request=request,
        name="persona_detail.html",
        context={"persona": persona, "runs": runs},
    )


def create_app() -> FastAPI:
    """Factory function for the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="0.0.0.0", port=8505, reload=True)
