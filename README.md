# StreamSociety

LLM-powered livestream simulation platform for studying comment selection, persona-driven audience dynamics, and avatar-based AI streaming.

## Overview

StreamSociety simulates a live streaming session where:
- **Viewer agents** (backed by Nemotron-Personas-Japan) generate comments based on their persona, interests, and emotional state
- A **Streamer agent** selects one comment per turn using a configurable **selection policy**
- Metrics (engagement, safety, topic diversity, sentiment shift) are computed and stored
- Results are visualized in a **web UI** and compared across policy variants

The system is fully self-contained with mock data, and can be connected to a real vLLM endpoint and MMDAgent-EX avatar bridge.

---

## Architecture

```
stream-society/
├── simulator/                # Core simulation engine
│   ├── schemas.py            # Pydantic v2 data models
│   ├── persona.py            # Persona loading and sampling
│   ├── viewer.py             # ViewerAgent (comment generation)
│   ├── streamer.py           # StreamerAgent (comment selection + response)
│   ├── simulation.py         # Main simulation loop
│   ├── metrics.py            # Engagement, safety, diversity metrics
│   ├── policy/               # Comment selection policies
│   │   ├── base.py           # Abstract SelectionPolicy
│   │   ├── random_policy.py  # Uniform random selection
│   │   ├── rule_based.py     # Heuristic: questions > sentiment > safety
│   │   ├── score_based.py    # Weighted score function
│   │   ├── contextual_bandit_stub.py  # LinUCB stub
│   │   └── factory.py        # create_policy(name)
│   └── adapters/             # LLM backends
│       ├── base.py           # Abstract LLMAdapter
│       ├── mock_adapter.py   # Template-based mock
│       └── vllm_adapter.py   # vLLM stub (TODO)
├── data/
│   └── personas/
│       └── sample_personas.jsonl  # 20 Japanese viewer personas
├── configs/
│   ├── streamer.yaml         # Streamer character config
│   └── experiments/          # Experiment YAML configs
├── ingestion/                # Persona data pipeline
├── analytics/                # Metrics and reporting
├── cli/                      # Typer CLI (ss command)
├── bridges/                  # Avatar bridges (mock + MMDAgent-EX stub)
├── web/                      # FastAPI + Jinja2 web UI
│   ├── app.py
│   ├── templates/
│   └── static/
├── tests/                    # pytest test suite
└── docs/                     # Research documentation
```

**Data Flow per Turn:**

```
Viewer Agents (x N)
      |
      | decide_comment(turn, topic)
      v
[CommentCandidate list]
      |
      | SelectionPolicy.select(candidates, context)
      v
[Selected CommentCandidate]
      |
      | LLMAdapter.generate_response(prompt, context)
      v
[StreamerResponse]
      |
      | update_state(was_selected, response)
      v
[Updated ViewerState x N]
      |
      | compute_turn_metrics(turn_log)
      v
[TurnLog] --> [RunSummary]
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### 2. Run the built-in demo

```bash
ss demo
```

This runs 3 policies (random, rule_based, score_based) for 20 turns with 10 viewers and prints a comparison table.

### 3. Run a custom experiment

```bash
ss simulate configs/experiments/demo_random.yaml
```

Override options:

```bash
ss simulate configs/experiments/demo_random.yaml --seed 99 --policy rule_based --output-dir outputs/custom
```

### 4. Evaluate a run

```bash
ss evaluate outputs/runs/<run_id>
```

### 5. Compare runs

```bash
ss report outputs/runs/run1 outputs/runs/run2 outputs/runs/run3
```

Export to CSV:

```bash
ss report outputs/runs/run1 outputs/runs/run2 --export-csv comparison.csv
```

### 6. Ingest personas

```bash
ss ingest-personas data/personas/sample_personas.jsonl --output-dir outputs/personas
```

### 7. Start web UI

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8080 --reload
```

Then open http://localhost:8080

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `ss demo` | Run end-to-end demo with all policies |
| `ss simulate <config>` | Run experiment from YAML config |
| `ss evaluate <run_dir>` | Show detailed metrics for a run |
| `ss report <run_dirs...>` | Compare metrics across runs |
| `ss ingest-personas <input>` | Run persona ingestion pipeline |

---

## Available Policies

| Policy | Description |
|--------|-------------|
| `random` | Uniform random selection |
| `rule_based` | Questions > high sentiment > low toxicity |
| `score_based` | Weighted: 0.3*sentiment + 0.2*safety + 0.3*novelty + 0.2*question |
| `contextual_bandit` | LinUCB stub (see `docs/contextual_bandit_todo.md`) |

---

## Configuration

### Experiment YAML

```yaml
experiment_name: my_experiment
seed: 42
num_viewers: 20
num_turns: 50
policy: score_based
streamer_config:
  name: Aoi
  topics: [gaming, music, anime, technology]
  response_style: friendly
persona_data: data/personas/sample_personas.jsonl
output_dir: outputs/runs
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# LLM Backend (optional, mock used by default)
LLM_BACKEND=mock
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct

# MMDAgent-EX Bridge (optional)
MMDAGENT_WS_URL=ws://localhost:9000/bridge
```

---

## Extension Points

### Connecting vLLM

1. Start vLLM server: `vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000`
2. Set `VLLM_BASE_URL=http://localhost:8000/v1`
3. Implement `VLLMAdapter.generate_response()` in `simulator/adapters/vllm_adapter.py`
   (see TODO comments in that file)

### Connecting MMDAgent-EX

1. Set `MMDAGENT_WS_URL=ws://<host>:9000/bridge`
2. Implement WebSocket protocol in `bridges/mmdagent_bridge.py`
   (see TODO comments in that file)

### Adding a New Policy

1. Create `simulator/policy/my_policy.py` inheriting from `SelectionPolicy`
2. Implement `select()` and optionally `update()`
3. Register in `simulator/policy/factory.py`

### Using Real Persona Data

1. Download Nemotron-Personas-Japan dataset
2. Use `ingestion/persona_ingestion.py` to normalize and convert
3. Point `persona_data` in your config to the output JSONL

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Research Notes

This platform is designed for studying:

- **Comment selection policy comparison**: How does policy choice affect engagement, diversity, and community health?
- **Persona-driven dynamics**: How do different viewer demographics affect comment patterns?
- **Sentiment evolution**: Does streamer response style influence community mood over time?
- **Safety vs. engagement tradeoffs**: Can we maintain high engagement while filtering toxic content?

Key metrics computed:
- `engagement_proxy`: average comments per turn / num_viewers
- `unique_participant_rate`: unique commenters / total viewers
- `topic_diversity`: Shannon entropy of topic distribution
- `safety_rate`: proportion of non-toxic comments
- `sentiment_shift`: sentiment delta between first and last turns

---

## Project Info

- Python 3.11+
- Framework: FastAPI + Jinja2 + HTMX
- CLI: Typer
- Data: Polars + PyArrow
- Formatter: black
- Linter: ruff
- Tests: pytest
