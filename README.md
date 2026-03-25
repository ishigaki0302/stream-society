# StreamSociety

LLM-powered livestream simulation platform for studying comment selection, persona-driven audience dynamics, and avatar-based AI streaming.

[![CI](https://github.com/ishigaki0302/stream-society/actions/workflows/ci.yml/badge.svg)](https://github.com/ishigaki0302/stream-society/actions/workflows/ci.yml)

---

## 実装状況

| フェーズ | 内容 | 状態 |
|---------|------|------|
| Phase 1 | scaffold / schemas / simulator 最小ループ / mock LLM | ✅ 完了 |
| Phase 2 | persona ingestion / viewer 内部状態 / 4方策 / metrics / CLI | ✅ 完了 |
| Phase 3 | Web UI（ライブ配信風・タイムラインスクラバー・比較ページ） | ✅ 完了 |
| Phase 4 | AItuber-Personas-Japan 取り込み・ストリーマーペルソナサンプリング | ✅ 完了 (#3) |
| Phase 4 | キャラクター別分析ダッシュボード | ✅ 完了 (#4) |
| Phase 4 | contextual bandit (LinUCB) 本実装 | ✅ 完了 (#5) |
| Phase 4 | topic diversity 指標修正 | ✅ 完了 (#6) |
| Phase 4 | CI/CD (GitHub Actions) | ✅ 完了 (#9) |
| Phase 5 | vLLM / Qwen2.5 adapter 実装 | 🔲 [#7](https://github.com/ishigaki0302/stream-society/issues/7) |
| Phase 5 | MMDAgent-EX WebSocket bridge 実装 | 🔲 [#8](https://github.com/ishigaki0302/stream-society/issues/8) |

---

## Overview

StreamSociety は AI ライブ配信セッションをシミュレートする研究基盤です:

- **Viewer agents** がペルソナ・興味・感情状態に基づいてコメントを生成
- **Streamer agent**（[DataPilot/AItuber-Personas-Japan](https://huggingface.co/datasets/DataPilot/AItuber-Personas-Japan) から取り込んだキャラクターを使用可能）がコメント選択方策でコメントを1件選び応答
- 指標（engagement・safety・topic diversity・sentiment shift）をターン毎に記録
- Web UI でタイムライン再生・方策比較・キャラクター別分析が可能

---

## Architecture

```
stream-society/
├── simulator/                # シミュレーションエンジン
│   ├── schemas.py            # Pydantic v2 データモデル（Persona, AItuberPersona, TurnLog, …）
│   ├── persona.py            # ペルソナ読み込み・サンプリング
│   ├── viewer.py             # ViewerAgent（コメント生成・内部状態管理）
│   ├── streamer.py           # StreamerAgent（コメント選択・応答生成）
│   ├── simulation.py         # メインシミュレーションループ
│   ├── metrics.py            # engagement / safety / diversity 指標
│   ├── policy/               # コメント選択方策
│   │   ├── random_policy.py  # ランダム選択
│   │   ├── rule_based.py     # ヒューリスティック（質問 > 感情 > 安全）
│   │   ├── score_based.py    # 重み付きスコア関数
│   │   └── contextual_bandit_stub.py  # LinUCB（numpy 本実装）
│   └── adapters/             # LLM バックエンド
│       ├── mock_adapter.py   # テンプレートベース mock（日本語）
│       └── vllm_adapter.py   # vLLM stub (TODO: #7)
├── ingestion/
│   ├── persona_ingestion.py       # Nemotron-Personas-Japan 取り込み
│   └── aituber_ingestion.py       # DataPilot/AItuber-Personas-Japan 取り込み
├── data/
│   └── personas/
│       ├── sample_personas.jsonl      # 日本語視聴者ペルソナ 20 名
│       └── aituber_sample.jsonl       # AItuber サンプル 5 キャラ（オフライン用）
├── configs/
│   ├── streamer.yaml               # ストリーマーキャラクター設定
│   └── experiments/                # 実験条件 YAML
├── analytics/                # レポート生成（Polars）
├── cli/                      # Typer CLI（ss コマンド）
├── bridges/                  # アバターブリッジ（mock + MMDAgent-EX stub）
├── web/                      # FastAPI + Jinja2 Web UI
│   ├── app.py
│   ├── templates/            # ライブ配信風デザイン
│   └── static/               # CSS / JS
└── tests/                    # pytest（71 テスト）
```

**ターン内データフロー:**

```
Viewer Agents (x N)
      │
      │ decide_comment(turn, topic)   ← 60% ストリーマートピック / 40% 自分のインタレスト
      ▼
[CommentCandidate list]  ← sentiment / toxicity / novelty / question_flag
      │
      │ SelectionPolicy.select(candidates, context)
      ▼
[Selected CommentCandidate]
      │
      │ LLMAdapter.generate_response(prompt, context)
      ▼
[StreamerResponse]
      │
      │ update_state(was_selected, response)
      ▼
[Updated ViewerState x N]  ← affinity / emotion_state / activity_level 更新
      │
      │ compute_turn_metrics(turn_log)
      ▼
outputs/runs/<run_id>/turns.jsonl + summary.json
```

---

## Quick Start

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. デモ実行（end-to-end）

```bash
python3 -m cli.main demo
```

random / rule_based / score_based / contextual_bandit の4方策を 10 視聴者 × 20 ターンで比較します。

### 3. カスタム実験の実行

```bash
python3 -m cli.main simulate configs/experiments/demo_random.yaml
python3 -m cli.main simulate configs/experiments/demo_random.yaml --seed 99 --policy rule_based
```

### 4. AItuber ペルソナの取り込み

```bash
# HuggingFace から 195 件フルダウンロード
python3 -m cli.main ingest-aituber --output data/personas/aituber_personas.jsonl

# ネットワーク未接続時はサンプル 5 件を使用
python3 -m cli.main ingest-aituber --use-sample
```

### 5. 結果の評価・比較

```bash
python3 -m cli.main evaluate outputs/runs/<run_id>
python3 -m cli.main report outputs/runs/run1 outputs/runs/run2
```

### 6. Web UI の起動

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8505 --reload
```

コンテナ外からは **http://localhost:10005** でアクセス（`docker -p 10005:8505`）。

| URL | 内容 |
|-----|------|
| `/` | ダッシュボード（run 一覧） |
| `/runs/<run_id>` | run 再生（タイムラインスクラバー） |
| `/compare` | 複数 run の方策比較 |
| `/personas` | AItuber キャラクター一覧 |
| `/personas/<id>` | キャラクター詳細・関連 run |

---

## CLI リファレンス

| コマンド | 説明 |
|---------|------|
| `python3 -m cli.main demo` | 全方策で end-to-end デモ実行 |
| `python3 -m cli.main simulate <config>` | YAML から実験実行 |
| `python3 -m cli.main evaluate <run_dir>` | run の詳細指標を表示 |
| `python3 -m cli.main report <run_dirs...>` | 複数 run を比較 |
| `python3 -m cli.main ingest-personas <input>` | 視聴者ペルソナ取り込み |
| `python3 -m cli.main ingest-aituber` | AItuber-Personas-Japan 取り込み |

---

## 選択方策

| 方策名 | 説明 | 実装状態 |
|--------|------|---------|
| `random` | 一様ランダム選択 | ✅ |
| `rule_based` | 質問優先 → 感情 → 安全性 | ✅ |
| `score_based` | 0.3×sentiment + 0.2×safety + 0.3×novelty + 0.2×question | ✅ |
| `contextual_bandit` | LinUCB（numpy 本実装、save_state/load_state 対応） | ✅ |

---

## ストリーマーペルソナ（AItuber-Personas-Japan）

[DataPilot/AItuber-Personas-Japan](https://huggingface.co/datasets/DataPilot/AItuber-Personas-Japan) から 195 キャラクターを取り込み可能。

各キャラクターは以下を持つ:
- **concept**: キャラクター設計書 Markdown
- **system_prompt**: LLM にそのまま入力できるシステムプロンプト
- **thema**: 配信テーマ 9〜10 選（title + content）

キャラクター例:

| 名前 | ジャンル | 性格 | 口調 |
|------|---------|------|------|
| 潮凪碧 | 雑談・癒し | 観察者・省エネ | 常体、だね/かな |
| 星宮せれん | オカルト系 | 中二病 | ギャル語 |
| 黒羽ミサ | オカルト系 | ミステリアス | ゴスロリ口調 |
| 武藤凛 | スポーツ系 | 猪突猛進 | 東北弁 |
| リュウ・ヴェルダンディ | オカルト系 | 自由奔放 | ラップ調 |

---

## 指標定義

| 指標 | 定義 |
|------|------|
| `engagement_proxy` | 平均コメント数 / ターン / 視聴者数 |
| `unique_participant_rate` | コメントしたユニーク視聴者 / 全視聴者 |
| `topic_diversity` | トピック分布のシャノンエントロピー（0=単調、高いほど多様） |
| `safety_rate` | 非有害コメントの割合 |
| `sentiment_shift` | 初回〜最終ターン間の平均感情スコア変化 |
| `persona_group_exposure` | ペルソナグループ別の選択回数 |

---

## 設定

### 実験 YAML

```yaml
experiment_name: my_experiment
seed: 42
num_viewers: 20
num_turns: 50
policy: contextual_bandit   # random / rule_based / score_based / contextual_bandit
streamer_persona: null       # AItuber persona_id（例: "aituber_000"）
streamer_config:
  name: Aoi
  topics: [gaming, music, anime, technology]
  response_style: friendly
persona_data: data/personas/sample_personas.jsonl
output_dir: outputs/runs
```

### 環境変数（`.env.example` 参照）

```bash
# Web UI（コンテナ外: http://localhost:10005）
WEB_PORT=8505

# LLM バックエンド（デフォルト: mock）
LLM_BACKEND=mock
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct

# MMDAgent-EX（デフォルト: mock）
MMDAGENT_BRIDGE=mock
MMDAGENT_WS_URL=ws://localhost:9000/bridge
```

---

## 拡張ポイント

### vLLM / Qwen2.5 の接続（[#7](https://github.com/ishigaki0302/stream-society/issues/7)）

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
# simulator/adapters/vllm_adapter.py の TODO を実装
```

### MMDAgent-EX の接続（[#8](https://github.com/ishigaki0302/stream-society/issues/8)）

```bash
export MMDAGENT_BRIDGE=mmdagent
export MMDAGENT_WS_URL=ws://localhost:9000/bridge
# bridges/mmdagent_bridge.py の TODO を実装
```

### 新しい方策の追加

```python
# simulator/policy/my_policy.py
class MyPolicy(SelectionPolicy):
    name = "my_policy"
    def select(self, candidates, context): ...

# simulator/policy/factory.py に登録
POLICIES["my_policy"] = MyPolicy
```

---

## テスト

```bash
pytest tests/ -v
# 71 tests — all passed
```

---

## 研究的背景

本プラットフォームは以下を研究対象とする:

- **コメント選択方策の比較**: random / rule_based / score_based / LinUCB の違いが engagement・diversity・community health にどう影響するか
- **ペルソナ駆動ダイナミクス**: 視聴者属性の違いがコメントパターンに与える影響
- **AItuber キャラクター効果**: キャラクターの性格・口調・ジャンルが視聴者動態に与える影響
- **感情の時系列変化**: ストリーマーの応答スタイルがコミュニティの感情に影響するか
- **安全性とエンゲージメントのトレードオフ**

---

## プロジェクト情報

- Python 3.11+
- フレームワーク: FastAPI + Jinja2
- CLI: Typer + Rich
- データ処理: Polars + NumPy
- フォーマッタ: black / ruff
- テスト: pytest（71 テスト）
- ライセンス: MIT
