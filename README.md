# StreamSociety

LLM-powered livestream simulation platform for studying comment selection, persona-driven audience dynamics, and avatar-based AI streaming.

## 実装状況

| フェーズ | 内容 | 状態 |
|---------|------|------|
| Phase 1 | scaffold / schemas / simulator最小ループ / mock LLM | ✅ 完了 |
| Phase 2 | persona ingestion / viewer内部状態 / 4方策 / metrics / CLI | ✅ 完了 |
| Phase 3 | Web UI（ライブ配信風・タイムラインスクラバー・比較ページ） | ✅ 完了 |
| Phase 4 | AItuber-Personas-Japan ストリーマーペルソナ取り込み | 🔲 [#3](https://github.com/ishigaki0302/stream-society/issues/3) |
| Phase 4 | キャラクター別分析ダッシュボード | 🔲 [#4](https://github.com/ishigaki0302/stream-society/issues/4) |
| Phase 4 | contextual bandit (LinUCB) 本実装 | 🔲 [#5](https://github.com/ishigaki0302/stream-society/issues/5) |
| Phase 4 | topic diversity 指標修正 | 🔲 [#6](https://github.com/ishigaki0302/stream-society/issues/6) |
| Phase 5 | vLLM / Qwen2.5 adapter 実装 | 🔲 [#7](https://github.com/ishigaki0302/stream-society/issues/7) |
| Phase 5 | MMDAgent-EX WebSocket bridge 実装 | 🔲 [#8](https://github.com/ishigaki0302/stream-society/issues/8) |
| Phase 5 | CI/CD (GitHub Actions) | 🔲 [#9](https://github.com/ishigaki0302/stream-society/issues/9) |

---

## Overview

StreamSociety simulates a live streaming session where:
- **Viewer agents** generate comments based on persona, interests, and emotional state
- A **Streamer agent** (AItuber character sampled from [DataPilot/AItuber-Personas-Japan](https://huggingface.co/datasets/DataPilot/AItuber-Personas-Japan)) selects one comment per turn using a configurable **selection policy**
- Metrics (engagement, safety, topic diversity, sentiment shift) are computed and stored per run
- Results are visualized in a **web UI** and compared across policy variants

The system is fully self-contained with mock data and can be connected to a real vLLM endpoint and MMDAgent-EX avatar bridge.

---

## Architecture

```
stream-society/
├── simulator/                # シミュレーションエンジン
│   ├── schemas.py            # Pydantic v2 データモデル
│   ├── persona.py            # ペルソナ読み込み・サンプリング
│   ├── viewer.py             # ViewerAgent（コメント生成・内部状態管理）
│   ├── streamer.py           # StreamerAgent（コメント選択・応答生成）
│   ├── simulation.py         # メインシミュレーションループ
│   ├── metrics.py            # engagement / safety / diversity 指標
│   ├── policy/               # コメント選択方策
│   │   ├── base.py           # 抽象 SelectionPolicy
│   │   ├── random_policy.py  # ランダム選択
│   │   ├── rule_based.py     # ヒューリスティック（質問 > 感情 > 安全）
│   │   ├── score_based.py    # 重み付きスコア関数
│   │   ├── contextual_bandit_stub.py  # LinUCB stub (TODO: #5)
│   │   └── factory.py        # create_policy(name)
│   └── adapters/             # LLM バックエンド
│       ├── base.py           # 抽象 LLMAdapter
│       ├── mock_adapter.py   # テンプレートベース mock（日本語）
│       └── vllm_adapter.py   # vLLM stub (TODO: #7)
├── data/
│   └── personas/
│       └── sample_personas.jsonl   # 日本語視聴者ペルソナ 20 名
├── configs/
│   ├── streamer.yaml               # ストリーマーキャラクター設定
│   └── experiments/                # 実験条件 YAML
│       ├── demo_random.yaml
│       └── demo_compare.yaml
├── ingestion/                # ペルソナデータパイプライン
├── analytics/                # レポート生成（Polars）
├── cli/                      # Typer CLI（ss コマンド）
├── bridges/                  # アバターブリッジ
│   ├── base.py               # 抽象 AvatarBridge
│   ├── mock_bridge.py        # Mock（ログ出力）
│   └── mmdagent_bridge.py    # MMDAgent-EX stub (TODO: #8)
├── web/                      # FastAPI + Jinja2 Web UI
│   ├── app.py
│   ├── templates/            # ライブ配信風デザイン
│   └── static/               # CSS / JS
├── tests/                    # pytest（44 テスト）
└── docs/
    └── contextual_bandit_todo.md
```

**ターン内データフロー:**

```
Viewer Agents (x N)
      |
      | decide_comment(turn, topic)
      v
[CommentCandidate list]  ← viewer_id / sentiment / toxicity / novelty / question_flag
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
[Updated ViewerState x N]  ← affinity / emotion_state / activity_level 更新
      |
      | compute_turn_metrics(turn_log)
      v
[TurnLog] → outputs/runs/<run_id>/turns.jsonl
            outputs/runs/<run_id>/summary.json
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

random / rule_based / score_based の3方策を 10 視聴者 × 20 ターンで比較し、結果を表示します。

### 3. カスタム実験の実行

```bash
python3 -m cli.main simulate configs/experiments/demo_random.yaml
```

オプション指定:

```bash
python3 -m cli.main simulate configs/experiments/demo_random.yaml \
  --seed 99 --policy rule_based --output-dir outputs/custom
```

### 4. 結果の評価

```bash
python3 -m cli.main evaluate outputs/runs/<run_id>
```

### 5. 複数 run の比較

```bash
python3 -m cli.main report outputs/runs/run1 outputs/runs/run2
```

CSV エクスポート:

```bash
python3 -m cli.main report outputs/runs/run1 outputs/runs/run2 --export-csv comparison.csv
```

### 6. Web UI の起動

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8080 --reload
```

`http://localhost:8080` でアクセス可能。

---

## CLI リファレンス

| コマンド | 説明 |
|---------|------|
| `python3 -m cli.main demo` | 全方策で end-to-end デモ実行 |
| `python3 -m cli.main simulate <config>` | YAML から実験実行 |
| `python3 -m cli.main evaluate <run_dir>` | run の詳細指標を表示 |
| `python3 -m cli.main report <run_dirs...>` | 複数 run を比較 |
| `python3 -m cli.main ingest-personas <input>` | ペルソナ取り込みパイプライン実行 |

---

## 選択方策

| 方策名 | 説明 | 実装状態 |
|--------|------|---------|
| `random` | 一様ランダム選択 | ✅ |
| `rule_based` | 質問優先 → 感情 → 安全性 | ✅ |
| `score_based` | 0.3×sentiment + 0.2×safety + 0.3×novelty + 0.2×question | ✅ |
| `contextual_bandit` | LinUCB スタブ（本実装は [#5](https://github.com/ishigaki0302/stream-society/issues/5)） | 🔲 stub |

---

## ストリーマーペルソナ

現在は `configs/streamer.yaml` に手動設定。今後は **DataPilot/AItuber-Personas-Japan**（195 キャラクター）からサンプリング予定（[#3](https://github.com/ishigaki0302/stream-society/issues/3)）。

データセット概要:
- 195 件のAItuberペルソナ（concept設計書 / system_prompt / 配信テーマ10選）
- ジャンル: 雑談・癒し / 学術 / 技術 / オカルト / 料理 / 音楽 など
- 性格: ツンデレ / 天然 / 知的クール / 中二病 / ダウナー系 など
- 口調: ギャル語 / 関西弁 / お嬢様言葉 / 古武士口調 など

キャラクター例:
- 潮凪碧（海辺出身・哲学的）/ 星宮せれん（中二病・ギャル語）/ 黒羽ミサ（ゴスロリ・タロット）/ 武藤凛（東北弁・格闘系）/ リュウ・ヴェルダンディ（サイバーパンク・ラップ占い）など

---

## 実験設定

### 実験 YAML の書き方

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

### 環境変数

`.env.example` を `.env` にコピーして設定:

```bash
# LLM バックエンド（デフォルト: mock）
LLM_BACKEND=mock
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct

# MMDAgent-EX ブリッジ（デフォルト: mock）
MMDAGENT_BRIDGE=mock
MMDAGENT_WS_URL=ws://localhost:9000/bridge
```

---

## 指標定義

| 指標 | 定義 |
|------|------|
| `engagement_proxy` | 平均コメント数 / ターン / 視聴者数 |
| `unique_participant_rate` | コメントした視聴者のユニーク率 |
| `topic_diversity` | トピック分布のシャノンエントロピー |
| `safety_rate` | 非有害コメントの割合 |
| `sentiment_shift` | 初回ターンと最終ターンの感情スコア差分 |
| `persona_group_exposure` | ペルソナグループ別の選択回数 |

---

## 拡張ポイント

### vLLM / Qwen2.5 の接続

```bash
# 1. vLLM サーバーを起動
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000

# 2. 環境変数を設定
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://localhost:8000/v1

# 3. simulator/adapters/vllm_adapter.py の TODO を実装
```

詳細は [#7](https://github.com/ishigaki0302/stream-society/issues/7)。

### MMDAgent-EX の接続

```bash
export MMDAGENT_BRIDGE=mmdagent
export MMDAGENT_WS_URL=ws://localhost:9000/bridge
# bridges/mmdagent_bridge.py の TODO を実装
```

詳細は [#8](https://github.com/ishigaki0302/stream-society/issues/8)。

### 新しい方策の追加

```python
# 1. simulator/policy/my_policy.py を作成
class MyPolicy(SelectionPolicy):
    name = "my_policy"
    def select(self, candidates, context): ...

# 2. simulator/policy/factory.py に登録
POLICIES["my_policy"] = MyPolicy
```

---

## テスト実行

```bash
pytest tests/ -v
# 44 tests in 0.13s — all passed
```

---

## 研究的背景

本プラットフォームは以下を研究対象とする:

- **コメント選択方策の比較**: 方策の違いがエンゲージメント・多様性・コミュニティ健全性にどう影響するか
- **ペルソナ駆動ダイナミクス**: 視聴者属性の違いがコメントパターンに与える影響
- **感情の時系列変化**: ストリーマーの応答スタイルがコミュニティの感情に影響するか
- **安全性とエンゲージメントのトレードオフ**: エンゲージメントを維持しながら有害コンテンツをフィルタリングできるか
- **キャラクター別効果**: AItuberのキャラクター設定が視聴者動態に与える影響（予定: [#4](https://github.com/ishigaki0302/stream-society/issues/4)）

---

## プロジェクト情報

- Python 3.11+
- フレームワーク: FastAPI + Jinja2
- CLI: Typer + Rich
- データ処理: Polars + PyArrow
- フォーマッタ: black / ruff
- テスト: pytest
- ライセンス: MIT
