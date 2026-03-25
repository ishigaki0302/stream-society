"""Analytics and reporting for stream-society simulation runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def load_run(run_dir: Path) -> Tuple:
    """Load a completed run from disk.

    Args:
        run_dir: Directory containing run results (turns.jsonl, summary.json).

    Returns:
        Tuple of (RunSummary, List[TurnLog]).
    """
    from simulator.schemas import RunSummary, TurnLog

    # Load summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = RunSummary(**json.load(f))

    # Load turn logs
    turns_path = run_dir / "turns.jsonl"
    turn_logs = []
    with open(turns_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turn_logs.append(TurnLog(**json.loads(line)))

    return summary, turn_logs


def compare_runs(run_dirs: List[Path]):
    """Load multiple runs and create a comparison DataFrame.

    Args:
        run_dirs: List of run directories to compare.

    Returns:
        polars.DataFrame with one row per run.
    """
    import polars as pl

    records = []
    for run_dir in run_dirs:
        try:
            summary, _ = load_run(run_dir)
            records.append(summary.model_dump())
        except Exception as e:
            logger.warning("Failed to load run from %s: %s", run_dir, e)

    if not records:
        return pl.DataFrame()

    return pl.DataFrame(records)


def print_summary(summary) -> None:
    """Pretty print a RunSummary using rich.

    Args:
        summary: RunSummary object to display.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(
        title=f"Run Summary: {summary.run_id}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    rows = [
        ("Run ID", summary.run_id),
        ("Experiment", summary.experiment_name),
        ("Policy", summary.policy),
        ("Turns", str(summary.num_turns)),
        ("Viewers", str(summary.num_viewers)),
        ("Total Comments", str(summary.total_comments)),
        ("Engagement Proxy", f"{summary.engagement_proxy:.4f}"),
        ("Unique Participant Rate", f"{summary.unique_participant_rate:.4f}"),
        ("Topic Diversity (entropy)", f"{summary.topic_diversity:.4f}"),
        ("Safety Rate", f"{summary.safety_rate:.4f}"),
        ("Sentiment Shift", f"{summary.sentiment_shift:+.4f}"),
    ]

    for metric, value in rows:
        table.add_row(metric, value)

    console.print(table)


def aggregate_by_character(run_dirs: List[Path]) -> List[Dict]:
    """複数 run から AItuber キャラクター別の指標集計を返す。

    run の summary.json に streamer_persona_id がある場合に集計対象とする。

    Returns list of dicts with:
        persona_id, name, policy, engagement_proxy, safety_rate,
        topic_diversity, sentiment_shift, run_count
    """
    from collections import defaultdict

    buckets: Dict[str, Dict] = defaultdict(
        lambda: {
            "persona_id": "",
            "name": "",
            "policy": "",
            "engagement_proxy_sum": 0.0,
            "safety_rate_sum": 0.0,
            "topic_diversity_sum": 0.0,
            "sentiment_shift_sum": 0.0,
            "run_count": 0,
        }
    )

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load summary from %s: %s", run_dir, e)
            continue

        persona_id = data.get("streamer_persona_id")
        if not persona_id:
            continue

        bucket = buckets[persona_id]
        bucket["persona_id"] = persona_id
        bucket["name"] = data.get("streamer_name", persona_id)
        bucket["policy"] = data.get("policy", "")
        bucket["engagement_proxy_sum"] += float(data.get("engagement_proxy", 0.0))
        bucket["safety_rate_sum"] += float(data.get("safety_rate", 0.0))
        bucket["topic_diversity_sum"] += float(data.get("topic_diversity", 0.0))
        bucket["sentiment_shift_sum"] += float(data.get("sentiment_shift", 0.0))
        bucket["run_count"] += 1

    results: List[Dict] = []
    for bucket in buckets.values():
        count = bucket["run_count"]
        results.append(
            {
                "persona_id": bucket["persona_id"],
                "name": bucket["name"],
                "policy": bucket["policy"],
                "engagement_proxy": bucket["engagement_proxy_sum"] / count,
                "safety_rate": bucket["safety_rate_sum"] / count,
                "topic_diversity": bucket["topic_diversity_sum"] / count,
                "sentiment_shift": bucket["sentiment_shift_sum"] / count,
                "run_count": count,
            }
        )

    return results


def export_metrics_csv(run_dirs: List[Path], output: Path) -> None:
    """Export comparison metrics to CSV.

    Args:
        run_dirs: List of run directories to include.
        output: Output CSV file path.
    """
    df = compare_runs(run_dirs)
    if df.is_empty():
        logger.warning("No runs loaded, skipping CSV export.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output)
    logger.info("Exported metrics CSV to %s", output)
