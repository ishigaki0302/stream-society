"""Analytics and reporting for stream-society simulation runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

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
