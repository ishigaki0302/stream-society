"""Command-line interface for stream-society."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich.console import Console

app = typer.Typer(
    name="ss",
    help="StreamSociety - AI Livestream Simulation Platform",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@app.command()
def simulate(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML config"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override seed from config"),
    policy: Optional[str] = typer.Option(None, "--policy", help="Override policy from config"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override output directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run a single simulation experiment from a YAML config file."""
    _setup_logging(verbose)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if seed is not None:
        cfg["seed"] = seed
    if policy is not None:
        cfg["policy"] = policy

    out_dir = output_dir or Path(cfg.get("output_dir", "outputs/runs"))

    # Build RunConfig
    from simulator.schemas import RunConfig

    run_id = f"{cfg.get('experiment_name', 'run')}_{cfg.get('seed', 42)}_{uuid.uuid4().hex[:6]}"
    run_config = RunConfig(
        run_id=run_id,
        experiment_name=cfg.get("experiment_name", "unnamed"),
        seed=cfg.get("seed", 42),
        num_viewers=cfg.get("num_viewers", 10),
        num_turns=cfg.get("num_turns", 20),
        policy=cfg.get("policy", "random"),
        streamer_config=cfg.get("streamer_config", {}),
        description=cfg.get("description", ""),
    )

    # Load personas
    persona_data_path = Path(cfg.get("persona_data", "data/personas/sample_personas.jsonl"))
    if not persona_data_path.is_absolute():
        # Resolve relative to project root (parent of cli/)
        persona_data_path = Path(__file__).parent.parent / persona_data_path

    from simulator.persona import load_personas
    from simulator.simulation import Simulation

    personas = load_personas(persona_data_path)
    console.print(f"[bold green]Loaded {len(personas)} personas[/bold green]")

    # Run simulation
    console.print(
        f"[bold]Starting simulation[/bold]: {run_config.experiment_name} "
        f"(policy={run_config.policy}, turns={run_config.num_turns}, viewers={run_config.num_viewers})"
    )

    sim = Simulation(config=run_config, personas=personas)
    sim.run()

    # Save results
    sim.save(out_dir)

    # Print summary
    from analytics.report import print_summary

    summary = sim.get_summary()
    print_summary(summary)

    console.print(f"\n[bold green]Results saved to:[/bold green] {out_dir / run_id}")


@app.command()
def evaluate(
    run_dir: Path = typer.Argument(..., help="Path to run directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Compute and display detailed metrics for a completed run."""
    _setup_logging(verbose)

    from analytics.report import load_run, print_summary
    from rich.table import Table

    summary, turn_logs = load_run(run_dir)
    print_summary(summary)

    # Per-turn breakdown
    table = Table(title="Per-Turn Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Turn", style="dim")
    table.add_column("Candidates")
    table.add_column("Questions")
    table.add_column("Avg Sentiment")
    table.add_column("Selected")

    for turn in turn_logs:
        m = turn.metrics
        selected = turn.selected_comment.text[:30] + "..." if turn.selected_comment else "-"
        table.add_row(
            str(turn.turn_id),
            str(m.get("num_candidates", 0)),
            str(m.get("num_questions", 0)),
            f"{m.get('avg_sentiment', 0):.3f}",
            selected,
        )

    console.print(table)


@app.command()
def report(
    run_dirs: List[Path] = typer.Argument(..., help="Run directories to compare"),
    export_csv: Optional[Path] = typer.Option(
        None, "--export-csv", help="Export comparison to CSV"
    ),
) -> None:
    """Compare metrics across multiple simulation runs."""
    from analytics.report import compare_runs, export_metrics_csv
    from rich.table import Table

    df = compare_runs(run_dirs)

    if df.is_empty():
        console.print("[red]No valid runs found.[/red]")
        raise typer.Exit(1)

    table = Table(title="Run Comparison", show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(col, style="cyan")

    for row in df.iter_rows():
        table.add_row(*[str(v) for v in row])

    console.print(table)

    if export_csv:
        export_metrics_csv(run_dirs, export_csv)
        console.print(f"[green]Exported to {export_csv}[/green]")


@app.command(name="ingest-personas")
def ingest_personas(
    input_path: Path = typer.Argument(..., help="Input JSONL file path"),
    output_dir: Path = typer.Option(
        Path("outputs/personas"), "--output-dir", help="Output directory"
    ),
) -> None:
    """Run persona ingestion pipeline."""
    from ingestion.persona_ingestion import (
        compute_distribution_report,
        load_from_file,
        save_to_jsonl,
        save_to_parquet,
    )

    personas = load_from_file(input_path)
    console.print(f"[green]Loaded {len(personas)} personas[/green]")

    report_data = compute_distribution_report(personas)
    console.print("\n[bold]Distribution Report:[/bold]")
    for key, value in report_data.items():
        console.print(f"  {key}: {value}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_to_jsonl(personas, output_dir / "personas_normalized.jsonl")
    save_to_parquet(personas, output_dir / "personas_normalized.parquet")

    console.print(f"\n[green]Ingestion complete. Output: {output_dir}[/green]")


@app.command(name="ingest-aituber")
def ingest_aituber(
    output: Path = typer.Option(
        Path("data/personas/aituber_personas.jsonl"),
        "--output",
        help="Output JSONL path for parsed personas",
    ),
    use_sample: bool = typer.Option(
        False,
        "--use-sample",
        help="Use bundled sample instead of HuggingFace download",
    ),
) -> None:
    """AItuber-Personas-Japan データセットを取り込む."""
    from ingestion.aituber_ingestion import (
        SAMPLE_FALLBACK,
        ingest_from_huggingface,
        load_from_jsonl,
        save_to_jsonl,
    )

    if use_sample:
        console.print(f"[cyan]サンプルファイルを使用: {SAMPLE_FALLBACK}[/cyan]")
        if not SAMPLE_FALLBACK.exists():
            console.print(f"[red]サンプルファイルが見つかりません: {SAMPLE_FALLBACK}[/red]")
            raise typer.Exit(1)
        personas = load_from_jsonl(SAMPLE_FALLBACK)
        # Resolve output relative to project root if not absolute
        out_path = output if output.is_absolute() else Path(__file__).parent.parent / output
        save_to_jsonl(personas, out_path)
        console.print(f"[green]{len(personas)} 件のペルソナを保存しました: {out_path}[/green]")
    else:
        console.print(
            "[cyan]HuggingFace からダウンロード中: DataPilot/AItuber-Personas-Japan[/cyan]"
        )
        out_path = output if output.is_absolute() else Path(__file__).parent.parent / output
        try:
            personas = ingest_from_huggingface(out_path)
            console.print(
                f"[green]{len(personas)} 件の有効なペルソナを取り込みました: {out_path}[/green]"
            )
        except ImportError as e:
            console.print(f"[red]エラー: {e}[/red]")
            console.print("[yellow]--use-sample フラグでサンプルデータを使用できます[/yellow]")
            raise typer.Exit(1)


@app.command()
def demo(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run the built-in end-to-end demo (10 viewers, 20 turns, random + rule_based)."""
    _setup_logging(verbose)

    from simulator.persona import load_personas
    from simulator.schemas import RunConfig
    from simulator.simulation import Simulation
    from analytics.report import print_summary

    project_root = Path(__file__).parent.parent
    persona_path = project_root / "data" / "personas" / "sample_personas.jsonl"
    personas = load_personas(persona_path)

    out_dir = project_root / "outputs" / "runs"

    streamer_cfg = {
        "name": "Aoi",
        "topics": ["gaming", "music", "anime", "technology"],
        "response_style": "friendly",
    }

    policies = ["random", "rule_based", "score_based"]
    summaries = []

    for policy_name in policies:
        console.print(f"\n[bold cyan]Running demo: policy={policy_name}[/bold cyan]")

        run_id = f"demo_{policy_name}_{uuid.uuid4().hex[:6]}"
        run_config = RunConfig(
            run_id=run_id,
            experiment_name=f"demo_{policy_name}",
            seed=42,
            num_viewers=10,
            num_turns=20,
            policy=policy_name,
            streamer_config=streamer_cfg,
            description=f"Demo run with {policy_name} policy",
        )

        sim = Simulation(config=run_config, personas=personas)
        sim.run()
        sim.save(out_dir)

        summary = sim.get_summary()
        summaries.append(summary)
        print_summary(summary)

    # Comparison table
    from rich.table import Table

    table = Table(title="Demo Comparison", show_header=True, header_style="bold yellow")
    table.add_column("Policy")
    table.add_column("Engagement")
    table.add_column("Unique Rate")
    table.add_column("Safety Rate")
    table.add_column("Sentiment Shift")
    table.add_column("Topic Diversity")

    for s in summaries:
        table.add_row(
            s.policy,
            f"{s.engagement_proxy:.4f}",
            f"{s.unique_participant_rate:.4f}",
            f"{s.safety_rate:.4f}",
            f"{s.sentiment_shift:+.4f}",
            f"{s.topic_diversity:.4f}",
        )

    console.print(table)
    console.print(f"\n[green]Demo complete! Results in {out_dir}[/green]")


if __name__ == "__main__":
    app()
