"""Rich display utilities for pipeline output."""

from __future__ import annotations

from typing import Any

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def display_dataframe(
    df: pl.DataFrame,
    title: str = "Data Preview",
    max_rows: int = 10,
    max_col_width: int = 40,
) -> None:
    """Display a Polars DataFrame as a Rich table."""
    table = Table(title=title, border_style="blue", show_lines=True)

    for col in df.columns:
        table.add_column(col, style="cyan", max_width=max_col_width)

    for row in df.head(max_rows).iter_rows():
        table.add_row(*[_truncate(str(v), max_col_width) for v in row])

    if len(df) > max_rows:
        table.add_row(*[f"... ({len(df)} total)" for _ in df.columns])

    console.print(table)


def display_profile(profile: Any) -> None:
    """Display a DataProfile as a Rich table."""
    table = Table(title=f"Data Profile: {profile.dataset_name}", border_style="green")
    table.add_column("Column", style="cyan")
    table.add_column("Type")
    table.add_column("Semantic")
    table.add_column("Nulls %", justify="right")
    table.add_column("Unique %", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Description", max_width=50)

    for col in profile.columns:
        q_color = "green" if col.quality_score >= 80 else "yellow" if col.quality_score >= 50 else "red"
        table.add_row(
            col.name,
            col.dtype,
            col.semantic_type or "-",
            f"{col.null_percentage:.1f}%",
            f"{col.unique_percentage:.1f}%",
            f"[{q_color}]{col.quality_score}[/{q_color}]",
            _truncate(col.description, 50),
        )

    console.print(table)
    console.print(
        Panel(
            f"Overall Quality: [bold]{'green' if profile.overall_quality_score >= 80 else 'yellow' if profile.overall_quality_score >= 50 else 'red'}]{profile.overall_quality_score}/100[/]\n\n"
            f"{profile.summary}",
            title="Summary",
            border_style="green",
        )
    )


def display_anomalies(report: Any) -> None:
    """Display an AnomalyReport as a Rich table."""
    if not report.anomalies:
        console.print("[green]No anomalies detected.[/green]")
        return

    table = Table(title=f"Anomalies: {report.dataset_name}", border_style="red")
    table.add_column("Row", justify="right")
    table.add_column("Column")
    table.add_column("Value", max_width=20)
    table.add_column("Type")
    table.add_column("Severity")
    table.add_column("Explanation", max_width=50)
    table.add_column("Action", max_width=30)

    severity_colors = {"critical": "red", "warning": "yellow", "info": "blue"}

    for a in report.anomalies[:20]:  # Limit display
        color = severity_colors.get(a.severity.value, "white")
        table.add_row(
            str(a.row_index),
            a.column,
            _truncate(a.value, 20),
            a.anomaly_type.value,
            f"[{color}]{a.severity.value.upper()}[/{color}]",
            _truncate(a.explanation, 50),
            _truncate(a.recommended_action, 30),
        )

    if len(report.anomalies) > 20:
        table.caption = f"Showing 20 of {len(report.anomalies)} anomalies"

    console.print(table)
    console.print(
        Panel(
            f"Total: [bold]{len(report.anomalies)}[/bold] | "
            f"Critical: [red]{report.critical_count}[/red] | "
            f"Warning: [yellow]{report.warning_count}[/yellow] | "
            f"Info: [blue]{report.info_count}[/blue]\n\n"
            f"{report.summary}",
            title="Anomaly Summary",
            border_style="red",
        )
    )


def display_pipeline_lineage(lineage: dict[str, list[str]]) -> None:
    """Display column lineage as a tree."""
    tree = Tree("[bold cyan]Column Lineage[/bold cyan]")
    for column, steps in sorted(lineage.items()):
        branch = tree.add(f"[cyan]{column}[/cyan]")
        for step in steps:
            branch.add(f"[green]{step}[/green]")
    console.print(tree)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
