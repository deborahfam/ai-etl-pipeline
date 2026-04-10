"""Run all FlowAI ETL demo pipelines.

Usage:
    python -m pipelines.run_all              # Standard suite (4 demos)
    python -m pipelines.run_all --quick        # Skip invoice pipeline (no vision)
    python -m pipelines.run_all --with-open-data   # Also run real open-data ETL demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

console = Console()


def main():
    quick_mode = "--quick" in sys.argv
    open_data = "--with-open-data" in sys.argv

    extra = ""
    if open_data:
        extra = "0. Open Data ETL — Titanic (real CSV + LLM profile/anomalies)\n"

    console.print(Panel(
        "[bold magenta]FlowAI ETL[/bold magenta] — Intelligent ETL Orchestration with Multimodal AI\n\n"
        "Running demo pipelines:\n"
        + extra
        + "  1. Sales Analytics (anomaly detection + profiling)\n"
        "  2. Customer Reviews NLP (sentiment + PII + entities)\n"
        + ("  3. Invoice Processing (LLM Vision extraction)\n" if not quick_mode else "")
        + "  4. Multi-Source Integration (schema mapping + advisor)",
        title="Full Demo Suite",
        border_style="magenta",
    ))

    start = time.time()
    results = []

    if open_data:
        console.print(Rule("[bold cyan]Open Data ETL (Titanic)[/bold cyan]"))
        try:
            from pipelines.demo_open_data_etl import main as demo_open
            ctx = demo_open()
            results.append(("Open Data ETL (Titanic)", "OK", ctx.run.duration_seconds if ctx else 0))
        except Exception as e:
            console.print(f"[red]Open data demo failed: {e}[/red]")
            results.append(("Open Data ETL (Titanic)", f"FAILED: {e}", 0))

    # Demo 1: Sales Analytics
    console.print(Rule("[bold cyan]Demo 1: Sales Analytics[/bold cyan]"))
    try:
        from pipelines.demo_sales_analytics import main as demo1
        ctx = demo1()
        results.append(("Sales Analytics", "OK", ctx.run.duration_seconds if ctx else 0))
    except Exception as e:
        console.print(f"[red]Demo 1 failed: {e}[/red]")
        results.append(("Sales Analytics", f"FAILED: {e}", 0))

    # Demo 2: Customer Reviews
    console.print(Rule("[bold cyan]Demo 2: Customer Reviews NLP[/bold cyan]"))
    try:
        from pipelines.demo_customer_reviews import main as demo2
        ctx = demo2()
        results.append(("Customer Reviews", "OK", ctx.run.duration_seconds if ctx else 0))
    except Exception as e:
        console.print(f"[red]Demo 2 failed: {e}[/red]")
        results.append(("Customer Reviews", f"FAILED: {e}", 0))

    # Demo 3: Invoice Processing (Vision)
    if not quick_mode:
        console.print(Rule("[bold cyan]Demo 3: Invoice Processing (Vision)[/bold cyan]"))
        try:
            from pipelines.demo_invoice_processing import main as demo3
            ctx = demo3()
            results.append(("Invoice Processing", "OK", ctx.run.duration_seconds if ctx else 0))
        except Exception as e:
            console.print(f"[red]Demo 3 failed: {e}[/red]")
            results.append(("Invoice Processing", f"FAILED: {e}", 0))

    # Demo 4: Multi-Source Integration
    console.print(Rule("[bold cyan]Demo 4: Multi-Source Integration[/bold cyan]"))
    try:
        from pipelines.demo_multiformat_ingestion import main as demo4
        ctx = demo4()
        results.append(("Multi-Source Integration", "OK", ctx.run.duration_seconds if ctx else 0))
    except Exception as e:
        console.print(f"[red]Demo 4 failed: {e}[/red]")
        results.append(("Multi-Source Integration", f"FAILED: {e}", 0))

    # Final summary
    total_time = time.time() - start
    console.print(Rule("[bold magenta]Final Summary[/bold magenta]"))

    from rich.table import Table
    table = Table(title="Pipeline Results", border_style="magenta")
    table.add_column("Pipeline")
    table.add_column("Status")
    table.add_column("Duration", justify="right")

    for name, status, duration in results:
        status_display = "[green]OK[/green]" if status == "OK" else f"[red]{status}[/red]"
        table.add_row(name, status_display, f"{duration:.1f}s")

    table.add_section()
    table.add_row("[bold]Total[/bold]", "", f"[bold]{total_time:.1f}s[/bold]")
    console.print(table)

    console.print(f"\nOutput files: [cyan]output/[/cyan]")
    output_dir = Path("output")
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            size = f.stat().st_size / 1024
            console.print(f"  {f.name} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
