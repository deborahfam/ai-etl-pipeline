"""Demo 3: Invoice Processing Pipeline (Vision)

Images → Extract (LLM Vision) → Validate → Clean → Load → Quality Report

Demonstrates:
- Multimodal AI: extracting structured data from document images
- LLM Vision with structured outputs (Pydantic schema)
- Cross-field validation (totals, line item math)
- Document processing pipeline inspired by Unstract
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import Pipeline, PipelineContext, extract, transform, ai_transform, load
from src.engine.models import InvoiceData
from src.intelligence.document_processor import process_documents
from src.transformers.cleaner import clean_dataframe
from src.intelligence.quality_scorer import score_quality
from src.loaders.duckdb_loader import load_to_duckdb
from src.loaders.file_loader import load_to_file
from src.llm.gateway import LLMGateway
from src.utils.display import display_dataframe, console
from src.utils.logging import setup_logging

load_dotenv()
setup_logging()


@extract(name="discover_invoices")
def discover_invoices(ctx: PipelineContext) -> pl.DataFrame:
    invoices_dir = Path(ctx.config.get("invoices_dir", "data/invoices"))
    image_files = sorted(invoices_dir.glob("*.png")) + sorted(invoices_dir.glob("*.jpg"))

    if not image_files:
        console.print(f"[yellow]No invoice images found in {invoices_dir}[/yellow]")
        console.print("[dim]Run: python data/generate_invoices.py to generate sample invoices[/dim]")
        return pl.DataFrame({"_source_file": []})

    console.print(f"\n[bold]Found {len(image_files)} invoice images[/bold]")
    for f in image_files:
        console.print(f"  [dim]{f.name}[/dim] ({f.stat().st_size / 1024:.1f} KB)")

    # Store paths for next step
    ctx.store["invoice_paths"] = image_files
    return pl.DataFrame({"_source_file": [f.name for f in image_files]})


@ai_transform(name="extract_invoice_data", depends_on=["discover_invoices"])
def extract_invoice_data(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    invoice_paths = ctx.store.get("invoice_paths", [])
    if not invoice_paths:
        return df

    console.print("\n[bold cyan]Extracting data from invoices using LLM Vision...[/bold cyan]")

    result_df, validation_issues = process_documents(
        document_paths=invoice_paths,
        response_model=InvoiceData,
        llm=llm,
        validate=True,
    )

    ctx.store["validation_issues"] = validation_issues

    if validation_issues:
        console.print(Panel(
            "\n".join(
                f"[yellow]- {v['source']}: {v['message']}[/yellow]"
                for v in validation_issues[:10]
            ),
            title="Extraction Validation Issues",
            border_style="yellow",
        ))

    display_dataframe(result_df, title="Extracted Invoice Data", max_rows=10)
    return result_df


@transform(name="clean_invoices", depends_on=["extract_invoice_data"])
def clean_invoices(df: pl.DataFrame) -> pl.DataFrame:
    if len(df) == 0:
        return df
    return clean_dataframe(df, normalize_strings=True)


@ai_transform(name="score_invoice_quality", depends_on=["clean_invoices"])
def score_invoice_quality(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    if len(df) == 0:
        return df

    report = score_quality(df, dataset_name="Extracted Invoices", llm=llm)
    ctx.store["quality_report"] = report

    console.print(Panel(
        f"Overall Quality: [bold]{report.overall_score}/100[/bold]\n\n"
        + "\n".join(f"  {d.name}: {d.score}/100" for d in report.dimensions)
        + (f"\n\n{report.summary}" if report.summary else ""),
        title="Invoice Data Quality",
        border_style="green" if report.overall_score >= 70 else "yellow",
    ))
    return df


@load(name="save_invoices", depends_on=["score_invoice_quality"])
def save_invoices(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    if len(df) == 0:
        console.print("[yellow]No data to save[/yellow]")
        return df

    output_dir = ctx.output_dir
    load_to_duckdb(df, "invoices", db_path=output_dir / "flowai.duckdb")
    load_to_file(df, output_dir / "invoices_extracted.csv")

    # Summary stats
    if "total" in df.columns:
        total_col = df["total"]
        if total_col.dtype in (pl.Float64, pl.Int64):
            console.print(Panel(
                f"Invoices processed: [bold]{len(df)}[/bold]\n"
                f"Total value: [bold green]${total_col.sum():,.2f}[/bold green]\n"
                f"Average: ${total_col.mean():,.2f}\n"
                f"Range: ${total_col.min():,.2f} - ${total_col.max():,.2f}",
                title="Invoice Summary",
                border_style="green",
            ))

    console.print(f"\n[green]Data saved to {output_dir}/[/green]")
    return df


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        name="Invoice Processing Pipeline",
        description="ETL pipeline: Images → LLM Vision Extraction → Validate → Clean → DuckDB",
    )
    pipeline.add_steps([
        discover_invoices,
        extract_invoice_data,
        clean_invoices,
        score_invoice_quality,
        save_invoices,
    ])
    return pipeline


def main():
    console.print(Panel(
        "[bold magenta]FlowAI ETL[/bold magenta] - Invoice Processing Pipeline (Vision)\n"
        "Demonstrates: LLM Vision for document data extraction,\n"
        "cross-field validation, quality scoring — multimodal AI",
        border_style="magenta",
    ))

    llm = LLMGateway.auto_detect()
    if not llm.has_vision:
        console.print("[red]No vision-capable LLM provider available![/red]")
        console.print("Configure ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        return None

    console.print(f"LLM Providers: [cyan]{llm.available_providers}[/cyan]")
    console.print(f"Vision support: [green]Yes[/green]\n")

    pipeline = build_pipeline()
    ctx = pipeline.run(llm=llm)

    console.print("\n")
    llm.cost_tracker.display(console)
    return ctx


if __name__ == "__main__":
    main()
