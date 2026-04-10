"""Demo 4: Multi-Source Integration Pipeline

CSV (sales) + PDF (document metadata) + JSON (products) + API (mock) →
    Schema Mapping → Merge & Reconcile → Clean → Pipeline Advisor → Load

Demonstrates:
- Multi-source data integration (tabular + document + API)
- LLM-powered schema mapping between heterogeneous sources
- Data reconciliation and merging
- Pipeline Advisor: AI suggests optimal transformations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import Pipeline, PipelineContext, extract, transform, ai_transform, load
from src.extractors.csv_extractor import extract_csv
from src.extractors.api_extractor import extract_from_mock_api
from src.extractors.pdf_extractor import extract_from_pdf
from src.transformers.cleaner import clean_dataframe
from src.transformers.semantic_mapper import map_schemas, apply_mapping
from src.intelligence.pipeline_advisor import advise_pipeline
from src.intelligence.profiler import auto_profile
from src.loaders.duckdb_loader import load_to_duckdb
from src.loaders.file_loader import load_to_file
from src.llm.gateway import LLMGateway
from src.utils.display import display_dataframe, display_profile, console
from src.utils.logging import setup_logging

load_dotenv()
setup_logging()


@extract(name="load_csv_source")
def load_csv_source(ctx: PipelineContext) -> pl.DataFrame:
    path = ctx.config.get("sales_path", "data/sales_transactions.csv")
    df = extract_csv(path)
    console.print(f"\n[bold]Source 1 (CSV):[/bold] {len(df)} rows, {len(df.columns)} cols")
    ctx.store["csv_data"] = df
    return df


@extract(name="load_pdf_source", depends_on=["load_csv_source"])
def load_pdf_source(ctx: PipelineContext, outputs: dict) -> pl.DataFrame:
    """Ingest a PDF as a fourth source (metadata / byte stats; optional LLM vision via extract_from_pdf)."""
    pdf_path = Path(ctx.config.get("pdf_path", "data/documents/report_sample.pdf"))
    if pdf_path.exists():
        pdf_df = extract_from_pdf(pdf_path, llm=None)
        ctx.store["pdf_data"] = pdf_df
        console.print(
            f"[bold]Source 2 (PDF):[/bold] {pdf_path.name} — "
            f"{len(pdf_df)} row(s) from document intake (non-vision preview)"
        )
    else:
        console.print(f"[yellow]PDF not found at {pdf_path} — run: python data/bootstrap_demo_data.py[/yellow]")
        ctx.store["pdf_data"] = pl.DataFrame()
    return outputs.get("load_csv_source", pl.DataFrame())


@extract(name="load_json_source", depends_on=["load_pdf_source"])
def load_json_source(ctx: PipelineContext, outputs: dict) -> pl.DataFrame:
    path = ctx.config.get("products_path", "data/products_catalog.json")
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        df = pl.DataFrame(data) if isinstance(data, list) else pl.DataFrame([data])
    else:
        # Mock product data if file doesn't exist
        df = pl.DataFrame([
            {"product_id": f"P{i:03d}", "name": f"Product {i}", "category": "General", "price": 10.0 + i}
            for i in range(1, 20)
        ])

    console.print(f"[bold]Source 3 (JSON):[/bold] {len(df)} rows, {len(df.columns)} cols")
    ctx.store["json_data"] = df
    return outputs.get("load_csv_source", df)


@extract(name="load_api_source", depends_on=["load_json_source"])
def load_api_source(ctx: PipelineContext, outputs: dict) -> pl.DataFrame:
    # Mock API data simulating an external inventory system
    mock_records = [
        {"item_code": f"SKU-{i:04d}", "item_name": f"Widget {chr(65 + i % 26)}", "stock_qty": 100 + i * 10,
         "unit_cost": round(5.0 + i * 1.5, 2), "warehouse": ["NYC", "LAX", "CHI", "MIA"][i % 4],
         "last_updated": f"2026-0{(i % 9) + 1}-{(i % 28) + 1:02d}"}
        for i in range(25)
    ]
    df = extract_from_mock_api(mock_records)
    console.print(f"[bold]Source 4 (API):[/bold] {len(df)} rows, {len(df.columns)} cols")
    ctx.store["api_data"] = df
    return outputs.get("load_csv_source", df)


@ai_transform(name="map_schemas", depends_on=["load_api_source"])
def map_schemas_step(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    """Use LLM to map schemas between CSV and API data sources."""
    csv_df = ctx.store.get("csv_data")
    api_df = ctx.store.get("api_data")

    if csv_df is None or api_df is None:
        return df

    console.print("\n[bold cyan]Schema Mapping: Inferring column correspondences with AI...[/bold cyan]")

    mapping = map_schemas(
        source_df=api_df,
        target_df=csv_df,
        llm=llm,
        context="API source is an inventory system; CSV source is sales transactions. Map matching fields.",
    )

    ctx.store["schema_mapping"] = mapping

    # Display mapping results
    table = Table(title="AI Schema Mapping Results", border_style="cyan")
    table.add_column("Source (API)")
    table.add_column("→")
    table.add_column("Target (CSV)")
    table.add_column("Confidence")
    table.add_column("Transform Hint")

    for m in mapping.mappings:
        conf_color = "green" if m.confidence >= 0.8 else "yellow" if m.confidence >= 0.5 else "red"
        table.add_row(
            m.source_column, "→", m.target_column,
            f"[{conf_color}]{m.confidence:.0%}[/{conf_color}]",
            m.transform_hint or "-",
        )

    if mapping.unmapped_source:
        for col in mapping.unmapped_source:
            table.add_row(col, "→", "[dim]no match[/dim]", "-", "-")

    console.print(table)

    if mapping.notes:
        console.print(f"[dim]Notes: {mapping.notes}[/dim]")

    return df


@ai_transform(name="merge_sources", depends_on=["map_schemas"])
def merge_sources(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    """Merge data from multiple sources."""
    csv_df = ctx.store.get("csv_data")
    json_df = ctx.store.get("json_data")

    if csv_df is None:
        return df

    # Enrich sales with product info from JSON catalog
    if json_df is not None and "product_id" in csv_df.columns and "product_id" in json_df.columns:
        # Select only useful columns from products to avoid conflicts
        product_cols = ["product_id"]
        for col in ["name", "category", "description"]:
            if col in json_df.columns:
                product_cols.append(col)

        products = json_df.select(product_cols).unique("product_id")
        merged = csv_df.join(products, on="product_id", how="left", suffix="_product")
        console.print(f"\n[bold]Merged sales with products:[/bold] {len(merged)} rows, {len(merged.columns)} cols")
        display_dataframe(merged, title="Merged Dataset", max_rows=5)
        return merged

    return csv_df


@transform(name="clean_merged", depends_on=["merge_sources"])
def clean_merged(df: pl.DataFrame) -> pl.DataFrame:
    return clean_dataframe(df, drop_duplicates=True, normalize_strings=True)


@ai_transform(name="advise_pipeline", depends_on=["clean_merged"])
def advise_pipeline_step(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    """Let AI suggest optimal transformations for this dataset."""
    console.print("\n[bold cyan]Pipeline Advisor: Analyzing dataset for recommendations...[/bold cyan]")

    advice = advise_pipeline(
        df,
        dataset_name="Multi-Source Integrated Dataset",
        llm=llm,
        target_use_case="Sales analytics and inventory optimization",
    )
    ctx.store["advice"] = advice

    # Display advice
    console.print(Panel(advice.dataset_summary, title="Dataset Analysis", border_style="cyan"))

    if advice.suggested_transforms:
        table = Table(title="Suggested Transformations", border_style="green")
        table.add_column("#", justify="right")
        table.add_column("Step")
        table.add_column("Priority")
        table.add_column("Description", max_width=50)
        table.add_column("Reason", max_width=40)

        for i, t in enumerate(advice.suggested_transforms, 1):
            p_color = {"high": "red", "medium": "yellow", "low": "green"}.get(t.priority, "white")
            table.add_row(
                str(i), t.step_name, f"[{p_color}]{t.priority}[/{p_color}]",
                t.description, t.reason,
            )
        console.print(table)

    if advice.warnings:
        console.print(Panel(
            "\n".join(f"[yellow]- {w}[/yellow]" for w in advice.warnings),
            title="Warnings",
            border_style="yellow",
        ))

    return df


@load(name="save_integrated", depends_on=["advise_pipeline"])
def save_integrated(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    output_dir = ctx.output_dir
    load_to_duckdb(df, "integrated_data", db_path=output_dir / "flowai.duckdb")
    load_to_file(df, output_dir / "integrated_data.parquet")
    console.print(f"\n[green]Data saved to {output_dir}/[/green]")
    return df


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        name="Multi-Source Integration Pipeline",
        description="ETL: CSV + PDF + JSON + API → Schema Mapping → Merge → Clean → AI Advisor → DuckDB",
    )
    pipeline.add_steps([
        load_csv_source,
        load_pdf_source,
        load_json_source,
        load_api_source,
        map_schemas_step,
        merge_sources,
        clean_merged,
        advise_pipeline_step,
        save_integrated,
    ])
    return pipeline


def main():
    console.print(Panel(
        "[bold magenta]FlowAI ETL[/bold magenta] - Multi-Source Integration Pipeline\n"
        "Demonstrates: CSV + PDF + JSON + API, AI schema mapping,\n"
        "data reconciliation, Pipeline Advisor — intelligent ETL",
        border_style="magenta",
    ))

    llm = LLMGateway.auto_detect()
    console.print(f"LLM Providers: [cyan]{llm.available_providers}[/cyan]\n")

    pipeline = build_pipeline()
    ctx = pipeline.run(llm=llm)

    console.print("\n")
    llm.cost_tracker.display(console)
    return ctx


if __name__ == "__main__":
    main()
