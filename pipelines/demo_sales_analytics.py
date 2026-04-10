"""Demo 1: Sales Analytics Pipeline

CSV → Clean → Validate → Detect Anomalies → Profile → Load → Report

Demonstrates:
- Full ETL pipeline with decorator-based step definition
- Statistical + LLM anomaly detection with natural language explanations
- Auto data profiling with semantic type inference
- DuckDB loading + Parquet export
- Rich console output with progress tracking
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import Pipeline, PipelineContext, extract, transform, ai_transform, load
from src.extractors.csv_extractor import extract_csv
from src.transformers.cleaner import clean_dataframe, normalize_column_names
from src.transformers.validator import validate_dataframe, validate_semantic
from src.intelligence.profiler import auto_profile
from src.intelligence.anomaly_detector import detect_anomalies
from src.intelligence.quality_scorer import score_quality
from src.loaders.duckdb_loader import load_to_duckdb
from src.loaders.file_loader import load_to_file
from src.llm.gateway import LLMGateway
from src.utils.display import display_dataframe, display_profile, display_anomalies, console
from src.utils.logging import setup_logging

load_dotenv()
setup_logging()


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

@extract(name="load_sales_data", description="Load sales transactions from CSV")
def load_sales_data(ctx: PipelineContext) -> pl.DataFrame:
    path = ctx.config.get("source_path", "data/sales_transactions.csv")
    df = extract_csv(path)
    console.print(f"\n[bold]Loaded {len(df)} raw transactions[/bold]")
    display_dataframe(df, title="Raw Sales Data", max_rows=5)
    return df


@transform(name="normalize_columns", depends_on=["load_sales_data"])
def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    return normalize_column_names(df)


@transform(
    name="clean_sales_data",
    depends_on=["normalize_columns"],
    description="Clean: deduplicate, handle nulls, normalize formats",
)
def clean_sales_data(df: pl.DataFrame) -> pl.DataFrame:
    # Coerce numeric columns
    numeric_cols = []
    for col in ["quantity", "unit_price", "total"]:
        if col in df.columns:
            numeric_cols.append(col)

    df = clean_dataframe(
        df,
        drop_duplicates=True,
        normalize_strings=True,
        coerce_numerics=numeric_cols if any(df[c].dtype == pl.Utf8 for c in numeric_cols if c in df.columns) else None,
        fill_nulls={"notes": "", "salesperson": "Unknown"},
    )
    return df


@transform(
    name="validate_sales",
    depends_on=["clean_sales_data"],
    description="Rule-based validation: nulls, ranges, custom checks",
)
def validate_sales(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    def check_total_matches(df: pl.DataFrame) -> str | None:
        """Check that total ≈ quantity × unit_price."""
        if all(c in df.columns for c in ["quantity", "unit_price", "total"]):
            mismatches = df.filter(
                (pl.col("total") - pl.col("quantity") * pl.col("unit_price")).abs() > 0.01
            )
            if len(mismatches) > 0:
                return f"{len(mismatches)} rows where total ≠ quantity × unit_price"
        return None

    not_null_cols = [c for c in ["transaction_id", "date", "quantity"] if c in df.columns]
    value_ranges = {}
    if "quantity" in df.columns and df["quantity"].dtype in (pl.Float64, pl.Int64):
        value_ranges["quantity"] = (0, 10000)
    if "unit_price" in df.columns and df["unit_price"].dtype in (pl.Float64, pl.Int64):
        value_ranges["unit_price"] = (0, 100000)

    df, result = validate_dataframe(
        df,
        not_null=not_null_cols,
        value_ranges=value_ranges,
        custom_checks=[check_total_matches],
    )

    if result.errors:
        console.print(Panel(
            "\n".join(f"[red]- {e}[/red]" for e in result.errors),
            title="Validation Issues",
            border_style="red",
        ))

    ctx.store["validation_result"] = result
    return df


@ai_transform(
    name="detect_sales_anomalies",
    depends_on=["validate_sales"],
    description="Statistical + LLM anomaly detection",
)
def detect_sales_anomalies(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    report = detect_anomalies(
        df,
        dataset_name="Sales Transactions",
        llm=llm,
        numeric_columns=numeric_cols,
    )

    ctx.store["anomaly_report"] = report
    display_anomalies(report)

    # Add anomaly flag column
    anomaly_rows = {a.row_index for a in report.anomalies}
    flags = ["anomaly" if i in anomaly_rows else "normal" for i in range(len(df))]
    df = df.with_columns(pl.Series("_anomaly_flag", flags))

    return df


@ai_transform(
    name="profile_sales",
    depends_on=["detect_sales_anomalies"],
    description="Auto-profile with LLM semantic descriptions",
)
def profile_sales(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    profile = auto_profile(df, dataset_name="Sales Transactions", llm=llm)
    ctx.store["profile"] = profile
    display_profile(profile)
    return df


@ai_transform(
    name="score_data_quality",
    depends_on=["profile_sales"],
    description="Multi-dimensional quality scoring",
)
def score_data_quality(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    report = score_quality(df, dataset_name="Sales Transactions", llm=llm)
    ctx.store["quality_report"] = report

    console.print(Panel(
        f"Overall Quality: [bold]{report.overall_score}/100[/bold]\n\n"
        + "\n".join(f"  {d.name}: {d.score}/100 - {d.details}" for d in report.dimensions)
        + (f"\n\n{report.summary}" if report.summary else ""),
        title="Data Quality Report",
        border_style="green" if report.overall_score >= 70 else "yellow",
    ))
    return df


@load(
    name="save_to_duckdb",
    depends_on=["score_data_quality"],
    description="Load cleaned data into DuckDB + Parquet",
)
def save_to_duckdb_step(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    output_dir = ctx.output_dir

    # Save to DuckDB
    load_to_duckdb(df, "sales_transactions", db_path=output_dir / "flowai.duckdb")

    # Save to Parquet
    load_to_file(df, output_dir / "sales_transactions_clean.parquet")

    # Save anomalies report
    report = ctx.store.get("anomaly_report")
    if report and report.anomalies:
        anomaly_dicts = [a.model_dump() for a in report.anomalies]
        anomaly_df = pl.DataFrame(anomaly_dicts)
        load_to_file(anomaly_df, output_dir / "sales_anomalies.csv")

    console.print(f"\n[green]Data saved to {output_dir}/[/green]")
    return df


# ---------------------------------------------------------------------------
# Pipeline Assembly
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        name="Sales Analytics Pipeline",
        description="ETL pipeline: CSV → Clean → Validate → Anomaly Detection → Profile → DuckDB",
    )
    pipeline.add_steps([
        load_sales_data,
        normalize_columns,
        clean_sales_data,
        validate_sales,
        detect_sales_anomalies,
        profile_sales,
        score_data_quality,
        save_to_duckdb_step,
    ])
    return pipeline


def main():
    console.print(Panel(
        "[bold magenta]FlowAI ETL[/bold magenta] - Sales Analytics Pipeline\n"
        "Demonstrates: CSV extraction, cleaning, validation, anomaly detection,\n"
        "data profiling, quality scoring — all powered by AI",
        border_style="magenta",
    ))

    llm = LLMGateway.auto_detect()
    console.print(f"LLM Providers: [cyan]{llm.available_providers}[/cyan]\n")

    pipeline = build_pipeline()
    ctx = pipeline.run(llm=llm)

    # Final summary
    console.print("\n")
    llm.cost_tracker.display(console)

    return ctx


if __name__ == "__main__":
    main()
