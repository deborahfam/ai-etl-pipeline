"""Demo 2: Customer Reviews NLP Pipeline

CSV → Clean → PII Redaction → Enrich (sentiment, entities, language) →
    Validate → Anomaly Detection → Load → Analytics Summary

Demonstrates:
- PII detection and redaction (regex + LLM hybrid)
- Sentiment analysis and entity extraction via LLM
- Multilingual text processing
- Rating/sentiment consistency checking
- Fake review detection
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
from src.transformers.cleaner import clean_dataframe
from src.transformers.enricher import enrich_text_column
from src.transformers.pii_redactor import redact_pii
from src.intelligence.anomaly_detector import detect_anomalies
from src.loaders.duckdb_loader import load_to_duckdb
from src.loaders.file_loader import load_to_file
from src.llm.gateway import LLMGateway
from src.utils.display import display_dataframe, display_anomalies, console
from src.utils.logging import setup_logging

load_dotenv()
setup_logging()


@extract(name="load_reviews")
def load_reviews(ctx: PipelineContext) -> pl.DataFrame:
    path = ctx.config.get("source_path", "data/customer_reviews.csv")
    df = extract_csv(path)
    console.print(f"\n[bold]Loaded {len(df)} reviews[/bold]")
    display_dataframe(df, title="Raw Reviews", max_rows=5)
    return df


@transform(name="clean_reviews", depends_on=["load_reviews"])
def clean_reviews(df: pl.DataFrame) -> pl.DataFrame:
    return clean_dataframe(
        df,
        drop_duplicates=True,
        normalize_strings=True,
        fill_nulls={"review_text": "", "language": "unknown"},
    )


@ai_transform(name="redact_pii", depends_on=["clean_reviews"])
def redact_pii_step(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    text_cols = [c for c in ["review_text", "customer_name", "email"] if c in df.columns]
    df_redacted, pii_report = redact_pii(
        df,
        text_columns=text_cols,
        llm=llm,
        use_regex=True,
        use_llm=True,
    )
    ctx.store["pii_report"] = pii_report

    console.print(Panel(
        f"PII Found: [bold red]{pii_report.total_pii_found}[/bold red]\n"
        f"Columns affected: {pii_report.columns_affected}\n"
        f"Types: {', '.join(set(e.pii_type for e in pii_report.entities))}",
        title="PII Redaction Report",
        border_style="red",
    ))
    return df_redacted


@ai_transform(name="enrich_reviews", depends_on=["redact_pii"])
def enrich_reviews(df: pl.DataFrame, llm: LLMGateway) -> pl.DataFrame:
    if "review_text" not in df.columns:
        return df

    df = enrich_text_column(
        df,
        text_column="review_text",
        llm=llm,
        operations=["sentiment", "entities", "category", "language"],
        batch_size=10,
    )

    # Show enrichment preview
    preview_cols = ["review_text", "sentiment_label", "sentiment_score", "category", "detected_language"]
    available = [c for c in preview_cols if c in df.columns]
    display_dataframe(df.select(available), title="Enriched Reviews", max_rows=8)
    return df


@transform(name="detect_inconsistencies", depends_on=["enrich_reviews"])
def detect_inconsistencies(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    """Check for rating vs. sentiment contradictions."""
    inconsistencies = []

    if "rating" in df.columns and "sentiment_label" in df.columns:
        for idx in range(len(df)):
            rating = df["rating"][idx]
            sentiment = df["sentiment_label"][idx]
            if rating is None or sentiment is None:
                continue

            try:
                rating_val = float(rating)
            except (ValueError, TypeError):
                continue

            if rating_val >= 4 and sentiment in ("negative",):
                inconsistencies.append({
                    "row": idx,
                    "rating": rating_val,
                    "sentiment": sentiment,
                    "issue": "High rating but negative sentiment",
                })
            elif rating_val <= 2 and sentiment in ("positive",):
                inconsistencies.append({
                    "row": idx,
                    "rating": rating_val,
                    "sentiment": sentiment,
                    "issue": "Low rating but positive sentiment",
                })

    if inconsistencies:
        table = Table(title="Rating/Sentiment Inconsistencies", border_style="yellow")
        table.add_column("Row")
        table.add_column("Rating")
        table.add_column("Sentiment")
        table.add_column("Issue")
        for inc in inconsistencies[:10]:
            table.add_row(str(inc["row"]), str(inc["rating"]), inc["sentiment"], inc["issue"])
        console.print(table)

    ctx.store["inconsistencies"] = inconsistencies

    # Add inconsistency flag
    incon_rows = {i["row"] for i in inconsistencies}
    flags = ["inconsistent" if i in incon_rows else "consistent" for i in range(len(df))]
    df = df.with_columns(pl.Series("_consistency_flag", flags))

    return df


@load(name="save_reviews", depends_on=["detect_inconsistencies"])
def save_reviews(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    output_dir = ctx.output_dir
    load_to_duckdb(df, "customer_reviews", db_path=output_dir / "flowai.duckdb")
    load_to_file(df, output_dir / "reviews_enriched.csv")

    # Analytics summary
    if "sentiment_label" in df.columns:
        sentiment_dist = df.group_by("sentiment_label").len().sort("len", descending=True)
        console.print("\n")
        display_dataframe(sentiment_dist, title="Sentiment Distribution")

    if "category" in df.columns:
        cat_dist = df.group_by("category").len().sort("len", descending=True)
        display_dataframe(cat_dist, title="Category Distribution")

    if "detected_language" in df.columns:
        lang_dist = df.group_by("detected_language").len().sort("len", descending=True)
        display_dataframe(lang_dist, title="Language Distribution")

    console.print(f"\n[green]Data saved to {output_dir}/[/green]")
    return df


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        name="Customer Reviews NLP Pipeline",
        description="ETL pipeline: CSV → Clean → PII Redaction → Enrichment → Consistency Check → DuckDB",
    )
    pipeline.add_steps([
        load_reviews,
        clean_reviews,
        redact_pii_step,
        enrich_reviews,
        detect_inconsistencies,
        save_reviews,
    ])
    return pipeline


def main():
    console.print(Panel(
        "[bold magenta]FlowAI ETL[/bold magenta] - Customer Reviews NLP Pipeline\n"
        "Demonstrates: PII redaction, sentiment analysis, entity extraction,\n"
        "multilingual processing, consistency checking — all powered by AI",
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
