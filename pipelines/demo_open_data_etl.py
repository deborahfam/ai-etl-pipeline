"""Demo: real open data + ETL + LLM (maps directly to a typical recruiter brief).

Titanic (public teaching dataset) → clean → validate → **LLM anomaly explanations**
→ **LLM-assisted profiling** (semantic description of columns/dataset) → DuckDB + report.

Run ``python3 data/fetch_open_datasets.py`` once if ``data/real/titanic.csv`` is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import Pipeline, PipelineContext, extract, transform, ai_transform, load
from src.extractors.csv_extractor import extract_csv
from src.transformers.cleaner import clean_dataframe, normalize_column_names
from src.transformers.validator import validate_dataframe
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

TITANIC_PATH = Path(__file__).resolve().parent.parent / "data" / "real" / "titanic.csv"


@extract(name="load_open_dataset")
def load_open_dataset(ctx: PipelineContext) -> pl.DataFrame:
    path = Path(ctx.config.get("dataset_path", TITANIC_PATH))
    if not path.exists():
        console.print(
            "[red]Missing dataset.[/red] Run:\n"
            "  [cyan]python3 data/fetch_open_datasets.py[/cyan]\n"
            "Sources are documented in [cyan]data/real/README.md[/cyan]."
        )
        raise FileNotFoundError(path)
    df = extract_csv(path)
    ctx.store["dataset_label"] = "Titanic (open data, Data Science Dojo mirror)"
    console.print(
        f"\n[bold]Loaded[/bold] {ctx.store['dataset_label']}: "
        f"{len(df)} rows × {len(df.columns)} columns"
    )
    display_dataframe(df, title="Raw sample", max_rows=5)
    return df


@transform(name="normalize_open_data", depends_on=["load_open_dataset"])
def normalize_open_data(df: pl.DataFrame) -> pl.DataFrame:
    return normalize_column_names(df)


@transform(name="clean_open_data", depends_on=["normalize_open_data"])
def clean_open_data(df: pl.DataFrame) -> pl.DataFrame:
    numeric = [c for c in ("age", "fare", "sibsp", "parch") if c in df.columns]
    df = clean_dataframe(
        df,
        drop_duplicates=True,
        normalize_strings=True,
        fill_nulls={"embarked": "Unknown"},
    )
    if "age" in df.columns:
        med = df["age"].median()
        if med is not None:
            df = df.with_columns(pl.col("age").fill_null(med))
    return df


@transform(name="validate_open_data", depends_on=["clean_open_data"])
def validate_open_data(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    not_null = [c for c in ("passenger_id", "survived", "pclass") if c in df.columns]
    ranges = {}
    if "age" in df.columns:
        ranges["age"] = (0.0, 120.0)
    if "fare" in df.columns:
        ranges["fare"] = (0.0, 1_000_000.0)
    df, vr = validate_dataframe(df, not_null=not_null, value_ranges=ranges)
    ctx.store["validation"] = vr
    if vr.errors:
        console.print(Panel("\n".join(f"[yellow]{e}[/yellow]" for e in vr.errors), title="Validation"))
    return df


@ai_transform(name="detect_anomalies_open", depends_on=["validate_open_data"])
def detect_anomalies_open(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    label = ctx.store.get("dataset_label", "Open dataset")
    numeric = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    report = detect_anomalies(df, dataset_name=label, llm=llm, numeric_columns=numeric)
    ctx.store["anomaly_report"] = report
    display_anomalies(report)
    bad = {a.row_index for a in report.anomalies}
    df = df.with_columns(
        pl.Series("_anomaly_flag", ["flagged" if i in bad else "ok" for i in range(len(df))])
    )
    return df


@ai_transform(name="profile_open_data", depends_on=["detect_anomalies_open"])
def profile_open_data(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    label = ctx.store.get("dataset_label", "Open dataset")
    profile = auto_profile(df, dataset_name=label, llm=llm)
    ctx.store["profile"] = profile
    display_profile(profile)
    return df


@ai_transform(name="quality_open_data", depends_on=["profile_open_data"])
def quality_open_data(df: pl.DataFrame, llm: LLMGateway, ctx: PipelineContext) -> pl.DataFrame:
    label = ctx.store.get("dataset_label", "Open dataset")
    q = score_quality(df, dataset_name=label, llm=llm)
    ctx.store["quality_report"] = q
    console.print(Panel(
        f"Quality score: [bold]{q.overall_score}/100[/bold]\n{q.summary or ''}",
        title="LLM-aware quality summary",
        border_style="cyan",
    ))
    return df


@load(name="persist_open_data", depends_on=["quality_open_data"])
def persist_open_data(df: pl.DataFrame, ctx: PipelineContext) -> pl.DataFrame:
    out = ctx.output_dir
    load_to_duckdb(df, "titanic_open_data", db_path=out / "flowai.duckdb")
    load_to_file(df, out / "titanic_open_clean.parquet")

    report_path = out / "RECRUITER_BRIEF_ALIGNMENT.md"
    profile = ctx.store.get("profile")
    anoms = ctx.store.get("anomaly_report")
    qual = ctx.store.get("quality_report")
    lines = [
        "# ETL + LLM — alignment with technical brief",
        "",
        "This run used a **real open dataset** (Titanic passenger manifest) with a full Python ETL and API-backed LLM.",
        "",
        "## Brief checklist",
        "",
        "| Requirement | How this repo addresses it |",
        "|-------------|----------------------------|",
        "| ETL orchestration in Python | `Pipeline`, `@extract` / `@transform` / `@load` / `@ai_transform`, DAG ordering |",
        "| Runnable examples | `python -m pipelines.demo_*` and `run_all` |",
        "| LLM via API | `LLMGateway` + provider adapters (Anthropic, OpenAI, OpenRouter, LM Studio) |",
        "| Text + images | NLP demos + `complete_vision` / invoice extraction |",
        "| Useful AI: dataset description | `auto_profile` → semantic column/dataset narrative |",
        "| Useful AI: anomalies in natural language | `detect_anomalies` → stats + NL explanations |",
        "",
        "## This run",
        "",
        f"- Rows loaded: **{len(df)}**",
        f"- Anomalies surfaced: **{len(anoms.anomalies) if anoms else 0}**",
        f"- Quality score: **{qual.overall_score if qual else 'n/a'}**/100",
        "",
        "### Dataset summary (from profiler)",
        "",
        profile.summary if profile and profile.summary else "_No summary returned._",
        "",
        "---",
        "_Generated by `pipelines.demo_open_data_etl`._",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[green]Saved[/green] DuckDB/Parquet + [cyan]{report_path.name}[/cyan]")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        name="Open Data ETL + LLM",
        description="Real Titanic CSV → clean → validate → LLM anomalies + profile + quality → DuckDB",
    ).add_steps([
        load_open_dataset,
        normalize_open_data,
        clean_open_data,
        validate_open_data,
        detect_anomalies_open,
        profile_open_data,
        quality_open_data,
        persist_open_data,
    ])


def main():
    console.print(Panel(
        "[bold magenta]Open data ETL + LLM[/bold magenta]\n"
        "Demonstrates the recruiter brief: real dataset, ETL, profiling, anomaly explanations.",
        border_style="magenta",
    ))
    llm = LLMGateway.auto_detect()
    console.print(f"LLM providers: [cyan]{llm.available_providers}[/cyan]\n")
    ctx = build_pipeline().run(llm=llm)
    console.print("\n")
    llm.cost_tracker.display(console)
    return ctx


if __name__ == "__main__":
    main()
