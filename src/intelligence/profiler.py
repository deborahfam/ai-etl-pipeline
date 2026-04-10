"""Auto Data Profiler: statistical analysis + LLM semantic descriptions."""

from __future__ import annotations

import json

import polars as pl

from src.engine.models import ColumnProfile, DataProfile
from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


def auto_profile(
    df: pl.DataFrame,
    dataset_name: str = "dataset",
    llm: LLMGateway | None = None,
    sample_size: int = 20,
) -> DataProfile:
    """Generate a comprehensive data profile with statistical and semantic analysis.

    Phase 1: Compute statistics for every column (nulls, uniques, distribution).
    Phase 2: Use LLM to generate semantic descriptions, infer types, detect issues.
    """
    columns: list[ColumnProfile] = []

    for col_name in df.columns:
        series = df[col_name]
        profile = _compute_column_stats(col_name, series, len(df))
        columns.append(profile)

    # Phase 2: LLM enrichment
    summary = ""
    relationships: list[str] = []
    recommendations: list[str] = []

    if llm is not None:
        llm_profile = _enrich_with_llm(df, columns, dataset_name, llm, sample_size)
        # Merge LLM results into column profiles
        for col_p in columns:
            llm_col = llm_profile.get(col_p.name, {})
            col_p.semantic_type = llm_col.get("semantic_type", col_p.semantic_type)
            col_p.description = llm_col.get("description", col_p.description)
            issues = llm_col.get("issues", [])
            col_p.issues.extend(issues)
            if issues:
                col_p.quality_score = max(0, col_p.quality_score - len(issues) * 10)

        summary = llm_profile.get("_summary", "")
        relationships = llm_profile.get("_relationships", [])
        recommendations = llm_profile.get("_recommendations", [])

    # Calculate overall quality
    quality_scores = [c.quality_score for c in columns]
    overall_quality = int(sum(quality_scores) / len(quality_scores)) if quality_scores else 100

    return DataProfile(
        dataset_name=dataset_name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        overall_quality_score=overall_quality,
        summary=summary,
        relationships=relationships,
        recommendations=recommendations,
    )


def _compute_column_stats(name: str, series: pl.Series, total_rows: int) -> ColumnProfile:
    """Compute statistical profile for a single column."""
    null_count = series.null_count()
    null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0.0
    unique_count = series.n_unique()
    unique_pct = (unique_count / total_rows * 100) if total_rows > 0 else 0.0

    # Sample values (non-null)
    non_null = series.drop_nulls()
    sample_vals = [str(v) for v in non_null.head(5).to_list()] if len(non_null) > 0 else []

    profile = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        null_count=null_count,
        null_percentage=round(null_pct, 2),
        unique_count=unique_count,
        unique_percentage=round(unique_pct, 2),
        sample_values=sample_vals,
    )

    # Numeric stats
    if series.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
        if len(non_null) > 0:
            profile.min_value = str(non_null.min())
            profile.max_value = str(non_null.max())
            profile.mean_value = float(non_null.mean()) if non_null.mean() is not None else None
            profile.std_value = float(non_null.std()) if non_null.std() is not None else None
    elif series.dtype == pl.Utf8:
        if len(non_null) > 0:
            lengths = non_null.str.len_bytes()
            profile.min_value = f"{lengths.min()} chars"
            profile.max_value = f"{lengths.max()} chars"

    # Compute quality score based on statistics
    score = 100
    if null_pct > 50:
        score -= 30
    elif null_pct > 20:
        score -= 15
    elif null_pct > 5:
        score -= 5

    if unique_pct < 1 and total_rows > 100:
        score -= 10  # Very low cardinality might indicate data issues

    profile.quality_score = max(0, score)
    return profile


def _enrich_with_llm(
    df: pl.DataFrame,
    columns: list[ColumnProfile],
    dataset_name: str,
    llm: LLMGateway,
    sample_size: int,
) -> dict:
    """Use LLM to add semantic descriptions to column profiles."""
    # Build concise representation
    col_info = []
    for col in columns:
        col_info.append({
            "name": col.name,
            "dtype": col.dtype,
            "nulls": f"{col.null_percentage}%",
            "unique": col.unique_count,
            "samples": col.sample_values[:3],
            "range": f"{col.min_value} - {col.max_value}" if col.min_value else None,
        })

    sample = df.head(sample_size).to_dicts()

    prompt = (
        f"You are a data profiling expert. Analyze this dataset '{dataset_name}'.\n\n"
        f"Column statistics:\n```json\n{json.dumps(col_info, indent=2)}\n```\n\n"
        f"Data sample:\n```json\n{json.dumps(sample, indent=2, default=str)}\n```\n\n"
        "For each column, provide:\n"
        '1. "semantic_type": What kind of data (email, currency, name, date, ID, category, free_text, etc.)\n'
        '2. "description": One-sentence business description\n'
        '3. "issues": List of data quality issues found (empty list if none)\n\n'
        "Also provide:\n"
        '- "_summary": 2-3 sentence dataset summary\n'
        '- "_relationships": Detected relationships between columns\n'
        '- "_recommendations": Suggestions for cleaning/transformation\n\n'
        'Return JSON with column names as keys, plus _summary, _relationships, _recommendations.'
    )

    try:
        resp = llm.complete(prompt, temperature=0.0)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"LLM profiling failed: {e}")
        return {}
