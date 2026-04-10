"""Multi-dimensional Data Quality Scorer with LLM explanation."""

from __future__ import annotations

import json

import polars as pl

from src.engine.models import QualityDimension, QualityReport
from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


def score_quality(
    df: pl.DataFrame,
    dataset_name: str = "dataset",
    llm: LLMGateway | None = None,
) -> QualityReport:
    """Score dataset quality across 5 dimensions.

    Dimensions:
        1. Completeness: % of non-null values
        2. Consistency: Format uniformity across columns
        3. Uniqueness: Duplicate detection
        4. Accuracy: Value plausibility (requires LLM)
        5. Freshness: Data recency (if date columns exist)
    """
    dimensions: list[QualityDimension] = []

    # 1. Completeness
    total_cells = len(df) * len(df.columns)
    null_cells = sum(df[col].null_count() for col in df.columns)
    completeness_score = int((1 - null_cells / total_cells) * 100) if total_cells > 0 else 100
    null_cols = {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    dimensions.append(QualityDimension(
        name="Completeness",
        score=completeness_score,
        details=f"{null_cells}/{total_cells} cells are null. Columns with nulls: {null_cols}" if null_cols else "No null values found",
    ))

    # 2. Consistency
    consistency_issues = []
    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    for col in string_cols:
        non_null = df[col].drop_nulls()
        if len(non_null) < 2:
            continue
        # Check for mixed case patterns
        has_upper = non_null.str.contains(r"[A-Z]").sum()
        has_lower = non_null.str.contains(r"[a-z]").sum()
        if has_upper > 0 and has_lower > 0 and has_upper < len(non_null) * 0.9:
            consistency_issues.append(f"{col}: mixed casing")

        # Check for leading/trailing whitespace
        trimmed = non_null.str.strip_chars()
        whitespace_count = (non_null != trimmed).sum()
        if whitespace_count > 0:
            consistency_issues.append(f"{col}: {whitespace_count} values with extra whitespace")

    consistency_score = max(0, 100 - len(consistency_issues) * 10)
    dimensions.append(QualityDimension(
        name="Consistency",
        score=consistency_score,
        details="; ".join(consistency_issues) if consistency_issues else "All columns have consistent formatting",
    ))

    # 3. Uniqueness
    total_rows = len(df)
    unique_rows = len(df.unique())
    dup_count = total_rows - unique_rows
    uniqueness_score = int((unique_rows / total_rows) * 100) if total_rows > 0 else 100
    dimensions.append(QualityDimension(
        name="Uniqueness",
        score=uniqueness_score,
        details=f"{dup_count} duplicate rows found" if dup_count > 0 else "No duplicate rows",
    ))

    # 4. Accuracy (statistical heuristics; LLM enhances this)
    accuracy_issues = []
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    for col in numeric_cols:
        series = df[col].drop_nulls()
        if len(series) < 5:
            continue
        # Detect extreme outliers
        mean = series.mean()
        std = series.std()
        if std and mean and std > 0:
            z_scores = ((series - mean) / std).abs()
            extreme = (z_scores > 5).sum()
            if extreme > 0:
                accuracy_issues.append(f"{col}: {extreme} extreme outliers (z>5)")

        # Detect negative values in positive-named columns
        if any(kw in col.lower() for kw in ["price", "amount", "total", "quantity"]):
            negatives = (series < 0).sum()
            if negatives > 0:
                accuracy_issues.append(f"{col}: {negatives} unexpected negative values")

    accuracy_score = max(0, 100 - len(accuracy_issues) * 15)
    dimensions.append(QualityDimension(
        name="Accuracy",
        score=accuracy_score,
        details="; ".join(accuracy_issues) if accuracy_issues else "No obvious accuracy issues detected",
    ))

    # 5. Freshness
    date_cols = [c for c in df.columns if df[c].dtype in (pl.Date, pl.Datetime)]
    if date_cols:
        import datetime as dt
        now = dt.datetime.now()
        freshness_details = []
        freshness_score = 100
        for col in date_cols:
            max_date = df[col].drop_nulls().max()
            if max_date:
                if isinstance(max_date, dt.date) and not isinstance(max_date, dt.datetime):
                    max_date = dt.datetime.combine(max_date, dt.time())
                age_days = (now - max_date).days
                freshness_details.append(f"{col}: latest date is {age_days} days ago")
                if age_days > 365:
                    freshness_score = min(freshness_score, 40)
                elif age_days > 90:
                    freshness_score = min(freshness_score, 70)
        dimensions.append(QualityDimension(
            name="Freshness",
            score=freshness_score,
            details="; ".join(freshness_details),
        ))
    else:
        dimensions.append(QualityDimension(
            name="Freshness",
            score=50,
            details="No date columns found to assess freshness",
        ))

    # Overall score (weighted average)
    weights = {"Completeness": 0.25, "Consistency": 0.20, "Uniqueness": 0.20, "Accuracy": 0.25, "Freshness": 0.10}
    overall = int(sum(
        d.score * weights.get(d.name, 0.2) for d in dimensions
    ))

    # LLM summary
    summary = ""
    critical_issues: list[str] = []
    recommendations: list[str] = []

    if llm is not None:
        summary, critical_issues, recommendations = _llm_quality_summary(
            df, dimensions, dataset_name, llm
        )

    return QualityReport(
        dataset_name=dataset_name,
        overall_score=overall,
        dimensions=dimensions,
        summary=summary or f"Quality score: {overall}/100",
        critical_issues=critical_issues,
        recommendations=recommendations,
    )


def _llm_quality_summary(
    df: pl.DataFrame,
    dimensions: list[QualityDimension],
    dataset_name: str,
    llm: LLMGateway,
) -> tuple[list[str], list[str], list[str]]:
    """Use LLM to generate quality summary and recommendations."""
    dim_info = [{"name": d.name, "score": d.score, "details": d.details} for d in dimensions]
    sample = df.head(10).to_dicts()

    prompt = (
        f"Analyze data quality for '{dataset_name}'.\n\n"
        f"Quality dimensions:\n```json\n{json.dumps(dim_info, indent=2)}\n```\n\n"
        f"Data sample:\n```json\n{json.dumps(sample, indent=2, default=str)}\n```\n\n"
        "Return JSON with:\n"
        '- "summary": 2-3 sentence natural language quality assessment\n'
        '- "critical_issues": List of critical issues requiring immediate attention\n'
        '- "recommendations": List of specific actionable recommendations\n'
    )

    try:
        resp = llm.complete(prompt, temperature=0.0)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return (
            result.get("summary", ""),
            result.get("critical_issues", []),
            result.get("recommendations", []),
        )
    except Exception as e:
        logger.warning(f"LLM quality summary failed: {e}")
        return "", [], []
