"""Anomaly Detector: statistical detection + LLM explanation."""

from __future__ import annotations

import json

import polars as pl

from src.engine.models import (
    Anomaly,
    AnomalyReport,
    AnomalyType,
    SeverityLevel,
)
from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


def detect_anomalies(
    df: pl.DataFrame,
    dataset_name: str = "dataset",
    llm: LLMGateway | None = None,
    numeric_columns: list[str] | None = None,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    business_rules: list[dict] | None = None,
    max_anomalies_for_llm: int = 30,
) -> AnomalyReport:
    """Two-phase anomaly detection: statistical + LLM explanation.

    Phase 1: Statistical anomaly detection
        - Z-score outliers for numeric columns
        - IQR-based outliers
        - Business rule violations
        - Null pattern anomalies
        - Date anomalies (future dates, impossible dates)

    Phase 2: LLM explanation
        - Classifies each anomaly (error, fraud, legitimate, system)
        - Explains in natural language why it's anomalous
        - Recommends actions
    """
    anomalies: list[Anomaly] = []

    # Auto-detect numeric columns
    if numeric_columns is None:
        numeric_columns = [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

    # Phase 1a: Z-score outliers
    for col in numeric_columns:
        series = df[col].drop_nulls()
        if len(series) < 10:
            continue
        mean = series.mean()
        std = series.std()
        if std is None or std == 0 or mean is None:
            continue

        for idx in range(len(df)):
            val = df[col][idx]
            if val is None:
                continue
            z_score = abs((val - mean) / std)
            if z_score > z_threshold:
                anomalies.append(Anomaly(
                    row_index=idx,
                    column=col,
                    value=str(val),
                    anomaly_type=AnomalyType.DATA_ERROR,
                    severity=SeverityLevel.WARNING if z_score < 5 else SeverityLevel.CRITICAL,
                    explanation=f"Z-score: {z_score:.2f} (threshold: {z_threshold})",
                    confidence=min(1.0, z_score / 10),
                ))

    # Phase 1b: IQR outliers
    for col in numeric_columns:
        series = df[col].drop_nulls()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        for idx in range(len(df)):
            val = df[col][idx]
            if val is None:
                continue
            if val < lower or val > upper:
                # Avoid duplicating z-score anomalies
                existing = any(
                    a.row_index == idx and a.column == col for a in anomalies
                )
                if not existing:
                    anomalies.append(Anomaly(
                        row_index=idx,
                        column=col,
                        value=str(val),
                        anomaly_type=AnomalyType.DATA_ERROR,
                        severity=SeverityLevel.WARNING,
                        explanation=f"IQR outlier: value {val} outside [{lower:.2f}, {upper:.2f}]",
                        confidence=0.7,
                    ))

    # Phase 1c: Business rule violations
    if business_rules:
        for rule in business_rules:
            rule_name = rule.get("name", "custom_rule")
            condition = rule.get("condition")  # Polars expression string
            if condition:
                try:
                    violated = df.filter(pl.sql_expr(condition))
                    for idx in range(len(violated)):
                        # Find original index
                        anomalies.append(Anomaly(
                            row_index=idx,
                            column=rule.get("column", "multiple"),
                            value=str(violated.row(idx)),
                            anomaly_type=AnomalyType.INCONSISTENCY,
                            severity=SeverityLevel.WARNING,
                            explanation=f"Business rule violation: {rule_name}",
                            confidence=0.9,
                        ))
                except Exception as e:
                    logger.warning(f"Business rule '{rule_name}' failed: {e}")

    # Phase 1d: Detect negative values in typically-positive columns
    positive_hints = ["price", "amount", "total", "quantity", "count", "cost", "revenue", "salary"]
    for col in numeric_columns:
        col_lower = col.lower()
        if any(hint in col_lower for hint in positive_hints):
            negatives = df.filter(pl.col(col) < 0)
            for i in range(len(negatives)):
                val = negatives[col][i]
                anomalies.append(Anomaly(
                    row_index=i,
                    column=col,
                    value=str(val),
                    anomaly_type=AnomalyType.DATA_ERROR,
                    severity=SeverityLevel.CRITICAL,
                    explanation=f"Negative value in column '{col}' which should be positive",
                    confidence=0.95,
                ))

    # Phase 1e: Detect duplicates with different IDs
    id_cols = [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    non_id_cols = [c for c in df.columns if c not in id_cols and c not in numeric_columns]
    if id_cols and len(non_id_cols) >= 2:
        check_cols = non_id_cols[:5]  # Check first 5 non-ID columns
        dupes = df.filter(pl.struct(check_cols).is_duplicated())
        if len(dupes) > 0 and len(dupes) < len(df) * 0.1:
            for i in range(min(len(dupes), 10)):
                anomalies.append(Anomaly(
                    row_index=i,
                    column=",".join(id_cols),
                    value="duplicate content with different ID",
                    anomaly_type=AnomalyType.POTENTIAL_FRAUD,
                    severity=SeverityLevel.WARNING,
                    explanation=f"Row appears to be a duplicate based on non-ID columns {check_cols}",
                    confidence=0.6,
                ))

    # Limit anomalies for LLM processing
    anomalies = anomalies[:max_anomalies_for_llm * 2]

    # Phase 2: LLM explanation and classification
    if llm is not None and anomalies:
        anomalies = _explain_with_llm(df, anomalies, dataset_name, llm, max_anomalies_for_llm)

    # Count severities
    critical = sum(1 for a in anomalies if a.severity == SeverityLevel.CRITICAL)
    warning = sum(1 for a in anomalies if a.severity == SeverityLevel.WARNING)
    info = sum(1 for a in anomalies if a.severity == SeverityLevel.INFO)

    report = AnomalyReport(
        dataset_name=dataset_name,
        total_rows=len(df),
        anomalies=anomalies,
        summary=f"Found {len(anomalies)} anomalies: {critical} critical, {warning} warnings, {info} info",
        critical_count=critical,
        warning_count=warning,
        info_count=info,
    )

    logger.info(f"Anomaly detection: {len(anomalies)} anomalies found in {len(df)} rows")
    return report


def _explain_with_llm(
    df: pl.DataFrame,
    anomalies: list[Anomaly],
    dataset_name: str,
    llm: LLMGateway,
    max_anomalies: int,
) -> list[Anomaly]:
    """Use LLM to explain and re-classify anomalies."""
    to_explain = anomalies[:max_anomalies]

    anomaly_data = []
    for a in to_explain:
        row_data = {}
        if a.row_index < len(df):
            row = df.row(a.row_index, named=True)
            row_data = {k: str(v)[:100] for k, v in row.items()}

        anomaly_data.append({
            "row_index": a.row_index,
            "column": a.column,
            "value": a.value,
            "statistical_reason": a.explanation,
            "row_context": row_data,
        })

    prompt = (
        f"You are a data quality expert analyzing anomalies in '{dataset_name}'.\n\n"
        f"Anomalies detected:\n```json\n{json.dumps(anomaly_data, indent=2, default=str)}\n```\n\n"
        "For each anomaly, provide:\n"
        '1. "type": One of: data_error, potential_fraud, legitimate_outlier, system_error, inconsistency\n'
        '2. "severity": One of: critical, warning, info\n'
        '3. "explanation": Human-readable explanation of WHY this is anomalous\n'
        '4. "action": Recommended action (investigate, correct, delete, flag, ignore)\n'
        '5. "confidence": 0.0-1.0 how confident you are this is a real anomaly\n\n'
        'Return JSON array with one object per anomaly, in the same order.'
    )

    try:
        resp = llm.complete(prompt, temperature=0.0)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        explanations = json.loads(text)

        if isinstance(explanations, dict) and "results" in explanations:
            explanations = explanations["results"]

        for i, (anomaly, expl) in enumerate(zip(to_explain, explanations)):
            try:
                anomaly.anomaly_type = AnomalyType(expl.get("type", anomaly.anomaly_type.value))
            except ValueError:
                pass
            try:
                anomaly.severity = SeverityLevel(expl.get("severity", anomaly.severity.value))
            except ValueError:
                pass
            anomaly.explanation = expl.get("explanation", anomaly.explanation)
            anomaly.recommended_action = expl.get("action", "investigate")
            anomaly.confidence = float(expl.get("confidence", anomaly.confidence))

    except Exception as e:
        logger.warning(f"LLM anomaly explanation failed: {e}")

    # Re-add unexplained anomalies
    remaining = anomalies[max_anomalies:]
    return to_explain + remaining
