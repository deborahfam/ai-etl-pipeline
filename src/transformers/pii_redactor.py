"""PII detection and redaction: hybrid regex + LLM approach."""

from __future__ import annotations

import json
import re

import polars as pl

from src.engine.models import PIIEntity, PIIReport
from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Regex patterns for common PII types
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


def redact_pii(
    df: pl.DataFrame,
    text_columns: list[str] | None = None,
    llm: LLMGateway | None = None,
    use_regex: bool = True,
    use_llm: bool = True,
    redaction_char: str = "***",
    sample_for_llm: int = 30,
) -> tuple[pl.DataFrame, PIIReport]:
    """Detect and redact PII from text columns.

    Uses a two-phase approach:
    1. Regex: Fast pattern matching for structured PII (emails, phones, SSNs)
    2. LLM: Contextual detection for unstructured PII (names, addresses)

    Args:
        df: Input DataFrame.
        text_columns: Columns to scan (default: all string columns).
        llm: LLMGateway for contextual PII detection.
        use_regex: Enable regex-based detection.
        use_llm: Enable LLM-based detection.
        redaction_char: Replacement string for detected PII.
        sample_for_llm: Number of rows to sample for LLM analysis.
    """
    if text_columns is None:
        text_columns = [c for c in df.columns if df[c].dtype == pl.Utf8]

    all_entities: list[PIIEntity] = []
    columns_affected: set[str] = set()

    # Phase 1: Regex-based detection
    if use_regex:
        for col in text_columns:
            for pii_type, pattern in PII_PATTERNS.items():
                compiled = re.compile(pattern)
                for idx, value in enumerate(df[col].to_list()):
                    if value is None:
                        continue
                    matches = compiled.findall(str(value))
                    for match in matches:
                        all_entities.append(
                            PIIEntity(
                                text=match,
                                pii_type=pii_type,
                                location=f"row {idx}, col {col}",
                                row_index=idx,
                                column=col,
                                redacted_value=redaction_char,
                            )
                        )
                        columns_affected.add(col)

    # Phase 2: LLM-based contextual detection
    if use_llm and llm is not None:
        llm_entities = _detect_pii_with_llm(df, text_columns, llm, sample_for_llm)
        all_entities.extend(llm_entities)
        for e in llm_entities:
            columns_affected.add(e.column)

    # Apply redactions
    df_redacted = df.clone()
    for entity in all_entities:
        col = entity.column
        if col in df_redacted.columns:
            df_redacted = df_redacted.with_columns(
                pl.col(col)
                .str.replace_all(re.escape(entity.text), redaction_char, literal=True)
                .alias(col)
            )

    report = PIIReport(
        total_pii_found=len(all_entities),
        entities=all_entities,
        columns_affected=sorted(columns_affected),
        summary=f"Found {len(all_entities)} PII instances across {len(columns_affected)} columns",
    )

    logger.info(f"PII redaction: found {len(all_entities)} entities in {len(columns_affected)} columns")
    return df_redacted, report


def _detect_pii_with_llm(
    df: pl.DataFrame,
    text_columns: list[str],
    llm: LLMGateway,
    sample_size: int,
) -> list[PIIEntity]:
    """Use LLM to detect contextual PII that regex misses."""
    entities: list[PIIEntity] = []

    sample = df.head(sample_size)
    sample_data = []
    for idx in range(len(sample)):
        row_texts = {}
        for col in text_columns:
            val = sample[col][idx]
            if val is not None:
                row_texts[col] = str(val)[:300]
        if row_texts:
            sample_data.append({"row": idx, "texts": row_texts})

    if not sample_data:
        return entities

    prompt = (
        "You are a PII detection expert. Analyze these text fields and identify ALL personally "
        "identifiable information (PII) including:\n"
        "- Person names (first, last, full names)\n"
        "- Physical addresses\n"
        "- Phone numbers (any format)\n"
        "- Dates of birth\n"
        "- Financial account numbers\n"
        "- Medical information\n"
        "- Any other identifying information\n\n"
        f"Data:\n```json\n{json.dumps(sample_data, indent=2)}\n```\n\n"
        'Return JSON array: [{"text": "exact PII text", "type": "PERSON|ADDRESS|PHONE|DOB|FINANCIAL|MEDICAL|OTHER", '
        '"row": row_index, "column": "column_name"}]\n'
        "If no PII found, return []"
    )

    try:
        resp = llm.complete(prompt, temperature=0.0)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        detected = json.loads(text)

        if isinstance(detected, list):
            for item in detected:
                entities.append(
                    PIIEntity(
                        text=item.get("text", ""),
                        pii_type=item.get("type", "OTHER").lower(),
                        location=f"row {item.get('row', 0)}, col {item.get('column', '')}",
                        row_index=item.get("row", 0),
                        column=item.get("column", ""),
                    )
                )
    except Exception as e:
        logger.warning(f"LLM PII detection failed: {e}")

    return entities
