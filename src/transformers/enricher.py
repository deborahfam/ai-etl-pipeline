"""LLM-powered data enrichment: sentiment analysis, entity extraction, classification."""

from __future__ import annotations

import json
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BatchEnrichment(BaseModel):
    """Schema for batch enrichment of multiple text records."""
    results: list[dict[str, Any]] = Field(default_factory=list)


def enrich_text_column(
    df: pl.DataFrame,
    text_column: str,
    llm: LLMGateway,
    operations: list[str] | None = None,
    batch_size: int = 10,
    language: str = "auto",
) -> pl.DataFrame:
    """Enrich a text column with AI-powered analysis.

    Operations available:
        - sentiment: Positive/negative/neutral with score
        - entities: Named entity recognition
        - category: Topic/category classification
        - language: Language detection
        - summary: Brief summary of text

    Args:
        df: Input DataFrame.
        text_column: Column containing text to analyze.
        llm: LLMGateway instance.
        operations: List of enrichment operations (default: all).
        batch_size: Number of records to process per LLM call.
        language: Expected language or "auto" for detection.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    ops = operations or ["sentiment", "entities", "category", "language"]

    texts = df[text_column].to_list()
    all_results: list[dict[str, Any]] = []

    # Process in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_idx = [j for j in range(i, min(i + batch_size, len(texts)))]
        logger.info(f"Enriching batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

        prompt = _build_enrichment_prompt(batch, batch_idx, ops, language)
        system = (
            "You are a data enrichment engine. Analyze each text and return structured results. "
            "Be precise and consistent. For sentiment scores, use -1.0 to 1.0 range."
        )

        try:
            resp = llm.complete(prompt, system=system, temperature=0.0)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            batch_results = json.loads(text)
            if isinstance(batch_results, dict) and "results" in batch_results:
                batch_results = batch_results["results"]
            all_results.extend(batch_results)
        except Exception as e:
            logger.error(f"Enrichment batch failed: {e}")
            # Fill with empty results
            for _ in batch:
                all_results.append({op: None for op in ops})

    # Pad results if needed
    while len(all_results) < len(df):
        all_results.append({op: None for op in ops})

    # Add enrichment columns to DataFrame
    if "sentiment" in ops:
        df = df.with_columns(
            pl.Series("sentiment_label", [_safe_get(r, "sentiment", "label", "unknown") for r in all_results]),
            pl.Series("sentiment_score", [_safe_get_float(r, "sentiment", "score") for r in all_results]),
        )

    if "entities" in ops:
        df = df.with_columns(
            pl.Series("entities", [json.dumps(_safe_get(r, "entities", default=[])) for r in all_results]),
        )

    if "category" in ops:
        df = df.with_columns(
            pl.Series("category", [_safe_get_str(r, "category") for r in all_results]),
        )

    if "language" in ops:
        df = df.with_columns(
            pl.Series("detected_language", [_safe_get_str(r, "language") for r in all_results]),
        )

    if "summary" in ops:
        df = df.with_columns(
            pl.Series("summary", [_safe_get_str(r, "summary") for r in all_results]),
        )

    logger.info(f"Enrichment complete: added columns for {ops}")
    return df


def _build_enrichment_prompt(
    texts: list[str], indices: list[int], operations: list[str], language: str
) -> str:
    ops_desc = {
        "sentiment": '"sentiment": {"label": "positive|negative|neutral|mixed", "score": float(-1 to 1)}',
        "entities": '"entities": [{"text": "...", "type": "PERSON|ORG|PRODUCT|LOCATION|DATE|MONEY"}]',
        "category": '"category": "main topic category string"',
        "language": '"language": "ISO 639-1 code (en, es, fr, de, etc.)"',
        "summary": '"summary": "one-sentence summary"',
    }

    schema_parts = [ops_desc[op] for op in operations if op in ops_desc]
    schema = "{\n  " + ",\n  ".join(schema_parts) + "\n}"

    texts_block = "\n".join(
        f"[{idx}] {text[:500]}" for idx, text in zip(indices, texts)
    )

    return (
        f"Analyze each text below and return a JSON array called 'results' with one object per text.\n\n"
        f"Expected schema per result:\n{schema}\n\n"
        f"Texts:\n{texts_block}\n\n"
        f'Return ONLY: {{"results": [...]}}'
    )


def _safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dict."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def _safe_get_str(d: dict, key: str) -> str:
    val = d.get(key, "")
    if isinstance(val, dict):
        return val.get("label", val.get("value", str(val)))
    return str(val) if val else ""


def _safe_get_float(d: dict, *keys: str) -> float:
    val = _safe_get(d, *keys, default=0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0
