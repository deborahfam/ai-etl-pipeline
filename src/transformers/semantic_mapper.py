"""LLM-powered schema mapping between heterogeneous data sources."""

from __future__ import annotations

import json

import polars as pl
from pydantic import BaseModel, Field

from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SchemaMapping(BaseModel):
    """Mapping between source and target column names."""
    source_column: str
    target_column: str
    confidence: float = Field(ge=0.0, le=1.0)
    transform_hint: str = ""


class SchemaMappingResult(BaseModel):
    mappings: list[SchemaMapping] = Field(default_factory=list)
    unmapped_source: list[str] = Field(default_factory=list)
    unmapped_target: list[str] = Field(default_factory=list)
    notes: str = ""


def map_schemas(
    source_df: pl.DataFrame,
    target_columns: list[str] | None = None,
    target_df: pl.DataFrame | None = None,
    llm: LLMGateway | None = None,
    context: str = "",
) -> SchemaMappingResult:
    """Use LLM to infer column mappings between source and target schemas.

    The LLM analyzes column names and sample data to suggest
    how source columns map to target columns.
    """
    if llm is None:
        raise ValueError("LLM gateway is required for semantic schema mapping")

    if target_columns is None and target_df is not None:
        target_columns = target_df.columns
    elif target_columns is None:
        raise ValueError("Either target_columns or target_df must be provided")

    # Build context with sample data
    source_sample = source_df.head(5).to_dicts()
    source_info = {
        "columns": source_df.columns,
        "dtypes": {col: str(source_df[col].dtype) for col in source_df.columns},
        "sample": source_sample,
    }

    target_info = {"columns": target_columns}
    if target_df is not None:
        target_info["dtypes"] = {col: str(target_df[col].dtype) for col in target_df.columns}
        target_info["sample"] = target_df.head(5).to_dicts()

    prompt = (
        "You are a data integration expert. Map source columns to target columns.\n\n"
        f"Source schema:\n```json\n{json.dumps(source_info, indent=2, default=str)}\n```\n\n"
        f"Target schema:\n```json\n{json.dumps(target_info, indent=2, default=str)}\n```\n\n"
        f"Context: {context}\n\n" if context else ""
        "For each source column, find the best matching target column.\n"
        "Consider: column names, data types, sample values, semantic meaning.\n"
        "If a source column has no good match, leave it unmapped.\n\n"
        "Return JSON:\n"
        '{"mappings": [{"source_column": "...", "target_column": "...", "confidence": 0.0-1.0, '
        '"transform_hint": "any transformation needed"}], '
        '"unmapped_source": ["cols with no match"], '
        '"unmapped_target": ["target cols with no source"], '
        '"notes": "any important observations"}'
    )

    resp = llm.complete(prompt, temperature=0.0)
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = SchemaMappingResult.model_validate(json.loads(text))
    logger.info(
        f"Schema mapping: {len(result.mappings)} mapped, "
        f"{len(result.unmapped_source)} unmapped source, "
        f"{len(result.unmapped_target)} unmapped target"
    )
    return result


def apply_mapping(
    df: pl.DataFrame,
    mapping: SchemaMappingResult,
    min_confidence: float = 0.5,
) -> pl.DataFrame:
    """Apply schema mapping to rename columns."""
    rename_map = {
        m.source_column: m.target_column
        for m in mapping.mappings
        if m.confidence >= min_confidence and m.source_column in df.columns
    }
    return df.rename(rename_map)
