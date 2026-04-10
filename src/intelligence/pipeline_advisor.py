"""Pipeline Advisor: LLM suggests optimal transformations for a dataset."""

from __future__ import annotations

import json

import polars as pl

from src.engine.models import PipelineAdvice, TransformSuggestion
from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


def advise_pipeline(
    df: pl.DataFrame,
    dataset_name: str = "dataset",
    llm: LLMGateway | None = None,
    target_use_case: str = "",
) -> PipelineAdvice:
    """Analyze a raw dataset and suggest an optimal ETL pipeline.

    The LLM examines column names, types, sample data, and statistics
    to recommend transformations, validations, and schema.
    """
    if llm is None:
        raise ValueError("LLM gateway required for pipeline advice")

    # Build comprehensive dataset summary
    col_info = []
    for col in df.columns:
        series = df[col]
        info = {
            "name": col,
            "dtype": str(series.dtype),
            "nulls": series.null_count(),
            "unique": series.n_unique(),
            "samples": [str(v) for v in series.drop_nulls().head(5).to_list()],
        }
        if series.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                info["min"] = str(non_null.min())
                info["max"] = str(non_null.max())
                info["mean"] = f"{non_null.mean():.2f}" if non_null.mean() is not None else None
        col_info.append(info)

    sample = df.head(10).to_dicts()

    use_case = f"\nTarget use case: {target_use_case}\n" if target_use_case else ""

    prompt = (
        f"You are a senior data engineer. Analyze this raw dataset '{dataset_name}' and "
        f"recommend an optimal ETL pipeline.{use_case}\n\n"
        f"Dataset info ({len(df)} rows, {len(df.columns)} columns):\n"
        f"```json\n{json.dumps(col_info, indent=2)}\n```\n\n"
        f"Data sample:\n```json\n{json.dumps(sample, indent=2, default=str)}\n```\n\n"
        "Provide:\n"
        '1. "dataset_summary": Brief analysis of what this data represents\n'
        '2. "suggested_transforms": Ordered list of recommended transformations:\n'
        '   Each with: "step_name", "description", "priority" (high/medium/low), "reason"\n'
        '3. "suggested_validations": List of validation checks to apply\n'
        '4. "suggested_schema": Recommended target schema {column: type}\n'
        '5. "warnings": Any concerns about data quality or usability\n\n'
        "Be specific and actionable. Consider: cleaning, dedup, type casting, "
        "normalization, enrichment, PII redaction, anomaly detection."
        "\nReturn ONLY valid JSON."
    )

    try:
        resp = llm.complete(prompt, temperature=0.0)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)

        transforms = []
        for t in result.get("suggested_transforms", []):
            transforms.append(TransformSuggestion(
                step_name=t.get("step_name", ""),
                description=t.get("description", ""),
                priority=t.get("priority", "medium"),
                reason=t.get("reason", ""),
            ))

        return PipelineAdvice(
            dataset_summary=result.get("dataset_summary", ""),
            suggested_transforms=transforms,
            suggested_validations=result.get("suggested_validations", []),
            suggested_schema=result.get("suggested_schema", {}),
            warnings=result.get("warnings", []),
        )

    except Exception as e:
        logger.error(f"Pipeline advice generation failed: {e}")
        return PipelineAdvice(
            dataset_summary=f"Analysis failed: {e}",
            warnings=[str(e)],
        )
